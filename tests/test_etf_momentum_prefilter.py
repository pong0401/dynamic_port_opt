from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_etf_momentum_prefilter_backtest import (  # noqa: E402
    ETFMomentumConfig,
    apply_momentum_filter,
    calculate_momentum_features,
    run_monthly_rebalance_backtest,
    run_optimizer_on_selected_universe,
)


def _price_panel(periods: int = 320) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=periods)
    x = np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "AAA": 100.0 * (1.0015 ** x),
            "BBB": 100.0 * (1.0010 ** x),
            "CCC": 100.0 * (1.0005 ** x),
            "DDD": 100.0 * (0.9995 ** x),
        },
        index=index,
    )


def test_momentum_features_do_not_use_future_prices() -> None:
    prices = _price_panel()
    rebalance_date = prices.index[260]
    baseline = calculate_momentum_features(prices, rebalance_date)

    changed_future = prices.copy()
    changed_future.loc[changed_future.index > rebalance_date, "AAA"] *= 50.0
    future_changed = calculate_momentum_features(changed_future, rebalance_date)

    pd.testing.assert_series_equal(
        baseline.set_index("ETF").loc["AAA", ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "sma200"]],
        future_changed.set_index("ETF").loc["AAA", ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "sma200"]],
        check_names=False,
    )


def test_insufficient_history_is_excluded_from_selection() -> None:
    prices = _price_panel()
    prices.loc[prices.index[:120], "BBB"] = np.nan
    features = calculate_momentum_features(prices, prices.index[-1])
    selected, ranked = apply_momentum_filter(features, top_n=3, min_n=1)

    assert "BBB" not in selected
    assert not bool(ranked.set_index("ETF").loc["BBB", "pass_momentum"])


def test_sma_filter_blocks_etf_below_sma200() -> None:
    prices = _price_panel()
    prices.loc[prices.index[-5:], "AAA"] *= 0.50
    features = calculate_momentum_features(prices, prices.index[-1])
    selected, ranked = apply_momentum_filter(features, top_n=3, min_n=1)

    assert not bool(ranked.set_index("ETF").loc["AAA", "pass_ma200"])
    assert "AAA" not in selected


def test_momentum_ranking_prefers_higher_score() -> None:
    momentum_df = pd.DataFrame(
        {
            "ETF": ["LOW", "HIGH", "MID"],
            "pass_momentum": [True, True, True],
            "momentum_score": [0.10, 0.90, 0.50],
        }
    )
    selected, ranked = apply_momentum_filter(momentum_df, top_n=2, min_n=1)

    assert ranked.iloc[0]["ETF"] == "HIGH"
    assert selected == ["HIGH", "MID"]


def test_optimizer_receives_only_selected_etfs() -> None:
    prices = _price_panel()
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    seen_assets: list[str] = []

    def spy_optimizer(cov: pd.DataFrame, momentum_signal: pd.Series, **kwargs) -> pd.Series:
        seen_assets.extend(cov.index.tolist())
        return pd.Series(1.0 / len(cov.index), index=cov.index)

    weights = run_optimizer_on_selected_universe(
        returns,
        ["AAA", "CCC"],
        prices.index[-1],
        ETFMomentumConfig(universe=("AAA", "BBB", "CCC", "DDD"), max_weight_per_asset=0.60),
        optimizer_fn=spy_optimizer,
    )

    assert seen_assets == ["AAA", "CCC"]
    assert set(weights.index) == {"AAA", "CCC"}


def test_fallback_to_cash_when_no_etf_passes_filter() -> None:
    prices = _price_panel()
    x = np.arange(len(prices), dtype=float)
    falling = pd.DataFrame(
        {column: 100.0 * (0.998 ** x) for column in prices.columns},
        index=prices.index,
    )
    config = ETFMomentumConfig(universe=tuple(falling.columns), top_n=3, min_n=1)
    results = run_monthly_rebalance_backtest(falling, config)
    weights = results["weight_history"]

    assert not weights.empty
    assert set(weights["ETF"].unique()) == {"CASH"}
    assert weights["weight"].eq(1.0).all()
