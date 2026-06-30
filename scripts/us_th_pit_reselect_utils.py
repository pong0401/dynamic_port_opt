from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynamic_factor_copula import (  # noqa: E402
    _parquet_column_names,
    build_factor_covariance,
    build_momentum_signal,
    compute_feature_table,
    compute_market_stress_signal,
    default_paths,
    get_set100_members_as_of,
    get_sp500_members_as_of,
    initialize_static_clusters,
    load_cached_market_data,
    load_overlay_compare_prices,
    load_set100_membership_intervals,
    load_sp500_membership_intervals,
    monthly_rebalance_dates,
    optimize_portfolio,
    optimize_risk_parity,
    run_dynamic_hmm,
    select_point_in_time_universe,
)
from run_us_th_joint_model import END_DATE, FEATURE_FLAGS, LOOKBACK_DAYS, N_CLUSTERS, START_DATE  # noqa: E402


PREFERRED_SHARE_CLASS = {
    "GOOGL": "GOOG",
}


def available_cached_columns(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(_parquet_column_names(str(path)))


def drop_duplicate_share_classes(pool: list[str]) -> list[str]:
    members = set(pool)
    return [
        ticker
        for ticker in pool
        if not (ticker in PREFERRED_SHARE_CLASS and PREFERRED_SHARE_CLASS[ticker] in members)
    ]


def load_full_us_th_thb_panel(
    include_overlay_assets: bool = False,
    overlay_asset_tickers: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str], list[str]]:
    paths = default_paths(ROOT)
    start_date = start_date or START_DATE
    end_date = end_date or END_DATE
    source_cols = available_cached_columns(paths.source_cache_root / "prices.parquet")
    extra_cols = available_cached_columns(paths.local_cache_root / "extra_prices.parquet")

    all_us = [
        ticker
        for ticker in load_sp500_membership_intervals(paths)["ticker"].dropna().astype(str).drop_duplicates()
        if ticker in source_cols
    ]
    all_th = [
        ticker
        for ticker in load_set100_membership_intervals(paths)["ticker"].dropna().astype(str).drop_duplicates()
        if ticker in extra_cols
    ]
    stock_tickers = list(dict.fromkeys(all_us + all_th))
    needed = stock_tickers + ["SPY", "^VIX"]
    cached_panel = load_cached_market_data(paths, tickers=needed)
    prices = cached_panel["prices"].loc[start_date:end_date].reindex(columns=needed).sort_index().ffill()
    volumes = cached_panel["volumes"].loc[start_date:end_date].reindex(columns=needed).fillna(0.0)

    overlay_asset_tickers = overlay_asset_tickers or ["GC=F", "BTC-USD"]
    overlay_tickers = list(dict.fromkeys(["SPY", *overlay_asset_tickers, "^VIX", "USDTHB=X"]))
    overlay_prices = load_overlay_compare_prices(
        paths,
        start_date=start_date,
        end_date=end_date,
        tickers=overlay_tickers,
    ).sort_index().ffill()
    fx = overlay_prices["USDTHB=X"].reindex(prices.index).ffill()

    us_price_df = prices.reindex(columns=[ticker for ticker in all_us if ticker in prices.columns]).mul(fx, axis=0)
    th_price_df = prices.reindex(columns=[ticker for ticker in all_th if ticker in prices.columns])

    frames = [us_price_df, th_price_df]
    if include_overlay_assets:
        overlay_asset_df = pd.DataFrame(index=prices.index)
        for ticker in overlay_asset_tickers:
            overlay_asset_df[ticker] = overlay_prices[ticker].reindex(prices.index).ffill().mul(fx)
        frames.append(overlay_asset_df)

    thb_prices = pd.concat(frames, axis=1)
    benchmark = overlay_prices["SPY"].reindex(prices.index).ffill().mul(fx).rename("benchmark")
    vol_proxy = overlay_prices["^VIX"].reindex(prices.index).ffill().rename("vol_proxy")
    common_index = thb_prices.index.intersection(benchmark.dropna().index).intersection(vol_proxy.dropna().index)
    thb_prices = thb_prices.reindex(common_index).ffill().dropna(how="all")
    volumes = volumes.reindex(thb_prices.index).reindex(columns=thb_prices.columns).fillna(0.0)
    benchmark = benchmark.reindex(thb_prices.index).ffill()
    vol_proxy = vol_proxy.reindex(thb_prices.index).ffill()
    return thb_prices, volumes, benchmark, vol_proxy, all_us, all_th


def build_asset_caps(
    us_tickers: list[str],
    th_tickers: list[str],
    gold_cap: float | None,
    btc_cap: float | None,
    us_cap: float,
    th_cap: float,
    bil_cap: float | None = None,
) -> dict[str, float]:
    caps: dict[str, float] = {}
    caps.update({ticker: us_cap for ticker in us_tickers})
    caps.update({ticker: th_cap for ticker in th_tickers})
    if gold_cap is not None:
        caps["GC=F"] = gold_cap
    if btc_cap is not None:
        caps["BTC-USD"] = btc_cap
    if bil_cap is not None:
        caps["BIL"] = bil_cap
    return caps


def weights_history_to_frame(history: dict[pd.Timestamp, pd.Series]) -> pd.DataFrame:
    rows = []
    for rebalance_date, weights in sorted(history.items()):
        rows.append(weights.rename(pd.Timestamp(rebalance_date)))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).fillna(0.0).sort_index(axis=1)


def run_joint_pit_reselect_model(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    benchmark: pd.Series,
    vol_proxy: pd.Series,
    us_all: list[str],
    th_all: list[str],
    us_assets: int,
    th_assets: int,
    objective_mode: str,
    max_weight: float,
    include_overlay_assets: bool = False,
    overlay_asset_tickers: list[str] | None = None,
    asset_caps: dict[str, float] | None = None,
    include_momentum: bool = True,
    include_momentum_features: bool | None = None,
    include_momentum_signal: bool | None = None,
    momentum_signal_mode: str = "mom_63",
) -> dict[str, object]:
    overlay_asset_tickers = overlay_asset_tickers or ["GC=F", "BTC-USD"]
    include_momentum_features = include_momentum if include_momentum_features is None else include_momentum_features
    include_momentum_signal = include_momentum if include_momentum_signal is None else include_momentum_signal
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    benchmark_ret = benchmark.pct_change(fill_method=None).rename("benchmark")
    vol_proxy_ret = vol_proxy.pct_change(fill_method=None).rename("vol_proxy")
    schedule = monthly_rebalance_dates(prices.index, lookback_days=LOOKBACK_DAYS, freq="ME")

    feature_history: dict[pd.Timestamp, pd.DataFrame] = {}
    market_stress_history: dict[pd.Timestamp, float] = {}
    universe_history: dict[pd.Timestamp, list[str]] = {}
    selected_split_history: dict[pd.Timestamp, dict[str, list[str]]] = {}

    for rebalance_date in schedule:
        loc = prices.index.get_loc(rebalance_date)
        train_index = prices.index[max(0, loc - LOOKBACK_DAYS + 1) : loc + 1]
        us_pool = [
            ticker
            for ticker in get_sp500_members_as_of(rebalance_date, default_paths(ROOT))
            if ticker in us_all and ticker in prices.columns
        ]
        us_pool = drop_duplicate_share_classes(us_pool)
        th_pool = [ticker for ticker in get_set100_members_as_of(rebalance_date, default_paths(ROOT)) if ticker in th_all and ticker in prices.columns]
        us_selected = select_point_in_time_universe(prices.loc[train_index], volumes.loc[train_index], us_pool, n_assets=us_assets)
        th_selected = select_point_in_time_universe(prices.loc[train_index], volumes.loc[train_index], th_pool, n_assets=th_assets)
        current_assets = list(dict.fromkeys(us_selected + th_selected))
        if include_overlay_assets:
            current_assets.extend([asset for asset in overlay_asset_tickers if asset in prices.columns])
        if not current_assets:
            continue
        train_returns = returns.reindex(train_index)[current_assets].dropna(axis=1, thresh=max(int(0.85 * len(train_index)), 60))
        if train_returns.shape[1] < max(N_CLUSTERS + 2, 6):
            continue
        current_assets = train_returns.columns.tolist()
        universe_history[rebalance_date] = current_assets
        selected_split_history[rebalance_date] = {
            "US": [ticker for ticker in current_assets if ticker in us_selected],
            "TH": [ticker for ticker in current_assets if ticker in th_selected],
            "Overlay": [ticker for ticker in current_assets if ticker in set(overlay_asset_tickers)],
        }
        market_stress_history[rebalance_date] = compute_market_stress_signal(
            benchmark_ret.reindex(train_index),
            vol_proxy_ret.reindex(train_index),
        )
        feature_table = compute_feature_table(
            train_returns,
            benchmark_ret.reindex(train_index),
            vol_proxy_ret.reindex(train_index),
            prices.reindex(train_index)[current_assets],
            include_momentum_features=include_momentum_features,
            feature_flags=FEATURE_FLAGS,
        )
        if feature_table.empty:
            continue
        feature_history[rebalance_date] = feature_table

    if not feature_history:
        raise RuntimeError("No feature history was available for the US/TH PIT reselect model.")

    first_date = min(feature_history)
    initial = initialize_static_clusters(feature_history[first_date], n_clusters=N_CLUSTERS)
    dynamic_state = run_dynamic_hmm(
        feature_history,
        initial_state=initial,
        gas_alpha=0.40,
        gas_beta=0.45,
        market_stress_history=market_stress_history,
        posterior_power=2.25,
    )
    static_post = pd.get_dummies(initial["labels"]).reindex(columns=range(N_CLUSTERS), fill_value=0.0)

    strategy_names = ["Equal Weight", "Risk Parity", "Static Copula", "Dynamic HMM Copula"]
    nav = {name: pd.Series(1.0, index=[schedule[0]]) for name in strategy_names}
    weights_history: dict[str, dict[pd.Timestamp, pd.Series]] = {name: {} for name in strategy_names}

    for idx, rebalance_date in enumerate(schedule[:-1]):
        next_date = schedule[idx + 1]
        if rebalance_date not in feature_history:
            continue
        loc = prices.index.get_loc(rebalance_date)
        train_index = prices.index[max(0, loc - LOOKBACK_DAYS + 1) : loc + 1]
        test_index = prices.index[(prices.index > rebalance_date) & (prices.index <= next_date)]
        if len(test_index) == 0:
            continue

        current_features = feature_history[rebalance_date]
        current_assets = current_features.index.tolist()
        train_returns = returns.loc[train_index, current_assets].dropna(how="all")
        bench_train = benchmark_ret.loc[train_index]
        if include_momentum_signal:
            momentum_signal = build_momentum_signal(current_features, mode=momentum_signal_mode)
        else:
            momentum_signal = pd.Series(0.0, index=current_features.index, dtype=float)

        eq_weights = pd.Series(1.0 / len(current_assets), index=current_assets)
        risk_parity_cov = train_returns.cov().reindex(index=current_assets, columns=current_assets).fillna(0.0)
        static_cov, _ = build_factor_covariance(
            train_returns,
            bench_train,
            static_post.reindex(current_assets).fillna(0.0),
            current_features,
            dynamic=False,
        )
        dyn_cov, _ = build_factor_covariance(
            train_returns,
            bench_train,
            dynamic_state["posterior_history"][rebalance_date].reindex(current_assets).fillna(0.0),
            current_features,
            dynamic=True,
            centroid_snapshot=dynamic_state["centroid_history"][rebalance_date],
        )

        active_caps = None
        if asset_caps is not None:
            active_caps = {asset: cap for asset, cap in asset_caps.items() if asset in current_assets}

        weights = {
            "Equal Weight": eq_weights,
            "Risk Parity": optimize_risk_parity(risk_parity_cov, max_weight=max_weight, asset_caps=active_caps),
            "Static Copula": optimize_portfolio(
                static_cov,
                momentum_signal,
                max_weight=max_weight,
                objective_mode=objective_mode,
                asset_caps=active_caps,
            ),
            "Dynamic HMM Copula": optimize_portfolio(
                dyn_cov,
                momentum_signal,
                max_weight=max_weight,
                objective_mode=objective_mode,
                asset_caps=active_caps,
            ),
        }
        period_returns = returns.reindex(test_index)[current_assets].fillna(0.0)
        for strategy, strategy_weights in weights.items():
            weights_history[strategy][rebalance_date] = strategy_weights
            weighted = period_returns.mul(strategy_weights, axis=1).sum(axis=1)
            starting_value = float(nav[strategy].iloc[-1])
            nav[strategy] = pd.concat([nav[strategy], starting_value * (1.0 + weighted).cumprod()])

    nav = {name: series[~series.index.duplicated(keep="last")].sort_index() for name, series in nav.items()}
    return {
        "nav": nav,
        "weights_history": weights_history,
        "universe_history": universe_history,
        "selected_split_history": selected_split_history,
        "initial_clusters": initial,
        "dynamic_state": dynamic_state,
        "optimizer_objective": objective_mode,
        "max_weight": max_weight,
        "asset_caps": asset_caps or {},
        "include_momentum": include_momentum,
        "include_momentum_features": include_momentum_features,
        "include_momentum_signal": include_momentum_signal,
        "momentum_signal_mode": momentum_signal_mode,
    }
