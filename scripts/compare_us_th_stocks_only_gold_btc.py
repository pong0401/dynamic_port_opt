from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in [SRC, SCRIPTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dynamic_factor_copula import (  # noqa: E402
    apply_daily_exposure_overlay,
    compute_port_opt_style_metrics,
    curve_from_returns,
    default_paths,
    load_overlay_compare_prices,
)
from run_us_th_joint_model import (  # noqa: E402
    BEST_ASSET_SWEEP_CASE,
    BEST_OBJECTIVE,
    DIME_STYLE_COMMISSION_BPS,
    END_DATE,
    SLIPPAGE_BPS,
    START_DATE,
    _build_ranked_us_th_universe,
    _load_thb_panel,
    _returns_from_asset_exposure,
    _run_model_on_prices,
    _weights_history_to_frame,
)


STOCK_ONLY_NAME = "US/TH stocks only reduce risk shift active market fee+slippage"
GOLD_BTC_NAME = "Side trigger realloc to active stock side, fee+slippage"


def _stock_only_side_trigger_exposure(
    prices: pd.DataFrame,
    sleeve_weight_history: pd.DataFrame,
    us_exposure: pd.Series,
    th_exposure: pd.Series,
) -> pd.DataFrame:
    index = prices.index[prices.index >= sleeve_weight_history.index.min()]
    weights = sleeve_weight_history.reindex(index).ffill().fillna(0.0)
    us_signal = us_exposure.reindex(index).ffill().bfill()
    th_signal = th_exposure.reindex(index).ffill().bfill()
    us_cols = [column for column in weights.columns if not str(column).endswith(".BK")]
    th_cols = [column for column in weights.columns if str(column).endswith(".BK")]

    exposure = pd.DataFrame(0.0, index=index, columns=weights.columns, dtype=float)
    exposure[us_cols] = weights[us_cols].mul(us_signal, axis=0)
    exposure[th_cols] = weights[th_cols].mul(th_signal, axis=0)

    for dt in index:
        idle = 1.0 - float(exposure.loc[dt, weights.columns].sum())
        if idle <= 1e-12:
            continue
        eligible_cols = []
        if float(us_signal.loc[dt]) >= 0.999:
            eligible_cols.extend(us_cols)
        if float(th_signal.loc[dt]) >= 0.999:
            eligible_cols.extend(th_cols)
        eligible_base = weights.loc[dt, eligible_cols]
        eligible_base = eligible_base[eligible_base > 0.0]
        if eligible_base.sum() > 0:
            exposure.loc[dt, eligible_base.index] += idle * eligible_base / eligible_base.sum()

    exposure["CASH"] = (1.0 - exposure.sum(axis=1)).clip(lower=0.0)
    return exposure.sort_index(axis=1)


def _summary_row(curve: pd.Series, traded: pd.Series) -> pd.Series:
    row = compute_port_opt_style_metrics(curve, risk_free_rate=0.03)
    row["Start"] = curve.dropna().index.min().date().isoformat()
    row["End"] = curve.dropna().index.max().date().isoformat()
    row["Objective"] = BEST_OBJECTIVE
    row["Fee Bps"] = DIME_STYLE_COMMISSION_BPS + SLIPPAGE_BPS
    row["Reallocate Stock Sleeve"] = True
    row["Avg Monthly Traded Notional"] = float(traded[traded > 0].mean())
    return row


def main() -> None:
    paths = default_paths(ROOT)
    us_tickers, th_tickers = _build_ranked_us_th_universe(
        BEST_ASSET_SWEEP_CASE["us_assets"],
        BEST_ASSET_SWEEP_CASE["th_assets"],
    )
    tickers = list(dict.fromkeys(us_tickers + th_tickers))
    prices, volumes, benchmark, vol_proxy, _ = _load_thb_panel(tickers)
    results = _run_model_on_prices(
        prices,
        volumes,
        benchmark,
        vol_proxy,
        objective_mode=BEST_OBJECTIVE,
        max_weight=BEST_ASSET_SWEEP_CASE["max_weight"],
    )
    sleeve_weight_history = _weights_history_to_frame(results["weights_history"]["Dynamic HMM Copula"])

    overlay_prices = load_overlay_compare_prices(
        paths,
        start_date=START_DATE,
        end_date=END_DATE,
        tickers=["SPY", "GC=F", "BTC-USD", "^VIX", "USDTHB=X"],
    ).dropna(subset=["SPY", "GC=F", "BTC-USD", "^VIX", "USDTHB=X"])
    set_index = pd.read_parquet(paths.local_cache_root / "extra_prices.parquet")["^SET.BK"].loc[START_DATE:END_DATE].ffill()
    asset_returns = prices.pct_change(fill_method=None).fillna(0.0)
    sample_returns = pd.Series(0.0, index=asset_returns.index, name="dummy")
    _, us_exposure_df = apply_daily_exposure_overlay(
        sample_returns,
        overlay_prices["SPY"].reindex(asset_returns.index).ffill(),
        overlay_prices["^VIX"].reindex(asset_returns.index).ffill(),
    )
    _, th_exposure_df = apply_daily_exposure_overlay(
        sample_returns,
        set_index.reindex(asset_returns.index).ffill(),
        None,
    )
    exposure = _stock_only_side_trigger_exposure(
        prices=prices,
        sleeve_weight_history=sleeve_weight_history,
        us_exposure=us_exposure_df["Daily Exposure"],
        th_exposure=th_exposure_df["Daily Exposure"],
    )
    returns, traded = _returns_from_asset_exposure(
        asset_returns,
        exposure,
        transaction_cost_bps=DIME_STYLE_COMMISSION_BPS + SLIPPAGE_BPS,
    )
    stock_only_curve = curve_from_returns(returns)
    stock_only_row = _summary_row(stock_only_curve, traded)

    existing_summary = pd.read_csv(paths.result_dir / "us_th_side_trigger_reallocation_summary_thb.csv", index_col=0)
    existing_curves = pd.read_csv(
        paths.result_dir / "us_th_side_trigger_reallocation_curves_thb.csv",
        index_col=0,
        parse_dates=True,
    )
    gold_btc_row = existing_summary.loc[GOLD_BTC_NAME].copy()
    gold_btc_curve = existing_curves[GOLD_BTC_NAME].dropna()

    comparison = pd.DataFrame(
        [
            stock_only_row.rename(STOCK_ONLY_NAME),
            gold_btc_row.rename(f"{GOLD_BTC_NAME} (with Gold/BTC 60/30/10)"),
        ]
    )
    comparison.index.name = "Strategy"
    comparison["CAGR Delta vs Stocks Only"] = comparison["CAGR"] - float(comparison.loc[STOCK_ONLY_NAME, "CAGR"])
    comparison["Sharpe Delta vs Stocks Only"] = comparison["Sharpe"] - float(comparison.loc[STOCK_ONLY_NAME, "Sharpe"])
    comparison["Max DD Delta vs Stocks Only"] = comparison["Max Drawdown"] - float(comparison.loc[STOCK_ONLY_NAME, "Max Drawdown"])
    comparison = comparison.sort_values("Sharpe", ascending=False)

    curves = pd.concat(
        {
            STOCK_ONLY_NAME: stock_only_curve,
            f"{GOLD_BTC_NAME} (with Gold/BTC 60/30/10)": gold_btc_curve,
        },
        axis=1,
    ).dropna(how="all")

    comparison.to_csv(paths.result_dir / "us_th_stocks_only_vs_gold_btc_side_trigger_comparison_thb.csv")
    curves.to_csv(paths.result_dir / "us_th_stocks_only_vs_gold_btc_side_trigger_curves_thb.csv")
    exposure.to_csv(paths.result_dir / "us_th_stocks_only_side_trigger_daily_asset_exposure_fee_slippage_thb.csv")
    stock_only_row.to_frame().T.to_csv(paths.result_dir / "us_th_stocks_only_side_trigger_fee_slippage_summary_thb.csv")

    print(comparison.to_string())


if __name__ == "__main__":
    main()
