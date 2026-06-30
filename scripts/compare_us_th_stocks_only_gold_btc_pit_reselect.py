from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for path in [SRC, SCRIPTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dynamic_factor_copula import (  # noqa: E402
    apply_daily_exposure_overlay,
    compare_trend_exposure,
    compute_port_opt_style_metrics,
    curve_from_returns,
    default_paths,
    load_overlay_compare_prices,
)
from run_us_th_joint_model import (  # noqa: E402
    BEST_OBJECTIVE,
    DIME_STYLE_COMMISSION_BPS,
    END_DATE,
    SLIPPAGE_BPS,
    START_DATE,
    _returns_from_asset_exposure,
    _side_trigger_asset_exposure,
)
from us_th_pit_reselect_utils import load_full_us_th_thb_panel, run_joint_pit_reselect_model, weights_history_to_frame  # noqa: E402


STOCK_ONLY_NAME = "US/TH stocks only reduce risk shift active market fee+slippage PIT reselect"
GOLD_BTC_NAME = "Side trigger realloc to active stock side, fee+slippage PIT reselect"


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
    row["Selection Rule"] = "Full PIT reselect every rebalance"
    row["Reallocate Stock Sleeve"] = True
    row["Avg Monthly Traded Notional"] = float(traded[traded > 0].mean())
    return row


def main() -> None:
    paths = default_paths(ROOT)
    prices, volumes, benchmark, vol_proxy, us_all, th_all = load_full_us_th_thb_panel(include_overlay_assets=False)
    results = run_joint_pit_reselect_model(
        prices=prices,
        volumes=volumes,
        benchmark=benchmark,
        vol_proxy=vol_proxy,
        us_all=us_all,
        th_all=th_all,
        us_assets=30,
        th_assets=30,
        objective_mode=BEST_OBJECTIVE,
        max_weight=0.08,
        include_overlay_assets=False,
    )
    sleeve_weight_history = weights_history_to_frame(results["weights_history"]["Dynamic HMM Copula"])

    overlay_prices = load_overlay_compare_prices(
        paths,
        start_date=START_DATE,
        end_date=END_DATE,
        tickers=["SPY", "GC=F", "BTC-USD", "^VIX", "USDTHB=X"],
    ).dropna(subset=["SPY", "GC=F", "BTC-USD", "^VIX", "USDTHB=X"])
    set_index = pd.read_parquet(paths.local_cache_root / "extra_prices.parquet")["^SET.BK"].loc[START_DATE:END_DATE].ffill()
    gold_thb = overlay_prices["GC=F"].mul(overlay_prices["USDTHB=X"])
    btc_thb = overlay_prices["BTC-USD"].mul(overlay_prices["USDTHB=X"])

    asset_prices = prices.copy()
    asset_prices["GOLD"] = gold_thb.reindex(asset_prices.index).ffill()
    asset_prices["BTC"] = btc_thb.reindex(asset_prices.index).ffill()
    asset_returns = asset_prices.pct_change(fill_method=None).fillna(0.0)

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
    gold_exposure = compare_trend_exposure(overlay_prices["GC=F"], 0.50)
    btc_exposure = compare_trend_exposure(overlay_prices["BTC-USD"], 0.00)

    stock_only_exposure = _stock_only_side_trigger_exposure(
        prices=prices,
        sleeve_weight_history=sleeve_weight_history,
        us_exposure=us_exposure_df["Daily Exposure"],
        th_exposure=th_exposure_df["Daily Exposure"],
    )
    stock_only_returns, stock_only_traded = _returns_from_asset_exposure(
        asset_returns.reindex(columns=stock_only_exposure.columns, fill_value=0.0),
        stock_only_exposure,
        transaction_cost_bps=DIME_STYLE_COMMISSION_BPS + SLIPPAGE_BPS,
    )
    stock_only_curve = curve_from_returns(stock_only_returns)
    stock_only_row = _summary_row(stock_only_curve, stock_only_traded)

    gold_btc_exposure = _side_trigger_asset_exposure(
        prices=asset_prices,
        sleeve_weight_history=sleeve_weight_history,
        us_exposure=us_exposure_df["Daily Exposure"],
        th_exposure=th_exposure_df["Daily Exposure"],
        gold_exposure=gold_exposure,
        btc_exposure=btc_exposure,
        reallocate_stock_sleeve=True,
    )
    gold_btc_returns, gold_btc_traded = _returns_from_asset_exposure(
        asset_returns.reindex(columns=gold_btc_exposure.columns, fill_value=0.0),
        gold_btc_exposure,
        transaction_cost_bps=DIME_STYLE_COMMISSION_BPS + SLIPPAGE_BPS,
    )
    gold_btc_curve = curve_from_returns(gold_btc_returns)
    gold_btc_row = _summary_row(gold_btc_curve, gold_btc_traded)

    comparison = pd.DataFrame(
        [
            stock_only_row.rename(STOCK_ONLY_NAME),
            gold_btc_row.rename(GOLD_BTC_NAME),
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
            GOLD_BTC_NAME: gold_btc_curve,
        },
        axis=1,
    ).dropna(how="all")

    comparison.to_csv(paths.result_dir / "us_th_stocks_only_vs_gold_btc_side_trigger_pit_reselect_comparison_thb.csv")
    curves.to_csv(paths.result_dir / "us_th_stocks_only_vs_gold_btc_side_trigger_pit_reselect_curves_thb.csv")
    gold_btc_exposure.to_csv(paths.result_dir / "us_th_side_trigger_pit_reselect_daily_asset_exposure_fee_slippage_thb.csv")
    stock_only_row.to_frame().T.to_csv(paths.result_dir / "us_th_stocks_only_side_trigger_pit_reselect_fee_slippage_summary_thb.csv")
    print(comparison.to_string())


if __name__ == "__main__":
    main()
