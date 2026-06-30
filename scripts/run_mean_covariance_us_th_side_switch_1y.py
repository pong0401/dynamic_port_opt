from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in [SRC, SCRIPTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dynamic_factor_copula import compute_metrics, default_paths  # noqa: E402
from run_mean_covariance_us_th_gold30_1y_lookback_latest_year import (  # noqa: E402
    BIL_CAP,
    BTC_CAP,
    DAILY_STRATEGY as BASE_DAILY_STRATEGY,
    GOLD_CAP,
    LOOKBACK_DAYS,
    OVERLAY_ASSETS,
    RISK_AVERSION,
    STOCK_CAP,
    TH_ASSETS,
    US_ASSETS,
    _sleeve_history,
    run_backtest,
)
from run_mean_covariance_us_th_overlay_gold30 import _best_signal_config, _close_trend_exposure  # noqa: E402


TH_MA_PERIODS = [50, 100, 150, 200, 250, 300]
TH_BELOW_EXPOSURES = [0.0, 0.25, 0.50, 0.75]
RESULT_PREFIX = "mean_covariance_us_th_gold30_stockcap8_1y_side_switch"


def _metrics_for_nav(nav: pd.Series, benchmark: pd.Series) -> pd.Series:
    benchmark_nav = benchmark.reindex(nav.index).ffill()
    benchmark_nav = benchmark_nav / benchmark_nav.iloc[0]
    return compute_metrics(nav, benchmark_nav=benchmark_nav)


def _trigger_exposures(
    prices: pd.DataFrame,
    benchmark: pd.Series,
    set_index: pd.Series,
    th_ma_period: int,
    th_below_exposure: float,
) -> pd.DataFrame:
    config = _best_signal_config()
    spy_cfg = config.loc["SPY"] if "SPY" in config.index else pd.Series({"MA Period": 300, "Below Exposure": 0.50})
    gold_cfg = config.loc["GOLD"] if "GOLD" in config.index else pd.Series({"MA Period": 50, "Below Exposure": 1.00})
    btc_cfg = config.loc["BTC"] if "BTC" in config.index else pd.Series({"MA Period": 50, "Below Exposure": 0.00})
    return pd.DataFrame(
        {
            "US Stock Exposure From SPY": _close_trend_exposure(
                benchmark,
                int(spy_cfg["MA Period"]),
                float(spy_cfg["Below Exposure"]),
            ),
            "TH Stock Exposure From SET": _close_trend_exposure(set_index, th_ma_period, th_below_exposure),
            "Gold Exposure From Gold Trend": _close_trend_exposure(
                prices["GC=F"],
                int(gold_cfg["MA Period"]),
                float(gold_cfg["Below Exposure"]),
            ),
            "BTC Exposure From BTC Trend": _close_trend_exposure(
                prices["BTC-USD"],
                int(btc_cfg["MA Period"]),
                float(btc_cfg["Below Exposure"]),
            ),
            "BIL Exposure": 1.0,
        }
    )


def _apply_cash_drag(
    daily_weights: pd.DataFrame,
    prices: pd.DataFrame,
    exposures: pd.DataFrame,
    benchmark: pd.Series,
    strategy: str,
) -> dict[str, object]:
    aligned_exposure = exposures.reindex(daily_weights.index).ffill().fillna(1.0)
    effective = pd.DataFrame(0.0, index=daily_weights.index, columns=daily_weights.columns, dtype=float)
    us_cols = [column for column in daily_weights.columns if not str(column).endswith(".BK") and column not in OVERLAY_ASSETS]
    th_cols = [column for column in daily_weights.columns if str(column).endswith(".BK")]
    if us_cols:
        effective[us_cols] = daily_weights[us_cols].mul(aligned_exposure["US Stock Exposure From SPY"], axis=0)
    if th_cols:
        effective[th_cols] = daily_weights[th_cols].mul(aligned_exposure["TH Stock Exposure From SET"], axis=0)
    if "GC=F" in daily_weights.columns:
        effective["GC=F"] = daily_weights["GC=F"] * aligned_exposure["Gold Exposure From Gold Trend"]
    if "BTC-USD" in daily_weights.columns:
        effective["BTC-USD"] = daily_weights["BTC-USD"] * aligned_exposure["BTC Exposure From BTC Trend"]
    if "BIL" in daily_weights.columns:
        effective["BIL"] = daily_weights["BIL"]
    effective["Cash / Reduced Exposure"] = (1.0 - effective.sum(axis=1)).clip(lower=0.0)
    asset_returns = prices.pct_change(fill_method=None).reindex(effective.index).reindex(columns=daily_weights.columns).fillna(0.0)
    returns = asset_returns.mul(effective.reindex(columns=daily_weights.columns), axis=1).sum(axis=1)
    nav = (1.0 + returns).cumprod().rename(strategy)
    metrics = _metrics_for_nav(nav, benchmark)
    metrics["Average Exposure"] = float(effective.reindex(columns=daily_weights.columns).sum(axis=1).mean())
    metrics["Minimum Exposure"] = float(effective.reindex(columns=daily_weights.columns).sum(axis=1).min())
    return {"nav": nav, "metrics": metrics, "effective_weights": effective}


def _apply_side_switch(
    daily_weights: pd.DataFrame,
    prices: pd.DataFrame,
    exposures: pd.DataFrame,
    benchmark: pd.Series,
    strategy: str,
) -> dict[str, object]:
    aligned_exposure = exposures.reindex(daily_weights.index).ffill().fillna(1.0)
    effective = pd.DataFrame(0.0, index=daily_weights.index, columns=daily_weights.columns, dtype=float)
    us_cols = [column for column in daily_weights.columns if not str(column).endswith(".BK") and column not in OVERLAY_ASSETS]
    th_cols = [column for column in daily_weights.columns if str(column).endswith(".BK")]

    for dt in daily_weights.index:
        base = daily_weights.loc[dt]
        us_base = base.reindex(us_cols).fillna(0.0)
        th_base = base.reindex(th_cols).fillna(0.0)
        us_sum = float(us_base.sum())
        th_sum = float(th_base.sum())
        stock_sum = us_sum + th_sum
        us_on = float(aligned_exposure.loc[dt, "US Stock Exposure From SPY"]) >= 0.999
        th_on = float(aligned_exposure.loc[dt, "TH Stock Exposure From SET"]) >= 0.999

        if us_on and th_on:
            effective.loc[dt, us_cols] = us_base
            effective.loc[dt, th_cols] = th_base
        elif (not us_on) and th_on and th_sum > 0.0:
            effective.loc[dt, th_cols] = stock_sum * th_base / th_sum
        elif us_on and (not th_on) and us_sum > 0.0:
            effective.loc[dt, us_cols] = stock_sum * us_base / us_sum
        else:
            effective.loc[dt, us_cols] = us_base * float(aligned_exposure.loc[dt, "US Stock Exposure From SPY"])
            effective.loc[dt, th_cols] = th_base * float(aligned_exposure.loc[dt, "TH Stock Exposure From SET"])

    if "GC=F" in daily_weights.columns:
        effective["GC=F"] = daily_weights["GC=F"] * aligned_exposure["Gold Exposure From Gold Trend"]
    if "BTC-USD" in daily_weights.columns:
        effective["BTC-USD"] = daily_weights["BTC-USD"] * aligned_exposure["BTC Exposure From BTC Trend"]
    if "BIL" in daily_weights.columns:
        effective["BIL"] = daily_weights["BIL"]
    effective["Cash / Reduced Exposure"] = (1.0 - effective.sum(axis=1)).clip(lower=0.0)
    asset_returns = prices.pct_change(fill_method=None).reindex(effective.index).reindex(columns=daily_weights.columns).fillna(0.0)
    returns = asset_returns.mul(effective.reindex(columns=daily_weights.columns), axis=1).sum(axis=1)
    nav = (1.0 + returns).cumprod().rename(strategy)
    metrics = _metrics_for_nav(nav, benchmark)
    metrics["Average Exposure"] = float(effective.reindex(columns=daily_weights.columns).sum(axis=1).mean())
    metrics["Minimum Exposure"] = float(effective.reindex(columns=daily_weights.columns).sum(axis=1).min())
    metrics["US Off TH On Days"] = int(((aligned_exposure["US Stock Exposure From SPY"] < 0.999) & (aligned_exposure["TH Stock Exposure From SET"] >= 0.999)).sum())
    metrics["TH Off US On Days"] = int(((aligned_exposure["TH Stock Exposure From SET"] < 0.999) & (aligned_exposure["US Stock Exposure From SPY"] >= 0.999)).sum())
    metrics["Both On Days"] = int(((aligned_exposure["US Stock Exposure From SPY"] >= 0.999) & (aligned_exposure["TH Stock Exposure From SET"] >= 0.999)).sum())
    metrics["Both Off Days"] = int(((aligned_exposure["US Stock Exposure From SPY"] < 0.999) & (aligned_exposure["TH Stock Exposure From SET"] < 0.999)).sum())
    return {"nav": nav, "metrics": metrics, "effective_weights": effective}


def _latest_frame(effective_weights: pd.DataFrame, strategy: str) -> pd.DataFrame:
    latest_date = pd.Timestamp(effective_weights.index.max())
    latest = effective_weights.iloc[-1].rename("Effective Weight").reset_index()
    latest.columns = ["Asset", "Effective Weight"]
    latest["Date"] = latest_date.date().isoformat()
    latest["Strategy"] = strategy
    latest["Sleeve"] = "US Equity"
    latest.loc[latest["Asset"].astype(str).str.endswith(".BK"), "Sleeve"] = "TH Equity"
    latest.loc[latest["Asset"].eq("GC=F"), "Sleeve"] = "Gold"
    latest.loc[latest["Asset"].eq("BTC-USD"), "Sleeve"] = "BTC"
    latest.loc[latest["Asset"].eq("BIL"), "Sleeve"] = "BIL"
    latest.loc[latest["Asset"].eq("Cash / Reduced Exposure"), "Sleeve"] = "Cash"
    return latest.sort_values("Effective Weight", ascending=False)


def main() -> None:
    paths = default_paths(ROOT)
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    base = run_backtest()
    daily_weights: pd.DataFrame = base["daily_weights"]
    prices: pd.DataFrame = base["prices"]
    benchmark: pd.Series = base["benchmark"]
    set_index: pd.Series = base["set_index"]

    param_rows = []
    param_curves = {}
    param_results = {}
    for ma_period in TH_MA_PERIODS:
        for below_exposure in TH_BELOW_EXPOSURES:
            strategy = f"TH SET daily exposure MA{ma_period} below{below_exposure:.0%} cash drag"
            exposures = _trigger_exposures(prices, benchmark, set_index, ma_period, below_exposure)
            result = _apply_cash_drag(daily_weights, prices, exposures, benchmark, strategy)
            row = result["metrics"].copy()
            row["Strategy"] = strategy
            row["TH MA Period"] = ma_period
            row["TH Below Exposure"] = below_exposure
            row["Start"] = result["nav"].index.min().date().isoformat()
            row["End"] = result["nav"].index.max().date().isoformat()
            param_rows.append(row)
            param_curves[strategy] = result["nav"]
            param_results[(ma_period, below_exposure)] = (exposures, result)

    param_summary = pd.DataFrame(param_rows).sort_values(["Sharpe", "CAGR"], ascending=False)
    best_param = param_summary.iloc[0]
    best_ma = int(best_param["TH MA Period"])
    best_below = float(best_param["TH Below Exposure"])
    best_exposures = param_results[(best_ma, best_below)][0]

    side_strategy = f"US+TH side-switch Mean Covariance Gold30 stock cap 8 1Y lookback THSET_MA{best_ma}_below{best_below:.0%}"
    side = _apply_side_switch(daily_weights, prices, best_exposures, benchmark, side_strategy)
    base_cash_strategy = f"{BASE_DAILY_STRATEGY} with optimized TH SET MA{best_ma}_below{best_below:.0%} cash drag"
    base_cash = param_results[(best_ma, best_below)][1]

    comparison_rows = []
    for strategy, result, variant in [
        (base_cash_strategy, base_cash, "Cash drag"),
        (side_strategy, side, "Side switch"),
    ]:
        row = result["metrics"].copy()
        row["Strategy"] = strategy
        row["Variant"] = variant
        row["TH MA Period"] = best_ma
        row["TH Below Exposure"] = best_below
        row["Start"] = result["nav"].index.min().date().isoformat()
        row["End"] = result["nav"].index.max().date().isoformat()
        row["US Assets"] = US_ASSETS
        row["TH Assets"] = TH_ASSETS
        row["Lookback Days"] = LOOKBACK_DAYS
        row["US Stock Cap"] = STOCK_CAP
        row["TH Stock Cap"] = STOCK_CAP
        row["Gold Cap"] = GOLD_CAP
        row["BTC Cap"] = BTC_CAP
        row["BIL Cap"] = BIL_CAP
        row["Risk Aversion"] = RISK_AVERSION
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows).sort_values(["Sharpe", "CAGR"], ascending=False)

    param_summary.to_csv(paths.result_dir / f"{RESULT_PREFIX}_th_set_param_sweep.csv", index=False)
    pd.DataFrame(param_curves).dropna(how="all").to_csv(paths.result_dir / f"{RESULT_PREFIX}_th_set_param_curves.csv")
    comparison.to_csv(paths.result_dir / f"{RESULT_PREFIX}_comparison.csv", index=False)
    pd.DataFrame({base_cash_strategy: base_cash["nav"], side_strategy: side["nav"]}).dropna(how="all").to_csv(
        paths.result_dir / f"{RESULT_PREFIX}_curves.csv"
    )
    best_exposures.reindex(daily_weights.index).ffill().to_csv(paths.result_dir / f"{RESULT_PREFIX}_trigger_exposure.csv")
    side["effective_weights"].to_csv(paths.result_dir / f"{RESULT_PREFIX}_side_switch_effective_weights.csv")
    _sleeve_history(side["effective_weights"]).to_csv(paths.result_dir / f"{RESULT_PREFIX}_side_switch_sleeve_weight_history.csv")
    _latest_frame(side["effective_weights"], side_strategy).to_csv(paths.result_dir / f"{RESULT_PREFIX}_side_switch_latest_weights.csv", index=False)

    print("Best TH SET daily exposure param")
    print(best_param.to_frame("Best").T.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nComparison")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nLatest side-switch weights")
    latest = _latest_frame(side["effective_weights"], side_strategy)
    print(latest.loc[latest["Effective Weight"] > 1e-8, ["Asset", "Sleeve", "Effective Weight"]].to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
