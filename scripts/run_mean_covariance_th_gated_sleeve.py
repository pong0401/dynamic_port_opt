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
    compute_port_opt_style_metrics,
    convert_usd_returns_to_local,
    curve_from_returns,
    default_paths,
    lag_close_signal_to_next_session,
    load_overlay_compare_prices,
)
from run_mean_covariance_gold30_with_th_sleeve import _run_th_sleeve  # noqa: E402


START_DATE = "2018-01-02"
END_DATE = "2026-04-29"
CORE_COLUMN = "Mean Covariance Gold30 stock-cap sweep stockcap8 mom_63 + asset-level daily exposure"
TH_WEIGHTS = [0.05, 0.10, 0.15, 0.20]
ABS_MA_PERIODS = [50, 75, 100, 125, 150, 175, 200, 250, 300]
RATIO_MA_PERIODS = [100, 150, 200]
MOM_PERIODS = [63, 126]
SLOPE_LOOKBACKS = [20, 40, 63]
RESULT_PREFIX = "mean_covariance_th_gated_sleeve"


def _load_core_returns_thb() -> pd.Series:
    paths = default_paths(ROOT)
    core_curve = pd.read_csv(
        paths.result_dir / "mean_covariance_stock_cap_sweep_daily_exposure_curves.csv",
        index_col=0,
        parse_dates=True,
    )[CORE_COLUMN].loc[START_DATE:END_DATE].dropna()
    fx = load_overlay_compare_prices(paths, start_date=START_DATE, end_date=END_DATE, tickers=["USDTHB=X"])["USDTHB=X"]
    fx_returns = fx.reindex(core_curve.index).ffill().pct_change(fill_method=None).fillna(0.0)
    core_returns_usd = core_curve.pct_change(fill_method=None).fillna(0.0)
    return convert_usd_returns_to_local(core_returns_usd, fx_returns).rename("CORE_THB")


def _load_market_signals(index: pd.DatetimeIndex) -> pd.DataFrame:
    paths = default_paths(ROOT)
    overlay = load_overlay_compare_prices(
        paths,
        start_date=START_DATE,
        end_date=END_DATE,
        tickers=["SPY", "USDTHB=X"],
    ).sort_index().ffill()
    spy_thb = overlay["SPY"].mul(overlay["USDTHB=X"]).reindex(index).ffill()
    set_index = pd.read_parquet(paths.local_cache_root / "extra_prices.parquet")["^SET.BK"].loc[START_DATE:END_DATE].reindex(index).ffill()
    return pd.DataFrame({"SPY_THB": spy_thb, "SET": set_index}).dropna()


def _gate_signal(
    market: pd.DataFrame,
    abs_ma: int,
    ratio_ma: int,
    mom_period: int,
    slope_lookback: int,
    mode: str,
) -> pd.Series:
    set_index = market["SET"]
    spy_thb = market["SPY_THB"]
    ratio = set_index / spy_thb
    set_ma = set_index.rolling(abs_ma, min_periods=max(20, abs_ma // 5)).mean()
    ratio_ma_series = ratio.rolling(ratio_ma, min_periods=max(20, ratio_ma // 5)).mean()
    abs_ok = set_index > set_ma
    ratio_ok = ratio > ratio_ma_series
    mom_ok = set_index.pct_change(mom_period, fill_method=None) > spy_thb.pct_change(mom_period, fill_method=None)
    abs_slope_ok = set_ma.diff(slope_lookback) > 0.0
    ratio_slope_ok = ratio_ma_series.diff(slope_lookback) > 0.0
    if mode == "absolute":
        raw = abs_ok
    elif mode == "absolute_relative":
        raw = abs_ok & ratio_ok
    elif mode == "relative_momentum":
        raw = ratio_ok & mom_ok
    elif mode == "all":
        raw = abs_ok & ratio_ok & mom_ok
    elif mode == "ma_slope":
        raw = abs_slope_ok
    elif mode == "ma_slope_relative":
        raw = abs_slope_ok & ratio_slope_ok
    elif mode == "ma_slope_price_confirm":
        raw = abs_slope_ok & abs_ok
    elif mode == "ma_slope_relative_momentum":
        raw = abs_slope_ok & ratio_slope_ok & mom_ok
    else:
        raise ValueError(f"Unknown gate mode: {mode}")
    return lag_close_signal_to_next_session(raw.astype(float), initial=0.0).rename("TH Gate")


def _dynamic_blend_returns(
    core_returns: pd.Series,
    th_returns: pd.Series,
    gate: pd.Series,
    th_weight: float,
) -> tuple[pd.Series, pd.DataFrame]:
    sleeves = pd.concat({"CORE": core_returns, "TH": th_returns}, axis=1).dropna()
    gate = gate.reindex(sleeves.index).ffill().fillna(0.0).clip(0.0, 1.0)
    weights = pd.DataFrame(index=sleeves.index)
    weights["TH"] = th_weight * gate
    weights["CORE"] = 1.0 - weights["TH"]
    returns = sleeves.mul(weights, axis=0).sum(axis=1).rename("Portfolio")
    return returns, weights


def _strategy_name(
    mode: str,
    th_weight: float,
    abs_ma: int,
    ratio_ma: int,
    mom_period: int,
    slope_lookback: int,
) -> str:
    parts = [f"TH gated sleeve {mode}", f"TH{int(th_weight * 100)}", f"SET_MA{abs_ma}"]
    if mode in {"absolute_relative", "relative_momentum", "all", "ma_slope_relative", "ma_slope_relative_momentum"}:
        parts.append(f"ratio_MA{ratio_ma}")
    if mode in {"relative_momentum", "all", "ma_slope_relative_momentum"}:
        parts.append(f"relmom{mom_period}")
    if slope_lookback:
        parts.append(f"slope{slope_lookback}")
    return " ".join(parts)


def _annual_returns(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy in curves.columns:
        curve = curves[strategy].dropna()
        annual = curve.groupby(curve.index.year).agg(["first", "last"])
        returns = annual["last"] / annual["first"] - 1.0
        for year, value in returns.items():
            rows.append({"Year": year, "Strategy": strategy, "Annual Return": value})
    return pd.DataFrame(rows)


def main() -> None:
    paths = default_paths(ROOT)
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    core_returns = _load_core_returns_thb()
    th_results = _run_th_sleeve()
    th_returns = th_results["nav"]["Static Copula"].pct_change(fill_method=None).fillna(0.0).reindex(core_returns.index)
    market = _load_market_signals(core_returns.index)
    core_returns = core_returns.reindex(market.index).fillna(0.0)
    th_returns = th_returns.reindex(market.index).fillna(0.0)

    rows = []
    curves = {"Core no-TH final": curve_from_returns(core_returns)}
    weight_history: dict[str, pd.DataFrame] = {}
    gate_history: dict[str, pd.Series] = {}
    modes = [
        "absolute",
        "absolute_relative",
        "relative_momentum",
        "all",
        "ma_slope",
        "ma_slope_relative",
        "ma_slope_price_confirm",
        "ma_slope_relative_momentum",
    ]
    for mode in modes:
        for abs_ma in ABS_MA_PERIODS:
            for ratio_ma in RATIO_MA_PERIODS:
                for mom_period in MOM_PERIODS:
                    slope_grid = SLOPE_LOOKBACKS if "slope" in mode else [0]
                    if mode == "absolute" and (ratio_ma != RATIO_MA_PERIODS[0] or mom_period != MOM_PERIODS[0]):
                        continue
                    if mode in {
                        "absolute_relative",
                        "ma_slope",
                        "ma_slope_relative",
                        "ma_slope_price_confirm",
                    } and mom_period != MOM_PERIODS[0]:
                        continue
                    if mode in {"absolute", "ma_slope", "ma_slope_price_confirm"} and ratio_ma != RATIO_MA_PERIODS[0]:
                        continue
                    for slope_lookback in slope_grid:
                        gate = _gate_signal(market, abs_ma, ratio_ma, mom_period, slope_lookback, mode)
                        for th_weight in TH_WEIGHTS:
                            strategy = _strategy_name(
                                mode,
                                th_weight,
                                abs_ma,
                                ratio_ma,
                                mom_period,
                                slope_lookback,
                            )
                            returns, weights = _dynamic_blend_returns(core_returns, th_returns, gate, th_weight)
                            curve = curve_from_returns(returns)
                            row = compute_port_opt_style_metrics(curve, risk_free_rate=0.03).to_dict()
                            row["Strategy"] = strategy
                            row["Gate Mode"] = mode
                            row["TH Weight When On"] = th_weight
                            row["SET MA Period"] = abs_ma
                            row["Ratio MA Period"] = ratio_ma
                            row["Relative Momentum Period"] = mom_period
                            row["Slope Lookback"] = slope_lookback
                            row["Average TH Weight"] = float(weights["TH"].mean())
                            row["TH On Days"] = int((weights["TH"] > 0.0).sum())
                            row["Start"] = curve.index.min().date().isoformat()
                            row["End"] = curve.index.max().date().isoformat()
                            rows.append(row)
                            curves[strategy] = curve
                            weight_history[strategy] = weights
                            gate_history[strategy] = gate.reindex(weights.index).ffill().fillna(0.0)

    summary = pd.DataFrame(rows).sort_values(["Sharpe", "CAGR"], ascending=False)
    best_strategy = str(summary.iloc[0]["Strategy"])
    curves_df = pd.DataFrame(curves).dropna(how="all")
    summary.to_csv(paths.result_dir / f"{RESULT_PREFIX}_summary_thb.csv", index=False)
    curves_df.to_csv(paths.result_dir / f"{RESULT_PREFIX}_curves_thb.csv")
    _annual_returns(curves_df[[CORE_COLUMN for CORE_COLUMN in ["Core no-TH final"] if CORE_COLUMN in curves_df.columns] + [best_strategy]]).to_csv(
        paths.result_dir / f"{RESULT_PREFIX}_annual_returns_thb.csv",
        index=False,
    )
    weight_history[best_strategy].to_csv(paths.result_dir / f"{RESULT_PREFIX}_best_weight_history_thb.csv")
    gate_history[best_strategy].rename("TH Gate").to_csv(paths.result_dir / f"{RESULT_PREFIX}_best_gate_history_thb.csv")

    latest_th_date = max(th_results["weights_history"]["Static Copula"])
    latest_th = th_results["weights_history"]["Static Copula"][latest_th_date].rename("TH Sleeve Internal Weight").reset_index()
    latest_th.columns = ["Asset", "TH Sleeve Internal Weight"]
    latest_th["Date"] = pd.Timestamp(latest_th_date).date().isoformat()
    latest_th["Best Strategy"] = best_strategy
    latest_th.sort_values("TH Sleeve Internal Weight", ascending=False).to_csv(
        paths.result_dir / f"{RESULT_PREFIX}_latest_th_internal_weights.csv",
        index=False,
    )

    cols = [
        "Strategy",
        "Gate Mode",
        "TH Weight When On",
        "SET MA Period",
        "Ratio MA Period",
        "Relative Momentum Period",
        "Slope Lookback",
        "CAGR",
        "Annual Vol",
        "Sharpe",
        "Max Drawdown",
        "Average TH Weight",
        "TH On Days",
    ]
    print(summary.head(12).reindex(columns=cols).to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nBest annual returns")
    annual = pd.read_csv(paths.result_dir / f"{RESULT_PREFIX}_annual_returns_thb.csv")
    print(annual.pivot(index="Year", columns="Strategy", values="Annual Return").to_string(float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
