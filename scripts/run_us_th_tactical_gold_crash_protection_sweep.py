from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dynamic_factor_copula import compute_port_opt_style_metrics, curve_from_returns, default_paths  # noqa: E402
from run_us_th_tactical_perf_momentum import (  # noqa: E402
    OVERLAY_MIX,
    RESULT_PREFIX,
    RISK_FREE_RATE,
    _best_tactical_daily_weight,
    _close_trend_exposure,
)
from run_us_th_tactical_gold_exposure_sweep import _load_overlay_inputs_from_cache  # noqa: E402


SELECTED_MIX = {"Equity": 0.65, "Gold": 0.25, "BTC": 0.10}
DD_WINDOWS = [126, 252]
WARN_DDS = [-0.08, -0.10, -0.12, -0.15]
CRASH_DDS = [-0.15, -0.18, -0.20, -0.25]
WARN_EXPOSURES = [0.50, 0.75]
CRASH_EXPOSURES = [0.25, 0.50]
RECOVERY_DDS = [-0.05, -0.08, -0.10]
PANIC_DDS = [-0.25, -0.30]
PANIC_MA_PERIODS = [200]
PANIC_MOM_PERIODS = [63]


def _gold_crash_exposure(
    gold_price: pd.Series,
    dd_window: int,
    warn_dd: float,
    crash_dd: float,
    warn_exposure: float,
    crash_exposure: float,
    recovery_dd: float,
    panic_dd: float | None = None,
    panic_ma_period: int = 200,
    panic_mom_period: int = 63,
) -> pd.Series:
    price = gold_price.astype(float).sort_index().ffill()
    rolling_high = price.rolling(dd_window, min_periods=max(20, dd_window // 4)).max()
    drawdown = price.div(rolling_high).sub(1.0)
    panic_ma = price.rolling(panic_ma_period, min_periods=max(20, panic_ma_period // 4)).mean()
    panic_mom = price.pct_change(panic_mom_period)
    active = 1.0
    values = []
    for date, dd in drawdown.items():
        panic = False
        if panic_dd is not None and pd.notna(dd):
            panic = (
                dd <= panic_dd
                and pd.notna(panic_ma.loc[date])
                and price.loc[date] < panic_ma.loc[date]
                and pd.notna(panic_mom.loc[date])
                and panic_mom.loc[date] < 0.0
            )
        if pd.isna(dd):
            active = 1.0
        elif panic:
            active = 0.0
        elif dd <= crash_dd:
            active = crash_exposure
        elif dd <= warn_dd:
            active = min(active, warn_exposure)
        elif dd >= recovery_dd:
            active = 1.0
        values.append(active)
    return pd.Series(values, index=drawdown.index, name="Gold Crash Exposure").shift(1).fillna(1.0)


def main() -> None:
    paths = default_paths(ROOT)
    monthly = pd.read_csv(paths.result_dir / f"{RESULT_PREFIX}_monthly_returns_thb.csv", index_col=0, parse_dates=True)
    comparison_curves = pd.read_csv(paths.result_dir / f"{RESULT_PREFIX}_comparison_curves_thb.csv", index_col=0, parse_dates=True).sort_index()
    tactical_summary = pd.read_csv(paths.result_dir / f"{RESULT_PREFIX}_tactical_exit_summary_thb.csv")
    tactical_weights = pd.read_csv(paths.result_dir / f"{RESULT_PREFIX}_tactical_exit_weight_history_thb.csv", index_col=0, parse_dates=True)

    daily_returns = comparison_curves[["US PIT optimized sleeve THB", "TH PIT optimized sleeve THB"]].pct_change(fill_method=None).fillna(0.0)
    index = daily_returns.index
    best_strategy, th_tactical_weight = _best_tactical_daily_weight(tactical_summary, tactical_weights, monthly, index)
    overlay_prices, signal_prices = _load_overlay_inputs_from_cache(index)

    asset_returns = pd.DataFrame(
        {
            "US Equity": daily_returns["US PIT optimized sleeve THB"],
            "TH Equity": daily_returns["TH PIT optimized sleeve THB"],
            "Gold": overlay_prices["Gold"].pct_change(fill_method=None).fillna(0.0),
            "BTC": overlay_prices["BTC"].pct_change(fill_method=None).fillna(0.0),
        },
        index=index,
    ).fillna(0.0)
    raw_weights = pd.DataFrame(
        {
            "US Equity": SELECTED_MIX["Equity"] * (1.0 - th_tactical_weight),
            "TH Equity": SELECTED_MIX["Equity"] * th_tactical_weight,
            "Gold": SELECTED_MIX["Gold"],
            "BTC": SELECTED_MIX["BTC"],
        },
        index=index,
    ).ffill().fillna(0.0)

    base_exposure = pd.DataFrame(
        {
            "US Equity": _close_trend_exposure(signal_prices["US Equity"], 300, 0.50),
            "TH Equity": _close_trend_exposure(signal_prices["TH Equity"], 200, 0.00),
            "BTC": _close_trend_exposure(signal_prices["BTC"], 50, 0.00),
        },
        index=index,
    ).ffill().fillna(1.0).clip(0.0, 1.0)

    rows = []
    curves: dict[str, pd.Series] = {}
    weights_by_strategy: dict[str, pd.DataFrame] = {}
    gold_signal = signal_prices["Gold"]
    panic_configs: list[tuple[float | None, int | None, int | None]] = [(None, None, None)]
    panic_configs.extend(
        (panic_dd, panic_ma_period, panic_mom_period)
        for panic_dd in PANIC_DDS
        for panic_ma_period in PANIC_MA_PERIODS
        for panic_mom_period in PANIC_MOM_PERIODS
    )
    for dd_window in DD_WINDOWS:
        for warn_dd in WARN_DDS:
            for crash_dd in CRASH_DDS:
                if crash_dd > warn_dd:
                    continue
                for warn_exposure in WARN_EXPOSURES:
                    for crash_exposure in CRASH_EXPOSURES:
                        if crash_exposure > warn_exposure:
                            continue
                        for recovery_dd in RECOVERY_DDS:
                            if recovery_dd <= warn_dd:
                                continue
                            for panic_dd, panic_ma_period, panic_mom_period in panic_configs:
                                gold_exposure = _gold_crash_exposure(
                                    gold_signal,
                                    dd_window,
                                    warn_dd,
                                    crash_dd,
                                    warn_exposure,
                                    crash_exposure,
                                    recovery_dd,
                                    panic_dd=panic_dd,
                                    panic_ma_period=panic_ma_period or 200,
                                    panic_mom_period=panic_mom_period or 63,
                                )
                                exposure = base_exposure.copy()
                                exposure["Gold"] = gold_exposure.reindex(index).ffill().fillna(1.0).clip(0.0, 1.0)
                                exposure = exposure[["US Equity", "TH Equity", "Gold", "BTC"]]
                                effective = raw_weights.mul(exposure, axis=0)
                                effective["Cash / Reduced Exposure"] = (
                                    1.0 - effective[["US Equity", "TH Equity", "Gold", "BTC"]].sum(axis=1)
                                ).clip(lower=0.0)
                                returns = asset_returns.mul(effective[["US Equity", "TH Equity", "Gold", "BTC"]], axis=0).sum(axis=1)
                                panic_label = "no panic" if panic_dd is None else (
                                    f"panic{panic_dd:.0%}/MA{panic_ma_period}/mom{panic_mom_period}->0"
                                )
                                strategy = (
                                    f"Gold25 crash DD{dd_window} warn{warn_dd:.0%}/exp{warn_exposure:.0%} "
                                    f"crash{crash_dd:.0%}/exp{crash_exposure:.0%} recover{recovery_dd:.0%} {panic_label}"
                                )
                                curve = curve_from_returns(returns)
                                row = compute_port_opt_style_metrics(curve.dropna(), risk_free_rate=RISK_FREE_RATE).to_dict()
                                row.update(
                                    {
                                        "Strategy": strategy,
                                        "Equity Weight": SELECTED_MIX["Equity"],
                                        "Gold Weight": SELECTED_MIX["Gold"],
                                        "BTC Weight": SELECTED_MIX["BTC"],
                                        "DD Window": dd_window,
                                        "Warn Drawdown": warn_dd,
                                        "Crash Drawdown": crash_dd,
                                        "Warn Exposure": warn_exposure,
                                        "Crash Exposure": crash_exposure,
                                        "Recovery Drawdown": recovery_dd,
                                        "Panic Drawdown": panic_dd,
                                        "Panic MA Period": panic_ma_period,
                                        "Panic Momentum Period": panic_mom_period,
                                        "Start": curve.dropna().index.min().date().isoformat(),
                                        "End": curve.dropna().index.max().date().isoformat(),
                                        "Average Gold Weight": float(effective["Gold"].mean()),
                                        "Average Cash / Reduced Exposure Weight": float(effective["Cash / Reduced Exposure"].mean()),
                                        "Gold Reduced Days": int((exposure["Gold"] < 1.0).sum()),
                                        "Gold Zero Days": int((exposure["Gold"] <= 0.0).sum()),
                                    }
                                )
                                rows.append(row)
                                curves[strategy] = curve
                                weights_by_strategy[strategy] = effective

    summary = pd.DataFrame(rows).sort_values(["Sharpe", "CAGR"], ascending=False)
    top = summary.head(20)["Strategy"].tolist()
    curves_df = pd.DataFrame({name: curves[name] for name in top}).dropna(how="all")
    weight_frames = []
    for strategy in top:
        out = weights_by_strategy[strategy].copy()
        out.insert(0, "Strategy", strategy)
        weight_frames.append(out.reset_index(names="Date"))
    weights_df = pd.concat(weight_frames, ignore_index=True) if weight_frames else pd.DataFrame()
    summary.to_csv(paths.result_dir / f"{RESULT_PREFIX}_gold_crash_protection_sweep_thb.csv", index=False)
    curves_df.to_csv(paths.result_dir / f"{RESULT_PREFIX}_gold_crash_protection_sweep_curves_thb.csv")
    weights_df.to_csv(paths.result_dir / f"{RESULT_PREFIX}_gold_crash_protection_sweep_weight_history_thb.csv", index=False)

    cols = [
        "Strategy",
        "CAGR",
        "Annual Vol",
        "Sharpe",
        "Max Drawdown",
        "Average Gold Weight",
        "Average Cash / Reduced Exposure Weight",
        "Gold Reduced Days",
        "Gold Zero Days",
    ]
    print(summary[cols].head(25).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
