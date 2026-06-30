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
    START_DATE,
    _best_tactical_daily_weight,
    _close_trend_exposure,
)


GOLD_MA_PERIODS = [25, 50, 75, 100, 150, 200, 250, 300]
GOLD_BELOW_EXPOSURES = [0.0, 0.25, 0.50, 0.75, 1.0]


def _load_overlay_inputs_from_cache(index: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = default_paths(ROOT)
    overlay = pd.read_parquet(paths.local_cache_root / "overlay_compare_prices.parquet").sort_index().ffill()
    overlay = overlay.loc[START_DATE:index.max()]
    fx = overlay["USDTHB=X"].ffill()
    set_index = pd.read_parquet(paths.local_cache_root / "extra_prices.parquet")["^SET.BK"].loc[START_DATE:index.max()].sort_index().ffill()
    full_index = index.union(overlay.index).union(set_index.index).sort_values()
    overlay = overlay.reindex(full_index).ffill()
    set_index = set_index.reindex(full_index).ffill()
    thb_prices = pd.DataFrame(
        {
            "Gold": overlay["GC=F"].mul(overlay["USDTHB=X"].ffill()),
            "BTC": overlay["BTC-USD"].mul(overlay["USDTHB=X"].ffill()),
        },
        index=full_index,
    ).reindex(index).ffill()
    signal_prices = pd.DataFrame(
        {
            "US Equity": overlay["SPY"],
            "TH Equity": set_index,
            "Gold": overlay["GC=F"],
            "BTC": overlay["BTC-USD"],
        },
        index=full_index,
    ).reindex(index).ffill()
    return thb_prices, signal_prices


def _metrics_row(curve: pd.Series, strategy: str, weights: pd.DataFrame, ma_period: int, below_exposure: float) -> dict[str, object]:
    row = compute_port_opt_style_metrics(curve.dropna(), risk_free_rate=RISK_FREE_RATE).to_dict()
    row.update(
        {
            "Strategy": strategy,
            "Gold MA Period": ma_period,
            "Gold Below Exposure": below_exposure,
            "Start": curve.dropna().index.min().date().isoformat(),
            "End": curve.dropna().index.max().date().isoformat(),
            "Average US Equity Weight": float(weights["US Equity"].mean()),
            "Average TH Equity Weight": float(weights["TH Equity"].mean()),
            "Average Gold Weight": float(weights["Gold"].mean()),
            "Average BTC Weight": float(weights["BTC"].mean()),
            "Average Cash / Reduced Exposure Weight": float(weights["Cash / Reduced Exposure"].mean()),
        }
    )
    return row


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
            "US Equity": OVERLAY_MIX["Equity"] * (1.0 - th_tactical_weight),
            "TH Equity": OVERLAY_MIX["Equity"] * th_tactical_weight,
            "Gold": OVERLAY_MIX["Gold"],
            "BTC": OVERLAY_MIX["BTC"],
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
    weight_frames = []
    for ma_period in GOLD_MA_PERIODS:
        for below_exposure in GOLD_BELOW_EXPOSURES:
            gold_exposure = _close_trend_exposure(signal_prices["Gold"], ma_period, below_exposure)
            exposure = base_exposure.copy()
            exposure["Gold"] = gold_exposure.reindex(index).ffill().fillna(1.0).clip(0.0, 1.0)
            exposure = exposure[["US Equity", "TH Equity", "Gold", "BTC"]]
            effective = raw_weights.mul(exposure, axis=0)
            effective["Cash / Reduced Exposure"] = (1.0 - effective[["US Equity", "TH Equity", "Gold", "BTC"]].sum(axis=1)).clip(lower=0.0)
            returns = asset_returns.mul(effective[["US Equity", "TH Equity", "Gold", "BTC"]], axis=0).sum(axis=1)
            strategy = (
                f"Tactical TH/Gold/BTC final with Gold MA{ma_period} "
                f"below{below_exposure:.0%} daily exposure ({best_strategy})"
            )
            curve = curve_from_returns(returns)
            rows.append(_metrics_row(curve, strategy, effective, ma_period, below_exposure))
            curves[strategy] = curve
            if len(curves) <= 12:
                out = effective.copy()
                out.insert(0, "Strategy", strategy)
                weight_frames.append(out.reset_index(names="Date"))

    summary = pd.DataFrame(rows).sort_values(["Sharpe", "CAGR"], ascending=False)
    top = summary.head(20)["Strategy"].tolist()
    curves_df = pd.DataFrame({name: curves[name] for name in top}).dropna(how="all")
    weights_df = pd.concat(weight_frames, ignore_index=True) if weight_frames else pd.DataFrame()

    summary.to_csv(paths.result_dir / f"{RESULT_PREFIX}_gold_daily_exposure_sweep_thb.csv", index=False)
    curves_df.to_csv(paths.result_dir / f"{RESULT_PREFIX}_gold_daily_exposure_sweep_curves_thb.csv")
    weights_df.to_csv(paths.result_dir / f"{RESULT_PREFIX}_gold_daily_exposure_sweep_weight_history_thb.csv", index=False)

    cols = [
        "Strategy",
        "Gold MA Period",
        "Gold Below Exposure",
        "CAGR",
        "Annual Vol",
        "Sharpe",
        "Max Drawdown",
        "Average Gold Weight",
        "Average Cash / Reduced Exposure Weight",
    ]
    print(summary[cols].head(20).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
