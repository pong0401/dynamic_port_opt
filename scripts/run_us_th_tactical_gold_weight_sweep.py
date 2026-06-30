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
    RESULT_PREFIX,
    RISK_FREE_RATE,
    _best_tactical_daily_weight,
    _close_trend_exposure,
)
from run_us_th_tactical_gold_exposure_sweep import _load_overlay_inputs_from_cache  # noqa: E402


GOLD_WEIGHTS = [0.20, 0.25, 0.30]
BTC_WEIGHT = 0.10


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

    exposure = pd.DataFrame(
        {
            "US Equity": _close_trend_exposure(signal_prices["US Equity"], 300, 0.50),
            "TH Equity": _close_trend_exposure(signal_prices["TH Equity"], 200, 0.00),
            "Gold": _close_trend_exposure(signal_prices["Gold"], 50, 1.00),
            "BTC": _close_trend_exposure(signal_prices["BTC"], 50, 0.00),
        },
        index=index,
    ).ffill().fillna(1.0).clip(0.0, 1.0)

    rows = []
    curves = {}
    weight_frames = []
    for gold_weight in GOLD_WEIGHTS:
        equity_weight = 1.0 - gold_weight - BTC_WEIGHT
        raw_weights = pd.DataFrame(
            {
                "US Equity": equity_weight * (1.0 - th_tactical_weight),
                "TH Equity": equity_weight * th_tactical_weight,
                "Gold": gold_weight,
                "BTC": BTC_WEIGHT,
            },
            index=index,
        ).ffill().fillna(0.0)

        for exposure_name, effective in [
            ("no daily exposure", raw_weights.assign(**{"Cash / Reduced Exposure": 0.0})),
            ("asset-level daily exposure", raw_weights.mul(exposure, axis=0)),
        ]:
            effective = effective.copy()
            if "Cash / Reduced Exposure" not in effective:
                effective["Cash / Reduced Exposure"] = (
                    1.0 - effective[["US Equity", "TH Equity", "Gold", "BTC"]].sum(axis=1)
                ).clip(lower=0.0)
            returns = asset_returns.mul(effective[["US Equity", "TH Equity", "Gold", "BTC"]], axis=0).sum(axis=1)
            strategy = (
                f"Tactical TH/Gold/BTC equity{equity_weight:.0%} gold{gold_weight:.0%} btc{BTC_WEIGHT:.0%} "
                f"{exposure_name} ({best_strategy})"
            )
            curve = curve_from_returns(returns)
            row = compute_port_opt_style_metrics(curve.dropna(), risk_free_rate=RISK_FREE_RATE).to_dict()
            row.update(
                {
                    "Strategy": strategy,
                    "Equity Weight": equity_weight,
                    "Gold Weight": gold_weight,
                    "BTC Weight": BTC_WEIGHT,
                    "Daily Exposure Mode": exposure_name,
                    "Start": curve.dropna().index.min().date().isoformat(),
                    "End": curve.dropna().index.max().date().isoformat(),
                    "Average US Equity Weight": float(effective["US Equity"].mean()),
                    "Average TH Equity Weight": float(effective["TH Equity"].mean()),
                    "Average Gold Weight": float(effective["Gold"].mean()),
                    "Average BTC Weight": float(effective["BTC"].mean()),
                    "Average Cash / Reduced Exposure Weight": float(effective["Cash / Reduced Exposure"].mean()),
                }
            )
            rows.append(row)
            curves[strategy] = curve
            out = effective.copy()
            out.insert(0, "Strategy", strategy)
            weight_frames.append(out.reset_index(names="Date"))

    summary = pd.DataFrame(rows).sort_values(["Sharpe", "CAGR"], ascending=False)
    curves_df = pd.DataFrame(curves).dropna(how="all")
    weights_df = pd.concat(weight_frames, ignore_index=True)
    summary.to_csv(paths.result_dir / f"{RESULT_PREFIX}_gold_weight_sweep_thb.csv", index=False)
    curves_df.to_csv(paths.result_dir / f"{RESULT_PREFIX}_gold_weight_sweep_curves_thb.csv")
    weights_df.to_csv(paths.result_dir / f"{RESULT_PREFIX}_gold_weight_sweep_weight_history_thb.csv", index=False)

    cols = [
        "Strategy",
        "Equity Weight",
        "Gold Weight",
        "BTC Weight",
        "Daily Exposure Mode",
        "CAGR",
        "Annual Vol",
        "Sharpe",
        "Max Drawdown",
        "Average Gold Weight",
        "Average Cash / Reduced Exposure Weight",
    ]
    print(summary[cols].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
