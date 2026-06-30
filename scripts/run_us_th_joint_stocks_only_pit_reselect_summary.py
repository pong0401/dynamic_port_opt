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

from dynamic_factor_copula import compute_port_opt_style_metrics, default_paths  # noqa: E402
from run_us_th_joint_model import BEST_OBJECTIVE  # noqa: E402
from us_th_pit_reselect_utils import load_full_us_th_thb_panel, run_joint_pit_reselect_model  # noqa: E402


def _summary_row(curve: pd.Series, strategy: str) -> dict[str, object]:
    row = compute_port_opt_style_metrics(curve, risk_free_rate=0.03).to_dict()
    row["Strategy"] = strategy
    row["Start"] = curve.dropna().index.min().date().isoformat()
    row["End"] = curve.dropna().index.max().date().isoformat()
    row["Objective"] = BEST_OBJECTIVE
    row["Selection Rule"] = "Full PIT reselect every rebalance"
    return row


def main() -> None:
    paths = default_paths(ROOT)
    prices, volumes, benchmark, vol_proxy, us_all, th_all = load_full_us_th_thb_panel(include_overlay_assets=False)

    us_only_results = run_joint_pit_reselect_model(
        prices=prices,
        volumes=volumes,
        benchmark=benchmark,
        vol_proxy=vol_proxy,
        us_all=us_all,
        th_all=[],
        us_assets=30,
        th_assets=0,
        objective_mode=BEST_OBJECTIVE,
        max_weight=0.08,
        include_overlay_assets=False,
    )
    us_th_results = run_joint_pit_reselect_model(
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

    rows = []
    curves = {}
    for strategy in ["Static Copula", "Dynamic HMM Copula"]:
        name = f"Joint US-only stocks {strategy} PIT reselect"
        curve = us_only_results["nav"][strategy].loc["2017-12-29":].mul(10_000.0)
        rows.append(_summary_row(curve, name))
        curves[name] = curve
    for strategy in ["Static Copula", "Dynamic HMM Copula"]:
        name = f"Joint US+TH stocks only {strategy} PIT reselect"
        curve = us_th_results["nav"][strategy].loc["2017-12-29":].mul(10_000.0)
        rows.append(_summary_row(curve, name))
        curves[name] = curve

    summary = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
    curve_df = pd.DataFrame(curves).dropna(how="all")
    summary.to_csv(paths.result_dir / "us_th_joint_stocks_only_pit_reselect_summary_thb.csv", index=False)
    curve_df.to_csv(paths.result_dir / "us_th_joint_stocks_only_pit_reselect_curves_thb.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
