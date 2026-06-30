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
from run_us_th_joint_model import OBJECTIVE_MODES  # noqa: E402
from us_th_pit_reselect_utils import (  # noqa: E402
    build_asset_caps,
    load_full_us_th_thb_panel,
    run_joint_pit_reselect_model,
    weights_history_to_frame,
)


TOP_N = 5


def _summary_row(curve: pd.Series, base_case: str, objective: str, top5_rank: int, us_cap: float, th_cap: float, gold_cap: float, btc_cap: float) -> dict[str, object]:
    row = compute_port_opt_style_metrics(curve, risk_free_rate=0.03).to_dict()
    row["Strategy"] = f"All-assets static capped [{base_case}] [{objective}] PIT reselect"
    row["Base Case"] = base_case
    row["Top 5 Rank"] = top5_rank
    row["Objective"] = objective
    row["Start"] = curve.dropna().index.min().date().isoformat()
    row["End"] = curve.dropna().index.max().date().isoformat()
    row["US Equity Max Weight"] = us_cap
    row["TH Equity Max Weight"] = th_cap
    row["Gold Max Weight"] = gold_cap
    row["BTC Max Weight"] = btc_cap
    row["Selection Rule"] = "Full PIT reselect every rebalance"
    return row


def main() -> None:
    paths = default_paths(ROOT)
    base_summary = pd.read_csv(paths.result_dir / "us_th_all_asset_cap_sweep_pit_reselect_summary_thb.csv").sort_values("Sharpe", ascending=False).head(TOP_N).reset_index(drop=True)
    prices, volumes, benchmark, vol_proxy, us_all, th_all = load_full_us_th_thb_panel(include_overlay_assets=True)
    rows = []
    curves = {}
    best_rows = []
    latest_rows = []

    for idx, case in base_summary.iterrows():
        base_case = str(case["Strategy"]).split("[", 1)[1].split("]")[0]
        us_cap = float(case["US Equity Max Weight"])
        th_cap = float(case["TH Equity Max Weight"])
        gold_cap = float(case["Gold Max Weight"])
        btc_cap = float(case["BTC Max Weight"])
        asset_caps = build_asset_caps(us_all, th_all, gold_cap, btc_cap, us_cap, th_cap)
        max_weight = max(us_cap, th_cap, gold_cap, btc_cap)

        case_rows = []
        for objective in OBJECTIVE_MODES:
            print(f"Running PIT reselect top-5 objective sweep: case={base_case}, objective={objective}")
            results = run_joint_pit_reselect_model(
                prices=prices,
                volumes=volumes,
                benchmark=benchmark,
                vol_proxy=vol_proxy,
                us_all=us_all,
                th_all=th_all,
                us_assets=30,
                th_assets=30,
                objective_mode=objective,
                max_weight=max_weight,
                include_overlay_assets=True,
                asset_caps=asset_caps,
            )
            curve = results["nav"]["Static Copula"].loc["2017-12-29":].mul(10_000.0)
            row = _summary_row(curve, base_case, objective, idx + 1, us_cap, th_cap, gold_cap, btc_cap)
            rows.append(row)
            case_rows.append(row)
            curves[row["Strategy"]] = curve

            weight_history = weights_history_to_frame(results["weights_history"]["Static Copula"])
            latest_date = weight_history.index.max()
            latest = weight_history.loc[latest_date].rename("Portfolio Weight").reset_index()
            latest.columns = ["Asset", "Portfolio Weight"]
            latest["Date"] = pd.Timestamp(latest_date).date().isoformat()
            latest["Strategy"] = row["Strategy"]
            latest["Base Case"] = base_case
            latest["Top 5 Rank"] = idx + 1
            latest["Objective"] = objective
            latest["US Equity Max Weight"] = us_cap
            latest["TH Equity Max Weight"] = th_cap
            latest["Gold Max Weight"] = gold_cap
            latest["BTC Max Weight"] = btc_cap
            latest["Sleeve"] = "US Equity"
            latest.loc[latest["Asset"].str.endswith(".BK"), "Sleeve"] = "TH Equity"
            latest.loc[latest["Asset"].eq("GC=F"), "Sleeve"] = "Gold"
            latest.loc[latest["Asset"].eq("BTC-USD"), "Sleeve"] = "BTC"
            latest_rows.append(latest.sort_values("Portfolio Weight", ascending=False))

        case_df = pd.DataFrame(case_rows).sort_values("Sharpe", ascending=False)
        best = case_df.iloc[0].to_dict()
        best["Source Case Sharpe"] = float(case["Sharpe"])
        best["Sharpe Delta vs Base Leader"] = float(best["Sharpe"]) - float(case["Sharpe"])
        best_rows.append(best)

    summary = pd.DataFrame(rows).sort_values(["Top 5 Rank", "Sharpe"], ascending=[True, False])
    best_by_case = pd.DataFrame(best_rows).sort_values("Top 5 Rank")
    curves_df = pd.DataFrame(curves).dropna(how="all")
    latest_df = pd.concat(latest_rows, ignore_index=True)
    summary.to_csv(paths.result_dir / "us_th_all_asset_cap_top5_objective_pit_reselect_sweep_thb.csv", index=False)
    best_by_case.to_csv(paths.result_dir / "us_th_all_asset_cap_top5_objective_pit_reselect_best_by_case_thb.csv", index=False)
    curves_df.to_csv(paths.result_dir / "us_th_all_asset_cap_top5_objective_pit_reselect_curves_thb.csv")
    latest_df.to_csv(paths.result_dir / "us_th_all_asset_cap_top5_objective_pit_reselect_latest_weights_thb.csv", index=False)
    print(best_by_case.to_string(index=False))


if __name__ == "__main__":
    main()
