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

from dynamic_factor_copula import compute_port_opt_style_metrics, default_paths  # noqa: E402
from run_us_th_joint_model import (  # noqa: E402
    END_DATE,
    OBJECTIVE_MODES,
    START_DATE,
    _load_thb_panel,
    _read_tickers,
    _run_model_on_prices,
)
from run_us_th_all_asset_cap_sweep import _build_asset_caps  # noqa: E402


TOP_N = 5


def _summary_row(
    curve: pd.Series,
    base_case: str,
    objective: str,
    top5_rank: int,
    us_cap: float,
    th_cap: float,
    gold_cap: float,
    btc_cap: float,
) -> dict[str, object]:
    row = compute_port_opt_style_metrics(curve, risk_free_rate=0.03).to_dict()
    row["Strategy"] = f"All-assets static capped [{base_case}] [{objective}]"
    row["Base Case"] = base_case
    row["Top 5 Rank"] = top5_rank
    row["Objective"] = objective
    row["Start"] = curve.dropna().index.min().date().isoformat()
    row["End"] = curve.dropna().index.max().date().isoformat()
    row["US Equity Max Weight"] = us_cap
    row["TH Equity Max Weight"] = th_cap
    row["Gold Max Weight"] = gold_cap
    row["BTC Max Weight"] = btc_cap
    return row


def main() -> None:
    paths = default_paths(ROOT)
    base_summary = pd.read_csv(paths.result_dir / "us_th_all_asset_cap_sweep_summary_thb.csv").sort_values("Sharpe", ascending=False).head(TOP_N).reset_index(drop=True)
    us_tickers = _read_tickers(paths.result_dir / "latest_us_hmm_members.csv")
    th_tickers = _read_tickers(paths.result_dir / "latest_th_hmm_members.csv")
    tickers = list(dict.fromkeys(us_tickers + th_tickers + ["GC=F", "BTC-USD"]))
    prices, volumes, benchmark, vol_proxy, _ = _load_thb_panel(tickers)

    rows: list[dict[str, object]] = []
    curves: dict[str, pd.Series] = {}
    best_rows: list[dict[str, object]] = []
    latest_rows: list[pd.DataFrame] = []

    for idx, case in base_summary.iterrows():
        base_case = str(case["Strategy"]).split("[", 1)[1].rstrip("]")
        us_cap = float(case["US Equity Max Weight"])
        th_cap = float(case["TH Equity Max Weight"])
        gold_cap = float(case["Gold Max Weight"])
        btc_cap = float(case["BTC Max Weight"])
        asset_caps = _build_asset_caps(us_tickers, th_tickers, gold_cap, btc_cap, us_cap, th_cap)
        max_weight = max(us_cap, th_cap, gold_cap, btc_cap)

        case_rows: list[dict[str, object]] = []
        for objective in OBJECTIVE_MODES:
            print(f"Running top-5 cap objective sweep: case={base_case}, objective={objective}")
            results = _run_model_on_prices(
                prices,
                volumes,
                benchmark,
                vol_proxy,
                objective_mode=objective,
                max_weight=max_weight,
                asset_caps=asset_caps,
            )
            curve = results["nav"]["Static Copula"].loc[START_DATE:END_DATE].mul(10_000.0)
            row = _summary_row(curve, base_case, objective, idx + 1, us_cap, th_cap, gold_cap, btc_cap)
            case_rows.append(row)
            rows.append(row)
            curves[row["Strategy"]] = curve

            weight_history = pd.DataFrame(
                [weights.rename(pd.Timestamp(rebalance_date)) for rebalance_date, weights in sorted(results["weights_history"]["Static Copula"].items())]
            ).fillna(0.0)
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
        best["Sharpe Delta vs Base Mean-Variance Leader"] = float(best["Sharpe"]) - float(case["Sharpe"])
        best_rows.append(best)

    summary = pd.DataFrame(rows).sort_values(["Top 5 Rank", "Sharpe"], ascending=[True, False])
    best_by_case = pd.DataFrame(best_rows).sort_values("Top 5 Rank")
    curves_df = pd.DataFrame(curves).dropna(how="all")
    latest_df = pd.concat(latest_rows, ignore_index=True)

    summary.to_csv(paths.result_dir / "us_th_all_asset_cap_top5_objective_sweep_thb.csv", index=False)
    best_by_case.to_csv(paths.result_dir / "us_th_all_asset_cap_top5_objective_best_by_case_thb.csv", index=False)
    curves_df.to_csv(paths.result_dir / "us_th_all_asset_cap_top5_objective_curves_thb.csv")
    latest_df.to_csv(paths.result_dir / "us_th_all_asset_cap_top5_objective_latest_weights_thb.csv", index=False)

    print("\nBest by case")
    print(best_by_case.to_string(index=False))


if __name__ == "__main__":
    main()
