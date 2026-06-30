from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "result"
DOC = ROOT / "doc" / "port_opt_advance"


def _read_summary(filename: str) -> pd.DataFrame:
    frame = pd.read_csv(RESULT / filename)
    if "Strategy" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "Strategy"})
    return frame


def _infer_curve_period(curve_file: str, strategy: str) -> tuple[str, str]:
    path = RESULT / curve_file
    if not path.exists():
        return "", ""
    curves = pd.read_csv(path, index_col=0, parse_dates=True)
    if strategy in curves.columns:
        series = curves[strategy].dropna()
    else:
        series = curves.dropna(how="all").iloc[:, 0].dropna() if not curves.empty else pd.Series(dtype=float)
    if series.empty:
        return "", ""
    return series.index.min().date().isoformat(), series.index.max().date().isoformat()


def _period_text(row: pd.Series) -> str:
    return f"{row.get('Start', '')} to {row.get('End', '')}"


def _stock_cap_mom63_final_row() -> pd.Series:
    summary_file = RESULT / "mean_covariance_stock_cap_sweep_daily_exposure_summary.csv"
    if not summary_file.exists():
        raise FileNotFoundError("Missing stock-cap sweep summary. Run scripts/run_mean_covariance_stock_cap_sweep.py first.")
    frame = pd.read_csv(summary_file)
    row = frame.loc[
        frame["Stock Cap"].round(6).eq(0.08)
        & frame["Signal Mode"].astype(str).eq("mom_63")
    ].iloc[0].copy()
    row["Start"], row["End"] = _infer_curve_period(
        "mean_covariance_stock_cap_sweep_daily_exposure_curves.csv",
        str(row["Strategy"]),
    )
    row["Step Order"] = 2.36
    row["Step"] = "2.3b-4 Final Mean Covariance Gold30 stock cap 8 mom_63"
    row["Precompute Port Growth File"] = str(Path("result") / "mean_covariance_stock_cap_sweep_daily_exposure_curves.csv")
    row["Precompute Period"] = _period_text(row)
    return row


def _write_gold30_latest_weights() -> tuple[pd.DataFrame, pd.DataFrame]:
    recheck_path = RESULT / "mean_covariance_gold30_asset_daily_recheck_today_weights.csv"
    sleeve_history_path = RESULT / "mean_covariance_gold30_asset_daily_sleeve_weight_history.csv"
    if recheck_path.exists() and sleeve_history_path.exists():
        latest = pd.read_csv(recheck_path)
        latest.to_csv(RESULT / "mean_covariance_gold30_asset_daily_latest_effective_weights.csv", index=False)
        sleeve_history = pd.read_csv(sleeve_history_path, index_col=0, parse_dates=True)
        return latest, sleeve_history

    effective_path = RESULT / "mean_covariance_gold_btc_bil_asset_daily_effective_weights.csv"
    effective = pd.read_csv(effective_path, index_col=0, parse_dates=True)
    gold30 = effective.loc[effective["Gold Cap"].round(6).eq(0.30)].drop(columns=["Gold Cap"])
    sleeve_map = {
        asset: (
            "Gold" if asset == "GC=F" else
            "BTC" if asset == "BTC-USD" else
            "BIL" if asset == "BIL" else
            "Cash / Reduced Exposure" if asset == "Cash / Reduced Exposure" else
            "US Equity"
        )
        for asset in gold30.columns
    }
    sleeve_history = gold30.rename(columns=sleeve_map).T.groupby(level=0).sum().T
    sleeve_history = sleeve_history.reindex(
        columns=["US Equity", "Gold", "BTC", "BIL", "Cash / Reduced Exposure"]
    ).dropna(axis=1, how="all")
    sleeve_history.to_csv(sleeve_history_path)

    latest_date = gold30.index.max()
    latest = (
        gold30.loc[latest_date]
        .rename("Effective Weight")
        .reset_index()
        .rename(columns={"index": "Asset"})
    )
    latest["Date"] = pd.Timestamp(latest_date).date().isoformat()
    latest["Strategy"] = "Mean Covariance + Gold/BTC/BIL capped Gold 30 + asset-level daily exposure"
    latest["Sleeve"] = "US Equity"
    latest.loc[latest["Asset"].eq("GC=F"), "Sleeve"] = "Gold"
    latest.loc[latest["Asset"].eq("BTC-USD"), "Sleeve"] = "BTC"
    latest.loc[latest["Asset"].eq("BIL"), "Sleeve"] = "BIL"
    latest.loc[latest["Asset"].eq("Cash / Reduced Exposure"), "Sleeve"] = "Cash / Reduced Exposure"
    latest["Effective Weight %"] = latest["Effective Weight"].mul(100.0)
    latest = latest.loc[latest["Effective Weight"].abs() > 1e-12].sort_values("Effective Weight", ascending=False)
    latest.to_csv(RESULT / "mean_covariance_gold30_asset_daily_latest_effective_weights.csv", index=False)
    return latest, sleeve_history


def main() -> None:
    sources = [
        (
            1.0,
            "1. Stock only",
            "pit_reselect_step1_stock_only_momentum_objective_maxweight_summary_thb.csv",
            "pit_reselect_step1_stock_only_momentum_objective_maxweight_curves_thb.csv",
        ),
        (
            2.1,
            "2.1 Equity + Gold/BTC/BIL allocation",
            "pit_reselect_step2_1_from_step1_momentum_equity_gold_btc_bil_allocation_summary_thb.csv",
            "pit_reselect_step2_1_from_step1_momentum_equity_gold_btc_bil_allocation_curves_thb.csv",
        ),
        (
            2.2,
            "2.2 Stocks + Gold/BTC/BIL one model",
            "pit_reselect_step2_2_from_step1_momentum_all_assets_with_bil_one_model_summary_thb.csv",
            "pit_reselect_step2_2_from_step1_momentum_all_assets_with_bil_one_model_curves_thb.csv",
        ),
        (
            2.3,
            "2.3 Capped one model from 2.1",
            "pit_reselect_step2_3_from_step1_momentum_all_assets_with_bil_capped_from_2_1_summary_thb.csv",
            "pit_reselect_step2_3_from_step1_momentum_all_assets_with_bil_capped_from_2_1_curves_thb.csv",
        ),
        (
            2.35,
            "2.3b No-TH mean covariance + asset daily exposure",
            "mean_covariance_gold_btc_bil_asset_daily_exposure_summary.csv",
            "mean_covariance_gold_btc_bil_asset_daily_exposure_curves.csv",
        ),
        (
            2.4,
            "2.4 Best stock assets + Gold/BTC/BIL/IEF reoptimized",
            "pit_reselect_step2_4_from_step1_momentum_best_stock_assets_gold_btc_bil_ief_reoptimized_summary_thb.csv",
            "pit_reselect_step2_4_from_step1_momentum_best_stock_assets_gold_btc_bil_ief_reoptimized_curves_thb.csv",
        ),
        (
            2.5,
            "2.5 Daily exposure on best 2.4",
            "pit_reselect_step2_5_daily_exposure_on_step2_4_summary_thb.csv",
            "pit_reselect_step2_5_daily_exposure_on_step2_4_curves_thb.csv",
        ),
    ]

    rows = []
    for order, step, summary_file, curve_file in sources:
        frame = _read_summary(summary_file)
        row = frame.sort_values("Sharpe", ascending=False).iloc[0].copy()
        if pd.isna(row.get("Start", np.nan)) or pd.isna(row.get("End", np.nan)) or not str(row.get("Start", "")).strip():
            start, end = _infer_curve_period(curve_file, str(row.get("Strategy", "")))
            row["Start"] = start
            row["End"] = end
        row["Step Order"] = order
        row["Step"] = step
        row["Precompute Port Growth File"] = str(Path("result") / curve_file)
        row["Precompute Period"] = _period_text(row)
        rows.append(row)
    rows.append(_stock_cap_mom63_final_row())

    final = pd.DataFrame(rows).sort_values("Step Order")
    best = final.loc[final["Step"].eq("2.3b-4 Final Mean Covariance Gold30 stock cap 8 mom_63")].iloc[0]
    ref_start = str(best.get("Start", ""))
    ref_end = str(best.get("End", ""))
    final["Same Period As Overall Best"] = final["Start"].astype(str).eq(ref_start) & final["End"].astype(str).eq(ref_end)
    final["Timing Note"] = np.where(
        final["Same Period As Overall Best"],
        "Comparable on exact same start/end dates as overall best",
        "Not exact same timing; compare Sharpe directionally or rerun on common overlap",
    )
    final.to_csv(RESULT / "pit_reselect_by_step_best_sharpe_summary_thb.csv", index=False)
    final[["Step", "Strategy", "Start", "End", "Sharpe", "Same Period As Overall Best", "Timing Note"]].to_csv(
        RESULT / "pit_reselect_by_step_timing_audit.csv",
        index=False,
    )
    latest_gold30, sleeve_history = _write_gold30_latest_weights()

    DOC.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PIT Reselect By Step - Port Opt Advance Handoff",
        "",
        "## Overall Best Sharpe",
        "",
        f"- Step: `{best['Step']}`",
        f"- Strategy: `{best['Strategy']}`",
        f"- Sharpe: `{float(best['Sharpe']):.4f}`",
        f"- CAGR: `{float(best['CAGR']):.4f}`",
        f"- Max Drawdown: `{float(best['Max Drawdown']):.4f}`",
        f"- Precompute port growth file path: `{best['Precompute Port Growth File']}`",
        f"- Precompute period: `{best['Precompute Period']}`",
        "",
        "## Latest Recommended Effective Weights",
        "",
        "- Strategy: `Mean Covariance Gold30 stock cap 8 mom_63 + asset-level daily exposure`",
        f"- Date: `{latest_gold30['Date'].iloc[0]}`",
        "- Source file: `result\\mean_covariance_gold30_asset_daily_latest_effective_weights.csv`",
        "- Sleeve history file: `result\\mean_covariance_gold30_asset_daily_sleeve_weight_history.csv`",
        "",
        "| Asset | Sleeve | Effective Weight |",
        "|---|---|---:|",
    ]
    for _, row in latest_gold30.iterrows():
        lines.append(f"| {row['Asset']} | {row['Sleeve']} | {float(row['Effective Weight']):.4f} |")
    lines.extend(
        [
        "",
        "## Timing Audit",
        "",
        f"- Overall-best comparison window: `{ref_start} to {ref_end}`",
        "- Rows marked false are not on the exact same start/end dates and should be compared directionally unless rerun on the common overlap.",
        "",
        "| Step | Strategy | Period | Sharpe | Same Timing |",
        "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in final.iterrows():
        lines.append(
            f"| {row['Step']} | {row['Strategy']} | {row['Precompute Period']} | {float(row['Sharpe']):.4f} | {bool(row['Same Period As Overall Best'])} |"
        )
    (DOC / "PIT_RESELECT_BY_STEP_HANDOFF.md").write_text("\n".join(lines), encoding="utf-8")
    print(final[["Step", "Strategy", "Start", "End", "Sharpe", "Same Period As Overall Best"]].to_string(index=False))


if __name__ == "__main__":
    main()
