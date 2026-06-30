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
from run_us_th_tactical_perf_momentum import RESULT_PREFIX, RISK_FREE_RATE  # noqa: E402
from us_th_pit_reselect_utils import (  # noqa: E402
    build_asset_caps,
    load_full_us_th_thb_panel,
    run_joint_pit_reselect_model,
    weights_history_to_frame,
)


OUTPUT_PREFIX = f"{RESULT_PREFIX}_final_model_optimizer_sweep"
OBJECTIVE_MODES = [
    "mean_variance",
    "min_vol_mom_tilt",
    "max_sharpe_mom",
    "risk_parity_mom_tilt",
]
MODEL_FAMILIES = [
    "Equal Weight",
    "Risk Parity",
    "Static Copula",
    "Dynamic HMM Copula",
]
OVERLAY_ASSETS = ["GC=F", "BTC-USD"]
US_ASSETS = 30
TH_ASSETS = 30
US_CAP = 0.08
TH_CAP = 0.08
GOLD_CAP = 0.30
BTC_CAP = 0.10


def _sleeve(asset: str) -> str:
    if asset == "GC=F":
        return "Gold"
    if asset == "BTC-USD":
        return "BTC"
    if asset.endswith(".BK"):
        return "TH Equity"
    return "US Equity"


def _summary_row(curve: pd.Series, strategy: str, model_family: str, objective: str) -> dict[str, object]:
    sample = curve.dropna()
    row = compute_port_opt_style_metrics(sample, risk_free_rate=RISK_FREE_RATE).to_dict()
    row.update(
        {
            "Strategy": strategy,
            "Model Family": model_family,
            "Objective": objective,
            "Start": sample.index.min().date().isoformat(),
            "End": sample.index.max().date().isoformat(),
            "US Assets": US_ASSETS,
            "TH Assets": TH_ASSETS,
            "US Cap": US_CAP,
            "TH Cap": TH_CAP,
            "Gold Cap": GOLD_CAP,
            "BTC Cap": BTC_CAP,
            "Overlay Assets Included": True,
            "Selection Rule": "US+TH PIT reselect with Gold/BTC included in one optimizer",
        }
    )
    return row


def main() -> None:
    paths = default_paths(ROOT)
    prices, volumes, benchmark, vol_proxy, us_all, th_all = load_full_us_th_thb_panel(
        include_overlay_assets=True,
        overlay_asset_tickers=OVERLAY_ASSETS,
    )
    asset_caps = build_asset_caps(
        us_tickers=us_all,
        th_tickers=th_all,
        gold_cap=GOLD_CAP,
        btc_cap=BTC_CAP,
        us_cap=US_CAP,
        th_cap=TH_CAP,
    )
    max_weight = max(US_CAP, TH_CAP, GOLD_CAP, BTC_CAP)

    rows: list[dict[str, object]] = []
    curves: dict[str, pd.Series] = {}
    latest_rows: list[pd.DataFrame] = []

    for objective in OBJECTIVE_MODES:
        print(f"Running final model/optimizer sweep objective={objective}", flush=True)
        results = run_joint_pit_reselect_model(
            prices=prices,
            volumes=volumes,
            benchmark=benchmark,
            vol_proxy=vol_proxy,
            us_all=us_all,
            th_all=th_all,
            us_assets=US_ASSETS,
            th_assets=TH_ASSETS,
            objective_mode=objective,
            max_weight=max_weight,
            include_overlay_assets=True,
            overlay_asset_tickers=OVERLAY_ASSETS,
            asset_caps=asset_caps,
            include_momentum=True,
            include_momentum_features=True,
            include_momentum_signal=True,
            momentum_signal_mode="mom_63",
        )
        for model_family in MODEL_FAMILIES:
            curve = results["nav"][model_family].dropna().mul(10_000.0)
            strategy = f"Final sweep {model_family} [{objective}] US/TH/Gold/BTC one optimizer"
            curves[strategy] = curve.rename(strategy)
            rows.append(_summary_row(curve, strategy, model_family, objective))

            weight_history = weights_history_to_frame(results["weights_history"][model_family])
            if weight_history.empty:
                continue
            latest_date = weight_history.index.max()
            latest = weight_history.loc[latest_date].rename("Portfolio Weight").reset_index()
            latest.columns = ["Asset", "Portfolio Weight"]
            latest = latest.loc[latest["Portfolio Weight"].abs().gt(1e-12)].copy()
            latest["Sleeve"] = latest["Asset"].map(_sleeve)
            latest["Date"] = pd.Timestamp(latest_date).date().isoformat()
            latest["Strategy"] = strategy
            latest["Model Family"] = model_family
            latest["Objective"] = objective
            latest_rows.append(latest.sort_values("Portfolio Weight", ascending=False))

    summary = pd.DataFrame(rows).sort_values(["Sharpe", "CAGR"], ascending=False)
    curves_df = pd.DataFrame(curves).dropna(how="all")
    latest_df = pd.concat(latest_rows, ignore_index=True) if latest_rows else pd.DataFrame()
    best = summary.head(1).copy()

    summary.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_summary_thb.csv", index=False)
    curves_df.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_curves_thb.csv")
    latest_df.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_latest_weights_thb.csv", index=False)
    best.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_best_thb.csv", index=False)
    print(summary.head(20).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
