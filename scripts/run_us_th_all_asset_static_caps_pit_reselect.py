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
from run_us_th_joint_model import ALL_ASSET_STATIC_OBJECTIVE  # noqa: E402
from us_th_pit_reselect_utils import build_asset_caps, load_full_us_th_thb_panel, run_joint_pit_reselect_model, weights_history_to_frame  # noqa: E402


def main() -> None:
    paths = default_paths(ROOT)
    prices, volumes, benchmark, vol_proxy, us_all, th_all = load_full_us_th_thb_panel(include_overlay_assets=True)
    asset_caps = build_asset_caps(us_all, th_all, 0.30, 0.10, 0.08, 0.08)
    results = run_joint_pit_reselect_model(
        prices=prices,
        volumes=volumes,
        benchmark=benchmark,
        vol_proxy=vol_proxy,
        us_all=us_all,
        th_all=th_all,
        us_assets=30,
        th_assets=30,
        objective_mode=ALL_ASSET_STATIC_OBJECTIVE,
        max_weight=0.30,
        include_overlay_assets=True,
        asset_caps=asset_caps,
    )
    curve = results["nav"]["Static Copula"].loc["2017-12-29":].mul(10_000.0)
    row = compute_port_opt_style_metrics(curve, risk_free_rate=0.03).to_dict()
    row["Strategy"] = "US/TH stocks + Gold/BTC all assets Static model capped rebalance PIT reselect"
    row["Start"] = curve.dropna().index.min().date().isoformat()
    row["End"] = curve.dropna().index.max().date().isoformat()
    row["Objective"] = ALL_ASSET_STATIC_OBJECTIVE
    row["Selection Rule"] = "Full PIT reselect every rebalance"
    summary = pd.DataFrame([row])
    summary.to_csv(paths.result_dir / "us_th_all_asset_static_caps_pit_reselect_summary_thb.csv", index=False)
    curve.to_frame(row["Strategy"]).to_csv(paths.result_dir / "us_th_all_asset_static_caps_pit_reselect_curves_thb.csv")

    weight_history = weights_history_to_frame(results["weights_history"]["Static Copula"])
    latest_date = weight_history.index.max()
    latest = weight_history.loc[latest_date].rename("Portfolio Weight").reset_index()
    latest.columns = ["Asset", "Portfolio Weight"]
    latest["Date"] = pd.Timestamp(latest_date).date().isoformat()
    latest["Sleeve"] = "US Equity"
    latest.loc[latest["Asset"].str.endswith(".BK"), "Sleeve"] = "TH Equity"
    latest.loc[latest["Asset"].eq("GC=F"), "Sleeve"] = "Gold"
    latest.loc[latest["Asset"].eq("BTC-USD"), "Sleeve"] = "BTC"
    latest.to_csv(paths.result_dir / "us_th_all_asset_static_caps_pit_reselect_latest_weights_thb.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
