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
from run_us_th_joint_model import ALL_ASSET_STATIC_OBJECTIVE  # noqa: E402
from us_th_pit_reselect_utils import (  # noqa: E402
    build_asset_caps,
    load_full_us_th_thb_panel,
    run_joint_pit_reselect_model,
    weights_history_to_frame,
)


US_CAPS = [0.06, 0.10]
TH_CAPS = [0.06, 0.10]
GOLD_CAPS = [0.20, 0.30, 0.40]
BTC_CAPS = [0.05, 0.10]


def _summary_row(curve: pd.Series, label: str, us_cap: float, th_cap: float, gold_cap: float, btc_cap: float) -> dict[str, object]:
    row = compute_port_opt_style_metrics(curve, risk_free_rate=0.03).to_dict()
    row["Strategy"] = label
    row["Start"] = curve.dropna().index.min().date().isoformat()
    row["End"] = curve.dropna().index.max().date().isoformat()
    row["Objective"] = ALL_ASSET_STATIC_OBJECTIVE
    row["US Equity Max Weight"] = us_cap
    row["TH Equity Max Weight"] = th_cap
    row["Gold Max Weight"] = gold_cap
    row["BTC Max Weight"] = btc_cap
    row["Selection Rule"] = "Full PIT reselect every rebalance"
    return row


def main() -> None:
    paths = default_paths(ROOT)
    prices, volumes, benchmark, vol_proxy, us_all, th_all = load_full_us_th_thb_panel(include_overlay_assets=True)
    rows = []
    curves = {}
    latest_rows = []

    for us_cap in US_CAPS:
        for th_cap in TH_CAPS:
            for gold_cap in GOLD_CAPS:
                for btc_cap in BTC_CAPS:
                    label = f"US{int(us_cap*100)}/TH{int(th_cap*100)}/Gold{int(gold_cap*100)}/BTC{int(btc_cap*100)}"
                    print(f"Running PIT reselect all-asset cap sweep: {label}")
                    asset_caps = build_asset_caps(us_all, th_all, gold_cap, btc_cap, us_cap, th_cap)
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
                        max_weight=max(us_cap, th_cap, gold_cap, btc_cap),
                        include_overlay_assets=True,
                        asset_caps=asset_caps,
                    )
                    curve = results["nav"]["Static Copula"].loc["2017-12-29":].mul(10_000.0)
                    strategy_name = f"All-assets static capped [{label}] PIT reselect"
                    rows.append(_summary_row(curve, strategy_name, us_cap, th_cap, gold_cap, btc_cap))
                    curves[strategy_name] = curve

                    weight_history = weights_history_to_frame(results["weights_history"]["Static Copula"])
                    latest_date = weight_history.index.max()
                    latest = weight_history.loc[latest_date].rename("Portfolio Weight").reset_index()
                    latest.columns = ["Asset", "Portfolio Weight"]
                    latest["Date"] = pd.Timestamp(latest_date).date().isoformat()
                    latest["Case"] = label
                    latest["US Equity Max Weight"] = us_cap
                    latest["TH Equity Max Weight"] = th_cap
                    latest["Gold Max Weight"] = gold_cap
                    latest["BTC Max Weight"] = btc_cap
                    latest["Sleeve"] = "US Equity"
                    latest.loc[latest["Asset"].str.endswith(".BK"), "Sleeve"] = "TH Equity"
                    latest.loc[latest["Asset"].eq("GC=F"), "Sleeve"] = "Gold"
                    latest.loc[latest["Asset"].eq("BTC-USD"), "Sleeve"] = "BTC"
                    latest_rows.append(latest.sort_values("Portfolio Weight", ascending=False))

    summary = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
    curves_df = pd.DataFrame(curves).dropna(how="all")
    latest_df = pd.concat(latest_rows, ignore_index=True)
    summary.to_csv(paths.result_dir / "us_th_all_asset_cap_sweep_pit_reselect_summary_thb.csv", index=False)
    curves_df.to_csv(paths.result_dir / "us_th_all_asset_cap_sweep_pit_reselect_curves_thb.csv")
    latest_df.to_csv(paths.result_dir / "us_th_all_asset_cap_sweep_pit_reselect_latest_weights_thb.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
