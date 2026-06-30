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
from run_us_th_joint_model import END_DATE, START_DATE, _build_ranked_us_th_universe, _load_thb_panel, _run_model_on_prices  # noqa: E402
from run_us_th_all_asset_cap_sweep import _build_asset_caps  # noqa: E402


ASSET_COUNTS = [30, 40, 50, 100]
BEST_OBJECTIVE = "mean_variance"
BEST_US_CAP = 0.06
BEST_TH_CAP = 0.06
BEST_GOLD_CAP = 0.40
BEST_BTC_CAP = 0.10


def _summary_row(curve: pd.Series, us_assets: int, th_assets: int) -> dict[str, object]:
    row = compute_port_opt_style_metrics(curve, risk_free_rate=0.03).to_dict()
    row["Strategy"] = f"All-assets static capped [US{us_assets}/TH{th_assets}/Gold40/BTC10]"
    row["Start"] = curve.dropna().index.min().date().isoformat()
    row["End"] = curve.dropna().index.max().date().isoformat()
    row["Objective"] = BEST_OBJECTIVE
    row["US Assets"] = us_assets
    row["TH Assets"] = th_assets
    row["US Equity Max Weight"] = BEST_US_CAP
    row["TH Equity Max Weight"] = BEST_TH_CAP
    row["Gold Max Weight"] = BEST_GOLD_CAP
    row["BTC Max Weight"] = BEST_BTC_CAP
    row["Selection Rule"] = "Top liquid PIT members by median dollar volume, availability >= 75%"
    return row


def main() -> None:
    paths = default_paths(ROOT)
    rows: list[dict[str, object]] = []
    curves: dict[str, pd.Series] = {}
    latest_rows: list[pd.DataFrame] = []

    for count in ASSET_COUNTS:
        print(f"Running best-cap asset-count sweep: US={count}, TH={count}")
        us_tickers, th_tickers = _build_ranked_us_th_universe(count, count)
        tickers = list(dict.fromkeys(us_tickers + th_tickers + ["GC=F", "BTC-USD"]))
        prices, volumes, benchmark, vol_proxy, _ = _load_thb_panel(tickers)
        asset_caps = _build_asset_caps(us_tickers, th_tickers, BEST_GOLD_CAP, BEST_BTC_CAP, BEST_US_CAP, BEST_TH_CAP)
        results = _run_model_on_prices(
            prices,
            volumes,
            benchmark,
            vol_proxy,
            objective_mode=BEST_OBJECTIVE,
            max_weight=max(BEST_US_CAP, BEST_TH_CAP, BEST_GOLD_CAP, BEST_BTC_CAP),
            asset_caps=asset_caps,
        )

        curve = results["nav"]["Static Copula"].loc[START_DATE:END_DATE].mul(10_000.0)
        row = _summary_row(curve, count, count)
        rows.append(row)
        curves[row["Strategy"]] = curve

        weight_history = pd.DataFrame(
            [weights.rename(pd.Timestamp(rebalance_date)) for rebalance_date, weights in sorted(results["weights_history"]["Static Copula"].items())]
        ).fillna(0.0)
        latest_date = weight_history.index.max()
        latest = weight_history.loc[latest_date].rename("Portfolio Weight").reset_index()
        latest.columns = ["Asset", "Portfolio Weight"]
        latest["Date"] = pd.Timestamp(latest_date).date().isoformat()
        latest["US Assets"] = count
        latest["TH Assets"] = count
        latest["Strategy"] = row["Strategy"]
        latest["Objective"] = BEST_OBJECTIVE
        latest["US Equity Max Weight"] = BEST_US_CAP
        latest["TH Equity Max Weight"] = BEST_TH_CAP
        latest["Gold Max Weight"] = BEST_GOLD_CAP
        latest["BTC Max Weight"] = BEST_BTC_CAP
        latest["Sleeve"] = "US Equity"
        latest.loc[latest["Asset"].str.endswith(".BK"), "Sleeve"] = "TH Equity"
        latest.loc[latest["Asset"].eq("GC=F"), "Sleeve"] = "Gold"
        latest.loc[latest["Asset"].eq("BTC-USD"), "Sleeve"] = "BTC"
        latest_rows.append(latest.sort_values("Portfolio Weight", ascending=False))

    summary = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
    curves_df = pd.DataFrame(curves).dropna(how="all")
    latest_df = pd.concat(latest_rows, ignore_index=True)

    summary.to_csv(paths.result_dir / "us_th_best_cap_asset_count_sweep_summary_thb.csv", index=False)
    curves_df.to_csv(paths.result_dir / "us_th_best_cap_asset_count_sweep_curves_thb.csv")
    latest_df.to_csv(paths.result_dir / "us_th_best_cap_asset_count_sweep_latest_weights_thb.csv", index=False)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
