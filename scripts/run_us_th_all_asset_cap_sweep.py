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

from dynamic_factor_copula import compute_port_opt_style_metrics, curve_from_returns, default_paths  # noqa: E402
from run_us_th_joint_model import (  # noqa: E402
    ALL_ASSET_STATIC_OBJECTIVE,
    END_DATE,
    START_DATE,
    _load_thb_panel,
    _read_tickers,
    _run_model_on_prices,
    _weights_history_to_frame,
)


US_CAPS = [0.06, 0.10]
TH_CAPS = [0.06, 0.10]
GOLD_CAPS = [0.20, 0.30, 0.40]
BTC_CAPS = [0.05, 0.10]


def _build_asset_caps(us_tickers: list[str], th_tickers: list[str], gold_cap: float, btc_cap: float, us_cap: float, th_cap: float) -> dict[str, float]:
    caps: dict[str, float] = {}
    caps.update({ticker: us_cap for ticker in us_tickers})
    caps.update({ticker: th_cap for ticker in th_tickers})
    caps["GC=F"] = gold_cap
    caps["BTC-USD"] = btc_cap
    return caps


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
    return row


def main() -> None:
    paths = default_paths(ROOT)
    us_tickers = _read_tickers(paths.result_dir / "latest_us_hmm_members.csv")
    th_tickers = _read_tickers(paths.result_dir / "latest_th_hmm_members.csv")
    tickers = list(dict.fromkeys(us_tickers + th_tickers + ["GC=F", "BTC-USD"]))
    prices, volumes, benchmark, vol_proxy, _ = _load_thb_panel(tickers)

    rows: list[dict[str, object]] = []
    curves: dict[str, pd.Series] = {}
    latest_rows: list[pd.DataFrame] = []

    for us_cap in US_CAPS:
        for th_cap in TH_CAPS:
            for gold_cap in GOLD_CAPS:
                for btc_cap in BTC_CAPS:
                    label = f"US{int(us_cap*100)}/TH{int(th_cap*100)}/Gold{int(gold_cap*100)}/BTC{int(btc_cap*100)}"
                    print(f"Running all-asset static cap sweep: {label}")
                    asset_caps = _build_asset_caps(us_tickers, th_tickers, gold_cap, btc_cap, us_cap, th_cap)
                    results = _run_model_on_prices(
                        prices,
                        volumes,
                        benchmark,
                        vol_proxy,
                        objective_mode=ALL_ASSET_STATIC_OBJECTIVE,
                        max_weight=max(us_cap, th_cap, gold_cap, btc_cap),
                        asset_caps=asset_caps,
                    )

                    curve = results["nav"]["Static Copula"].loc[START_DATE:END_DATE].mul(10_000.0)
                    strategy_name = f"All-assets static capped [{label}]"
                    curves[strategy_name] = curve
                    rows.append(_summary_row(curve, strategy_name, us_cap, th_cap, gold_cap, btc_cap))

                    weight_history = _weights_history_to_frame(results["weights_history"]["Static Copula"])
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

    summary.to_csv(paths.result_dir / "us_th_all_asset_cap_sweep_summary_thb.csv", index=False)
    curves_df.to_csv(paths.result_dir / "us_th_all_asset_cap_sweep_curves_thb.csv")
    latest_df.to_csv(paths.result_dir / "us_th_all_asset_cap_sweep_latest_weights_thb.csv", index=False)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
