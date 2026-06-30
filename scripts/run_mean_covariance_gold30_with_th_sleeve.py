from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynamic_factor_copula import (  # noqa: E402
    backtest_dynamic_factor_copula,
    compare_rebalanced_portfolio,
    compute_port_opt_style_metrics,
    convert_usd_returns_to_local,
    curve_from_returns,
    default_paths,
    load_overlay_compare_prices,
)


START_DATE = "2016-01-01"
END_DATE = "2026-04-29"
FEATURE_FLAGS = {"resid_vol": False, "drawdown": False, "downside_beta": False}
TH_WEIGHTS = [0.00, 0.05, 0.10, 0.15, 0.20]
CORE_RAW_COLUMN = "Mean Covariance + Gold/BTC/BIL capped Gold 30"
CORE_DAILY_COLUMN = "Mean Covariance + Gold/BTC/BIL capped Gold 30 + asset-level daily exposure"


def _run_th_sleeve() -> dict[str, object]:
    return backtest_dynamic_factor_copula(
        start_date=START_DATE,
        end_date=END_DATE,
        n_assets=30,
        n_clusters=4,
        lookback_days=504,
        rebalance_freq="ME",
        max_weight=0.10,
        point_in_time_liquid=True,
        universe_mode="set100_pit",
        benchmark_ticker="^SET.BK",
        vol_proxy_ticker="",
        include_momentum=True,
        include_momentum_features=True,
        include_momentum_signal=True,
        momentum_signal_mode="mom_63",
        optimizer_objective="mean_variance",
        feature_flags=FEATURE_FLAGS,
        paths=default_paths(ROOT),
    )


def _load_core_returns() -> pd.DataFrame:
    paths = default_paths(ROOT)
    raw_curves = pd.read_csv(paths.result_dir / "mean_covariance_gold_btc_bil_capped_curve.csv", index_col=0, parse_dates=True)
    daily_curves = pd.read_csv(paths.result_dir / "mean_covariance_gold_btc_bil_asset_daily_exposure_curves.csv", index_col=0, parse_dates=True)
    core_usd = pd.DataFrame(
        {
            "Core Raw Gold30": raw_curves[CORE_RAW_COLUMN],
            "Core Daily Exposure Gold30": daily_curves[CORE_DAILY_COLUMN],
        }
    ).dropna(how="all")
    fx = load_overlay_compare_prices(
        paths,
        start_date=START_DATE,
        end_date=END_DATE,
        tickers=["USDTHB=X"],
    )["USDTHB=X"].reindex(core_usd.index).ffill()
    fx_returns = fx.pct_change(fill_method=None).fillna(0.0)
    core_returns = core_usd.pct_change(fill_method=None).fillna(0.0)
    return pd.DataFrame(
        {
            column: convert_usd_returns_to_local(core_returns[column], fx_returns)
            for column in core_returns.columns
        }
    )


def _evaluate(core_returns: pd.DataFrame, th_returns: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    curves = {}
    for core_column in core_returns.columns:
        for th_weight in TH_WEIGHTS:
            core_weight = 1.0 - th_weight
            sleeve_returns = pd.concat(
                {
                    "CORE": core_returns[core_column],
                    "TH": th_returns,
                },
                axis=1,
            ).dropna()
            port_returns = compare_rebalanced_portfolio(
                sleeve_returns,
                weights=pd.Series({"CORE": core_weight, "TH": th_weight}, dtype=float),
                rebalance_months=1,
            )
            curve = curve_from_returns(port_returns)
            strategy = f"{core_column} + TH sleeve {int(th_weight * 100)}"
            row = compute_port_opt_style_metrics(curve, risk_free_rate=0.03).to_dict()
            row["Strategy"] = strategy
            row["Core Variant"] = core_column
            row["Core Weight"] = core_weight
            row["TH Weight"] = th_weight
            row["Currency"] = "THB"
            rows.append(row)
            curves[strategy] = curve
    return pd.DataFrame(rows).sort_values(["Sharpe", "CAGR"], ascending=False), pd.DataFrame(curves).dropna(how="all")


def main() -> None:
    paths = default_paths(ROOT)
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    core_returns = _load_core_returns()
    th_results = _run_th_sleeve()
    th_returns = th_results["nav"]["Static Copula"].pct_change(fill_method=None).fillna(0.0)
    summary, curves = _evaluate(core_returns, th_returns)
    summary.to_csv(paths.result_dir / "mean_covariance_gold30_with_th_sleeve_summary_thb.csv", index=False)
    curves.to_csv(paths.result_dir / "mean_covariance_gold30_with_th_sleeve_curves_thb.csv")

    latest_date = max(th_results["weights_history"]["Static Copula"])
    latest = th_results["weights_history"]["Static Copula"][latest_date].rename("TH Sleeve Weight").reset_index()
    latest.columns = ["Asset", "TH Sleeve Weight"]
    latest["Date"] = pd.Timestamp(latest_date).date().isoformat()
    latest.sort_values("TH Sleeve Weight", ascending=False).to_csv(
        paths.result_dir / "mean_covariance_gold30_with_th_sleeve_latest_th_weights.csv",
        index=False,
    )

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(latest.loc[latest["TH Sleeve Weight"] > 1e-8].sort_values("TH Sleeve Weight", ascending=False).to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
