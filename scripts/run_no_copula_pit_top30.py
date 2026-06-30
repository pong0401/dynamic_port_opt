from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynamic_factor_copula import (  # noqa: E402
    backtest_dynamic_factor_copula,
    build_momentum_signal,
    compute_metrics,
    default_paths,
    optimize_portfolio,
)


START_DATE = "2012-01-01"
END_DATE = "2026-04-30"
N_ASSETS = 30
N_CLUSTERS = 4
LOOKBACK_DAYS = 504
MAX_WEIGHT = 0.08
FEATURE_FLAGS = {"resid_vol": False, "drawdown": False, "downside_beta": False}
OBJECTIVE_MODES = ["mean_variance", "min_vol_mom_tilt", "max_sharpe_mom"]
REBALANCE_MONTHS = [1, 2, 3, 6]


def _turnover(history: dict[pd.Timestamp, pd.Series]) -> float:
    ordered_dates = sorted(history)
    if len(ordered_dates) < 2:
        return float("nan")
    turns = []
    for prev_date, curr_date in zip(ordered_dates[:-1], ordered_dates[1:]):
        curr = history[curr_date]
        prev = history[prev_date].reindex(curr.index.union(history[prev_date].index), fill_value=0.0)
        curr = curr.reindex(prev.index, fill_value=0.0)
        turns.append(0.5 * np.abs(curr - prev).sum())
    return float(np.mean(turns))


def run_sample_cov_optimizer(
    base_results: dict[str, object],
    objective_mode: str,
    lookback_days: int = LOOKBACK_DAYS,
    max_weight: float = MAX_WEIGHT,
) -> tuple[pd.Series, dict[pd.Timestamp, pd.Series]]:
    panel = base_results["panel"]
    prices: pd.DataFrame = panel["prices"]
    returns: pd.DataFrame = panel["returns"]
    schedule: list[pd.Timestamp] = base_results["schedule"]
    feature_history: dict[pd.Timestamp, pd.DataFrame] = base_results["feature_history"]

    nav = pd.Series(1.0, index=[schedule[0]], name=f"Sample Cov Optimizer [{objective_mode}]")
    weights_history: dict[pd.Timestamp, pd.Series] = {}

    for idx, rebalance_date in enumerate(schedule[:-1]):
        next_date = schedule[idx + 1]
        if rebalance_date not in feature_history:
            continue

        loc = prices.index.get_loc(rebalance_date)
        train_index = prices.index[max(0, loc - lookback_days + 1) : loc + 1]
        test_index = prices.index[(prices.index > rebalance_date) & (prices.index <= next_date)]
        if len(test_index) == 0:
            continue

        current_features = feature_history[rebalance_date]
        current_assets = current_features.index.tolist()
        train_returns = returns.loc[train_index, current_assets].dropna(how="all")
        sample_cov = train_returns.cov().reindex(index=current_assets, columns=current_assets).fillna(0.0)
        momentum_signal = build_momentum_signal(current_features, mode="mom_63")
        weights = optimize_portfolio(
            sample_cov,
            momentum_signal,
            max_weight=max_weight,
            objective_mode=objective_mode,
        )
        weights_history[rebalance_date] = weights

        period_returns = returns.reindex(test_index)[current_assets].fillna(0.0)
        weighted = period_returns.mul(weights, axis=1).sum(axis=1)
        nav = pd.concat([nav, float(nav.iloc[-1]) * (1.0 + weighted).cumprod()])

    nav = nav[~nav.index.duplicated(keep="last")].sort_index()
    return nav, weights_history


def rebalance_freq(months: int) -> str:
    return "ME" if months == 1 else f"{months}ME"


def main() -> None:
    paths = default_paths(ROOT)
    summary_rows = []
    curves = {}
    no_copula_histories = {}

    for months in REBALANCE_MONTHS:
        base_results = backtest_dynamic_factor_copula(
            start_date=START_DATE,
            end_date=END_DATE,
            n_assets=N_ASSETS,
            n_clusters=N_CLUSTERS,
            lookback_days=LOOKBACK_DAYS,
            rebalance_freq=rebalance_freq(months),
            max_weight=MAX_WEIGHT,
            universe_mode="sp500_pit",
            include_momentum=True,
            include_momentum_features=True,
            include_momentum_signal=True,
            momentum_signal_mode="mom_63",
            optimizer_objective="mean_variance",
            feature_flags=FEATURE_FLAGS,
            paths=paths,
        )

        benchmark_nav = base_results["nav"]["Benchmark"]
        for model_name in ["Static Copula", "Dynamic HMM Copula", "Equal Weight", "Risk Parity"]:
            metrics = base_results["metrics"].loc[model_name].copy()
            metrics["Strategy"] = model_name
            metrics["Covariance Model"] = model_name
            metrics["Optimizer Objective"] = "mean_variance" if "Copula" in model_name else "n/a"
            metrics["Rebalance Months"] = months
            summary_rows.append(metrics)
            curves[f"{model_name} {months}M"] = base_results["nav"][model_name].rename(f"{model_name} {months}M")

        for objective_mode in OBJECTIVE_MODES:
            nav, weights_history = run_sample_cov_optimizer(
                base_results,
                objective_mode,
                lookback_days=LOOKBACK_DAYS,
                max_weight=MAX_WEIGHT,
            )
            label = f"Sample Cov Optimizer [{objective_mode}]"
            metrics = compute_metrics(nav, benchmark_nav=benchmark_nav)
            metrics["Turnover"] = _turnover(weights_history)
            metrics["Strategy"] = label
            metrics["Covariance Model"] = "Sample Covariance"
            metrics["Optimizer Objective"] = objective_mode
            metrics["Rebalance Months"] = months
            summary_rows.append(metrics)
            curves[f"{label} {months}M"] = nav.rename(f"{label} {months}M")
            no_copula_histories[f"{label} {months}M"] = weights_history

    paths.result_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows).sort_values(["Sharpe", "CAGR"], ascending=False)
    column_order = [
        "Strategy",
        "Covariance Model",
        "Optimizer Objective",
        "Rebalance Months",
        "Total Return",
        "CAGR",
        "Annual Vol",
        "Sharpe",
        "Sortino",
        "Max Drawdown",
        "Benchmark Relative Return",
        "Turnover",
    ]
    summary = summary.reindex(columns=column_order)
    summary.to_csv(paths.result_dir / "no_copula_pit_top30_optimizer_summary.csv", index=False)
    pd.DataFrame(curves).dropna(how="all").to_csv(paths.result_dir / "no_copula_pit_top30_optimizer_curves.csv")

    covariance_compare = summary.loc[
        summary["Strategy"].isin(["Static Copula", "Dynamic HMM Copula", "Sample Cov Optimizer [mean_variance]"])
    ].copy()
    covariance_compare.to_csv(paths.result_dir / "pit_reselect_copula_vs_no_copula_rebalance_sweep.csv", index=False)

    latest_rows = []
    for label, history in no_copula_histories.items():
        if not history:
            continue
        latest_date = max(history)
        latest = history[latest_date].rename("Portfolio Weight").reset_index()
        latest.columns = ["Ticker", "Portfolio Weight"]
        latest.insert(0, "Strategy", label)
        latest.insert(1, "Date", latest_date)
        latest_rows.append(latest.sort_values("Portfolio Weight", ascending=False))
    if latest_rows:
        pd.concat(latest_rows, ignore_index=True).to_csv(
            paths.result_dir / "no_copula_pit_top30_optimizer_latest_weights.csv",
            index=False,
        )

    print(summary.to_string(float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
