from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynamic_factor_copula import (  # noqa: E402
    EPSILON,
    backtest_dynamic_factor_copula,
    build_momentum_signal,
    compute_metrics,
    default_paths,
    optimize_portfolio,
    optimize_risk_parity,
)


START_DATE = "2012-01-01"
END_DATE = "2026-04-30"
N_ASSETS = 30
N_CLUSTERS = 4
LOOKBACK_DAYS = 504
MAX_WEIGHT_SWEEP = [0.08, 0.10, 0.15, 0.20]
FEATURE_FLAGS = {"resid_vol": False, "drawdown": False, "downside_beta": False}


def _turnover(history: dict[pd.Timestamp, pd.Series]) -> float:
    ordered_dates = sorted(history)
    if len(ordered_dates) < 2:
        return float("nan")
    turns = []
    for prev_date, curr_date in zip(ordered_dates[:-1], ordered_dates[1:]):
        prev = history[prev_date]
        curr = history[curr_date].reindex(prev.index.union(history[curr_date].index), fill_value=0.0)
        prev = prev.reindex(curr.index, fill_value=0.0)
        turns.append(0.5 * np.abs(curr - prev).sum())
    return float(np.mean(turns))


def optimize_min_vol(cov: pd.DataFrame, max_weight: float) -> pd.Series:
    assets = cov.index
    cov = cov.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    cov_matrix = cov.to_numpy(dtype=float)
    n_assets = len(assets)
    caps = pd.Series(max_weight, index=assets, dtype=float)
    if float(caps.sum()) < 1.0 - EPSILON:
        raise ValueError("Portfolio caps must sum to at least 100%.")

    x0 = caps / caps.sum()
    bounds = [(0.0, float(caps.loc[asset])) for asset in assets]
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]

    def objective(x: np.ndarray) -> float:
        return float(x @ cov_matrix @ x)

    result = minimize(objective, x0=x0, bounds=bounds, constraints=constraints, method="SLSQP")
    if not result.success:
        return pd.Series(x0, index=assets)
    weights = pd.Series(result.x, index=assets).clip(lower=0.0)
    return weights / weights.sum()


def run_sample_cov_strategy(
    base_results: dict[str, object],
    strategy: str,
    max_weight: float,
) -> tuple[pd.Series, dict[pd.Timestamp, pd.Series]]:
    panel = base_results["panel"]
    prices: pd.DataFrame = panel["prices"]
    returns: pd.DataFrame = panel["returns"]
    schedule: list[pd.Timestamp] = base_results["schedule"]
    feature_history: dict[pd.Timestamp, pd.DataFrame] = base_results["feature_history"]

    label = f"{strategy} max{int(max_weight * 100)}"
    nav = pd.Series(1.0, index=[schedule[0]], name=label)
    weights_history: dict[pd.Timestamp, pd.Series] = {}

    for idx, rebalance_date in enumerate(schedule[:-1]):
        next_date = schedule[idx + 1]
        if rebalance_date not in feature_history:
            continue

        loc = prices.index.get_loc(rebalance_date)
        train_index = prices.index[max(0, loc - LOOKBACK_DAYS + 1) : loc + 1]
        test_index = prices.index[(prices.index > rebalance_date) & (prices.index <= next_date)]
        if len(test_index) == 0:
            continue

        current_features = feature_history[rebalance_date]
        current_assets = current_features.index.tolist()
        train_returns = returns.loc[train_index, current_assets].dropna(how="all")
        cov = train_returns.cov().reindex(index=current_assets, columns=current_assets).fillna(0.0)

        if strategy == "Mean Covariance":
            momentum_signal = build_momentum_signal(current_features, mode="mom_63")
            weights = optimize_portfolio(
                cov,
                momentum_signal,
                max_weight=max_weight,
                objective_mode="mean_variance",
            )
        elif strategy == "Risk Parity":
            weights = optimize_risk_parity(cov, max_weight=max_weight)
        elif strategy == "Min Vol":
            weights = optimize_min_vol(cov, max_weight=max_weight)
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        weights_history[rebalance_date] = weights
        period_returns = returns.reindex(test_index)[current_assets].fillna(0.0)
        weighted = period_returns.mul(weights, axis=1).sum(axis=1)
        nav = pd.concat([nav, float(nav.iloc[-1]) * (1.0 + weighted).cumprod()])

    nav = nav[~nav.index.duplicated(keep="last")].sort_index()
    return nav, weights_history


def main() -> None:
    paths = default_paths(ROOT)
    paths.result_dir.mkdir(parents=True, exist_ok=True)

    base_results = backtest_dynamic_factor_copula(
        start_date=START_DATE,
        end_date=END_DATE,
        n_assets=N_ASSETS,
        n_clusters=N_CLUSTERS,
        lookback_days=LOOKBACK_DAYS,
        rebalance_freq="ME",
        max_weight=MAX_WEIGHT_SWEEP[0],
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

    rows = []
    curves = {}
    latest_rows = []
    for max_weight in MAX_WEIGHT_SWEEP:
        for strategy in ["Mean Covariance", "Risk Parity", "Min Vol"]:
            nav, weights_history = run_sample_cov_strategy(base_results, strategy, max_weight=max_weight)
            label = f"{strategy} max{int(max_weight * 100)}"
            metrics = compute_metrics(nav, benchmark_nav=benchmark_nav)
            metrics["Turnover"] = _turnover(weights_history)
            metrics["Strategy"] = strategy
            metrics["Max Weight"] = max_weight
            metrics["Rebalance Months"] = 1
            metrics["Covariance Model"] = "Sample Covariance"
            rows.append(metrics)
            curves[label] = nav.rename(label)

            if weights_history:
                latest_date = max(weights_history)
                latest = weights_history[latest_date].rename("Portfolio Weight").reset_index()
                latest.columns = ["Ticker", "Portfolio Weight"]
                latest.insert(0, "Strategy", strategy)
                latest.insert(1, "Max Weight", max_weight)
                latest.insert(2, "Date", latest_date)
                latest_rows.append(latest.sort_values("Portfolio Weight", ascending=False))

    summary = pd.DataFrame(rows).sort_values(["Sharpe", "CAGR"], ascending=False)
    column_order = [
        "Strategy",
        "Max Weight",
        "Rebalance Months",
        "Covariance Model",
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
    summary.to_csv(paths.result_dir / "covariance_objective_max_weight_sweep.csv", index=False)
    pd.DataFrame(curves).dropna(how="all").to_csv(paths.result_dir / "covariance_objective_max_weight_curves.csv")
    if latest_rows:
        pd.concat(latest_rows, ignore_index=True).to_csv(
            paths.result_dir / "covariance_objective_max_weight_latest_weights.csv",
            index=False,
        )

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
