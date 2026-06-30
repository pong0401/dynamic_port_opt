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
    build_momentum_signal,
    compute_feature_table,
    default_paths,
    get_sp500_members_as_of,
    monthly_rebalance_dates,
    optimize_portfolio,
    select_point_in_time_universe,
)
from run_mean_covariance_penalty_sweep import (  # noqa: E402
    _apply_asset_level_daily_exposure,
    _concentration_stats,
)
from run_mean_covariance_with_overlay_caps import (  # noqa: E402
    FEATURE_FLAGS,
    LOOKBACK_DAYS,
    OVERLAY_ASSETS,
    STOCK_CAP,
    US_ASSETS,
    _load_us_overlay_panel,
    _metrics_for_nav,
    _turnover,
)


GOLD_CAP = 0.30
BTC_CAP = 0.05
BIL_CAP = 0.00
RISK_AVERSION = 8.0
SIGNAL_MODES = ["mom_63", "rank_63", "zscore_63", "blend_21_63"]
MOMENTUM_STRENGTH_SWEEP = [0.50, 1.00]
STRATEGY_PREFIX = "Mean Covariance Gold30 signal sweep"


def _strategy_name(signal_mode: str, momentum_strength: float, daily: bool = False) -> str:
    strategy = f"{STRATEGY_PREFIX} {signal_mode} mom{momentum_strength:g}"
    if daily:
        strategy = f"{strategy} + asset-level daily exposure"
    return strategy


def main() -> None:
    paths = default_paths(ROOT)
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    prices, volumes, benchmark, vol_proxy, us_all = _load_us_overlay_panel()
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    benchmark_ret = benchmark.pct_change(fill_method=None).rename("benchmark")
    vol_proxy_ret = vol_proxy.pct_change(fill_method=None).rename("vol_proxy")
    schedule = monthly_rebalance_dates(prices.index, lookback_days=LOOKBACK_DAYS, freq="ME")
    configs = [(signal_mode, momentum_strength) for signal_mode in SIGNAL_MODES for momentum_strength in MOMENTUM_STRENGTH_SWEEP]

    navs = {config: pd.Series(1.0, index=[schedule[0]], name=_strategy_name(*config)) for config in configs}
    weights_history: dict[tuple[str, float], dict[pd.Timestamp, pd.Series]] = {config: {} for config in configs}
    daily_weight_frames: dict[tuple[str, float], list[pd.DataFrame]] = {config: [] for config in configs}

    for idx, rebalance_date in enumerate(schedule[:-1]):
        next_date = schedule[idx + 1]
        loc = prices.index.get_loc(rebalance_date)
        train_index = prices.index[max(0, loc - LOOKBACK_DAYS + 1) : loc + 1]
        test_index = prices.index[(prices.index > rebalance_date) & (prices.index <= next_date)]
        if len(test_index) == 0:
            continue
        sp500_pool = [
            ticker
            for ticker in get_sp500_members_as_of(rebalance_date, paths)
            if ticker in us_all and ticker in prices.columns
        ]
        us_selected = select_point_in_time_universe(
            prices.loc[train_index, sp500_pool],
            volumes.loc[train_index, sp500_pool],
            sp500_pool,
            n_assets=US_ASSETS,
        )
        current_assets = list(dict.fromkeys(us_selected + OVERLAY_ASSETS))
        train_returns = returns.reindex(train_index)[current_assets].dropna(axis=1, thresh=max(int(0.85 * len(train_index)), 60))
        current_assets = train_returns.columns.tolist()
        if len(current_assets) < 6:
            continue
        features = compute_feature_table(
            train_returns,
            benchmark_ret.reindex(train_index),
            vol_proxy_ret.reindex(train_index),
            prices.loc[train_index, current_assets],
            include_momentum_features=True,
            feature_flags=FEATURE_FLAGS,
        )
        if features.empty:
            continue
        sample_cov = train_returns.cov().reindex(index=current_assets, columns=current_assets).fillna(0.0)
        asset_caps = {asset: STOCK_CAP for asset in current_assets}
        asset_caps.update({asset: cap for asset, cap in {"GC=F": GOLD_CAP, "BTC-USD": BTC_CAP, "BIL": BIL_CAP}.items() if asset in current_assets})
        period_returns = returns.reindex(test_index)[current_assets].fillna(0.0)
        for config in configs:
            signal_mode, momentum_strength = config
            momentum_signal = build_momentum_signal(features, mode=signal_mode)
            weights = optimize_portfolio(
                sample_cov,
                momentum_signal,
                max_weight=max(STOCK_CAP, GOLD_CAP, BTC_CAP, BIL_CAP),
                risk_aversion=RISK_AVERSION,
                objective_mode="mean_variance",
                asset_caps=asset_caps,
                momentum_strength=momentum_strength,
            )
            weights_history[config][rebalance_date] = weights
            daily_weight_frames[config].append(
                pd.DataFrame(
                    np.tile(weights.reindex(current_assets).fillna(0.0).to_numpy(), (len(test_index), 1)),
                    index=test_index,
                    columns=current_assets,
                )
            )
            weighted = period_returns.mul(weights, axis=1).sum(axis=1)
            navs[config] = pd.concat([navs[config], float(navs[config].iloc[-1]) * (1.0 + weighted).cumprod()])

    rows = []
    curves = {}
    daily_rows = []
    daily_curves = {}
    latest_daily_rows = []
    for config in configs:
        signal_mode, momentum_strength = config
        strategy = _strategy_name(*config)
        nav = navs[config][~navs[config].index.duplicated(keep="last")].sort_index()
        latest_weight = weights_history[config][max(weights_history[config])] if weights_history[config] else pd.Series(dtype=float)
        metrics = _metrics_for_nav(nav, benchmark)
        metrics["Turnover"] = _turnover(weights_history[config])
        metrics["Strategy"] = strategy
        metrics["Signal Mode"] = signal_mode
        metrics["Momentum Strength"] = momentum_strength
        for key, value in _concentration_stats(latest_weight).items():
            metrics[f"Latest {key}"] = value
        rows.append(metrics)
        curves[strategy] = nav.rename(strategy)

        daily_weights = pd.concat(daily_weight_frames[config]).sort_index() if daily_weight_frames[config] else pd.DataFrame()
        daily_weights = daily_weights.loc[~daily_weights.index.duplicated(keep="last")].fillna(0.0)
        daily = _apply_asset_level_daily_exposure(daily_weights, prices, benchmark)
        daily_strategy = _strategy_name(*config, daily=True)
        daily_metrics = daily["metrics"].copy()
        daily_metrics["Strategy"] = daily_strategy
        daily_metrics["Base Strategy"] = strategy
        daily_metrics["Signal Mode"] = signal_mode
        daily_metrics["Momentum Strength"] = momentum_strength
        effective = daily["effective_weights"]
        latest_effective = effective.drop(columns=["Cash / Reduced Exposure"], errors="ignore").iloc[-1] if not effective.empty else pd.Series(dtype=float)
        for key, value in _concentration_stats(latest_effective).items():
            daily_metrics[f"Latest {key}"] = value
        daily_rows.append(daily_metrics)
        daily_curves[daily_strategy] = daily["nav"].rename(daily_strategy)
        if not effective.empty:
            latest = effective.iloc[-1].rename("Effective Weight").reset_index().rename(columns={"index": "Asset"})
            latest["Strategy"] = daily_strategy
            latest["Date"] = pd.Timestamp(effective.index.max()).date().isoformat()
            latest["Signal Mode"] = signal_mode
            latest["Momentum Strength"] = momentum_strength
            latest_daily_rows.append(latest.sort_values("Effective Weight", ascending=False))

    summary = pd.DataFrame(rows).sort_values(["Sharpe", "CAGR"], ascending=False)
    daily_summary = pd.DataFrame(daily_rows).sort_values(["Sharpe", "CAGR"], ascending=False)
    summary.to_csv(paths.result_dir / "mean_covariance_signal_sweep_summary.csv", index=False)
    daily_summary.to_csv(paths.result_dir / "mean_covariance_signal_sweep_daily_exposure_summary.csv", index=False)
    pd.DataFrame(curves).dropna(how="all").to_csv(paths.result_dir / "mean_covariance_signal_sweep_curves.csv")
    pd.DataFrame(daily_curves).dropna(how="all").to_csv(paths.result_dir / "mean_covariance_signal_sweep_daily_exposure_curves.csv")
    if latest_daily_rows:
        pd.concat(latest_daily_rows, ignore_index=True).to_csv(paths.result_dir / "mean_covariance_signal_sweep_latest_effective_weights.csv", index=False)

    cols = ["Strategy", "Signal Mode", "Momentum Strength", "CAGR", "Sharpe", "Max Drawdown", "Latest Effective N", "Latest Top 5 Weight", "Latest Top 10 Weight", "Latest Gold Weight"]
    print(daily_summary.reindex(columns=cols).to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
