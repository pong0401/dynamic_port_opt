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
    _parquet_column_names,
    build_momentum_signal,
    compute_feature_table,
    default_paths,
    get_sp500_members_as_of,
    load_overlay_compare_prices,
    load_sp500_membership_intervals,
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
MOMENTUM_STRENGTH = 1.0
COV_SHRINK_SWEEP = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]
STRATEGY_PREFIX = "Mean Covariance Gold30 covariance shrink"


def _strategy_name(cov_shrink: float, daily: bool = False) -> str:
    strategy = f"{STRATEGY_PREFIX} shrink{cov_shrink:g}"
    if daily:
        strategy = f"{strategy} + asset-level daily exposure"
    return strategy


def _shrink_cov(cov: pd.DataFrame, shrink: float) -> pd.DataFrame:
    cov = cov.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    diag = pd.DataFrame(np.diag(np.diag(cov.to_numpy(dtype=float))), index=cov.index, columns=cov.columns)
    return (1.0 - shrink) * cov + shrink * diag


def _read_parquet_some(path: Path, columns: list[str], start: str = "2016-01-01", end: str = "2026-04-29") -> pd.DataFrame:
    available = set(_parquet_column_names(str(path)))
    wanted = [column for column in columns if column in available]
    if not wanted:
        return pd.DataFrame()
    frames = []
    for idx in range(0, len(wanted), 80):
        frame = pd.read_parquet(path, columns=wanted[idx : idx + 80]).loc[start:end]
        frames.append(frame.astype("float32", copy=False))
    return pd.concat(frames, axis=1, copy=False) if frames else pd.DataFrame()


def _combine_source_local(source: pd.DataFrame, local: pd.DataFrame) -> pd.DataFrame:
    if source.empty:
        return local
    if local.empty:
        return source
    cols = list(dict.fromkeys([*source.columns, *local.columns]))
    out = source.reindex(columns=cols)
    for column in local.columns:
        out[column] = out[column].combine_first(local[column]) if column in out.columns else local[column]
    return out


def _load_us_overlay_panel_light(use_precomputed_assets: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    paths = default_paths(ROOT)
    precomputed_assets_path = paths.result_dir / "mean_covariance_gold_btc_bil_asset_daily_effective_weights.csv"
    if use_precomputed_assets and precomputed_assets_path.exists():
        pre_cols = pd.read_csv(precomputed_assets_path, nrows=1).columns.tolist()
        precomputed_us = [
            col for col in pre_cols
            if col not in {"Date", "Gold Cap", "Cash / Reduced Exposure", *OVERLAY_ASSETS}
        ]
    else:
        precomputed_us = []
    source_price_cols = set(_parquet_column_names(str(paths.source_cache_root / "prices.parquet")))
    local_prices = paths.local_cache_root / "extra_prices.parquet"
    if local_prices.exists():
        source_price_cols |= set(_parquet_column_names(str(local_prices)))
    if use_precomputed_assets and precomputed_us:
        us_all = [ticker for ticker in precomputed_us if ticker in source_price_cols]
    else:
        us_all = [
            ticker
            for ticker in load_sp500_membership_intervals(paths)["ticker"].dropna().astype(str).drop_duplicates()
            if ticker in source_price_cols
        ]
    wanted = us_all + ["SPY", "^VIX"]
    source_prices = _read_parquet_some(paths.source_cache_root / "prices.parquet", wanted)
    source_volumes = _read_parquet_some(paths.source_cache_root / "volumes.parquet", wanted)
    local_prices_df = _read_parquet_some(paths.local_cache_root / "extra_prices.parquet", wanted) if local_prices.exists() else pd.DataFrame()
    local_volumes_path = paths.local_cache_root / "extra_volumes.parquet"
    local_volumes_df = _read_parquet_some(local_volumes_path, wanted) if local_volumes_path.exists() else pd.DataFrame()

    stock_prices = _combine_source_local(source_prices, local_prices_df)
    stock_volumes = _combine_source_local(source_volumes, local_volumes_df)
    stock_prices = stock_prices.reindex(columns=us_all).sort_index().ffill()
    stock_volumes = stock_volumes.reindex(columns=us_all).sort_index().fillna(0.0)

    overlay = load_overlay_compare_prices(
        paths,
        start_date="2016-01-01",
        end_date="2026-04-29",
        tickers=["SPY", "^VIX", *OVERLAY_ASSETS],
    ).sort_index().ffill()
    prices = pd.concat([stock_prices, overlay[OVERLAY_ASSETS].reindex(stock_prices.index).ffill()], axis=1)
    volumes = stock_volumes.reindex(columns=prices.columns).fillna(0.0)
    volumes.loc[:, OVERLAY_ASSETS] = 1.0
    benchmark = overlay["SPY"].reindex(prices.index).ffill().rename("benchmark")
    vol_proxy = overlay["^VIX"].reindex(prices.index).ffill().rename("vol_proxy")
    common_index = prices.index.intersection(benchmark.dropna().index).intersection(vol_proxy.dropna().index)
    return prices.reindex(common_index).ffill(), volumes.reindex(common_index).fillna(0.0), benchmark.reindex(common_index), vol_proxy.reindex(common_index), us_all


def main() -> None:
    paths = default_paths(ROOT)
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    prices, volumes, benchmark, vol_proxy, us_all = _load_us_overlay_panel()
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    benchmark_ret = benchmark.pct_change(fill_method=None).rename("benchmark")
    vol_proxy_ret = vol_proxy.pct_change(fill_method=None).rename("vol_proxy")
    schedule = monthly_rebalance_dates(prices.index, lookback_days=LOOKBACK_DAYS, freq="ME")

    navs = {
        shrink: pd.Series(1.0, index=[schedule[0]], name=_strategy_name(shrink))
        for shrink in COV_SHRINK_SWEEP
    }
    weights_history: dict[float, dict[pd.Timestamp, pd.Series]] = {shrink: {} for shrink in COV_SHRINK_SWEEP}
    daily_weight_frames: dict[float, list[pd.DataFrame]] = {shrink: [] for shrink in COV_SHRINK_SWEEP}

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
        train_returns = returns.reindex(train_index)[current_assets].dropna(
            axis=1,
            thresh=max(int(0.85 * len(train_index)), 60),
        )
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
        momentum_signal = build_momentum_signal(features, mode="mom_63")
        sample_cov = train_returns.cov().reindex(index=current_assets, columns=current_assets).fillna(0.0)
        asset_caps = {asset: STOCK_CAP for asset in current_assets}
        asset_caps.update(
            {
                asset: cap
                for asset, cap in {"GC=F": GOLD_CAP, "BTC-USD": BTC_CAP, "BIL": BIL_CAP}.items()
                if asset in current_assets
            }
        )
        period_returns = returns.reindex(test_index)[current_assets].fillna(0.0)

        for shrink in COV_SHRINK_SWEEP:
            cov = _shrink_cov(sample_cov, shrink)
            weights = optimize_portfolio(
                cov,
                momentum_signal,
                max_weight=max(STOCK_CAP, GOLD_CAP, BTC_CAP, BIL_CAP),
                risk_aversion=RISK_AVERSION,
                objective_mode="mean_variance",
                asset_caps=asset_caps,
                momentum_strength=MOMENTUM_STRENGTH,
            )
            weights_history[shrink][rebalance_date] = weights
            daily_weight_frames[shrink].append(
                pd.DataFrame(
                    np.tile(weights.reindex(current_assets).fillna(0.0).to_numpy(), (len(test_index), 1)),
                    index=test_index,
                    columns=current_assets,
                )
            )
            weighted = period_returns.mul(weights, axis=1).sum(axis=1)
            navs[shrink] = pd.concat([navs[shrink], float(navs[shrink].iloc[-1]) * (1.0 + weighted).cumprod()])

    rows = []
    curves = {}
    daily_rows = []
    daily_curves = {}
    latest_daily_rows = []
    for shrink in COV_SHRINK_SWEEP:
        strategy = _strategy_name(shrink)
        nav = navs[shrink][~navs[shrink].index.duplicated(keep="last")].sort_index()
        latest_weight = weights_history[shrink][max(weights_history[shrink])] if weights_history[shrink] else pd.Series(dtype=float)
        metrics = _metrics_for_nav(nav, benchmark)
        metrics["Turnover"] = _turnover(weights_history[shrink])
        metrics["Strategy"] = strategy
        metrics["Cov Shrink"] = shrink
        metrics["Risk Aversion"] = RISK_AVERSION
        metrics["Momentum Strength"] = MOMENTUM_STRENGTH
        metrics["Gold Cap"] = GOLD_CAP
        metrics["BTC Cap"] = BTC_CAP
        metrics["BIL Cap"] = BIL_CAP
        metrics["US Stock Cap"] = STOCK_CAP
        for key, value in _concentration_stats(latest_weight).items():
            metrics[f"Latest {key}"] = value
        rows.append(metrics)
        curves[strategy] = nav.rename(strategy)

        daily_weights = pd.concat(daily_weight_frames[shrink]).sort_index() if daily_weight_frames[shrink] else pd.DataFrame()
        daily_weights = daily_weights.loc[~daily_weights.index.duplicated(keep="last")].fillna(0.0)
        daily = _apply_asset_level_daily_exposure(daily_weights, prices, benchmark)
        daily_strategy = _strategy_name(shrink, daily=True)
        daily_metrics = daily["metrics"].copy()
        daily_metrics["Strategy"] = daily_strategy
        daily_metrics["Base Strategy"] = strategy
        daily_metrics["Cov Shrink"] = shrink
        daily_metrics["Risk Aversion"] = RISK_AVERSION
        daily_metrics["Momentum Strength"] = MOMENTUM_STRENGTH
        daily_metrics["Gold Cap"] = GOLD_CAP
        daily_metrics["BTC Cap"] = BTC_CAP
        daily_metrics["BIL Cap"] = BIL_CAP
        daily_metrics["US Stock Cap"] = STOCK_CAP
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
            latest["Cov Shrink"] = shrink
            latest_daily_rows.append(latest.sort_values("Effective Weight", ascending=False))

    summary = pd.DataFrame(rows).sort_values(["Sharpe", "CAGR"], ascending=False)
    daily_summary = pd.DataFrame(daily_rows).sort_values(["Sharpe", "CAGR"], ascending=False)
    summary.to_csv(paths.result_dir / "mean_covariance_cov_shrink_sweep_summary.csv", index=False)
    daily_summary.to_csv(paths.result_dir / "mean_covariance_cov_shrink_sweep_daily_exposure_summary.csv", index=False)
    pd.DataFrame(curves).dropna(how="all").to_csv(paths.result_dir / "mean_covariance_cov_shrink_sweep_curves.csv")
    pd.DataFrame(daily_curves).dropna(how="all").to_csv(paths.result_dir / "mean_covariance_cov_shrink_sweep_daily_exposure_curves.csv")
    if latest_daily_rows:
        pd.concat(latest_daily_rows, ignore_index=True).to_csv(
            paths.result_dir / "mean_covariance_cov_shrink_sweep_latest_effective_weights.csv",
            index=False,
        )

    cols = [
        "Strategy",
        "Cov Shrink",
        "CAGR",
        "Sharpe",
        "Max Drawdown",
        "Latest Effective N",
        "Latest Top 5 Weight",
        "Latest Top 10 Weight",
        "Latest Gold Weight",
    ]
    print(daily_summary.reindex(columns=cols).to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
