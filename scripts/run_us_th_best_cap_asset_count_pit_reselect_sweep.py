from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in [SRC, SCRIPTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dynamic_factor_copula import (  # noqa: E402
    build_factor_covariance,
    build_momentum_signal,
    compute_feature_table,
    compute_port_opt_style_metrics,
    curve_from_returns,
    default_paths,
    get_set100_members_as_of,
    get_sp500_members_as_of,
    initialize_static_clusters,
    load_cached_market_data,
    load_overlay_compare_prices,
    load_set100_membership_intervals,
    load_sp500_membership_intervals,
    monthly_rebalance_dates,
    optimize_portfolio,
    select_point_in_time_universe,
)
from run_us_th_joint_model import END_DATE, FEATURE_FLAGS, LOOKBACK_DAYS, N_CLUSTERS, START_DATE  # noqa: E402
from run_us_th_all_asset_cap_sweep import _build_asset_caps  # noqa: E402


ASSET_COUNTS = [30, 40, 50, 100]
BEST_OBJECTIVE = "mean_variance"
BEST_US_CAP = 0.06
BEST_TH_CAP = 0.06
BEST_GOLD_CAP = 0.40
BEST_BTC_CAP = 0.10


def _load_full_thb_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str], list[str]]:
    paths = default_paths(ROOT)
    cached = load_cached_market_data(paths)
    source_cols = set(cached["prices"].columns)
    extra_prices = pd.read_parquet(paths.local_cache_root / "extra_prices.parquet")
    extra_cols = set(extra_prices.columns)

    all_us = [
        ticker
        for ticker in load_sp500_membership_intervals(paths)["ticker"].dropna().astype(str).drop_duplicates()
        if ticker in source_cols
    ]
    all_th = [
        ticker
        for ticker in load_set100_membership_intervals(paths)["ticker"].dropna().astype(str).drop_duplicates()
        if ticker in extra_cols
    ]
    asset_tickers = list(dict.fromkeys(all_us + all_th + ["GC=F", "BTC-USD"]))
    needed = list(dict.fromkeys(asset_tickers + ["SPY", "^VIX"]))
    prices = cached["prices"].loc[START_DATE:END_DATE].reindex(columns=needed).sort_index().ffill()
    volumes = cached["volumes"].loc[START_DATE:END_DATE].reindex(columns=needed).fillna(0.0)

    overlay_prices = load_overlay_compare_prices(
        paths,
        start_date=START_DATE,
        end_date=END_DATE,
        tickers=["SPY", "GC=F", "BTC-USD", "^VIX", "USDTHB=X"],
    ).sort_index().ffill()
    fx = overlay_prices["USDTHB=X"].reindex(prices.index).ffill()

    thb_prices = pd.DataFrame(index=prices.index)
    th_assets = [ticker for ticker in asset_tickers if ticker.endswith(".BK")]
    usd_assets = [ticker for ticker in asset_tickers if not ticker.endswith(".BK")]
    if th_assets:
        thb_prices[th_assets] = prices[th_assets]
    for ticker in usd_assets:
        if ticker in prices.columns and prices[ticker].notna().any():
            thb_prices[ticker] = prices[ticker].mul(fx)
        elif ticker in overlay_prices.columns:
            thb_prices[ticker] = overlay_prices[ticker].reindex(prices.index).ffill().mul(fx)

    benchmark = overlay_prices["SPY"].reindex(prices.index).ffill().mul(fx).rename("benchmark")
    vol_proxy = overlay_prices["^VIX"].reindex(prices.index).ffill().rename("vol_proxy")
    common_index = thb_prices.index.intersection(benchmark.dropna().index).intersection(vol_proxy.dropna().index)
    thb_prices = thb_prices.reindex(common_index).ffill().dropna(how="all")
    volumes = volumes.reindex(thb_prices.index).reindex(columns=thb_prices.columns).fillna(0.0)
    benchmark = benchmark.reindex(thb_prices.index).ffill()
    vol_proxy = vol_proxy.reindex(thb_prices.index).ffill()
    return thb_prices, volumes, benchmark, vol_proxy, all_us, all_th


def _summary_row(curve: pd.Series, us_assets: int, th_assets: int) -> dict[str, object]:
    row = compute_port_opt_style_metrics(curve, risk_free_rate=0.03).to_dict()
    row["Strategy"] = f"All-assets static capped [US{us_assets}/TH{th_assets}/Gold40/BTC10] PIT reselect"
    row["Start"] = curve.dropna().index.min().date().isoformat()
    row["End"] = curve.dropna().index.max().date().isoformat()
    row["Objective"] = BEST_OBJECTIVE
    row["US Assets"] = us_assets
    row["TH Assets"] = th_assets
    row["US Equity Max Weight"] = BEST_US_CAP
    row["TH Equity Max Weight"] = BEST_TH_CAP
    row["Gold Max Weight"] = BEST_GOLD_CAP
    row["BTC Max Weight"] = BEST_BTC_CAP
    row["Selection Rule"] = "PIT reselect every rebalance using median dollar volume, availability >= 90%"
    return row


def _run_static_pit_reselect_backtest(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    benchmark: pd.Series,
    vol_proxy: pd.Series,
    us_all: list[str],
    th_all: list[str],
    us_assets: int,
    th_assets: int,
) -> tuple[pd.Series, dict[pd.Timestamp, pd.Series], list[dict[str, object]]]:
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    benchmark_ret = benchmark.pct_change(fill_method=None).rename("benchmark")
    vol_proxy_ret = vol_proxy.pct_change(fill_method=None).rename("vol_proxy")
    schedule = monthly_rebalance_dates(prices.index, lookback_days=LOOKBACK_DAYS, freq="ME")
    nav = pd.Series(1.0, index=[schedule[0]], dtype=float)
    weights_history: dict[pd.Timestamp, pd.Series] = {}
    universe_rows: list[dict[str, object]] = []

    for idx, rebalance_date in enumerate(schedule[:-1]):
        next_date = schedule[idx + 1]
        loc = prices.index.get_loc(rebalance_date)
        train_index = prices.index[max(0, loc - LOOKBACK_DAYS + 1) : loc + 1]
        test_index = prices.index[(prices.index > rebalance_date) & (prices.index <= next_date)]
        if len(test_index) == 0:
            continue

        us_pool = [ticker for ticker in get_sp500_members_as_of(rebalance_date, default_paths(ROOT)) if ticker in us_all and ticker in prices.columns]
        th_pool = [ticker for ticker in get_set100_members_as_of(rebalance_date, default_paths(ROOT)) if ticker in th_all and ticker in prices.columns]
        us_selected = select_point_in_time_universe(prices.loc[train_index], volumes.loc[train_index], us_pool, n_assets=us_assets)
        th_selected = select_point_in_time_universe(prices.loc[train_index], volumes.loc[train_index], th_pool, n_assets=th_assets)
        selected_assets = list(dict.fromkeys(us_selected + th_selected + ["GC=F", "BTC-USD"]))

        train_returns = returns.reindex(train_index)[selected_assets].dropna(axis=1, thresh=max(int(0.85 * len(train_index)), 60))
        current_assets = train_returns.columns.tolist()
        if len(current_assets) < max(N_CLUSTERS + 2, 6):
            continue

        current_prices = prices.reindex(train_index)[current_assets]
        current_features = compute_feature_table(
            train_returns,
            benchmark_ret.reindex(train_index),
            vol_proxy_ret.reindex(train_index),
            current_prices,
            include_momentum_features=True,
            feature_flags=FEATURE_FLAGS,
        )
        if current_features.empty:
            continue

        static_init = initialize_static_clusters(current_features, n_clusters=N_CLUSTERS)
        static_post = pd.get_dummies(static_init["labels"]).reindex(index=current_features.index, columns=range(N_CLUSTERS), fill_value=0.0)
        momentum_signal = build_momentum_signal(current_features, mode="mom_63")
        static_cov, _ = build_factor_covariance(
            train_returns[current_features.index],
            benchmark_ret.reindex(train_index),
            static_post.fillna(0.0),
            current_features,
            dynamic=False,
        )
        current_asset_caps = _build_asset_caps(us_selected, th_selected, BEST_GOLD_CAP, BEST_BTC_CAP, BEST_US_CAP, BEST_TH_CAP)
        current_asset_caps = {asset: cap for asset, cap in current_asset_caps.items() if asset in current_features.index}
        static_weights = optimize_portfolio(
            static_cov,
            momentum_signal,
            max_weight=max(BEST_US_CAP, BEST_TH_CAP, BEST_GOLD_CAP, BEST_BTC_CAP),
            objective_mode=BEST_OBJECTIVE,
            asset_caps=current_asset_caps,
        )
        weights_history[rebalance_date] = static_weights
        for asset in current_assets:
            universe_rows.append(
                {
                    "Date": pd.Timestamp(rebalance_date).date().isoformat(),
                    "Asset": asset,
                    "US Assets Target": us_assets,
                    "TH Assets Target": th_assets,
                    "Selected": True,
                    "Sleeve": "US Equity" if not asset.endswith(".BK") and asset not in {"GC=F", "BTC-USD"} else (
                        "TH Equity" if asset.endswith(".BK") else ("Gold" if asset == "GC=F" else "BTC")
                    ),
                }
            )

        period_returns = returns.reindex(test_index)[current_features.index].fillna(0.0)
        weighted = period_returns.mul(static_weights.reindex(period_returns.columns).fillna(0.0), axis=1).sum(axis=1)
        starting_value = float(nav.iloc[-1])
        period_nav = starting_value * (1.0 + weighted).cumprod()
        nav = pd.concat([nav, period_nav])

    return nav[~nav.index.duplicated(keep="last")].sort_index(), weights_history, universe_rows


def main() -> None:
    paths = default_paths(ROOT)
    prices, volumes, benchmark, vol_proxy, us_all, th_all = _load_full_thb_panel()
    rows: list[dict[str, object]] = []
    curves: dict[str, pd.Series] = {}
    latest_rows: list[pd.DataFrame] = []
    universe_history_rows: list[dict[str, object]] = []

    for count in ASSET_COUNTS:
        print(f"Running full PIT reselect sweep: US={count}, TH={count}")
        curve, weights_history, universe_rows = _run_static_pit_reselect_backtest(
            prices,
            volumes,
            benchmark,
            vol_proxy,
            us_all,
            th_all,
            us_assets=count,
            th_assets=count,
        )
        row = _summary_row(curve.mul(10_000.0), count, count)
        rows.append(row)
        curves[row["Strategy"]] = curve.mul(10_000.0)
        universe_history_rows.extend(universe_rows)

        if weights_history:
            latest_date = max(weights_history)
            latest = weights_history[latest_date].rename("Portfolio Weight").reset_index()
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
    latest_df = pd.concat(latest_rows, ignore_index=True) if latest_rows else pd.DataFrame()
    universe_df = pd.DataFrame(universe_history_rows)

    summary.to_csv(paths.result_dir / "us_th_best_cap_asset_count_pit_reselect_sweep_summary_thb.csv", index=False)
    curves_df.to_csv(paths.result_dir / "us_th_best_cap_asset_count_pit_reselect_sweep_curves_thb.csv")
    latest_df.to_csv(paths.result_dir / "us_th_best_cap_asset_count_pit_reselect_sweep_latest_weights_thb.csv", index=False)
    universe_df.to_csv(paths.result_dir / "us_th_best_cap_asset_count_pit_reselect_universe_history_thb.csv", index=False)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
