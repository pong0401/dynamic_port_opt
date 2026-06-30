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
    compute_metrics,
    default_paths,
    get_sp500_members_as_of,
    lag_close_signal_to_next_session,
    load_cached_market_data,
    load_overlay_compare_prices,
    load_sp500_membership_intervals,
    monthly_rebalance_dates,
    optimize_portfolio,
    select_point_in_time_universe,
)


START_DATE = "2016-01-01"
END_DATE = "2026-04-29"
LOOKBACK_DAYS = 504
US_ASSETS = 30
STOCK_CAP = 0.10
OVERLAY_ASSETS = ["GC=F", "BTC-USD", "BIL"]
OVERLAY_CAPS = {"GC=F": 0.40, "BTC-USD": 0.05, "BIL": 0.00}
GOLD_CAP_SWEEP = [0.20, 0.30, 0.40]
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


def _load_us_overlay_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    paths = default_paths(ROOT)
    cached = load_cached_market_data(paths)
    source_cols = set(cached["prices"].columns)
    us_all = [
        ticker
        for ticker in load_sp500_membership_intervals(paths)["ticker"].dropna().astype(str).drop_duplicates()
        if ticker in source_cols
    ]

    cached_panel = load_cached_market_data(paths, tickers=us_all + ["SPY", "^VIX"])
    stock_prices = cached_panel["prices"].loc[START_DATE:END_DATE].reindex(columns=us_all).sort_index().ffill()
    stock_volumes = cached_panel["volumes"].loc[START_DATE:END_DATE].reindex(columns=us_all).fillna(0.0)

    overlay = load_overlay_compare_prices(
        paths,
        start_date=START_DATE,
        end_date=END_DATE,
        tickers=["SPY", "^VIX", *OVERLAY_ASSETS],
    ).sort_index().ffill()
    prices = pd.concat(
        [
            stock_prices,
            overlay[OVERLAY_ASSETS].reindex(stock_prices.index).ffill(),
        ],
        axis=1,
    )
    volumes = stock_volumes.reindex(columns=prices.columns).fillna(0.0)
    volumes.loc[:, OVERLAY_ASSETS] = 1.0
    benchmark = overlay["SPY"].reindex(prices.index).ffill().rename("benchmark")
    vol_proxy = overlay["^VIX"].reindex(prices.index).ffill().rename("vol_proxy")
    common_index = prices.index.intersection(benchmark.dropna().index).intersection(vol_proxy.dropna().index)
    return prices.reindex(common_index).ffill(), volumes.reindex(common_index).fillna(0.0), benchmark.reindex(common_index), vol_proxy.reindex(common_index), us_all


def _best_signal_config() -> pd.DataFrame:
    config_path = default_paths(ROOT).result_dir / "best_param_step3b_best_signal_config_used.csv"
    if config_path.exists():
        return pd.read_csv(config_path, index_col=0)
    return pd.DataFrame(
        {
            "Asset": {"SPY": "S&P 500", "GOLD": "Gold", "BTC": "BTC"},
            "MA Period": {"SPY": 300, "GOLD": 50, "BTC": 50},
            "Below Exposure": {"SPY": 0.50, "GOLD": 1.00, "BTC": 0.00},
        }
    )


def _close_trend_exposure(price: pd.Series, ma_period: int, below_exposure: float) -> pd.Series:
    price = price.astype(float).sort_index().ffill()
    min_periods = max(20, int(ma_period * 0.20))
    ma = price.rolling(ma_period, min_periods=min_periods).mean()
    signal = pd.Series(1.0, index=price.index, dtype=float)
    signal.loc[price < ma] = below_exposure
    signal.loc[ma.isna()] = 1.0
    return lag_close_signal_to_next_session(signal, initial=1.0)


def _metrics_for_nav(nav: pd.Series, benchmark: pd.Series) -> pd.Series:
    benchmark_nav = benchmark.reindex(nav.index).ffill()
    benchmark_nav = benchmark_nav / benchmark_nav.iloc[0]
    return compute_metrics(nav, benchmark_nav=benchmark_nav)


def run_backtest(gold_cap: float = OVERLAY_CAPS["GC=F"]) -> dict[str, object]:
    paths = default_paths(ROOT)
    prices, volumes, benchmark, vol_proxy, us_all = _load_us_overlay_panel()
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    benchmark_ret = benchmark.pct_change(fill_method=None).rename("benchmark")
    vol_proxy_ret = vol_proxy.pct_change(fill_method=None).rename("vol_proxy")
    schedule = monthly_rebalance_dates(prices.index, lookback_days=LOOKBACK_DAYS, freq="ME")

    overlay_caps = dict(OVERLAY_CAPS)
    overlay_caps["GC=F"] = gold_cap
    nav = pd.Series(1.0, index=[schedule[0]], name=f"Mean Covariance + Gold/BTC/BIL capped Gold {int(gold_cap * 100)}")
    weights_history: dict[pd.Timestamp, pd.Series] = {}
    universe_history: dict[pd.Timestamp, list[str]] = {}
    daily_weight_frames: list[pd.DataFrame] = []

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
            prices.reindex(train_index),
            volumes.reindex(train_index),
            sp500_pool,
            n_assets=US_ASSETS,
        )
        current_assets = list(dict.fromkeys(us_selected + OVERLAY_ASSETS))
        train_returns = returns.reindex(train_index)[current_assets].dropna(axis=1, thresh=max(int(0.85 * len(train_index)), 60))
        current_assets = train_returns.columns.tolist()
        if len(current_assets) < 6:
            continue
        universe_history[rebalance_date] = current_assets

        features = compute_feature_table(
            train_returns,
            benchmark_ret.reindex(train_index),
            vol_proxy_ret.reindex(train_index),
            prices.reindex(train_index)[current_assets],
            include_momentum_features=True,
            feature_flags=FEATURE_FLAGS,
        )
        if features.empty:
            continue
        momentum_signal = build_momentum_signal(features, mode="mom_63")
        sample_cov = train_returns.cov().reindex(index=current_assets, columns=current_assets).fillna(0.0)
        asset_caps = {asset: STOCK_CAP for asset in current_assets}
        asset_caps.update({asset: cap for asset, cap in overlay_caps.items() if asset in current_assets})
        weights = optimize_portfolio(
            sample_cov,
            momentum_signal,
            max_weight=max([STOCK_CAP, *overlay_caps.values()]),
            objective_mode="mean_variance",
            asset_caps=asset_caps,
        )
        weights_history[rebalance_date] = weights

        period_returns = returns.reindex(test_index)[current_assets].fillna(0.0)
        daily_weight_frames.append(
            pd.DataFrame(
                np.tile(weights.reindex(current_assets).fillna(0.0).to_numpy(), (len(test_index), 1)),
                index=test_index,
                columns=current_assets,
            )
        )
        weighted = period_returns.mul(weights, axis=1).sum(axis=1)
        nav = pd.concat([nav, float(nav.iloc[-1]) * (1.0 + weighted).cumprod()])

    nav = nav[~nav.index.duplicated(keep="last")].sort_index()
    metrics = _metrics_for_nav(nav, benchmark)
    metrics["Turnover"] = _turnover(weights_history)
    daily_weights = pd.concat(daily_weight_frames).sort_index() if daily_weight_frames else pd.DataFrame()
    daily_weights = daily_weights.loc[~daily_weights.index.duplicated(keep="last")].fillna(0.0)
    return {
        "nav": nav,
        "metrics": metrics,
        "weights_history": weights_history,
        "universe_history": universe_history,
        "overlay_caps": overlay_caps,
        "daily_weights": daily_weights,
        "prices": prices,
        "benchmark": benchmark,
    }


def apply_asset_level_daily_exposure(results: dict[str, object]) -> dict[str, object]:
    prices: pd.DataFrame = results["prices"]
    benchmark: pd.Series = results["benchmark"]
    daily_weights: pd.DataFrame = results["daily_weights"]
    if daily_weights.empty:
        return {
            "nav": pd.Series(dtype=float),
            "metrics": pd.Series(dtype=float),
            "exposure_history": pd.DataFrame(),
            "effective_weights": pd.DataFrame(),
        }

    config = _best_signal_config()
    spy_cfg = config.loc["SPY"] if "SPY" in config.index else pd.Series({"MA Period": 300, "Below Exposure": 0.50})
    gold_cfg = config.loc["GOLD"] if "GOLD" in config.index else pd.Series({"MA Period": 50, "Below Exposure": 1.00})
    btc_cfg = config.loc["BTC"] if "BTC" in config.index else pd.Series({"MA Period": 50, "Below Exposure": 0.00})

    stock_exposure = _close_trend_exposure(
        benchmark,
        ma_period=int(spy_cfg["MA Period"]),
        below_exposure=float(spy_cfg["Below Exposure"]),
    )
    gold_exposure = _close_trend_exposure(
        prices["GC=F"],
        ma_period=int(gold_cfg["MA Period"]),
        below_exposure=float(gold_cfg["Below Exposure"]),
    )
    btc_exposure = _close_trend_exposure(
        prices["BTC-USD"],
        ma_period=int(btc_cfg["MA Period"]),
        below_exposure=float(btc_cfg["Below Exposure"]),
    )

    exposure = pd.DataFrame(1.0, index=daily_weights.index, columns=daily_weights.columns, dtype=float)
    stock_cols = [column for column in exposure.columns if column not in OVERLAY_ASSETS]
    exposure.loc[:, stock_cols] = stock_exposure.reindex(exposure.index).ffill().fillna(1.0).to_numpy()[:, None]
    if "GC=F" in exposure.columns:
        exposure["GC=F"] = gold_exposure.reindex(exposure.index).ffill().fillna(1.0)
    if "BTC-USD" in exposure.columns:
        exposure["BTC-USD"] = btc_exposure.reindex(exposure.index).ffill().fillna(1.0)
    if "BIL" in exposure.columns:
        exposure["BIL"] = 1.0

    effective_weights = daily_weights.mul(exposure).clip(lower=0.0)
    asset_returns = prices.pct_change(fill_method=None).reindex(effective_weights.index).reindex(columns=effective_weights.columns).fillna(0.0)
    exposed_returns = asset_returns.mul(effective_weights, axis=1).sum(axis=1)
    nav = (1.0 + exposed_returns).cumprod().rename("Daily Exposure NAV")
    metrics = _metrics_for_nav(nav, benchmark)
    metrics["Average Exposure"] = float(effective_weights.sum(axis=1).mean())
    metrics["Minimum Exposure"] = float(effective_weights.sum(axis=1).min())

    exposure_history = pd.DataFrame(
        {
            "Stock Exposure From SPY": stock_exposure,
            "Gold Exposure From Gold Trend": gold_exposure,
            "BTC Exposure From BTC Trend": btc_exposure,
            "BIL Exposure": 1.0,
        }
    ).reindex(effective_weights.index).ffill().fillna(1.0)
    effective_weights["Cash / Reduced Exposure"] = (1.0 - effective_weights.sum(axis=1)).clip(lower=0.0)
    return {
        "nav": nav,
        "metrics": metrics,
        "exposure_history": exposure_history,
        "effective_weights": effective_weights,
    }


def main() -> None:
    paths = default_paths(ROOT)
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    curves = {}
    latest_rows = []
    daily_exposure_rows = []
    daily_exposure_curves = {}
    exposure_history_frames = []
    effective_weight_frames = []
    for gold_cap in GOLD_CAP_SWEEP:
        results = run_backtest(gold_cap=gold_cap)
        strategy = f"Mean Covariance + Gold/BTC/BIL capped Gold {int(gold_cap * 100)}"
        row = results["metrics"].copy()
        row["Strategy"] = strategy
        row["US Assets"] = US_ASSETS
        row["US Stock Cap"] = STOCK_CAP
        row["Gold Cap"] = gold_cap
        row["BTC Cap"] = OVERLAY_CAPS["BTC-USD"]
        row["BIL Cap"] = OVERLAY_CAPS["BIL"]
        row["Rebalance Months"] = 1
        row["Covariance Model"] = "Sample Covariance"
        row["Objective"] = "mean_variance + mom_63"
        summary_rows.append(row)
        curves[strategy] = results["nav"].rename(strategy)
        daily_exposure = apply_asset_level_daily_exposure(results)
        daily_strategy = f"{strategy} + asset-level daily exposure"
        daily_row = daily_exposure["metrics"].copy()
        daily_row["Strategy"] = daily_strategy
        daily_row["Base Strategy"] = strategy
        daily_row["US Assets"] = US_ASSETS
        daily_row["US Stock Cap"] = STOCK_CAP
        daily_row["Gold Cap"] = gold_cap
        daily_row["BTC Cap"] = OVERLAY_CAPS["BTC-USD"]
        daily_row["BIL Cap"] = OVERLAY_CAPS["BIL"]
        daily_row["Rebalance Months"] = 1
        daily_row["Covariance Model"] = "Sample Covariance"
        daily_row["Objective"] = "mean_variance + mom_63"
        daily_row["Stock Exposure Signal"] = "SPY best-param trend"
        daily_row["Gold Exposure Signal"] = "Gold own best-param trend"
        daily_row["BTC Exposure Signal"] = "BTC own best-param trend"
        daily_exposure_rows.append(daily_row)
        daily_exposure_curves[daily_strategy] = daily_exposure["nav"].rename(daily_strategy)

        if not daily_exposure["exposure_history"].empty:
            exposure_frame = daily_exposure["exposure_history"].copy()
            exposure_frame["Gold Cap"] = gold_cap
            exposure_history_frames.append(exposure_frame)
        if not daily_exposure["effective_weights"].empty:
            effective_frame = daily_exposure["effective_weights"].copy()
            effective_frame["Gold Cap"] = gold_cap
            effective_weight_frames.append(effective_frame)

        if results["weights_history"]:
            latest_date = max(results["weights_history"])
            latest = results["weights_history"][latest_date].rename("Portfolio Weight").reset_index()
            latest.columns = ["Asset", "Portfolio Weight"]
            latest["Date"] = pd.Timestamp(latest_date).date().isoformat()
            latest["Strategy"] = strategy
            latest["Gold Cap"] = gold_cap
            latest["Sleeve"] = "US Equity"
            latest.loc[latest["Asset"].eq("GC=F"), "Sleeve"] = "Gold"
            latest.loc[latest["Asset"].eq("BTC-USD"), "Sleeve"] = "BTC"
            latest.loc[latest["Asset"].eq("BIL"), "Sleeve"] = "BIL"
            latest_rows.append(latest.sort_values("Portfolio Weight", ascending=False))

    summary = pd.DataFrame(summary_rows).sort_values(["Sharpe", "CAGR"], ascending=False)
    column_order = [
        "Strategy",
        "US Assets",
        "US Stock Cap",
        "Gold Cap",
        "BTC Cap",
        "BIL Cap",
        "Rebalance Months",
        "Covariance Model",
        "Objective",
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
    summary.to_csv(paths.result_dir / "mean_covariance_gold_btc_bil_capped_summary.csv", index=False)
    pd.DataFrame(curves).dropna(how="all").to_csv(paths.result_dir / "mean_covariance_gold_btc_bil_capped_curve.csv")
    if daily_exposure_rows:
        daily_summary = pd.DataFrame(daily_exposure_rows).sort_values(["Sharpe", "CAGR"], ascending=False)
        daily_summary.to_csv(paths.result_dir / "mean_covariance_gold_btc_bil_asset_daily_exposure_summary.csv", index=False)
        pd.DataFrame(daily_exposure_curves).dropna(how="all").to_csv(
            paths.result_dir / "mean_covariance_gold_btc_bil_asset_daily_exposure_curves.csv"
        )
    if exposure_history_frames:
        pd.concat(exposure_history_frames).to_csv(
            paths.result_dir / "mean_covariance_gold_btc_bil_asset_daily_exposure_history.csv"
        )
    if effective_weight_frames:
        pd.concat(effective_weight_frames).to_csv(
            paths.result_dir / "mean_covariance_gold_btc_bil_asset_daily_effective_weights.csv"
        )

    if latest_rows:
        pd.concat(latest_rows, ignore_index=True).to_csv(
            paths.result_dir / "mean_covariance_gold_btc_bil_capped_latest_weights.csv",
            index=False,
        )

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    if daily_exposure_rows:
        print(pd.DataFrame(daily_exposure_rows).sort_values(["Sharpe", "CAGR"], ascending=False).to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    if latest_rows:
        latest_all = pd.concat(latest_rows, ignore_index=True)
        active = latest_all.loc[latest_all["Portfolio Weight"] > 1e-8]
        print(active[["Strategy", "Asset", "Sleeve", "Portfolio Weight"]].to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
