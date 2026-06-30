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
    build_momentum_signal,
    compute_feature_table,
    compute_metrics,
    default_paths,
    get_set100_members_as_of,
    get_sp500_members_as_of,
    lag_close_signal_to_next_session,
    load_cached_market_data,
    monthly_rebalance_dates,
    optimize_portfolio,
    select_point_in_time_universe,
)
from us_th_pit_reselect_utils import drop_duplicate_share_classes, load_full_us_th_thb_panel  # noqa: E402


START_DATE = "2016-01-01"
END_DATE = "2026-04-29"
LOOKBACK_DAYS = 504
US_ASSETS = 30
TH_ASSETS = 30
STOCK_CAP = 0.10
GOLD_CAP = 0.30
BTC_CAP = 0.05
BIL_CAP = 0.00
OVERLAY_ASSETS = ["GC=F", "BTC-USD", "BIL"]
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


def _best_signal_config() -> pd.DataFrame:
    path = default_paths(ROOT).result_dir / "best_param_step3b_best_signal_config_used.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0)
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


def _load_set_index_thb(index: pd.DatetimeIndex) -> pd.Series:
    paths = default_paths(ROOT)
    cached = load_cached_market_data(paths, tickers=["^SET.BK"])
    return cached["prices"]["^SET.BK"].reindex(index).ffill().rename("^SET.BK")


def run_backtest() -> dict[str, object]:
    paths = default_paths(ROOT)
    prices, volumes, benchmark, vol_proxy, us_all, th_all = load_full_us_th_thb_panel(
        include_overlay_assets=True,
        overlay_asset_tickers=OVERLAY_ASSETS,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    benchmark_ret = benchmark.pct_change(fill_method=None).rename("benchmark")
    vol_proxy_ret = vol_proxy.pct_change(fill_method=None).rename("vol_proxy")
    schedule = monthly_rebalance_dates(prices.index, lookback_days=LOOKBACK_DAYS, freq="ME")

    nav = pd.Series(1.0, index=[schedule[0]], name="US+TH Mean Covariance + Gold/BTC/BIL Gold30")
    weights_history: dict[pd.Timestamp, pd.Series] = {}
    selected_split_history: dict[pd.Timestamp, dict[str, list[str]]] = {}
    daily_weight_frames: list[pd.DataFrame] = []

    for idx, rebalance_date in enumerate(schedule[:-1]):
        next_date = schedule[idx + 1]
        loc = prices.index.get_loc(rebalance_date)
        train_index = prices.index[max(0, loc - LOOKBACK_DAYS + 1) : loc + 1]
        test_index = prices.index[(prices.index > rebalance_date) & (prices.index <= next_date)]
        if len(test_index) == 0:
            continue

        us_pool = [
            ticker
            for ticker in get_sp500_members_as_of(rebalance_date, paths)
            if ticker in us_all and ticker in prices.columns
        ]
        us_pool = drop_duplicate_share_classes(us_pool)
        th_pool = [
            ticker
            for ticker in get_set100_members_as_of(rebalance_date, paths)
            if ticker in th_all and ticker in prices.columns
        ]
        us_selected = select_point_in_time_universe(prices.reindex(train_index), volumes.reindex(train_index), us_pool, n_assets=US_ASSETS)
        th_selected = select_point_in_time_universe(prices.reindex(train_index), volumes.reindex(train_index), th_pool, n_assets=TH_ASSETS)
        current_assets = list(dict.fromkeys(us_selected + th_selected + OVERLAY_ASSETS))
        train_returns = returns.reindex(train_index)[current_assets].dropna(axis=1, thresh=max(int(0.85 * len(train_index)), 60))
        current_assets = train_returns.columns.tolist()
        if len(current_assets) < 6:
            continue

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
        asset_caps.update({"GC=F": GOLD_CAP, "BTC-USD": BTC_CAP, "BIL": BIL_CAP})
        weights = optimize_portfolio(
            sample_cov,
            momentum_signal,
            max_weight=max(STOCK_CAP, GOLD_CAP, BTC_CAP, BIL_CAP),
            objective_mode="mean_variance",
            asset_caps={asset: cap for asset, cap in asset_caps.items() if asset in current_assets},
        )
        weights_history[rebalance_date] = weights
        selected_split_history[rebalance_date] = {
            "US": [asset for asset in current_assets if asset in us_selected],
            "TH": [asset for asset in current_assets if asset in th_selected],
            "Overlay": [asset for asset in current_assets if asset in OVERLAY_ASSETS],
        }

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
    daily_weights = pd.concat(daily_weight_frames).sort_index() if daily_weight_frames else pd.DataFrame()
    daily_weights = daily_weights.loc[~daily_weights.index.duplicated(keep="last")].fillna(0.0)
    metrics = _metrics_for_nav(nav, benchmark)
    metrics["Turnover"] = _turnover(weights_history)
    return {
        "nav": nav,
        "metrics": metrics,
        "weights_history": weights_history,
        "daily_weights": daily_weights,
        "selected_split_history": selected_split_history,
        "prices": prices,
        "benchmark": benchmark,
        "set_index": _load_set_index_thb(prices.index),
    }


def apply_asset_level_daily_exposure(results: dict[str, object]) -> dict[str, object]:
    prices: pd.DataFrame = results["prices"]
    benchmark: pd.Series = results["benchmark"]
    set_index: pd.Series = results["set_index"]
    daily_weights: pd.DataFrame = results["daily_weights"]
    if daily_weights.empty:
        return {"nav": pd.Series(dtype=float), "metrics": pd.Series(dtype=float), "exposure_history": pd.DataFrame(), "effective_weights": pd.DataFrame()}

    config = _best_signal_config()
    spy_cfg = config.loc["SPY"] if "SPY" in config.index else pd.Series({"MA Period": 300, "Below Exposure": 0.50})
    gold_cfg = config.loc["GOLD"] if "GOLD" in config.index else pd.Series({"MA Period": 50, "Below Exposure": 1.00})
    btc_cfg = config.loc["BTC"] if "BTC" in config.index else pd.Series({"MA Period": 50, "Below Exposure": 0.00})
    us_exposure = _close_trend_exposure(benchmark, int(spy_cfg["MA Period"]), float(spy_cfg["Below Exposure"]))
    th_exposure = _close_trend_exposure(set_index, int(spy_cfg["MA Period"]), float(spy_cfg["Below Exposure"]))
    gold_exposure = _close_trend_exposure(prices["GC=F"], int(gold_cfg["MA Period"]), float(gold_cfg["Below Exposure"]))
    btc_exposure = _close_trend_exposure(prices["BTC-USD"], int(btc_cfg["MA Period"]), float(btc_cfg["Below Exposure"]))

    exposure = pd.DataFrame(1.0, index=daily_weights.index, columns=daily_weights.columns, dtype=float)
    us_cols = [column for column in exposure.columns if not str(column).endswith(".BK") and column not in OVERLAY_ASSETS]
    th_cols = [column for column in exposure.columns if str(column).endswith(".BK")]
    if us_cols:
        exposure.loc[:, us_cols] = us_exposure.reindex(exposure.index).ffill().fillna(1.0).to_numpy()[:, None]
    if th_cols:
        exposure.loc[:, th_cols] = th_exposure.reindex(exposure.index).ffill().fillna(1.0).to_numpy()[:, None]
    if "GC=F" in exposure.columns:
        exposure["GC=F"] = gold_exposure.reindex(exposure.index).ffill().fillna(1.0)
    if "BTC-USD" in exposure.columns:
        exposure["BTC-USD"] = btc_exposure.reindex(exposure.index).ffill().fillna(1.0)
    if "BIL" in exposure.columns:
        exposure["BIL"] = 1.0

    effective_weights = daily_weights.mul(exposure).clip(lower=0.0)
    asset_returns = prices.pct_change(fill_method=None).reindex(effective_weights.index).reindex(columns=effective_weights.columns).fillna(0.0)
    exposed_returns = asset_returns.mul(effective_weights, axis=1).sum(axis=1)
    nav = (1.0 + exposed_returns).cumprod().rename("US+TH Daily Exposure NAV")
    metrics = _metrics_for_nav(nav, benchmark)
    metrics["Average Exposure"] = float(effective_weights.sum(axis=1).mean())
    metrics["Minimum Exposure"] = float(effective_weights.sum(axis=1).min())
    exposure_history = pd.DataFrame(
        {
            "US Stock Exposure From SPY": us_exposure,
            "TH Stock Exposure From SET": th_exposure,
            "Gold Exposure From Gold Trend": gold_exposure,
            "BTC Exposure From BTC Trend": btc_exposure,
            "BIL Exposure": 1.0,
        }
    ).reindex(effective_weights.index).ffill().fillna(1.0)
    effective_weights["Cash / Reduced Exposure"] = (1.0 - effective_weights.sum(axis=1)).clip(lower=0.0)
    return {"nav": nav, "metrics": metrics, "exposure_history": exposure_history, "effective_weights": effective_weights}


def _latest_weights_frame(results: dict[str, object], strategy: str) -> pd.DataFrame:
    history = results["weights_history"]
    if not history:
        return pd.DataFrame()
    latest_date = max(history)
    latest = history[latest_date].rename("Portfolio Weight").reset_index()
    latest.columns = ["Asset", "Portfolio Weight"]
    latest["Date"] = pd.Timestamp(latest_date).date().isoformat()
    latest["Strategy"] = strategy
    latest["Sleeve"] = "US Equity"
    latest.loc[latest["Asset"].astype(str).str.endswith(".BK"), "Sleeve"] = "TH Equity"
    latest.loc[latest["Asset"].eq("GC=F"), "Sleeve"] = "Gold"
    latest.loc[latest["Asset"].eq("BTC-USD"), "Sleeve"] = "BTC"
    latest.loc[latest["Asset"].eq("BIL"), "Sleeve"] = "BIL"
    return latest.sort_values("Portfolio Weight", ascending=False)


def main() -> None:
    paths = default_paths(ROOT)
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    raw = run_backtest()
    daily = apply_asset_level_daily_exposure(raw)

    raw_row = raw["metrics"].copy()
    raw_row["Strategy"] = "US+TH Mean Covariance + Gold/BTC/BIL Gold30"
    raw_row["Daily Exposure"] = "No"
    daily_row = daily["metrics"].copy()
    daily_row["Strategy"] = "US+TH Mean Covariance + Gold/BTC/BIL Gold30 + asset daily exposure"
    daily_row["Daily Exposure"] = "US=SPY trend, TH=SET trend, Gold/BTC own trend"
    summary = pd.DataFrame([raw_row, daily_row]).sort_values("Sharpe", ascending=False)
    for column, value in {
        "US Assets": US_ASSETS,
        "TH Assets": TH_ASSETS,
        "US Stock Cap": STOCK_CAP,
        "TH Stock Cap": STOCK_CAP,
        "Gold Cap": GOLD_CAP,
        "BTC Cap": BTC_CAP,
        "BIL Cap": BIL_CAP,
        "Covariance Model": "Sample Covariance",
        "Objective": "mean_variance + mom_63",
    }.items():
        summary[column] = value
    summary.to_csv(paths.result_dir / "mean_covariance_us_th_gold_btc_bil_gold30_summary.csv", index=False)
    pd.DataFrame(
        {
            "Raw": raw["nav"],
            "Asset Daily Exposure": daily["nav"],
        }
    ).dropna(how="all").to_csv(paths.result_dir / "mean_covariance_us_th_gold_btc_bil_gold30_curves.csv")
    _latest_weights_frame(raw, "US+TH Mean Covariance + Gold/BTC/BIL Gold30").to_csv(
        paths.result_dir / "mean_covariance_us_th_gold_btc_bil_gold30_latest_weights.csv",
        index=False,
    )
    daily["exposure_history"].to_csv(paths.result_dir / "mean_covariance_us_th_gold_btc_bil_gold30_daily_exposure_history.csv")
    daily["effective_weights"].to_csv(paths.result_dir / "mean_covariance_us_th_gold_btc_bil_gold30_daily_effective_weights.csv")

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    latest = _latest_weights_frame(raw, "US+TH Mean Covariance + Gold/BTC/BIL Gold30")
    print(latest.loc[latest["Portfolio Weight"] > 1e-8, ["Asset", "Sleeve", "Portfolio Weight"]].to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
