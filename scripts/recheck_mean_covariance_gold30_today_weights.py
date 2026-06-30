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
    lag_close_signal_to_next_session,
    load_cached_market_data,
    load_overlay_compare_prices,
    load_sp500_membership_intervals,
    optimize_portfolio,
    select_point_in_time_universe,
)


START_DATE = "2016-01-01"
LOOKBACK_DAYS = 504
US_ASSETS = 30
STOCK_CAP = 0.08
GOLD_CAP = 0.30
BTC_CAP = 0.05
BIL_CAP = 0.00
OVERLAY_ASSETS = ["GC=F", "BTC-USD", "BIL"]
FEATURE_FLAGS = {"resid_vol": False, "drawdown": False, "downside_beta": False}
STRATEGY = "Mean Covariance Gold30 stock cap 8 mom_63 + asset-level daily exposure"


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


def _latest_common_close(overlay: pd.DataFrame) -> pd.Timestamp:
    required = ["SPY", "^VIX", "GC=F", "BIL"]
    common = overlay.dropna(subset=required)
    if common.empty:
        raise ValueError("No common latest close found for SPY, ^VIX, Gold, and BIL.")
    return pd.Timestamp(common.index.max())


def main() -> None:
    paths = default_paths(ROOT)
    paths.result_dir.mkdir(parents=True, exist_ok=True)

    source_cols = set(_parquet_column_names(str(paths.source_cache_root / "prices.parquet")))
    local_prices = paths.local_cache_root / "extra_prices.parquet"
    if local_prices.exists():
        source_cols |= set(_parquet_column_names(str(local_prices)))
    us_all = [
        ticker
        for ticker in load_sp500_membership_intervals(paths)["ticker"].dropna().astype(str).drop_duplicates()
        if ticker in source_cols
    ]
    overlay_full = load_overlay_compare_prices(
        paths,
        start_date=START_DATE,
        tickers=["SPY", "^VIX", *OVERLAY_ASSETS],
    ).sort_index()
    as_of = _latest_common_close(overlay_full)

    cached_panel = load_cached_market_data(paths, tickers=us_all + ["SPY", "^VIX"])
    stock_prices = cached_panel["prices"].loc[START_DATE:as_of].reindex(columns=us_all).sort_index().ffill()
    stock_volumes = cached_panel["volumes"].loc[START_DATE:as_of].reindex(columns=us_all).fillna(0.0)
    overlay = overlay_full.loc[START_DATE:as_of, ["SPY", "^VIX", *OVERLAY_ASSETS]].sort_index().ffill()

    prices = pd.concat(
        [stock_prices, overlay[OVERLAY_ASSETS].reindex(stock_prices.index).ffill()],
        axis=1,
    )
    volumes = stock_volumes.reindex(columns=prices.columns).fillna(0.0)
    volumes.loc[:, OVERLAY_ASSETS] = 1.0
    benchmark = overlay["SPY"].reindex(prices.index).ffill().rename("benchmark")
    vol_proxy = overlay["^VIX"].reindex(prices.index).ffill().rename("vol_proxy")
    common_index = prices.index.intersection(benchmark.dropna().index).intersection(vol_proxy.dropna().index)
    prices = prices.reindex(common_index).ffill()
    volumes = volumes.reindex(common_index).fillna(0.0)
    benchmark = benchmark.reindex(common_index)
    vol_proxy = vol_proxy.reindex(common_index)

    loc = prices.index.get_loc(as_of)
    train_index = prices.index[max(0, loc - LOOKBACK_DAYS + 1) : loc + 1]
    sp500_pool = [
        ticker
        for ticker in get_sp500_members_as_of(as_of, paths)
        if ticker in us_all and ticker in prices.columns
    ]
    us_selected = select_point_in_time_universe(
        prices.loc[train_index, sp500_pool],
        volumes.loc[train_index, sp500_pool],
        sp500_pool,
        n_assets=US_ASSETS,
    )
    current_assets = list(dict.fromkeys(us_selected + OVERLAY_ASSETS))
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    train_returns = returns.reindex(train_index)[current_assets].dropna(
        axis=1,
        thresh=max(int(0.85 * len(train_index)), 60),
    )
    current_assets = train_returns.columns.tolist()
    benchmark_ret = benchmark.pct_change(fill_method=None).rename("benchmark")
    vol_proxy_ret = vol_proxy.pct_change(fill_method=None).rename("vol_proxy")

    features = compute_feature_table(
        train_returns,
        benchmark_ret.reindex(train_index),
        vol_proxy_ret.reindex(train_index),
        prices.loc[train_index, current_assets],
        include_momentum_features=True,
        feature_flags=FEATURE_FLAGS,
    )
    momentum_signal = build_momentum_signal(features, mode="mom_63")
    sample_cov = train_returns.cov().reindex(index=current_assets, columns=current_assets).fillna(0.0)
    asset_caps = {asset: STOCK_CAP for asset in current_assets}
    asset_caps.update({"GC=F": GOLD_CAP, "BTC-USD": BTC_CAP, "BIL": BIL_CAP})
    asset_caps = {asset: cap for asset, cap in asset_caps.items() if asset in current_assets}
    raw_weights = optimize_portfolio(
        sample_cov,
        momentum_signal,
        max_weight=max(STOCK_CAP, GOLD_CAP, BTC_CAP, BIL_CAP),
        objective_mode="mean_variance",
        asset_caps=asset_caps,
    ).sort_values(ascending=False)

    config = _best_signal_config()
    spy_cfg = config.loc["SPY"] if "SPY" in config.index else pd.Series({"MA Period": 300, "Below Exposure": 0.50})
    gold_cfg = config.loc["GOLD"] if "GOLD" in config.index else pd.Series({"MA Period": 50, "Below Exposure": 1.00})
    btc_cfg = config.loc["BTC"] if "BTC" in config.index else pd.Series({"MA Period": 50, "Below Exposure": 0.00})
    stock_exposure = _close_trend_exposure(benchmark, int(spy_cfg["MA Period"]), float(spy_cfg["Below Exposure"]))
    gold_exposure = _close_trend_exposure(prices["GC=F"], int(gold_cfg["MA Period"]), float(gold_cfg["Below Exposure"]))
    btc_exposure = _close_trend_exposure(prices["BTC-USD"], int(btc_cfg["MA Period"]), float(btc_cfg["Below Exposure"]))

    exposure_by_asset = pd.Series(1.0, index=raw_weights.index, dtype=float)
    stock_assets = [asset for asset in raw_weights.index if asset not in OVERLAY_ASSETS]
    exposure_by_asset.loc[stock_assets] = float(stock_exposure.reindex([as_of]).ffill().iloc[-1])
    if "GC=F" in exposure_by_asset.index:
        exposure_by_asset.loc["GC=F"] = float(gold_exposure.reindex([as_of]).ffill().iloc[-1])
    if "BTC-USD" in exposure_by_asset.index:
        exposure_by_asset.loc["BTC-USD"] = float(btc_exposure.reindex([as_of]).ffill().iloc[-1])
    if "BIL" in exposure_by_asset.index:
        exposure_by_asset.loc["BIL"] = 1.0

    effective = raw_weights.mul(exposure_by_asset).clip(lower=0.0)
    cash_weight = max(0.0, 1.0 - float(effective.sum()))
    if cash_weight > 1e-12:
        effective.loc["Cash / Reduced Exposure"] = cash_weight

    latest = effective.rename("Effective Weight").reset_index().rename(columns={"index": "Asset"})
    latest["Raw Optimizer Weight"] = latest["Asset"].map(raw_weights).fillna(0.0)
    latest["Daily Exposure"] = latest["Asset"].map(exposure_by_asset).fillna(1.0)
    latest["Date"] = as_of.date().isoformat()
    latest["Strategy"] = STRATEGY
    latest["Sleeve"] = "US Equity"
    latest.loc[latest["Asset"].eq("GC=F"), "Sleeve"] = "Gold"
    latest.loc[latest["Asset"].eq("BTC-USD"), "Sleeve"] = "BTC"
    latest.loc[latest["Asset"].eq("BIL"), "Sleeve"] = "BIL"
    latest.loc[latest["Asset"].eq("Cash / Reduced Exposure"), "Sleeve"] = "Cash / Reduced Exposure"
    latest["Effective Weight %"] = latest["Effective Weight"].mul(100.0)
    latest["Raw Optimizer Weight %"] = latest["Raw Optimizer Weight"].mul(100.0)
    latest = latest.loc[latest["Effective Weight"].abs() > 1e-12].sort_values("Effective Weight", ascending=False)

    sleeve_latest = latest.groupby("Sleeve", as_index=False)["Effective Weight"].sum()
    sleeve_latest["Date"] = as_of.date().isoformat()
    sleeve_latest["Effective Weight %"] = sleeve_latest["Effective Weight"].mul(100.0)
    sleeve_latest = sleeve_latest.sort_values("Effective Weight", ascending=False)

    meta = pd.DataFrame(
        [
            {
                "Date": as_of.date().isoformat(),
                "Strategy": STRATEGY,
                "Train Start": pd.Timestamp(train_index.min()).date().isoformat(),
                "Train End": pd.Timestamp(train_index.max()).date().isoformat(),
                "Lookback Days": len(train_index),
                "Selected US Assets": len([asset for asset in current_assets if asset not in OVERLAY_ASSETS]),
                "US Stock Cap": STOCK_CAP,
                "Signal Mode": "mom_63",
                "Stock Exposure": float(stock_exposure.loc[as_of]),
                "Gold Exposure": float(gold_exposure.loc[as_of]),
                "BTC Exposure": float(btc_exposure.loc[as_of]) if "BTC-USD" in prices.columns else np.nan,
                "Weight Timing Note": "Fresh recheck from latest common close in cache; signals are lagged by one session.",
            }
        ]
    )

    latest_path = paths.result_dir / "mean_covariance_gold30_asset_daily_latest_effective_weights.csv"
    today_path = paths.result_dir / "mean_covariance_gold30_asset_daily_recheck_today_weights.csv"
    sleeve_path = paths.result_dir / "mean_covariance_gold30_asset_daily_recheck_today_sleeve_weights.csv"
    history_path = paths.result_dir / "mean_covariance_gold30_asset_daily_sleeve_weight_history.csv"
    meta_path = paths.result_dir / "mean_covariance_gold30_asset_daily_recheck_today_meta.csv"
    latest.to_csv(latest_path, index=False)
    latest.to_csv(today_path, index=False)
    sleeve_latest.to_csv(sleeve_path, index=False)
    history_row = sleeve_latest.set_index("Sleeve")["Effective Weight"].reindex(
        ["US Equity", "Gold", "BTC", "BIL", "Cash / Reduced Exposure"],
        fill_value=0.0,
    )
    history_row.name = as_of
    if history_path.exists():
        history = pd.read_csv(history_path, index_col=0, parse_dates=True)
        history = pd.concat([history, history_row.to_frame().T], axis=0)
        history = history[~history.index.duplicated(keep="last")].sort_index()
    else:
        history = history_row.to_frame().T
    history.to_csv(history_path)
    meta.to_csv(meta_path, index=False)

    print(meta.to_string(index=False))
    print(latest[["Asset", "Sleeve", "Effective Weight", "Raw Optimizer Weight", "Daily Exposure"]].to_string(index=False))
    print(sleeve_latest.to_string(index=False))


if __name__ == "__main__":
    main()
