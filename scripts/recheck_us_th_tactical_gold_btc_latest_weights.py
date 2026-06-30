from __future__ import annotations

from pathlib import Path
import json
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
YFINANCE_CACHE = ROOT / ".yfinance"
YFINANCE_CACHE.mkdir(parents=True, exist_ok=True)
yf.set_tz_cache_location(str(YFINANCE_CACHE))

from dynamic_factor_copula import (  # noqa: E402
    default_paths,
    load_set100_membership_intervals,
    load_sp500_membership_intervals,
)
from us_th_pit_reselect_utils import drop_duplicate_share_classes, run_joint_pit_reselect_model  # noqa: E402

from run_us_th_tactical_perf_momentum import (  # noqa: E402
    FEATURE_FLAGS,
    PRIMARY_MODEL,
    RISK_FREE_RATE,
    START_DATE,
    _close_trend_exposure,
    _daily_weight_from_monthly,
    _monthly_returns,
    _monthly_weight_signal,
    _nav_to_returns,
)


SELECTED_MIX = {"Equity": 0.65, "Gold": 0.25, "BTC": 0.10}
STRATEGY = "Final Best Sharpe Tactical TH/Gold/BTC 65/25/10 Gold crash protection"
RESULT_PREFIX = "us_th_tactical_perf_momentum_final_best"
FRESH_LOOKBACK_CALENDAR_DAYS = 850
FRESH_START_DATE = (pd.Timestamp.today().normalize() - pd.Timedelta(days=FRESH_LOOKBACK_CALENDAR_DAYS)).date().isoformat()
FRESH_BATCH_SIZE = 80
OVERLAY_TICKERS = ["SPY", "^VIX", "GC=F", "BTC-USD", "USDTHB=X", "^SET.BK"]
GOLD_DD_WINDOW = 252
GOLD_WARN_DD = -0.08
GOLD_WARN_EXPOSURE = 0.50
GOLD_CRASH_DD = -0.20
GOLD_CRASH_EXPOSURE = 0.50
GOLD_RECOVERY_DD = -0.05
GOLD_PANIC_DD = -0.30
GOLD_PANIC_MA = 200
GOLD_PANIC_MOM = 63


def _latest_common_close() -> pd.Timestamp:
    paths = default_paths(ROOT)
    overlay = load_overlay_compare_prices(
        paths,
        start_date=START_DATE,
        tickers=["SPY", "^VIX", "GC=F", "BTC-USD", "USDTHB=X"],
    ).sort_index()
    overlay_latest = overlay.dropna(subset=["SPY", "^VIX", "GC=F", "BTC-USD", "USDTHB=X"]).index.max()
    set_latest = pd.read_parquet(paths.local_cache_root / "extra_prices.parquet")["^SET.BK"].dropna().index.max()
    return pd.Timestamp(min(overlay_latest, set_latest))


def _extract_yfinance_close_volume(raw: pd.DataFrame, tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}
    for ticker in tickers:
        sub = None
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker in raw.columns.get_level_values(0):
                sub = raw[ticker]
            elif "Close" in raw.columns.get_level_values(0) and ticker in raw["Close"].columns:
                close = raw["Close"][ticker].dropna().rename(ticker)
                volume = raw.get("Volume", pd.DataFrame(index=raw.index)).get(ticker, pd.Series(index=raw.index, dtype=float))
                prices[ticker] = close
                volumes[ticker] = volume.reindex(close.index).fillna(0.0).rename(ticker)
                continue
        elif "Close" in raw.columns and len(tickers) == 1:
            sub = raw
        if sub is None or "Close" not in sub.columns:
            continue
        close = sub["Close"].dropna().rename(ticker)
        if close.empty:
            continue
        volume = sub.get("Volume", pd.Series(index=sub.index, dtype=float)).reindex(close.index).fillna(0.0).rename(ticker)
        prices[ticker] = close
        volumes[ticker] = volume
    return pd.DataFrame(prices).sort_index(), pd.DataFrame(volumes).sort_index()


def _download_yfinance_panel(tickers: list[str], start_date: str, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    price_frames = []
    volume_frames = []
    unique = list(dict.fromkeys(tickers))
    for i in range(0, len(unique), FRESH_BATCH_SIZE):
        batch = unique[i : i + FRESH_BATCH_SIZE]
        raw = yf.download(
            batch,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        prices, volumes = _extract_yfinance_close_volume(raw, batch)
        if not prices.empty:
            price_frames.append(prices)
            volume_frames.append(volumes)
    if not price_frames:
        raise RuntimeError("Fresh yfinance download returned no usable price data.")
    prices = pd.concat(price_frames, axis=1).sort_index()
    volumes = pd.concat(volume_frames, axis=1).sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated()].ffill()
    volumes = volumes.loc[:, ~volumes.columns.duplicated()].fillna(0.0)
    return prices, volumes


def _gold_crash_protection_exposure(gold_price: pd.Series) -> pd.Series:
    price = gold_price.astype(float).sort_index().ffill()
    rolling_high = price.rolling(GOLD_DD_WINDOW, min_periods=max(20, GOLD_DD_WINDOW // 4)).max()
    drawdown = price.div(rolling_high).sub(1.0)
    panic_ma = price.rolling(GOLD_PANIC_MA, min_periods=max(20, GOLD_PANIC_MA // 4)).mean()
    panic_mom = price.pct_change(GOLD_PANIC_MOM)
    active = 1.0
    values = []
    for date, dd in drawdown.items():
        panic = (
            pd.notna(dd)
            and dd <= GOLD_PANIC_DD
            and pd.notna(panic_ma.loc[date])
            and price.loc[date] < panic_ma.loc[date]
            and pd.notna(panic_mom.loc[date])
            and panic_mom.loc[date] < 0.0
        )
        if pd.isna(dd):
            active = 1.0
        elif panic:
            active = 0.0
        elif dd <= GOLD_CRASH_DD:
            active = GOLD_CRASH_EXPOSURE
        elif dd <= GOLD_WARN_DD:
            active = min(active, GOLD_WARN_EXPOSURE)
        elif dd >= GOLD_RECOVERY_DD:
            active = 1.0
        values.append(active)
    return pd.Series(values, index=drawdown.index, name="Gold Crash Protection Exposure").shift(1).fillna(1.0)


def _fresh_us_th_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str], list[str], pd.DataFrame, pd.Timestamp]:
    paths = default_paths(ROOT)
    tomorrow = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).date().isoformat()
    sp500_intervals = load_sp500_membership_intervals(paths)
    set100_intervals = load_set100_membership_intervals(paths)
    all_us = drop_duplicate_share_classes(sp500_intervals["ticker"].dropna().astype(str).drop_duplicates().tolist())
    all_th = set100_intervals["ticker"].dropna().astype(str).drop_duplicates().tolist()
    overlay_prices, overlay_volumes = _download_yfinance_panel(OVERLAY_TICKERS, FRESH_START_DATE, tomorrow)
    prices_raw, volumes_raw = _download_yfinance_panel(all_us + all_th, FRESH_START_DATE, tomorrow)
    prices_raw = pd.concat([prices_raw, overlay_prices], axis=1).loc[:, lambda frame: ~frame.columns.duplicated()].sort_index()
    volumes_raw = pd.concat([volumes_raw, overlay_volumes], axis=1).loc[:, lambda frame: ~frame.columns.duplicated()].sort_index()
    required_overlay = ["SPY", "^VIX", "GC=F", "BTC-USD", "USDTHB=X", "^SET.BK"]
    missing_overlay = [ticker for ticker in required_overlay if ticker not in prices_raw.columns]
    if missing_overlay:
        raise RuntimeError(f"Fresh yfinance missing required overlay tickers: {missing_overlay}")
    overlay_common = prices_raw[required_overlay].dropna()
    if overlay_common.empty:
        raise RuntimeError("Fresh yfinance has no common overlay close.")
    as_of = pd.Timestamp(overlay_common.index.max())
    fx = prices_raw["USDTHB=X"].ffill()
    us_cols = [ticker for ticker in all_us if ticker in prices_raw.columns]
    th_cols = [ticker for ticker in all_th if ticker in prices_raw.columns]
    us_price_df = prices_raw[us_cols].mul(fx, axis=0)
    th_price_df = prices_raw[th_cols]
    thb_prices = pd.concat([us_price_df, th_price_df], axis=1).loc[:as_of].ffill()
    volumes = volumes_raw.reindex(thb_prices.index).reindex(columns=thb_prices.columns).fillna(0.0)
    benchmark = prices_raw["SPY"].mul(fx).reindex(thb_prices.index).ffill().rename("benchmark")
    vol_proxy = prices_raw["^VIX"].reindex(thb_prices.index).ffill().rename("vol_proxy")
    common_index = thb_prices.index.intersection(benchmark.dropna().index).intersection(vol_proxy.dropna().index)
    thb_prices = thb_prices.reindex(common_index).ffill().dropna(how="all")
    volumes = volumes.reindex(thb_prices.index).fillna(0.0)
    benchmark = benchmark.reindex(thb_prices.index).ffill()
    vol_proxy = vol_proxy.reindex(thb_prices.index).ffill()
    overlay = prices_raw[required_overlay].loc[:as_of].ffill()
    return thb_prices, volumes, benchmark, vol_proxy, us_cols, th_cols, overlay, as_of


def _run_us_sleeve(end_date: str) -> dict[str, object]:
    prices, volumes, benchmark, vol_proxy, us_all, _ = load_full_us_th_thb_panel(
        include_overlay_assets=False,
        start_date=START_DATE,
        end_date=end_date,
    )
    return run_joint_pit_reselect_model(
        prices=prices,
        volumes=volumes,
        benchmark=benchmark,
        vol_proxy=vol_proxy,
        us_all=us_all,
        th_all=[],
        us_assets=30,
        th_assets=0,
        objective_mode="mean_variance",
        max_weight=0.08,
        include_overlay_assets=False,
        include_momentum=True,
        momentum_signal_mode="mom_63",
    )


def _run_th_sleeve(end_date: str) -> dict[str, object]:
    return backtest_dynamic_factor_copula(
        start_date=START_DATE,
        end_date=end_date,
        n_assets=30,
        n_clusters=4,
        lookback_days=504,
        rebalance_freq="ME",
        max_weight=0.08,
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


def _load_overlay_prices(index: pd.DatetimeIndex, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = default_paths(ROOT)
    overlay = load_overlay_compare_prices(
        paths,
        start_date=START_DATE,
        end_date=end_date,
        tickers=["SPY", "^VIX", "GC=F", "BTC-USD", "USDTHB=X"],
    ).sort_index().ffill()
    fx = overlay["USDTHB=X"].ffill()
    set_index = pd.read_parquet(paths.local_cache_root / "extra_prices.parquet")["^SET.BK"].loc[START_DATE:end_date].sort_index().ffill()
    full_index = index.union(overlay.index).union(set_index.index).sort_values()
    overlay = overlay.reindex(full_index).ffill()
    set_index = set_index.reindex(full_index).ffill()
    thb_prices = pd.DataFrame(
        {
            "S&P 500 ETF THB": overlay["SPY"].mul(fx.reindex(full_index).ffill()),
            "SET Index THB proxy": set_index,
            "Gold": overlay["GC=F"].mul(fx.reindex(full_index).ffill()),
            "BTC": overlay["BTC-USD"].mul(fx.reindex(full_index).ffill()),
        },
        index=full_index,
    ).reindex(index).ffill()
    signal_prices = pd.DataFrame(
        {
            "US Equity": overlay["SPY"],
            "TH Equity": set_index,
            "Gold": overlay["GC=F"],
            "BTC": overlay["BTC-USD"],
        },
        index=full_index,
    ).reindex(index).ffill()
    return thb_prices, signal_prices


def _coingecko_btc_daily(days: int = 120) -> pd.Series:
    end = int(time.time())
    start = end - days * 86_400
    url = (
        "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
        f"?vs_currency=usd&from={start}&to={end}"
    )
    data = json.loads(urllib.request.urlopen(url, timeout=20).read().decode())
    raw = pd.DataFrame(data["prices"], columns=["ts", "price"])
    raw["Date"] = pd.to_datetime(raw["ts"], unit="ms", utc=True).dt.tz_convert(None)
    return raw.set_index("Date")["price"].resample("D").last().dropna().rename("BTC-USD")


def _latest_btc_exposure_from_coingecko() -> dict[str, object] | None:
    try:
        btc = _coingecko_btc_daily()
    except Exception:
        return None
    if btc.empty:
        return None
    ma50 = btc.rolling(50, min_periods=20).mean()
    latest_date = pd.Timestamp(btc.index[-1])
    latest_price = float(btc.iloc[-1])
    latest_ma = float(ma50.iloc[-1])
    exposure = 1.0 if latest_price >= latest_ma else 0.0
    return {
        "date": latest_date,
        "price": latest_price,
        "ma50": latest_ma,
        "exposure": exposure,
        "source": "CoinGecko market_chart/range",
    }


def _latest_weights(history: dict[pd.Timestamp, pd.Series], sleeve: str, multiplier: float, date: pd.Timestamp) -> pd.DataFrame:
    latest_date = max(history)
    latest = history[latest_date].rename("Internal Weight").reset_index()
    latest.columns = ["Asset", "Internal Weight"]
    latest["Sleeve"] = sleeve
    latest["Sleeve Multiplier"] = multiplier
    latest["Effective Weight"] = latest["Internal Weight"].mul(multiplier)
    latest["Internal Weight Date"] = pd.Timestamp(latest_date).date().isoformat()
    latest["Date"] = date.date().isoformat()
    return latest


def main() -> None:
    paths = default_paths(ROOT)
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    prices, volumes, benchmark, vol_proxy, us_all, th_all, overlay_raw, as_of = _fresh_us_th_panel()
    end_date = as_of.date().isoformat()

    us_results = run_joint_pit_reselect_model(
        prices=prices,
        volumes=volumes,
        benchmark=benchmark,
        vol_proxy=vol_proxy,
        us_all=us_all,
        th_all=[],
        us_assets=30,
        th_assets=0,
        objective_mode="mean_variance",
        max_weight=0.08,
        include_overlay_assets=False,
        include_momentum=True,
        momentum_signal_mode="mom_63",
    )
    set_benchmark = overlay_raw["^SET.BK"].reindex(prices.index).ffill().rename("benchmark")
    th_results = run_joint_pit_reselect_model(
        prices=prices,
        volumes=volumes,
        benchmark=set_benchmark,
        vol_proxy=vol_proxy,
        us_all=[],
        th_all=th_all,
        us_assets=0,
        th_assets=30,
        objective_mode="mean_variance",
        max_weight=0.08,
        include_overlay_assets=False,
        include_momentum=True,
        momentum_signal_mode="mom_63",
    )

    us_nav = us_results["nav"][PRIMARY_MODEL].dropna()
    th_nav = th_results["nav"][PRIMARY_MODEL].dropna()
    common_index = us_nav.index.union(th_nav.index).sort_values()
    daily_returns = pd.DataFrame(
        {
            "US PIT optimized sleeve THB": _nav_to_returns(us_nav).reindex(common_index).fillna(0.0),
            "TH PIT optimized sleeve THB": _nav_to_returns(th_nav).reindex(common_index).fillna(0.0),
        }
    ).loc[:end_date].fillna(0.0)
    fx = overlay_raw["USDTHB=X"].reindex(daily_returns.index).ffill()
    overlay_prices = pd.DataFrame(
        {
            "S&P 500 ETF THB": overlay_raw["SPY"].reindex(daily_returns.index).ffill().mul(fx),
            "SET Index THB proxy": overlay_raw["^SET.BK"].reindex(daily_returns.index).ffill(),
            "Gold": overlay_raw["GC=F"].reindex(daily_returns.index).ffill().mul(fx),
            "BTC": overlay_raw["BTC-USD"].reindex(daily_returns.index).ffill().mul(fx),
        },
        index=daily_returns.index,
    ).ffill()
    signal_prices = pd.DataFrame(
        {
            "US Equity": overlay_raw["SPY"].reindex(daily_returns.index).ffill(),
            "TH Equity": overlay_raw["^SET.BK"].reindex(daily_returns.index).ffill(),
            "Gold": overlay_raw["GC=F"].reindex(daily_returns.index).ffill(),
            "BTC": overlay_raw["BTC-USD"].reindex(daily_returns.index).ffill(),
        },
        index=daily_returns.index,
    ).ffill()

    sleeve_curves = (1.0 + daily_returns).cumprod().mul(10_000.0)
    monthly = _monthly_returns(
        pd.concat(
            [
                sleeve_curves,
                overlay_prices[["S&P 500 ETF THB", "SET Index THB proxy"]]
                .div(overlay_prices[["S&P 500 ETF THB", "SET Index THB proxy"]].iloc[0])
                .mul(10_000.0),
            ],
            axis=1,
        )
    )
    th_monthly_weight = _monthly_weight_signal(
        monthly,
        mode="relative_return",
        allocation_method="binary",
        lookback=1,
        th_weight=0.30,
        entry=0.0,
        exit_threshold=0.0,
        min_hold=0,
        exit_confirm=1,
        us_col="S&P 500 ETF THB",
        th_col="SET Index THB proxy",
    )
    th_daily_weight = _daily_weight_from_monthly(th_monthly_weight, daily_returns.index)
    th_tactical_weight = float(th_daily_weight.loc[:as_of].iloc[-1])

    exposure = pd.DataFrame(
        {
            "US Equity": _close_trend_exposure(signal_prices["US Equity"], 300, 0.50),
            "TH Equity": _close_trend_exposure(signal_prices["TH Equity"], 200, 0.00),
            "Gold": _gold_crash_protection_exposure(signal_prices["Gold"]),
            "BTC": _close_trend_exposure(signal_prices["BTC"], 50, 0.00),
        }
    ).reindex(daily_returns.index).ffill().fillna(1.0).clip(0.0, 1.0)
    latest_exposure = exposure.loc[:as_of].iloc[-1]
    output_date = as_of

    raw_sleeve = pd.Series(
        {
            "US Equity": SELECTED_MIX["Equity"] * (1.0 - th_tactical_weight),
            "TH Equity": SELECTED_MIX["Equity"] * th_tactical_weight,
            "Gold": SELECTED_MIX["Gold"],
            "BTC": SELECTED_MIX["BTC"],
        },
        dtype=float,
    )
    effective_sleeve = raw_sleeve.mul(latest_exposure)
    effective_sleeve["Cash / Reduced Exposure"] = max(0.0, 1.0 - float(effective_sleeve.sum()))

    us_multiplier = float(effective_sleeve["US Equity"])
    th_multiplier = float(effective_sleeve["TH Equity"])
    security_frames = [
        _latest_weights(us_results["weights_history"][PRIMARY_MODEL], "US Equity", us_multiplier, output_date),
        _latest_weights(th_results["weights_history"][PRIMARY_MODEL], "TH Equity", th_multiplier, output_date),
        pd.DataFrame(
            [
                {
                    "Asset": "GC=F",
                    "Internal Weight": 1.0,
                    "Sleeve": "Gold",
                    "Sleeve Multiplier": float(effective_sleeve["Gold"]),
                    "Effective Weight": float(effective_sleeve["Gold"]),
                    "Internal Weight Date": output_date.date().isoformat(),
                    "Date": output_date.date().isoformat(),
                },
                {
                    "Asset": "BTC-USD",
                    "Internal Weight": 1.0,
                    "Sleeve": "BTC",
                    "Sleeve Multiplier": float(effective_sleeve["BTC"]),
                    "Effective Weight": float(effective_sleeve["BTC"]),
                    "Internal Weight Date": output_date.date().isoformat(),
                    "Date": output_date.date().isoformat(),
                },
                {
                    "Asset": "Cash / Reduced Exposure",
                    "Internal Weight": 1.0,
                    "Sleeve": "Cash / Reduced Exposure",
                    "Sleeve Multiplier": float(effective_sleeve["Cash / Reduced Exposure"]),
                    "Effective Weight": float(effective_sleeve["Cash / Reduced Exposure"]),
                    "Internal Weight Date": output_date.date().isoformat(),
                    "Date": output_date.date().isoformat(),
                },
            ]
        ),
    ]
    security_weights = pd.concat(security_frames, ignore_index=True)
    security_weights["Strategy"] = STRATEGY
    sleeve_raw_map = raw_sleeve.to_dict()
    sleeve_exposure_map = latest_exposure.to_dict()
    security_weights["Raw Sleeve Weight"] = security_weights["Sleeve"].map(sleeve_raw_map).fillna(security_weights["Sleeve Multiplier"])
    security_weights["Daily Exposure"] = security_weights["Sleeve"].map(sleeve_exposure_map).fillna(1.0)
    security_weights["Effective Weight %"] = security_weights["Effective Weight"].mul(100.0)
    us_internal_date = pd.Timestamp(max(us_results["weights_history"][PRIMARY_MODEL])).date().isoformat()
    th_internal_date = pd.Timestamp(max(th_results["weights_history"][PRIMARY_MODEL])).date().isoformat()
    keep_zero_sleeves = {"Gold", "BTC", "Cash / Reduced Exposure"}
    security_weights = security_weights.loc[
        security_weights["Effective Weight"].abs().gt(1e-12) | security_weights["Sleeve"].isin(keep_zero_sleeves)
    ].sort_values(
        "Effective Weight",
        ascending=False,
    )

    sleeve_weights = effective_sleeve.rename("Effective Weight").reset_index().rename(columns={"index": "Sleeve"})
    sleeve_weights["Raw Sleeve Weight"] = sleeve_weights["Sleeve"].map(raw_sleeve).fillna(0.0)
    sleeve_weights["Daily Exposure"] = sleeve_weights["Sleeve"].map(latest_exposure).fillna(1.0)
    sleeve_weights["Date"] = output_date.date().isoformat()
    sleeve_weights["Strategy"] = STRATEGY
    sleeve_weights["Effective Weight %"] = sleeve_weights["Effective Weight"].mul(100.0)

    meta = pd.DataFrame(
        [
            {
                "Date": output_date.date().isoformat(),
                "Strategy": STRATEGY,
                "Tactical Rule": "proxy_regime relative_return binary lb1 cap30 entry0 exit0 hold0 confirm1",
                "Overlay Mix": "Equity/Gold/BTC 65/25/10",
                "Daily Exposure": (
                    "US SPY MA300 below50%; TH SET MA200 below0%; "
                    "Gold DD252 warn-8%->50%, crash-20%->50%, panic-30% + below MA200 + mom63<0 -> 0%, recover-5%; "
                    "BTC MA50 below0%"
                ),
                "TH Tactical Weight Inside Equity Sleeve": th_tactical_weight,
                "US Sleeve Internal Weight Date": us_internal_date,
                "TH Sleeve Internal Weight Date": th_internal_date,
                "Risk Free Rate": RISK_FREE_RATE,
                "Fresh Start Date": FRESH_START_DATE,
                "Fresh Lookback Calendar Days": FRESH_LOOKBACK_CALENDAR_DAYS,
                "BTC Price Source": "yfinance fresh BTC-USD",
                "BTC Price": float(signal_prices["BTC"].loc[:as_of].iloc[-1]),
                "BTC MA50": float(signal_prices["BTC"].rolling(50, min_periods=20).mean().loc[:as_of].iloc[-1]),
                "BTC Daily Exposure": float(latest_exposure["BTC"]),
                "Gold Price": float(signal_prices["Gold"].loc[:as_of].iloc[-1]),
                "Gold DD252": float(signal_prices["Gold"].loc[:as_of].iloc[-1] / signal_prices["Gold"].rolling(GOLD_DD_WINDOW, min_periods=max(20, GOLD_DD_WINDOW // 4)).max().loc[:as_of].iloc[-1] - 1.0),
                "Gold Daily Exposure": float(latest_exposure["Gold"]),
                "Timing Note": (
                    "Fresh yfinance download at run time; no price cache used for US stocks, TH stocks, Gold, BTC, FX, SPY, VIX, or SET. "
                    "Month-end tactical signal and close trend exposure are lagged before use."
                ),
            }
        ]
    )

    security_path = paths.result_dir / f"{RESULT_PREFIX}_latest_effective_security_weights_thb.csv"
    sleeve_path = paths.result_dir / f"{RESULT_PREFIX}_latest_effective_sleeve_weights_thb.csv"
    meta_path = paths.result_dir / f"{RESULT_PREFIX}_latest_meta.csv"
    security_weights.to_csv(security_path, index=False)
    sleeve_weights.to_csv(sleeve_path, index=False)
    meta.to_csv(meta_path, index=False)

    print(meta.to_string(index=False))
    print(sleeve_weights.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(
        security_weights[
            ["Asset", "Sleeve", "Effective Weight", "Internal Weight", "Raw Sleeve Weight", "Daily Exposure", "Sleeve Multiplier"]
        ].head(40).to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )


if __name__ == "__main__":
    main()
