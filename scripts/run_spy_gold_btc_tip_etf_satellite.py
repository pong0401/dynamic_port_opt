from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynamic_factor_copula import (  # noqa: E402
    compute_port_opt_style_metrics,
    curve_from_returns,
    default_paths,
    lag_close_signal_to_next_session,
    monthly_rebalance_dates,
)

OUTPUT_PREFIX = "spy_gold_btc_tip_etf_satellite"
SOURCE_PRICE_CACHE = "spy_gold_btc_tip_etf_allocation_prices.parquet"
START_DATE = "2016-01-01"
END_DATE = "2026-04-29"
INITIAL_VALUE = 10_000.0
RISK_FREE_RATE = 0.03

CORE_WEIGHTS = {"SPY": 0.45, "Gold": 0.30, "BTC": 0.10, "TIP": 0.15}
ETF_ASSETS = ["SPMO", "MTUM", "SCHG", "XLK", "EWT", "EWY", "EWJ", "INDA", "MCHI"]
PRICE_TICKERS = ["SPY", "GC=F", "BTC-USD", "TIP"] + ETF_ASSETS
TICKER_TO_ASSET = {"GC=F": "Gold", "BTC-USD": "BTC"}
ASSETS = ["SPY", "Gold", "BTC", "TIP"] + ETF_ASSETS


@dataclass(frozen=True)
class SatelliteConfig:
    name: str
    etf_bucket: float
    top_n: int
    require_ma200: bool = True
    require_positive_3m: bool = True
    require_positive_6m: bool = True
    etf_daily_exposure: bool = True


def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=tickers)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.xs("Close", axis=1, level=1)
    else:
        close = raw
    close = close.reindex(columns=tickers).sort_index()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.astype(float)


def load_prices() -> pd.DataFrame:
    paths = default_paths(ROOT)
    paths.local_cache_root.mkdir(parents=True, exist_ok=True)
    cache_file = paths.local_cache_root / SOURCE_PRICE_CACHE
    out_cache = paths.local_cache_root / f"{OUTPUT_PREFIX}_prices.parquet"
    cached = pd.DataFrame()
    for candidate in [cache_file, out_cache]:
        if candidate.exists():
            cached = pd.read_parquet(candidate)
            cached.index = pd.to_datetime(cached.index)
            cached = cached.reindex(columns=PRICE_TICKERS).sort_index().loc[START_DATE:END_DATE].ffill()
            if all(t in cached.columns and not cached[t].dropna().empty for t in PRICE_TICKERS):
                cached.to_parquet(out_cache)
                return cached
    missing = [ticker for ticker in PRICE_TICKERS if cached.empty or ticker not in cached.columns or cached[ticker].dropna().empty]
    if missing:
        if yf is None:
            raise RuntimeError(f"Missing prices for {missing} and yfinance is unavailable")
        yf.set_tz_cache_location(str(paths.local_cache_root / ".yfinance"))
        raw = yf.download(
            missing,
            start=START_DATE,
            end=(pd.Timestamp(END_DATE) + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=False,
        )
        downloaded = _extract_close(raw, missing).sort_index().loc[START_DATE:END_DATE].ffill()
        cached = pd.concat([cached, downloaded], axis=1) if not cached.empty else downloaded
    prices = cached.loc[:, ~cached.columns.duplicated(keep="last")].reindex(columns=PRICE_TICKERS).sort_index().loc[START_DATE:END_DATE].ffill()
    prices.to_parquet(out_cache)
    return prices


def asset_prices_from_tickers(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.rename(columns=TICKER_TO_ASSET).reindex(columns=ASSETS).dropna(how="all")


def trend_exposure(price: pd.Series, ma_period: int, below_exposure: float) -> pd.Series:
    price = price.astype(float).sort_index().ffill()
    ma = price.rolling(ma_period, min_periods=max(20, int(ma_period * 0.20))).mean()
    signal = pd.Series(1.0, index=price.index, dtype=float)
    signal.loc[price < ma] = below_exposure
    signal.loc[ma.isna()] = 1.0
    return lag_close_signal_to_next_session(signal, initial=1.0)


def build_exposures(prices: pd.DataFrame) -> pd.DataFrame:
    exposure = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
    exposure["SPY"] = trend_exposure(prices["SPY"], 300, 0.50)
    exposure["Gold"] = trend_exposure(prices["Gold"], 50, 1.00)
    exposure["BTC"] = trend_exposure(prices["BTC"], 50, 0.00)
    for asset in ETF_ASSETS:
        if asset in prices:
            exposure[asset] = trend_exposure(prices[asset], 200, 0.00)
    return exposure.reindex(prices.index).ffill().fillna(1.0).clip(0.0, 1.0)


def momentum_rank(prices: pd.DataFrame, rebalance_date: pd.Timestamp, config: SatelliteConfig) -> pd.DataFrame:
    history = prices.loc[:rebalance_date, ETF_ASSETS].ffill()
    rows = []
    for asset in ETF_ASSETS:
        series = history[asset].dropna() if asset in history else pd.Series(dtype=float)
        if len(series) < 253:
            rows.append({"ETF": asset, "pass": False, "score": np.nan})
            continue
        latest = float(series.iloc[-1])
        ret_1m = latest / float(series.iloc[-22]) - 1.0
        ret_3m = latest / float(series.iloc[-64]) - 1.0
        ret_6m = latest / float(series.iloc[-127]) - 1.0
        ret_12m = latest / float(series.iloc[-253]) - 1.0
        sma200 = float(series.iloc[-200:].mean())
        passes = True
        if config.require_ma200:
            passes &= latest > sma200
        if config.require_positive_3m:
            passes &= ret_3m > 0.0
        if config.require_positive_6m:
            passes &= ret_6m > 0.0
        rows.append(
            {
                "ETF": asset,
                "price": latest,
                "ret_1m": ret_1m,
                "ret_3m": ret_3m,
                "ret_6m": ret_6m,
                "ret_12m": ret_12m,
                "sma200": sma200,
                "pass": passes,
            }
        )
    frame = pd.DataFrame(rows)
    score = pd.Series(0.0, index=frame.index, dtype=float)
    for column, weight in {"ret_1m": 0.20, "ret_3m": 0.30, "ret_6m": 0.40, "ret_12m": 0.10}.items():
        score += weight * frame[column].rank(pct=True).fillna(0.0)
    frame["score"] = score
    return frame.sort_values(["pass", "score"], ascending=[False, False]).reset_index(drop=True)


def satellite_weights(selected: list[str], bucket: float) -> pd.Series:
    weights = pd.Series(CORE_WEIGHTS, dtype=float)
    if not selected or bucket <= 0.0:
        return weights
    tip_funding = min(CORE_WEIGHTS["TIP"], bucket * 0.50)
    spy_funding = bucket - tip_funding
    weights["TIP"] -= tip_funding
    weights["SPY"] -= spy_funding
    if weights["SPY"] < 0.30 - 1e-12:
        shortfall = 0.30 - weights["SPY"]
        weights["SPY"] += shortfall
        weights["TIP"] -= shortfall
    if weights["TIP"] < -1e-12:
        raise RuntimeError("ETF bucket funding is infeasible with the configured core weights.")
    for asset in selected:
        weights.loc[asset] = bucket / len(selected)
    return weights / weights.sum()


def run_variant(prices: pd.DataFrame, returns: pd.DataFrame, exposure: pd.DataFrame, config: SatelliteConfig) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    schedule = monthly_rebalance_dates(prices.index, lookback_days=252, freq="ME")
    weights = pd.DataFrame(0.0, index=prices.index, columns=ASSETS)
    selection_rows = []
    rank_rows = []
    for idx, rebalance_date in enumerate(schedule[:-1]):
        next_date = schedule[idx + 1]
        test_index = prices.index[(prices.index > rebalance_date) & (prices.index <= next_date)]
        if len(test_index) == 0:
            continue
        ranks = momentum_rank(prices, rebalance_date, config)
        selected = ranks.loc[ranks["pass"].fillna(False), "ETF"].head(config.top_n).astype(str).tolist()
        monthly_weights = satellite_weights(selected, config.etf_bucket)
        weights.loc[test_index, :] = 0.0
        weights.loc[test_index, monthly_weights.index] = monthly_weights.to_numpy(dtype=float)
        ranks = ranks.copy()
        ranks["rebalance_date"] = rebalance_date
        ranks["selected"] = ranks["ETF"].isin(selected)
        rank_rows.append(ranks)
        selection_rows.append(
            {
                "rebalance_date": rebalance_date,
                "next_rebalance_date": next_date,
                "selected_etfs": ",".join(selected),
                "selected_count": len(selected),
                "etf_bucket": config.etf_bucket if selected else 0.0,
                "raw_spy_weight": float(monthly_weights.get("SPY", 0.0)),
                "raw_tip_weight": float(monthly_weights.get("TIP", 0.0)),
            }
        )
    weights = weights.loc[weights.sum(axis=1).gt(0.0)].copy()
    aligned_returns = returns.reindex(weights.index).fillna(0.0)
    effective = weights.copy()
    for asset in ["SPY", "Gold", "BTC"]:
        effective[asset] = effective[asset] * exposure.reindex(weights.index)[asset].ffill().fillna(1.0)
    if config.etf_daily_exposure:
        for asset in ETF_ASSETS:
            effective[asset] = effective[asset] * exposure.reindex(weights.index)[asset].ffill().fillna(1.0)
    strategy_returns = aligned_returns.mul(effective, axis=1).sum(axis=1)
    curve = curve_from_returns(strategy_returns, initial=INITIAL_VALUE).rename(config.name)
    selection = pd.DataFrame(selection_rows)
    ranking = pd.concat(rank_rows, ignore_index=True) if rank_rows else pd.DataFrame()
    effective.insert(0, "Strategy", config.name)
    effective.insert(0, "Date", effective.index)
    return curve, selection, effective.reset_index(drop=True), ranking


def metrics_row(curve: pd.Series, config: SatelliteConfig, effective: pd.DataFrame) -> dict[str, object]:
    row = compute_port_opt_style_metrics(curve.dropna(), risk_free_rate=RISK_FREE_RATE).to_dict()
    weight_cols = [c for c in effective.columns if c in ASSETS]
    avg = effective[weight_cols].mean()
    latest = effective.loc[effective["Date"].eq(effective["Date"].max()), weight_cols].iloc[-1]
    row.update(
        {
            "Strategy": config.name,
            "ETF Bucket": config.etf_bucket,
            "Top N": config.top_n,
            "ETF Daily Exposure": config.etf_daily_exposure,
            "Start": curve.dropna().index.min().date().isoformat(),
            "End": curve.dropna().index.max().date().isoformat(),
            "Average SPY Weight": float(avg.get("SPY", 0.0)),
            "Average Gold Weight": float(avg.get("Gold", 0.0)),
            "Average BTC Weight": float(avg.get("BTC", 0.0)),
            "Average TIP Weight": float(avg.get("TIP", 0.0)),
            "Average ETF Weight": float(avg.reindex(ETF_ASSETS).fillna(0.0).sum()),
            "Latest SPY Weight": float(latest.get("SPY", 0.0)),
            "Latest Gold Weight": float(latest.get("Gold", 0.0)),
            "Latest BTC Weight": float(latest.get("BTC", 0.0)),
            "Latest TIP Weight": float(latest.get("TIP", 0.0)),
            "Latest ETF Weight": float(latest.reindex(ETF_ASSETS).fillna(0.0).sum()),
        }
    )
    return row


def main() -> None:
    paths = default_paths(ROOT)
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    ticker_prices = load_prices()
    prices = asset_prices_from_tickers(ticker_prices).ffill()
    returns = prices.pct_change(fill_method=None).where(prices.notna()).fillna(0.0)
    exposure = build_exposures(prices)

    configs = [SatelliteConfig("Core SPY/Gold/BTC/TIP 45/30/10/15 daily exposure", 0.0, 0)]
    for bucket in [0.10, 0.15, 0.20]:
        for top_n in [1, 3, 5]:
            configs.append(SatelliteConfig(f"Core + ETF satellite {int(bucket*100)} top{top_n} momentum daily exposure", bucket, top_n))
            configs.append(SatelliteConfig(f"Core + ETF satellite {int(bucket*100)} top{top_n} momentum raw ETF", bucket, top_n, etf_daily_exposure=False))

    curves = {}
    summary_rows = []
    selection_frames = []
    effective_frames = []
    ranking_frames = []
    for config in configs:
        curve, selection, effective, ranking = run_variant(prices, returns, exposure, config)
        curves[config.name] = curve
        summary_rows.append(metrics_row(curve, config, effective))
        if not selection.empty:
            selection.insert(0, "Strategy", config.name)
            selection_frames.append(selection)
        effective_frames.append(effective)
        if not ranking.empty:
            ranking.insert(0, "Strategy", config.name)
            ranking_frames.append(ranking)

    summary = pd.DataFrame(summary_rows).sort_values(["Sharpe", "CAGR"], ascending=False)
    pd.DataFrame(curves).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_curves.csv")
    summary.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_summary.csv", index=False)
    pd.concat(selection_frames, ignore_index=True).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_selection_history.csv", index=False)
    pd.concat(effective_frames, ignore_index=True).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_effective_weights.csv", index=False)
    pd.concat(ranking_frames, ignore_index=True).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_momentum_ranking.csv", index=False)

    cols = ["Strategy", "CAGR", "Annual Vol", "Sharpe", "Max Drawdown", "Average ETF Weight", "Latest ETF Weight"]
    print(summary[cols].head(20).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()