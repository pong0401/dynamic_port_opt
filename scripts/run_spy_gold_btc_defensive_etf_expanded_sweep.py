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

OUTPUT_PREFIX = "spy_gold_btc_defensive_etf_expanded_sweep"
START_DATE = "2016-01-01"
END_DATE = "2026-04-29"
INITIAL_VALUE = 10_000.0
RISK_FREE_RATE = 0.03
CORE_BASE = {"SPY": 0.45, "Gold": 0.30, "BTC": 0.10}
DEFENSIVE_ASSETS = ["TIP", "BIL", "IEF"]
CORE_ETFS = ["SPMO", "MTUM", "SCHG", "XLK", "EWY", "EWJ", "INDA"]
REGION_ETFS = ["EWW", "EIDO", "VNM", "KSA", "EPOL"]
SECTOR_ETFS = ["XLI", "XLF", "XLE", "XLV", "XLP", "XLU", "XAR", "ITA"]
ALL_ETFS = list(dict.fromkeys(CORE_ETFS + REGION_ETFS + SECTOR_ETFS))
PRICE_TICKERS = ["SPY", "GC=F", "BTC-USD"] + DEFENSIVE_ASSETS + ALL_ETFS
TICKER_TO_ASSET = {"GC=F": "Gold", "BTC-USD": "BTC"}
UNIVERSE_VARIANTS = {
    "core_liquid": CORE_ETFS,
    "region_plus": CORE_ETFS + ["EWW", "EIDO", "VNM", "KSA"],
    "sector_plus": CORE_ETFS + ["XLI", "XLF", "XLE", "XLV", "XLP", "XLU", "XAR"],
    "compact_best": ["SPMO", "SCHG", "XLK", "EWY", "EWJ", "INDA", "EWW", "EIDO", "XAR"],
    "all_expanded": ALL_ETFS,
}


@dataclass(frozen=True)
class Config:
    defensive_asset: str
    universe_name: str
    etfs: tuple[str, ...]
    bucket: float
    top_n: int
    funding_mode: str

    @property
    def name(self) -> str:
        return f"{self.defensive_asset} core + {self.universe_name} bucket{self.bucket:.0%} top{self.top_n} fund_{self.funding_mode}"


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
    cache_file = paths.local_cache_root / f"{OUTPUT_PREFIX}_prices.parquet"
    cached = pd.DataFrame()
    if cache_file.exists():
        cached = pd.read_parquet(cache_file)
        cached.index = pd.to_datetime(cached.index)
        cached = cached.reindex(columns=PRICE_TICKERS).sort_index().loc[START_DATE:END_DATE].ffill()
        if all(t in cached.columns and not cached[t].dropna().empty for t in PRICE_TICKERS):
            return cached
    # Reuse prior cache for overlapping tickers.
    prior = paths.local_cache_root / "spy_gold_btc_tip_etf_allocation_prices.parquet"
    if prior.exists():
        old = pd.read_parquet(prior)
        old.index = pd.to_datetime(old.index)
        cached = old.reindex(columns=PRICE_TICKERS).sort_index().loc[START_DATE:END_DATE].ffill()
    missing = [t for t in PRICE_TICKERS if cached.empty or t not in cached.columns or cached[t].dropna().empty]
    if missing:
        if yf is None:
            raise RuntimeError(f"Missing prices for {missing}; yfinance unavailable")
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
    prices = cached.loc[:, ~cached.columns.duplicated(keep="last")].reindex(columns=PRICE_TICKERS)
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    prices = prices.loc[prices.index.notna()].sort_index().loc[START_DATE:END_DATE].ffill()
    prices.to_parquet(cache_file)
    return prices


def asset_prices(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.rename(columns=TICKER_TO_ASSET).reindex(columns=["SPY", "Gold", "BTC"] + DEFENSIVE_ASSETS + ALL_ETFS).dropna(how="all")


def trend_exposure(price: pd.Series, ma_period: int, below_exposure: float) -> pd.Series:
    price = price.astype(float).sort_index().ffill()
    ma = price.rolling(ma_period, min_periods=max(20, int(ma_period * 0.20))).mean()
    signal = pd.Series(1.0, index=price.index, dtype=float)
    signal.loc[price < ma] = below_exposure
    signal.loc[ma.isna()] = 1.0
    return lag_close_signal_to_next_session(signal, initial=1.0)


def build_exposure(prices: pd.DataFrame) -> pd.DataFrame:
    exposure = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
    exposure["SPY"] = trend_exposure(prices["SPY"], 300, 0.50)
    exposure["Gold"] = trend_exposure(prices["Gold"], 50, 1.00)
    exposure["BTC"] = trend_exposure(prices["BTC"], 50, 0.00)
    return exposure.ffill().fillna(1.0).clip(0.0, 1.0)


def momentum_rank(prices: pd.DataFrame, date: pd.Timestamp, etfs: tuple[str, ...]) -> pd.DataFrame:
    hist = prices.loc[:date, list(etfs)].ffill()
    rows = []
    for asset in etfs:
        s = hist[asset].dropna() if asset in hist else pd.Series(dtype=float)
        if len(s) < 253:
            rows.append({"ETF": asset, "pass": False, "score": np.nan})
            continue
        latest = float(s.iloc[-1])
        ret_1m = latest / float(s.iloc[-22]) - 1.0
        ret_3m = latest / float(s.iloc[-64]) - 1.0
        ret_6m = latest / float(s.iloc[-127]) - 1.0
        ret_12m = latest / float(s.iloc[-253]) - 1.0
        sma200 = float(s.iloc[-200:].mean())
        rows.append({"ETF": asset, "ret_1m": ret_1m, "ret_3m": ret_3m, "ret_6m": ret_6m, "ret_12m": ret_12m, "pass": latest > sma200 and ret_3m > 0.0 and ret_6m > 0.0})
    df = pd.DataFrame(rows)
    score = pd.Series(0.0, index=df.index, dtype=float)
    for col, weight in {"ret_1m": 0.20, "ret_3m": 0.30, "ret_6m": 0.40, "ret_12m": 0.10}.items():
        score += weight * df[col].rank(pct=True).fillna(0.0)
    df["score"] = score
    return df.sort_values(["pass", "score"], ascending=[False, False]).reset_index(drop=True)


def core_weights(defensive_asset: str) -> pd.Series:
    w = pd.Series(CORE_BASE, dtype=float)
    w[defensive_asset] = 0.15
    return w


def fund_weights(config: Config, selected: list[str]) -> pd.Series:
    w = core_weights(config.defensive_asset)
    if not selected:
        return w
    if config.funding_mode == "half":
        defensive_funding = min(float(w[config.defensive_asset]), config.bucket * 0.50)
        spy_funding = config.bucket - defensive_funding
    elif config.funding_mode == "spy2_def1":
        spy_funding = config.bucket * 2.0 / 3.0
        defensive_funding = config.bucket - spy_funding
    elif config.funding_mode == "spy":
        spy_funding = config.bucket
        defensive_funding = 0.0
    else:
        spy_funding = 0.0
        defensive_funding = config.bucket
    w["SPY"] -= spy_funding
    w[config.defensive_asset] -= defensive_funding
    if w["SPY"] < 0.30 - 1e-12 or w[config.defensive_asset] < -1e-12:
        return pd.Series(dtype=float)
    for asset in selected:
        w[asset] = config.bucket / len(selected)
    return w / w.sum()


def run_variant(prices: pd.DataFrame, returns: pd.DataFrame, exposure: pd.DataFrame, config: Config) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    schedule = monthly_rebalance_dates(prices.index, lookback_days=252, freq="ME")
    cols = ["SPY", "Gold", "BTC"] + DEFENSIVE_ASSETS + ALL_ETFS
    weights = pd.DataFrame(0.0, index=prices.index, columns=cols)
    selections = []
    for i, date in enumerate(schedule[:-1]):
        next_date = schedule[i + 1]
        test_idx = prices.index[(prices.index > date) & (prices.index <= next_date)]
        if len(test_idx) == 0:
            continue
        ranks = momentum_rank(prices, date, config.etfs)
        selected = ranks.loc[ranks["pass"].fillna(False), "ETF"].head(config.top_n).astype(str).tolist()
        w = fund_weights(config, selected)
        if w.empty:
            continue
        weights.loc[test_idx, w.index] = w.to_numpy(dtype=float)
        selections.append({"Strategy": config.name, "rebalance_date": date, "next_rebalance_date": next_date, "selected_etfs": ",".join(selected), "selected_count": len(selected), "bucket_used": config.bucket if selected else 0.0})
    weights = weights.loc[weights.sum(axis=1).gt(0.0)].copy()
    effective = weights.copy()
    for asset in ["SPY", "Gold", "BTC"]:
        effective[asset] *= exposure.reindex(effective.index)[asset].ffill().fillna(1.0)
    strat_ret = returns.reindex(effective.index).fillna(0.0).mul(effective, axis=1).sum(axis=1)
    curve = curve_from_returns(strat_ret, initial=INITIAL_VALUE).rename(config.name)
    latest = effective.iloc[-1].rename("Effective Weight").reset_index().rename(columns={"index": "Asset"})
    latest = latest.loc[latest["Effective Weight"].abs().gt(1e-12)].copy()
    latest.insert(0, "Strategy", config.name)
    latest.insert(1, "Date", effective.index.max())
    return curve, pd.DataFrame(selections), latest


def metrics_row(curve: pd.Series, config: Config, sel: pd.DataFrame, latest: pd.DataFrame) -> dict[str, object]:
    clean = curve.dropna()
    row = compute_port_opt_style_metrics(clean, risk_free_rate=RISK_FREE_RATE).to_dict()
    latest_weights = latest.set_index("Asset")["Effective Weight"] if not latest.empty else pd.Series(dtype=float)
    active_months = int(sel["selected_count"].gt(0).sum()) if not sel.empty else 0
    total_months = int(sel.shape[0]) if not sel.empty else 0
    row.update({
        "Strategy": config.name,
        "Defensive Asset": config.defensive_asset,
        "Universe": config.universe_name,
        "ETF Bucket": config.bucket,
        "Top N": config.top_n,
        "Funding Mode": config.funding_mode,
        "Start": clean.index.min().date().isoformat(),
        "End": clean.index.max().date().isoformat(),
        "Active Rate": active_months / total_months if total_months else np.nan,
        "Latest SPY Weight": float(latest_weights.get("SPY", 0.0)),
        "Latest Gold Weight": float(latest_weights.get("Gold", 0.0)),
        "Latest BTC Weight": float(latest_weights.get("BTC", 0.0)),
        "Latest Defensive Weight": float(latest_weights.get(config.defensive_asset, 0.0)),
        "Latest ETF Weight": float(latest_weights.reindex(ALL_ETFS).fillna(0.0).sum()),
        "Latest ETF Assets": ",".join(latest.loc[latest["Asset"].isin(ALL_ETFS), "Asset"].astype(str).tolist()),
    })
    return row


def coverage(prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in PRICE_TICKERS:
        s = prices[ticker].dropna() if ticker in prices else pd.Series(dtype=float)
        rows.append({"Ticker": ticker, "Asset": TICKER_TO_ASSET.get(ticker, ticker), "Available": not s.empty, "First Date": s.index.min().date().isoformat() if not s.empty else "", "Last Date": s.index.max().date().isoformat() if not s.empty else "", "Observations": int(s.shape[0])})
    return pd.DataFrame(rows)


def main() -> None:
    paths = default_paths(ROOT)
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    raw_prices = load_prices()
    prices = asset_prices(raw_prices).ffill()
    # Exclude ETFs with less than roughly 10 years of usable data from full-period tests.
    usable_etfs = [e for e in ALL_ETFS if e in prices and prices[e].dropna().shape[0] >= 2520]
    universes = {name: [e for e in etfs if e in usable_etfs] for name, etfs in UNIVERSE_VARIANTS.items()}
    returns = prices.pct_change(fill_method=None).where(prices.notna()).fillna(0.0)
    exposure = build_exposure(prices)
    configs = []
    for defensive in DEFENSIVE_ASSETS:
        if prices[defensive].dropna().shape[0] < 2520:
            continue
        for uname, etfs in universes.items():
            if not etfs:
                continue
            for bucket in [0.15, 0.20]:
                for top_n in [1, 2]:
                    for funding in ["half", "spy2_def1", "spy"]:
                        if funding == "spy" and CORE_BASE["SPY"] - bucket < 0.30 - 1e-12:
                            continue
                        configs.append(Config(defensive, uname, tuple(etfs), bucket, top_n, funding))
    curves = {}
    summaries = []
    selection_frames = []
    latest_frames = []
    for config in configs:
        curve, sel, latest = run_variant(prices, returns, exposure, config)
        curves[config.name] = curve
        summaries.append(metrics_row(curve, config, sel, latest))
        if not sel.empty:
            selection_frames.append(sel)
        latest_frames.append(latest)
    summary = pd.DataFrame(summaries).sort_values(["Sharpe", "CAGR"], ascending=False)
    coverage(raw_prices).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_coverage.csv", index=False)
    summary.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_summary.csv", index=False)
    pd.DataFrame(curves).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_curves.csv")
    pd.concat(selection_frames, ignore_index=True).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_selection_history.csv", index=False)
    pd.concat(latest_frames, ignore_index=True).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_latest_weights.csv", index=False)
    cols = ["Strategy", "CAGR", "Annual Vol", "Sharpe", "Max Drawdown", "Active Rate", "Latest ETF Assets", "Latest ETF Weight"]
    print(summary[cols].head(30).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()