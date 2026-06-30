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
SCRIPTS = ROOT / "scripts"
for path in [SRC, SCRIPTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dynamic_factor_copula import compute_port_opt_style_metrics, curve_from_returns, default_paths, lag_close_signal_to_next_session, monthly_rebalance_dates  # noqa: E402
import run_spy_gold_btc_bil_etf_gold_exposure_sweep as gold_sweep  # noqa: E402
import run_spy_gold_btc_defensive_etf_expanded_sweep as expanded  # noqa: E402

OUTPUT_PREFIX = "spy_gold_btc_bil_gold_dd_exact_variants"
START_DATE = expanded.START_DATE
END_DATE = expanded.END_DATE
INITIAL_VALUE = expanded.INITIAL_VALUE
RISK_FREE_RATE = expanded.RISK_FREE_RATE
CORE = {"SPY": 0.45, "Gold": 0.30, "BTC": 0.10, "BIL": 0.15}
VARIANTS = {
    "variant_1_cash": ["SPMO", "MTUM", "SCHG", "XLK", "EWY", "EWJ", "INDA", "BIL", "SGOV", "IEF"],
    "variant_2_region": ["SPMO", "SCHG", "XLK", "EWY", "EWJ", "INDA", "EWW", "EIDO", "VNM", "KSA"],
    "variant_3_sector": ["SPMO", "SCHG", "XLK", "XLI", "XLF", "XLE", "XLV", "XLP", "XLU", "XAR"],
    "variant_4_best_compact": ["SPMO", "SCHG", "XLK", "EWY", "EWJ", "INDA", "EWW", "EIDO", "XAR", "BIL"],
}
BASE_TICKERS = ["SPY", "GC=F", "BTC-USD", "BIL"]
ALL_CANDIDATES = list(dict.fromkeys(asset for assets in VARIANTS.values() for asset in assets))
PRICE_TICKERS = list(dict.fromkeys(BASE_TICKERS + ALL_CANDIDATES))
TICKER_TO_ASSET = {"GC=F": "Gold", "BTC-USD": "BTC"}
GOLD_RULE = gold_sweep.GoldRule("dd252_warn8_crash20_half", "dd_simple", dd_window=252, warn_dd=-0.08, crash_dd=-0.20, warn_exposure=0.5, crash_exposure=0.5)


@dataclass(frozen=True)
class Config:
    variant: str
    candidates: tuple[str, ...]
    bucket: float
    top_n: int
    funding_mode: str
    min_history_days: int

    @property
    def name(self) -> str:
        hist = "short" if self.min_history_days < 2520 else "full"
        return f"{self.variant} {hist} bucket{self.bucket:.0%} top{self.top_n} fund_{self.funding_mode} gold_dd"


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
    for candidate in [cache_file, paths.local_cache_root / "spy_gold_btc_defensive_etf_expanded_sweep_prices.parquet"]:
        if candidate.exists():
            frame = pd.read_parquet(candidate)
            frame.index = pd.to_datetime(frame.index)
            cached = frame.reindex(columns=PRICE_TICKERS).sort_index().loc[START_DATE:END_DATE].ffill()
            if all(t in cached.columns and not cached[t].dropna().empty for t in PRICE_TICKERS):
                cached.to_parquet(cache_file)
                return cached
    missing = [t for t in PRICE_TICKERS if cached.empty or t not in cached.columns or cached[t].dropna().empty]
    if missing:
        if yf is None:
            raise RuntimeError(f"Missing prices for {missing}; yfinance unavailable")
        yf.set_tz_cache_location(str(paths.local_cache_root / ".yfinance"))
        raw = yf.download(missing, start=START_DATE, end=(pd.Timestamp(END_DATE) + pd.Timedelta(days=1)).date().isoformat(), auto_adjust=True, progress=False, group_by="column", threads=False)
        downloaded = _extract_close(raw, missing).sort_index().loc[START_DATE:END_DATE].ffill()
        cached = pd.concat([cached, downloaded], axis=1) if not cached.empty else downloaded
    prices = cached.loc[:, ~cached.columns.duplicated(keep="last")].reindex(columns=PRICE_TICKERS)
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    prices = prices.loc[prices.index.notna()].sort_index().loc[START_DATE:END_DATE].ffill()
    prices.to_parquet(cache_file)
    return prices


def asset_prices(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.rename(columns=TICKER_TO_ASSET).loc[:, ~raw.rename(columns=TICKER_TO_ASSET).columns.duplicated(keep="last")].reindex(columns=list(dict.fromkeys(["SPY", "Gold", "BTC", "BIL"] + ALL_CANDIDATES))).dropna(how="all")


def trend_exposure(price: pd.Series, ma_period: int, below_exposure: float) -> pd.Series:
    price = price.astype(float).sort_index().ffill()
    ma = price.rolling(ma_period, min_periods=max(20, int(ma_period * 0.20))).mean()
    signal = pd.Series(1.0, index=price.index, dtype=float)
    signal.loc[price < ma] = below_exposure
    signal.loc[ma.isna()] = 1.0
    return lag_close_signal_to_next_session(signal, initial=1.0)


def momentum_rank(prices: pd.DataFrame, date: pd.Timestamp, candidates: tuple[str, ...]) -> pd.DataFrame:
    hist = prices.loc[:date, list(candidates)].ffill()
    rows = []
    for asset in candidates:
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


def funded_weights(selected: list[str], config: Config) -> pd.Series:
    weights = pd.Series(CORE, dtype=float)
    if not selected:
        return weights
    if config.funding_mode == "spy2_bil1":
        spy_funding = config.bucket * 2.0 / 3.0
        bil_funding = config.bucket - spy_funding
    elif config.funding_mode == "half":
        spy_funding = config.bucket * 0.5
        bil_funding = config.bucket * 0.5
    elif config.funding_mode == "spy":
        spy_funding = config.bucket
        bil_funding = 0.0
    else:
        spy_funding = 0.0
        bil_funding = config.bucket
    weights["SPY"] -= spy_funding
    weights["BIL"] -= bil_funding
    if weights["SPY"] < 0.30 - 1e-12 or weights["BIL"] < -1e-12:
        return pd.Series(dtype=float)
    for asset in selected:
        weights[asset] = weights.get(asset, 0.0) + config.bucket / len(selected)
    return weights / weights.sum()


def run_variant(prices: pd.DataFrame, returns: pd.DataFrame, config: Config) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    schedule = monthly_rebalance_dates(prices.index, lookback_days=252, freq="ME")
    cols = list(dict.fromkeys(["SPY", "Gold", "BTC", "BIL"] + ALL_CANDIDATES))
    weights = pd.DataFrame(0.0, index=prices.index, columns=cols)
    selection_rows = []
    for idx, date in enumerate(schedule[:-1]):
        next_date = schedule[idx + 1]
        test_idx = prices.index[(prices.index > date) & (prices.index <= next_date)]
        if len(test_idx) == 0:
            continue
        ranks = momentum_rank(prices, date, config.candidates)
        selected = ranks.loc[ranks["pass"].fillna(False), "ETF"].head(config.top_n).astype(str).tolist()
        w = funded_weights(selected, config)
        if w.empty:
            continue
        weights.loc[test_idx, w.index] = w.to_numpy(dtype=float)
        selection_rows.append({"Strategy": config.name, "rebalance_date": date, "next_rebalance_date": next_date, "selected_etfs": ",".join(selected), "selected_count": len(selected)})
    weights = weights.loc[weights.sum(axis=1).gt(0.0)].copy()
    exposure = pd.DataFrame(1.0, index=weights.index, columns=weights.columns)
    exposure["SPY"] = trend_exposure(prices["SPY"], 300, 0.50).reindex(weights.index).ffill().fillna(1.0)
    exposure["Gold"] = gold_sweep.gold_exposure(prices["Gold"], GOLD_RULE).reindex(weights.index).ffill().fillna(1.0)
    exposure["BTC"] = trend_exposure(prices["BTC"], 50, 0.00).reindex(weights.index).ffill().fillna(1.0)
    effective = weights * exposure
    strategy_returns = returns.reindex(effective.index).fillna(0.0).mul(effective, axis=1).sum(axis=1)
    curve = curve_from_returns(strategy_returns, initial=INITIAL_VALUE).rename(config.name)
    latest = effective.iloc[-1].rename("Effective Weight").reset_index().rename(columns={"index": "Asset"})
    latest = latest.loc[latest["Effective Weight"].abs().gt(1e-12)].copy()
    latest.insert(0, "Strategy", config.name)
    latest.insert(1, "Date", effective.index.max())
    return curve, pd.DataFrame(selection_rows), latest, exposure.reset_index(drop=True)


def metrics_row(curve: pd.Series, config: Config, sel: pd.DataFrame, latest: pd.DataFrame, exposure: pd.DataFrame) -> dict[str, object]:
    clean = curve.dropna()
    row = compute_port_opt_style_metrics(clean, risk_free_rate=RISK_FREE_RATE).to_dict()
    latest_weights = latest.set_index("Asset")["Effective Weight"] if not latest.empty else pd.Series(dtype=float)
    active_months = int(sel["selected_count"].gt(0).sum()) if not sel.empty else 0
    total_months = int(sel.shape[0]) if not sel.empty else 0
    row.update({"Strategy": config.name, "Variant": config.variant, "Bucket": config.bucket, "Top N": config.top_n, "Funding Mode": config.funding_mode, "Min History Days": config.min_history_days, "Gold Rule": GOLD_RULE.name, "Start": clean.index.min().date().isoformat(), "End": clean.index.max().date().isoformat(), "Active Rate": active_months / total_months if total_months else np.nan, "Average Gold Exposure": float(exposure["Gold"].mean()), "Latest SPY Weight": float(latest_weights.get("SPY", 0.0)), "Latest Gold Weight": float(latest_weights.get("Gold", 0.0)), "Latest BTC Weight": float(latest_weights.get("BTC", 0.0)), "Latest BIL Weight": float(latest_weights.get("BIL", 0.0)), "Latest Satellite Weight": float(latest_weights.reindex(config.candidates).fillna(0.0).sum()), "Latest Satellite Assets": ",".join(latest.loc[latest["Asset"].isin(config.candidates), "Asset"].astype(str).tolist())})
    return row


def coverage(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in PRICE_TICKERS:
        s = raw[ticker].dropna() if ticker in raw else pd.Series(dtype=float)
        rows.append({"Ticker": ticker, "Asset": TICKER_TO_ASSET.get(ticker, ticker), "Available": not s.empty, "First Date": s.index.min().date().isoformat() if not s.empty else "", "Last Date": s.index.max().date().isoformat() if not s.empty else "", "Observations": int(s.shape[0])})
    return pd.DataFrame(rows)


def main() -> None:
    paths = default_paths(ROOT)
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    raw = load_prices()
    prices = asset_prices(raw).ffill()
    returns = prices.pct_change(fill_method=None).where(prices.notna()).fillna(0.0)
    configs = []
    for min_history_days in [2520, 253]:
        for variant, assets in VARIANTS.items():
            usable = [asset for asset in assets if asset in prices and prices[asset].dropna().shape[0] >= min_history_days]
            if not usable:
                continue
            for bucket in [0.15, 0.20]:
                for top_n in [1, 2]:
                    for funding in ["spy2_bil1", "half", "spy"]:
                        if funding == "spy" and CORE["SPY"] - bucket < 0.30 - 1e-12:
                            continue
                        configs.append(Config(variant, tuple(usable), bucket, top_n, funding, min_history_days))
    curves = {}
    summaries = []
    selections = []
    latest_frames = []
    exposure_frames = []
    for config in configs:
        curve, sel, latest, exposure = run_variant(prices, returns, config)
        curves[config.name] = curve
        summaries.append(metrics_row(curve, config, sel, latest, exposure))
        if not sel.empty:
            selections.append(sel)
        latest_frames.append(latest)
        exposure_frames.append(exposure.assign(Strategy=config.name))
    summary = pd.DataFrame(summaries).sort_values(["Sharpe", "CAGR"], ascending=False)
    coverage(raw).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_coverage.csv", index=False)
    summary.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_summary.csv", index=False)
    pd.DataFrame(curves).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_curves.csv")
    pd.concat(selections, ignore_index=True).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_selection_history.csv", index=False)
    pd.concat(latest_frames, ignore_index=True).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_latest_weights.csv", index=False)
    cols = ["Strategy", "CAGR", "Annual Vol", "Sharpe", "Max Drawdown", "Active Rate", "Latest Satellite Assets", "Latest Satellite Weight"]
    print(summary[cols].head(30).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()