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
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for path in [SCRIPTS, SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dynamic_factor_copula import compute_port_opt_style_metrics, curve_from_returns, default_paths, monthly_rebalance_dates  # noqa: E402
import run_spy_gold_btc_bil_leverage_etf_sweep as lev  # noqa: E402

OUTPUT_PREFIX = "spy_gold_btc_bil_long_short_leverage_sweep"
SHORT_ETFS = ("SH", "SDS", "SPXU", "PSQ", "QID", "SQQQ", "TECS", "SOXS")
PRICE_TICKERS = list(dict.fromkeys([*lev.PRICE_TICKERS, *SHORT_ETFS]))


@dataclass(frozen=True)
class Config:
    baseline: str
    long_bucket: float
    long_top_n: int
    long_funding: str
    long_filter: str
    short_bucket: float
    short_top_n: int
    short_funding: str
    short_filter: str
    short_candidates: tuple[str, ...]

    @property
    def name(self) -> str:
        return (
            f"{self.baseline} long{self.long_bucket:.0%}top{self.long_top_n}_{self.long_filter}_{self.long_funding} "
            f"short{self.short_bucket:.0%}top{self.short_top_n}_{self.short_filter}_{self.short_funding}"
        )


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
    for candidate in [cache_file, paths.local_cache_root / "spy_gold_btc_bil_leverage_etf_sweep_prices.parquet"]:
        if candidate.exists():
            frame = pd.read_parquet(candidate)
            frame.index = pd.to_datetime(frame.index)
            cached = frame.reindex(columns=PRICE_TICKERS).sort_index().loc[lev.START_DATE:lev.END_DATE].ffill()
            if all(t in cached.columns and not cached[t].dropna().empty for t in PRICE_TICKERS):
                cached.to_parquet(cache_file)
                return cached
    missing = [t for t in PRICE_TICKERS if cached.empty or t not in cached.columns or cached[t].dropna().empty]
    if missing:
        if yf is None:
            raise RuntimeError(f"Missing prices for {missing}; yfinance unavailable")
        yf.set_tz_cache_location(str(paths.local_cache_root / ".yfinance"))
        raw = yf.download(
            missing,
            start=lev.START_DATE,
            end=(pd.Timestamp(lev.END_DATE) + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=False,
        )
        downloaded = _extract_close(raw, missing).sort_index().loc[lev.START_DATE:lev.END_DATE].ffill()
        cached = pd.concat([cached, downloaded], axis=1) if not cached.empty else downloaded
    prices = cached.loc[:, ~cached.columns.duplicated(keep="last")].reindex(columns=PRICE_TICKERS)
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    prices = prices.loc[prices.index.notna()].sort_index().loc[lev.START_DATE:lev.END_DATE].ffill()
    prices.to_parquet(cache_file)
    return prices


def asset_prices(raw: pd.DataFrame) -> pd.DataFrame:
    renamed = raw.rename(columns=lev.TICKER_TO_ASSET)
    cols = list(dict.fromkeys(["SPY", "Gold", "BTC", "BIL", "IEF", *lev.ALL_CANDIDATES, *SHORT_ETFS]))
    return renamed.loc[:, ~renamed.columns.duplicated(keep="last")].reindex(columns=cols).dropna(how="all")


def risk_off_signal(prices: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "none":
        return pd.Series(False, index=prices.index)
    risk_on = lev.risk_on_signal(prices, {
        "below_ma200": "spy_ma200",
        "below_ma300": "spy_ma300",
        "ma200_or_dd8": "spy_ma200_dd8",
    }[mode])
    return ~risk_on


def select_short(prices: pd.DataFrame, date: pd.Timestamp, candidates: tuple[str, ...], top_n: int) -> list[str]:
    ranks = lev.momentum_rank(prices, date, candidates, require_trend=False)
    passed = ranks.loc[ranks["pass"].fillna(False), "ETF"].head(top_n).astype(str).tolist()
    return passed


def apply_funding(weights: pd.Series, selected: list[str], bucket: float, funding: str) -> pd.Series:
    if not selected or bucket <= 0.0:
        return weights
    out = weights.copy()
    if funding == "bil":
        out["BIL"] = out.get("BIL", 0.0) - bucket
    elif funding == "spy":
        out["SPY"] = out.get("SPY", 0.0) - bucket
    else:
        out["SPY"] = out.get("SPY", 0.0) - bucket * 0.5
        out["BIL"] = out.get("BIL", 0.0) - bucket * 0.5
    if out.get("SPY", 0.0) < 0.20 - 1e-12 or out.get("BIL", 0.0) < -1e-12:
        return pd.Series(dtype=float)
    for asset in selected:
        out[asset] = out.get(asset, 0.0) + bucket / len(selected)
    return out / out.sum()


def run_variant(prices: pd.DataFrame, returns: pd.DataFrame, config: Config):
    schedule = monthly_rebalance_dates(prices.index, lookback_days=252, freq="ME")
    cols = list(dict.fromkeys(["SPY", "Gold", "BTC", "BIL", "IEF", *lev.ALL_CANDIDATES, *SHORT_ETFS]))
    weights = pd.DataFrame(0.0, index=prices.index, columns=cols)
    selections = []
    long_risk_on = lev.risk_on_signal(prices, config.long_filter)
    short_risk_off = risk_off_signal(prices, config.short_filter)
    for idx, date in enumerate(schedule[:-1]):
        next_date = schedule[idx + 1]
        test_idx = prices.index[(prices.index > date) & (prices.index <= next_date)]
        if len(test_idx) == 0:
            continue
        w, base_selected = lev.baseline_weights(prices, date, config.baseline)
        long_selected = []
        if bool(long_risk_on.reindex([date]).ffill().fillna(False).iloc[0]) and config.long_bucket > 0.0:
            ranks = lev.momentum_rank(prices, date, lev.LEVERAGE_ETFS)
            long_selected = ranks.loc[ranks["pass"].fillna(False), "ETF"].head(config.long_top_n).astype(str).tolist()
        w2 = lev.funded_weights(w, long_selected, config.long_bucket, config.long_funding)
        if w2.empty:
            long_selected = []
            w2 = w
        short_selected = []
        if bool(short_risk_off.reindex([date]).ffill().fillna(False).iloc[0]) and config.short_bucket > 0.0:
            short_selected = select_short(prices, date, config.short_candidates, config.short_top_n)
        w3 = apply_funding(w2, short_selected, config.short_bucket, config.short_funding)
        if w3.empty:
            short_selected = []
            w3 = w2
        weights.loc[test_idx, w3.index] = w3.to_numpy(dtype=float)
        selections.append({
            "Strategy": config.name,
            "rebalance_date": date,
            "next_rebalance_date": next_date,
            "base_selected": ",".join(base_selected),
            "long_selected": ",".join(long_selected),
            "short_selected": ",".join(short_selected),
            "long_count": len(long_selected),
            "short_count": len(short_selected),
        })
    weights = weights.loc[weights.sum(axis=1).gt(0.0)].copy()
    exposure = pd.DataFrame(1.0, index=weights.index, columns=weights.columns)
    exposure["SPY"] = lev.trend_exposure(prices["SPY"], 300, 0.50).reindex(weights.index).ffill().fillna(1.0)
    exposure["Gold"] = lev.gold_sweep.gold_exposure(prices["Gold"], lev.GOLD_RULE).reindex(weights.index).ffill().fillna(1.0)
    exposure["BTC"] = lev.trend_exposure(prices["BTC"], 50, 0.00).reindex(weights.index).ffill().fillna(1.0)
    effective = weights * exposure
    strat_ret = returns.reindex(effective.index).fillna(0.0).mul(effective, axis=1).sum(axis=1)
    curve = curve_from_returns(strat_ret, initial=lev.INITIAL_VALUE).rename(config.name)
    latest = effective.iloc[-1].rename("Effective Weight").reset_index().rename(columns={"index": "Asset"})
    latest = latest.loc[latest["Effective Weight"].abs().gt(1e-12)].copy()
    latest.insert(0, "Strategy", config.name)
    latest.insert(1, "Date", effective.index.max())
    return curve, pd.DataFrame(selections), latest


def metrics_row(curve: pd.Series, config: Config, sel: pd.DataFrame, latest: pd.DataFrame) -> dict[str, object]:
    clean = curve.dropna()
    row = compute_port_opt_style_metrics(clean, risk_free_rate=lev.RISK_FREE_RATE).to_dict()
    latest_weights = latest.set_index("Asset")["Effective Weight"] if not latest.empty else pd.Series(dtype=float)
    long_active = int(sel["long_count"].gt(0).sum()) if not sel.empty else 0
    short_active = int(sel["short_count"].gt(0).sum()) if not sel.empty else 0
    total = int(sel.shape[0]) if not sel.empty else 0
    row.update({
        "Strategy": config.name,
        "Baseline": config.baseline,
        "Long Bucket": config.long_bucket,
        "Long Top N": config.long_top_n,
        "Long Funding": config.long_funding,
        "Long Filter": config.long_filter,
        "Short Bucket": config.short_bucket,
        "Short Top N": config.short_top_n,
        "Short Funding": config.short_funding,
        "Short Filter": config.short_filter,
        "Start": clean.index.min().date().isoformat(),
        "End": clean.index.max().date().isoformat(),
        "Long Active Rate": long_active / total if total else np.nan,
        "Short Active Rate": short_active / total if total else np.nan,
        "Latest SPY Weight": float(latest_weights.get("SPY", 0.0)),
        "Latest Gold Weight": float(latest_weights.get("Gold", 0.0)),
        "Latest BTC Weight": float(latest_weights.get("BTC", 0.0)),
        "Latest BIL Weight": float(latest_weights.get("BIL", 0.0)),
        "Latest Long Leverage Weight": float(latest_weights.reindex(lev.LEVERAGE_ETFS).fillna(0.0).sum()),
        "Latest Short Weight": float(latest_weights.reindex(SHORT_ETFS).fillna(0.0).sum()),
        "Latest Long Assets": ",".join(latest.loc[latest["Asset"].isin(lev.LEVERAGE_ETFS), "Asset"].astype(str).tolist()),
        "Latest Short Assets": ",".join(latest.loc[latest["Asset"].isin(SHORT_ETFS), "Asset"].astype(str).tolist()),
    })
    return row


def coverage(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in PRICE_TICKERS:
        s = raw[ticker].dropna() if ticker in raw else pd.Series(dtype=float)
        rows.append({
            "Ticker": ticker,
            "Asset": lev.TICKER_TO_ASSET.get(ticker, ticker),
            "Available": not s.empty,
            "First Date": s.index.min().date().isoformat() if not s.empty else "",
            "Last Date": s.index.max().date().isoformat() if not s.empty else "",
            "Observations": int(s.shape[0]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    paths = default_paths(ROOT)
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    raw = load_prices()
    prices = asset_prices(raw).ffill()
    returns = prices.pct_change(fill_method=None).where(prices.notna()).fillna(0.0)
    usable_short = tuple(asset for asset in SHORT_ETFS if asset in prices and prices[asset].dropna().shape[0] >= 2520)
    configs = []
    for baseline in ["conservative_cash", "balanced_etf"]:
        for long_bucket in [0.05]:
            for long_top_n in [2]:
                for long_filter in ["spy_ma300", "spy_ma200_dd8"]:
                    for short_bucket in [0.05, 0.10]:
                        for short_top_n in [1, 2]:
                            for short_funding in ["bil", "half"]:
                                for short_filter in ["below_ma300", "ma200_or_dd8"]:
                                    configs.append(Config(baseline, long_bucket, long_top_n, "spy", long_filter, short_bucket, short_top_n, short_funding, short_filter, usable_short))
    curves = {}
    summaries = []
    selections = []
    latest_frames = []
    for config in configs:
        curve, sel, latest = run_variant(prices, returns, config)
        curves[config.name] = curve
        summaries.append(metrics_row(curve, config, sel, latest))
        selections.append(sel)
        latest_frames.append(latest)
    summary = pd.DataFrame(summaries).sort_values(["Sharpe", "CAGR"], ascending=False)
    coverage(raw).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_coverage.csv", index=False)
    summary.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_summary.csv", index=False)
    pd.DataFrame(curves).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_curves.csv")
    pd.concat(selections, ignore_index=True).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_selection_history.csv", index=False)
    pd.concat(latest_frames, ignore_index=True).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_latest_weights.csv", index=False)
    cols = ["Strategy", "CAGR", "Annual Vol", "Sharpe", "Max Drawdown", "Long Active Rate", "Short Active Rate", "Latest Long Assets", "Latest Short Assets"]
    print(summary[cols].head(40).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()

