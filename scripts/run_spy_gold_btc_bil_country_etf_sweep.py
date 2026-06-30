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

from dynamic_factor_copula import compute_port_opt_style_metrics, curve_from_returns, default_paths, monthly_rebalance_dates  # noqa: E402
import run_spy_gold_btc_bil_leverage_etf_sweep as lev  # noqa: E402

OUTPUT_PREFIX = "spy_gold_btc_bil_country_etf_sweep"

COUNTRY_ETFS = (
    # Americas
    "EWC", "EWA", "EWW", "EWZ", "ARGT", "ECH", "EPU", "GXG",
    # Europe country funds
    "EWG", "EWU", "EWQ", "EWL", "EWP", "EWI", "EWN", "EWD", "EDEN", "EIRL", "NORW", "EPOL", "GREK", "TUR",
    # Asia Pacific / EM country funds
    "EWJ", "DXJ", "EWT", "EWY", "INDA", "EPI", "MCHI", "KWEB", "EWS", "EWM", "EIDO", "VNM", "THD", "ENZL", "PAK", "KSA", "QAT", "UAE", "EIS",
    # Africa / frontier-ish with long enough history when available
    "EZA", "EGPT",
)
PRICE_TICKERS = list(dict.fromkeys([*lev.BASE_TICKERS, *lev.ALL_CANDIDATES, *COUNTRY_ETFS]))


@dataclass(frozen=True)
class Config:
    baseline: str
    universe_name: str
    candidates: tuple[str, ...]
    bucket: float
    top_n: int
    funding: str
    min_history_days: int

    @property
    def name(self) -> str:
        hist = "short" if self.min_history_days < 2520 else "full"
        return f"{self.baseline} + country {self.universe_name} {hist} bucket{self.bucket:.0%} top{self.top_n} fund_{self.funding}"


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
    for candidate in [
        cache_file,
        paths.local_cache_root / "spy_gold_btc_bil_leverage_etf_sweep_prices.parquet",
        paths.local_cache_root / "spy_gold_btc_bil_gold_dd_exact_variants_prices.parquet",
        paths.local_cache_root / "spy_gold_btc_defensive_etf_expanded_sweep_prices.parquet",
    ]:
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
    cols = list(dict.fromkeys(["SPY", "Gold", "BTC", "BIL", "IEF", *lev.ALL_CANDIDATES, *COUNTRY_ETFS]))
    return renamed.loc[:, ~renamed.columns.duplicated(keep="last")].reindex(columns=cols).dropna(how="all")


def universe_variants(usable: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    asia = tuple(x for x in ("EWJ", "DXJ", "EWT", "EWY", "INDA", "EPI", "MCHI", "KWEB", "EWS", "EWM", "EIDO", "VNM", "THD", "ENZL") if x in usable)
    europe = tuple(x for x in ("EWG", "EWU", "EWQ", "EWL", "EWP", "EWI", "EWN", "EWD", "EDEN", "EIRL", "NORW", "EPOL", "GREK", "TUR") if x in usable)
    americas = tuple(x for x in ("EWC", "EWA", "EWW", "EWZ", "ARGT", "ECH", "EPU", "GXG") if x in usable)
    em = tuple(x for x in ("EWW", "EWZ", "ARGT", "ECH", "EPU", "GXG", "INDA", "EPI", "MCHI", "KWEB", "EWT", "EWY", "EIDO", "VNM", "THD", "TUR", "EPOL", "EZA", "EGPT", "KSA", "QAT", "UAE") if x in usable)
    return {
        "country_all": usable,
        "country_asia": asia,
        "country_europe": europe,
        "country_americas": americas,
        "country_em": em,
    }


def run_variant(prices: pd.DataFrame, returns: pd.DataFrame, config: Config):
    schedule = monthly_rebalance_dates(prices.index, lookback_days=252, freq="ME")
    cols = list(dict.fromkeys(["SPY", "Gold", "BTC", "BIL", "IEF", *lev.ALL_CANDIDATES, *COUNTRY_ETFS]))
    weights = pd.DataFrame(0.0, index=prices.index, columns=cols)
    selections = []
    for idx, date in enumerate(schedule[:-1]):
        next_date = schedule[idx + 1]
        test_idx = prices.index[(prices.index > date) & (prices.index <= next_date)]
        if len(test_idx) == 0:
            continue
        base_w, base_selected = lev.baseline_weights(prices, date, config.baseline)
        ranks = lev.momentum_rank(prices, date, config.candidates)
        selected = ranks.loc[ranks["pass"].fillna(False), "ETF"].head(config.top_n).astype(str).tolist()
        w = lev.funded_weights(base_w, selected, config.bucket, config.funding)
        if w.empty:
            selected = []
            w = base_w
        weights.loc[test_idx, w.index] = w.to_numpy(dtype=float)
        selections.append({
            "Strategy": config.name,
            "rebalance_date": date,
            "next_rebalance_date": next_date,
            "base_selected": ",".join(base_selected),
            "country_selected": ",".join(selected),
            "country_count": len(selected),
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
    active = int(sel["country_count"].gt(0).sum()) if not sel.empty else 0
    total = int(sel.shape[0]) if not sel.empty else 0
    row.update({
        "Strategy": config.name,
        "Baseline": config.baseline,
        "Universe": config.universe_name,
        "Bucket": config.bucket,
        "Top N": config.top_n,
        "Funding": config.funding,
        "Min History Days": config.min_history_days,
        "Start": clean.index.min().date().isoformat(),
        "End": clean.index.max().date().isoformat(),
        "Country Active Rate": active / total if total else np.nan,
        "Latest SPY Weight": float(latest_weights.get("SPY", 0.0)),
        "Latest Gold Weight": float(latest_weights.get("Gold", 0.0)),
        "Latest BTC Weight": float(latest_weights.get("BTC", 0.0)),
        "Latest BIL Weight": float(latest_weights.get("BIL", 0.0)),
        "Latest Country Weight": float(latest_weights.reindex(config.candidates).fillna(0.0).sum()),
        "Latest Country Assets": ",".join(latest.loc[latest["Asset"].isin(config.candidates), "Asset"].astype(str).tolist()),
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
    configs = []
    for min_history_days in [2520]:
        usable = tuple(asset for asset in COUNTRY_ETFS if asset in prices and prices[asset].dropna().shape[0] >= min_history_days)
        for universe_name, candidates in universe_variants(usable).items():
            if not candidates:
                continue
            for baseline in ["conservative_cash", "balanced_etf"]:
                for bucket in [0.05, 0.10, 0.15]:
                    for top_n in [1, 2]:
                        for funding in ["spy", "spy2_bil1"]:
                            configs.append(Config(baseline, universe_name, candidates, bucket, top_n, funding, min_history_days))
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
    cols = ["Strategy", "CAGR", "Annual Vol", "Sharpe", "Max Drawdown", "Country Active Rate", "Latest Country Assets", "Latest Country Weight"]
    print(summary[cols].head(40).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()

