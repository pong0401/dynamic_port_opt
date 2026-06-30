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
import run_spy_gold_btc_bil_adaptive_gold_country_winner as adaptive  # noqa: E402
import run_spy_gold_btc_bil_country_etf_sweep as country  # noqa: E402
import run_spy_gold_btc_bil_leverage_etf_sweep as lev  # noqa: E402

OUTPUT_PREFIX = "spy_gold_btc_bil_combined_etf_universe_current_best"

CORE_MOMENTUM_ETFS = ("SPMO", "MTUM", "SCHG", "XLK", "EWY", "EWJ", "INDA")
SECTOR_ETFS = ("XLI", "XLF", "XLE", "XLV", "XLP", "XLU", "XAR", "ITA", "SMH")
REGION_ETFS = ("EWW", "EIDO", "VNM", "KSA", "EPOL")
ALL_CANDIDATE_ASSETS = tuple(dict.fromkeys([*CORE_MOMENTUM_ETFS, *SECTOR_ETFS, *REGION_ETFS, *country.COUNTRY_ETFS]))
PRICE_TICKERS = list(dict.fromkeys([*lev.BASE_TICKERS, *lev.ALL_CANDIDATES, *ALL_CANDIDATE_ASSETS]))


@dataclass(frozen=True)
class Config:
    name: str
    candidates: tuple[str, ...]
    bucket: float
    top_n: int
    funding: str
    gold_boost: float


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
        paths.local_cache_root / "spy_gold_btc_bil_country_etf_sweep_prices.parquet",
        paths.local_cache_root / "spy_gold_btc_defensive_etf_expanded_sweep_prices.parquet",
        paths.local_cache_root / "spy_gold_btc_bil_leverage_etf_sweep_prices.parquet",
    ]:
        if not candidate.exists():
            continue
        frame = pd.read_parquet(candidate)
        frame.index = pd.to_datetime(frame.index)
        cached = pd.concat([cached, frame], axis=1) if not cached.empty else frame
    if not cached.empty:
        cached = cached.loc[:, ~cached.columns.duplicated(keep="last")].reindex(columns=PRICE_TICKERS)
        cached = cached.sort_index().loc[lev.START_DATE:lev.END_DATE].ffill()
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
    cols = list(dict.fromkeys(["SPY", "Gold", "BTC", "BIL", "IEF", *lev.ALL_CANDIDATES, *ALL_CANDIDATE_ASSETS]))
    return renamed.loc[:, ~renamed.columns.duplicated(keep="last")].reindex(columns=cols).dropna(how="all")


def usable_assets(prices: pd.DataFrame, assets: tuple[str, ...], min_history_days: int = 2520) -> tuple[str, ...]:
    return tuple(asset for asset in assets if asset in prices and prices[asset].dropna().shape[0] >= min_history_days)


def universe_variants(prices: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    core = usable_assets(prices, CORE_MOMENTUM_ETFS)
    sector = usable_assets(prices, SECTOR_ETFS)
    country_all = usable_assets(prices, country.COUNTRY_ETFS)
    region = usable_assets(prices, REGION_ETFS)
    return {
        "country_only": country_all,
        "momentum_core": core,
        "sector_only": sector,
        "momentum_sector": tuple(dict.fromkeys([*core, *sector])),
        "momentum_country": tuple(dict.fromkeys([*core, *country_all])),
        "sector_country": tuple(dict.fromkeys([*sector, *country_all])),
        "momentum_sector_country": tuple(dict.fromkeys([*core, *sector, *country_all])),
        "all_with_region": tuple(dict.fromkeys([*core, *sector, *region, *country_all])),
    }


def run_variant(prices: pd.DataFrame, returns: pd.DataFrame, config: Config):
    schedule = monthly_rebalance_dates(prices.index, lookback_days=252, freq="ME")
    cols = list(dict.fromkeys(["SPY", "Gold", "BTC", "BIL", "IEF", *lev.ALL_CANDIDATES, *ALL_CANDIDATE_ASSETS]))
    weights = pd.DataFrame(0.0, index=prices.index, columns=cols)
    selections = []
    for idx, date in enumerate(schedule[:-1]):
        next_date = schedule[idx + 1]
        test_idx = prices.index[(prices.index > date) & (prices.index <= next_date)]
        if len(test_idx) == 0:
            continue
        try:
            base_w, base_selected = lev.baseline_weights(prices, date, "conservative_cash")
        except KeyError:
            base_w = pd.Series({"SPY": 0.45, "Gold": 0.30, "BTC": 0.10, "BIL": 0.15}, dtype=float)
            base_selected = []
        ranks = lev.momentum_rank(prices, date, config.candidates)
        selected = ranks.loc[ranks["pass"].fillna(False), "ETF"].head(config.top_n).astype(str).tolist()
        w = lev.funded_weights(base_w, selected, config.bucket, config.funding)
        if w.empty:
            selected = []
            w = base_w
        if adaptive.spy_risk_off(prices, date, "spy_below_ma200_or_dd8"):
            w = adaptive.apply_gold_boost(w, config.gold_boost, "spy")
        weights.loc[test_idx, w.index] = w.to_numpy(dtype=float)
        selections.append(
            {
                "Strategy": config.name,
                "rebalance_date": date,
                "next_rebalance_date": next_date,
                "base_selected": ",".join(base_selected),
                "selected_etfs": ",".join(selected),
                "selected_count": len(selected),
            }
        )
    weights = weights.loc[weights.sum(axis=1).gt(0.0)].copy()
    dd_rule = adaptive.GoldRule(
        "dd252_warn8_crash20_half",
        "dd_simple",
        warn_dd=-0.08,
        crash_dd=-0.20,
        warn_exposure=0.50,
        crash_exposure=0.50,
    )
    exposure = pd.DataFrame(1.0, index=weights.index, columns=weights.columns)
    exposure["SPY"] = adaptive.trend_exposure(prices["SPY"], 300, 0.50).reindex(weights.index).ffill().fillna(1.0)
    exposure["Gold"] = adaptive.gold_exposure(prices["Gold"], dd_rule).reindex(weights.index).ffill().fillna(1.0)
    exposure["BTC"] = adaptive.trend_exposure(prices["BTC"], 50, 0.00).reindex(weights.index).ffill().fillna(1.0)
    effective = weights * exposure
    strat_ret = returns.reindex(effective.index).fillna(0.0).mul(effective, axis=1).sum(axis=1)
    curve = curve_from_returns(strat_ret, initial=lev.INITIAL_VALUE).rename(config.name)
    latest = effective.iloc[-1].rename("Effective Weight").reset_index().rename(columns={"index": "Asset"})
    latest = latest.loc[latest["Effective Weight"].abs().gt(1e-12)].copy()
    latest.insert(0, "Strategy", config.name)
    latest.insert(1, "Date", effective.index.max())
    return curve, pd.DataFrame(selections), latest, exposure


def metrics_row(curve: pd.Series, config: Config, sel: pd.DataFrame, latest: pd.DataFrame, exposure: pd.DataFrame) -> dict[str, object]:
    clean = curve.dropna()
    row = compute_port_opt_style_metrics(clean, risk_free_rate=lev.RISK_FREE_RATE).to_dict()
    latest_weights = latest.set_index("Asset")["Effective Weight"] if not latest.empty else pd.Series(dtype=float)
    row.update(
        {
            "Strategy": config.name,
            "Universe": config.name.split(" ")[1],
            "Bucket": config.bucket,
            "Top N": config.top_n,
            "Funding": config.funding,
            "Gold Boost": config.gold_boost,
            "Start": clean.index.min().date().isoformat(),
            "End": clean.index.max().date().isoformat(),
            "Active Rate": float(sel["selected_count"].gt(0).mean()) if not sel.empty else np.nan,
            "Average Gold Exposure": float(exposure["Gold"].mean()),
            "Latest SPY Weight": float(latest_weights.get("SPY", 0.0)),
            "Latest Gold Weight": float(latest_weights.get("Gold", 0.0)),
            "Latest BTC Weight": float(latest_weights.get("BTC", 0.0)),
            "Latest BIL Weight": float(latest_weights.get("BIL", 0.0)),
            "Latest ETF Weight": float(latest_weights.reindex(config.candidates).fillna(0.0).sum()),
            "Latest ETF Assets": ",".join(latest.loc[latest["Asset"].isin(config.candidates), "Asset"].astype(str).tolist()),
        }
    )
    return row


def main() -> None:
    paths = default_paths(ROOT)
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    raw = load_prices()
    prices = asset_prices(raw).ffill()
    returns = prices.pct_change(fill_method=None).where(prices.notna()).fillna(0.0)

    configs = []
    for universe_name, candidates in universe_variants(prices).items():
        if not candidates:
            continue
        for bucket in [0.05, 0.08, 0.10]:
            for top_n in [1, 2, 3]:
                name = f"current_best {universe_name} bucket{bucket:.0%} top{top_n} boost16"
                configs.append(Config(name, candidates, bucket, top_n, "spy", 0.16))

    curves = {}
    summaries = []
    selections = []
    latest_frames = []
    for config in configs:
        curve, sel, latest, exposure = run_variant(prices, returns, config)
        curves[config.name] = curve
        summaries.append(metrics_row(curve, config, sel, latest, exposure))
        selections.append(sel)
        latest_frames.append(latest)

    summary = pd.DataFrame(summaries).sort_values(["Sharpe", "CAGR"], ascending=False)
    summary.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_summary.csv", index=False)
    pd.DataFrame(curves).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_curves.csv")
    pd.concat(selections, ignore_index=True).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_selection_history.csv", index=False)
    pd.concat(latest_frames, ignore_index=True).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_latest_weights.csv", index=False)
    cols = [
        "Strategy",
        "CAGR",
        "Annual Vol",
        "Sharpe",
        "Max Drawdown",
        "Active Rate",
        "Latest ETF Assets",
        "Latest ETF Weight",
    ]
    print(summary[cols].head(40).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
