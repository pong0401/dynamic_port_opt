from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynamic_factor_copula import (  # noqa: E402
    _ensure_yfinance_cache,
    default_paths,
    get_sp500_members_as_of,
    load_cached_market_data,
)

try:
    import yfinance as yf
except Exception as exc:  # pragma: no cover
    raise RuntimeError("yfinance is required to refresh latest cache") from exc


OVERLAY_TICKERS = ["SPY", "GC=F", "BTC-USD", "BIL", "IEF", "^VIX", "USDTHB=X"]


def merge_parquet(path: Path, new_data: pd.DataFrame) -> pd.DataFrame:
    new_data = new_data.copy()
    new_data.index = pd.to_datetime(new_data.index).tz_localize(None)
    new_data = new_data.sort_index()
    combined = new_data
    if path.exists():
        existing = pd.read_parquet(path)
        existing.index = pd.to_datetime(existing.index).tz_localize(None)
        combined = new_data.combine_first(existing)
    combined = combined.sort_index()
    combined.to_parquet(path)
    return combined


def download_close_volume(tickers: list[str], start: str, end: str, chunk_size: int = 80) -> tuple[pd.DataFrame, pd.DataFrame]:
    close_frames: list[pd.DataFrame] = []
    volume_frames: list[pd.DataFrame] = []
    for idx in range(0, len(tickers), chunk_size):
        chunk = tickers[idx : idx + chunk_size]
        print(f"Downloading {idx + 1}-{idx + len(chunk)} of {len(tickers)}")
        raw = yf.download(
            chunk,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        if raw.empty:
            continue

        closes = {}
        volumes = {}
        for ticker in chunk:
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker not in raw.columns.get_level_values(0):
                    continue
                sub = raw[ticker]
            else:
                sub = raw
            if "Close" not in sub.columns:
                continue
            close = sub["Close"].dropna()
            if close.empty:
                continue
            closes[ticker] = close
            volumes[ticker] = sub.get("Volume", pd.Series(index=close.index, dtype=float)).reindex(close.index).fillna(0.0)
        if closes:
            close_frames.append(pd.DataFrame(closes))
            volume_frames.append(pd.DataFrame(volumes))

    close_df = pd.concat(close_frames, axis=1).sort_index() if close_frames else pd.DataFrame()
    volume_df = pd.concat(volume_frames, axis=1).sort_index() if volume_frames else pd.DataFrame()
    return close_df, volume_df


def main() -> None:
    paths = default_paths(ROOT)
    _ensure_yfinance_cache(paths)
    paths.local_cache_root.mkdir(parents=True, exist_ok=True)

    today = pd.Timestamp.today().normalize()
    end = (today + pd.Timedelta(days=1)).date().isoformat()

    cached = load_cached_market_data(paths)
    source_end = pd.Timestamp(cached["prices"].index.max()).normalize()
    start = max(source_end - pd.Timedelta(days=10), pd.Timestamp("2026-04-01")).date().isoformat()

    current_members = get_sp500_members_as_of(today, paths)
    tickers = sorted(set(current_members + OVERLAY_TICKERS))
    print(f"Refresh window: {start} to {end}")
    print(f"Tickers: {len(tickers)} current S&P/overlay tickers")

    prices, volumes = download_close_volume(tickers, start=start, end=end)
    if prices.empty:
        raise RuntimeError("No latest prices downloaded.")

    extra_prices_path = paths.local_cache_root / "extra_prices.parquet"
    extra_volumes_path = paths.local_cache_root / "extra_volumes.parquet"
    combined_prices = merge_parquet(extra_prices_path, prices)
    combined_volumes = merge_parquet(extra_volumes_path, volumes)

    overlay_cols = [ticker for ticker in OVERLAY_TICKERS if ticker in prices.columns]
    if overlay_cols:
        overlay_path = paths.local_cache_root / "overlay_compare_prices.parquet"
        overlay_existing = pd.read_parquet(overlay_path) if overlay_path.exists() else pd.DataFrame()
        if not overlay_existing.empty:
            overlay_existing.index = pd.to_datetime(overlay_existing.index).tz_localize(None)
        overlay_new = prices[overlay_cols]
        overlay_combined = overlay_new.combine_first(overlay_existing).sort_index()
        overlay_combined.to_parquet(overlay_path)
        print(f"overlay_compare_prices.parquet end: {overlay_combined.index.max().date().isoformat()}")

    print(f"extra_prices.parquet end: {combined_prices.index.max().date().isoformat()} columns={combined_prices.shape[1]}")
    print(f"extra_volumes.parquet end: {combined_volumes.index.max().date().isoformat()} columns={combined_volumes.shape[1]}")
    print("Downloaded latest rows:")
    print(prices.tail().to_string())


if __name__ == "__main__":
    main()
