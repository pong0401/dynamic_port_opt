from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable, List
import warnings

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynamic_factor_copula import (  # noqa: E402
    THAI_SYMBOL_EXCLUDE,
    _ensure_yfinance_cache,
    _normalize_set_symbol,
    default_paths,
    ensure_assets_available,
    load_set100_membership_intervals,
    yf,
)


START_DATE = "2003-02-04"
END_DATE = "2026-05-12"
THAI_BENCHMARK = "^SET.BK"
THAI_VOL_PROXY = ""


def chunked(values: List[str], size: int) -> Iterable[List[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def build_target_tickers() -> List[str]:
    intervals = load_set100_membership_intervals(default_paths(ROOT))
    tickers = sorted(
        {
            _normalize_set_symbol(ticker)
            for ticker in intervals["ticker"].dropna().tolist()
            if _normalize_set_symbol(ticker) and _normalize_set_symbol(ticker) not in THAI_SYMBOL_EXCLUDE
        }
    )
    return tickers


def main() -> None:
    paths = default_paths(ROOT)
    _ensure_yfinance_cache(paths)
    warnings.simplefilter("ignore", category=FutureWarning)
    tickers = build_target_tickers()
    print(f"SET100 unique tickers: {len(tickers)}")
    if not tickers:
        raise SystemExit("No SET100 tickers found.")
    if yf is None:
        raise SystemExit("yfinance is unavailable in this environment.")

    downloaded_all: List[str] = []
    missing_all: List[str] = []
    for batch in chunked(tickers, 60):
        frames = ensure_assets_available(batch, START_DATE, END_DATE, paths)
        available = [ticker for ticker in batch if ticker in frames["prices"].columns]
        downloaded_all.extend(available)
        missing_all.extend([ticker for ticker in batch if ticker not in frames["prices"].columns])
        print(f"Loaded batch {batch[0]}..{batch[-1]} | available={len(available)} missing={len(batch) - len(available)}")

    # Benchmark is stored in the same extra cache path so set100_pit can use benchmark_ticker='^SET.BK'
    bench_frames = ensure_assets_available([THAI_BENCHMARK], START_DATE, END_DATE, paths)
    benchmark_available = THAI_BENCHMARK in bench_frames["prices"].columns
    print(f"Benchmark {THAI_BENCHMARK} available: {benchmark_available}")

    local_prices = paths.local_cache_root / "extra_prices.parquet"
    local_volumes = paths.local_cache_root / "extra_volumes.parquet"
    prices = pd.read_parquet(local_prices) if local_prices.exists() else pd.DataFrame()
    volumes = pd.read_parquet(local_volumes) if local_volumes.exists() else pd.DataFrame()

    thai_price_cols = [col for col in prices.columns if col.endswith(".BK") or col == THAI_BENCHMARK]
    thai_volume_cols = [col for col in volumes.columns if col.endswith(".BK") or col == THAI_BENCHMARK]
    print(f"Cached Thai price columns: {len(thai_price_cols)}")
    print(f"Cached Thai volume columns: {len(thai_volume_cols)}")

    summary = pd.DataFrame(
        {
            "ticker": sorted(set(downloaded_all + missing_all + ([THAI_BENCHMARK] if benchmark_available else []))),
        }
    )
    summary["in_cache"] = summary["ticker"].isin(thai_price_cols)
    summary["is_benchmark"] = summary["ticker"].eq(THAI_BENCHMARK)
    summary["kind"] = summary["ticker"].map(lambda ticker: "benchmark" if ticker == THAI_BENCHMARK else "set100_member")
    summary.to_csv(paths.result_dir / "thai_set100_cache_status.csv", index=False)

    print("Thai cache status written to thai_set100_cache_status.csv")
    print(summary["in_cache"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
