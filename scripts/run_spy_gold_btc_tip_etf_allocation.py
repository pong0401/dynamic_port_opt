from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception as exc:  # pragma: no cover
    raise RuntimeError("yfinance is required for this backtest") from exc

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynamic_factor_copula import (  # noqa: E402
    compute_port_opt_style_metrics,
    curve_from_returns,
    default_paths,
    lag_close_signal_to_next_session,
)

OUTPUT_PREFIX = "spy_gold_btc_tip_etf_allocation"
START_DATE = "2016-01-01"
END_DATE = "2026-04-29"
INITIAL_VALUE = 10_000.0
RISK_FREE_RATE = 0.03
N_CANDIDATES = 20_000
RANDOM_SEED = 20260620

CORE_ASSETS = ["SPY", "Gold", "BTC", "TIP"]
ETF_ASSETS = ["SPMO", "MTUM", "SCHG", "XLK", "EWT", "EWY", "EWJ", "INDA", "MCHI"]
PRICE_TICKERS = ["SPY", "GC=F", "BTC-USD", "TIP"] + ETF_ASSETS
TICKER_TO_ASSET = {"GC=F": "Gold", "BTC-USD": "BTC"}

MIN_WEIGHTS = {
    "SPY": 0.30,
    "Gold": 0.10,
    "BTC": 0.05,
    "TIP": 0.00,
}
MAX_WEIGHTS = {
    "SPY": 0.70,
    "Gold": 0.40,
    "BTC": 0.10,
    "TIP": 0.30,
    **{asset: 0.15 for asset in ETF_ASSETS},
}
ETF_GROUP_CAP = 0.40


def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=tickers)
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"]
        else:
            close = raw.xs("Close", axis=1, level=1)
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
        cached = cached.reindex(columns=PRICE_TICKERS).sort_index()
        cached = cached.loc[START_DATE:END_DATE].ffill()
        if all(ticker in cached.columns and not cached[ticker].dropna().empty for ticker in PRICE_TICKERS):
            return cached

    missing = [ticker for ticker in PRICE_TICKERS if cached.empty or ticker not in cached.columns or cached[ticker].dropna().empty]
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
    prices = pd.concat([cached, downloaded], axis=1) if not cached.empty else downloaded
    prices = prices.loc[:, ~prices.columns.duplicated(keep="last")].reindex(columns=PRICE_TICKERS).sort_index().loc[START_DATE:END_DATE].ffill()
    prices.to_parquet(cache_file)
    return prices


def asset_prices_from_tickers(prices: pd.DataFrame) -> pd.DataFrame:
    out = prices.rename(columns=TICKER_TO_ASSET).copy()
    return out.reindex(columns=CORE_ASSETS + ETF_ASSETS).dropna(how="all")


def trend_exposure(price: pd.Series, ma_period: int, below_exposure: float) -> pd.Series:
    price = price.astype(float).sort_index().ffill()
    ma = price.rolling(ma_period, min_periods=max(20, int(ma_period * 0.20))).mean()
    signal = pd.Series(1.0, index=price.index, dtype=float)
    signal.loc[price < ma] = below_exposure
    signal.loc[ma.isna()] = 1.0
    return lag_close_signal_to_next_session(signal, initial=1.0)


def exposed_returns(asset_prices: pd.DataFrame) -> pd.DataFrame:
    returns = asset_prices.pct_change(fill_method=None).where(asset_prices.notna()).fillna(0.0)
    exposed = returns.copy()
    exposed["SPY"] = returns["SPY"] * trend_exposure(asset_prices["SPY"], 300, 0.50).reindex(returns.index).ffill().fillna(1.0)
    exposed["Gold"] = returns["Gold"] * trend_exposure(asset_prices["Gold"], 50, 1.00).reindex(returns.index).ffill().fillna(1.0)
    exposed["BTC"] = returns["BTC"] * trend_exposure(asset_prices["BTC"], 50, 0.00).reindex(returns.index).ffill().fillna(1.0)
    return exposed


def random_weight_grid(assets: list[str], n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[pd.Series] = []
    min_series = pd.Series({asset: MIN_WEIGHTS.get(asset, 0.0) for asset in assets}, dtype=float)
    max_series = pd.Series({asset: MAX_WEIGHTS.get(asset, 1.0) for asset in assets}, dtype=float)
    if min_series.sum() > 1.0:
        raise RuntimeError("Minimum weights are infeasible.")

    # Include simple anchor candidates so the random search cannot miss obvious core mixes.
    anchors = [
        {"SPY": 0.50, "Gold": 0.30, "BTC": 0.10, "TIP": 0.10},
        {"SPY": 0.45, "Gold": 0.30, "BTC": 0.10, "TIP": 0.15},
        {"SPY": 0.40, "Gold": 0.30, "BTC": 0.10, "TIP": 0.20},
        {"SPY": 0.40, "Gold": 0.25, "BTC": 0.10, "TIP": 0.15, "SPMO": 0.10},
        {"SPY": 0.40, "Gold": 0.25, "BTC": 0.10, "TIP": 0.15, "MTUM": 0.10},
        {"SPY": 0.40, "Gold": 0.25, "BTC": 0.10, "TIP": 0.15, "SCHG": 0.10},
        {"SPY": 0.40, "Gold": 0.25, "BTC": 0.10, "TIP": 0.15, "XLK": 0.10},
    ]
    for anchor in anchors:
        row = pd.Series(0.0, index=assets)
        for asset, weight in anchor.items():
            if asset in row.index:
                row.loc[asset] = weight
        if abs(row.sum() - 1.0) < 1e-9:
            rows.append(row)

    attempts = 0
    while len(rows) < n and attempts < n * 40:
        attempts += 1
        residual = 1.0 - float(min_series.sum())
        draw = rng.dirichlet(np.ones(len(assets))) * residual
        row = min_series + pd.Series(draw, index=assets)
        if (row > max_series + 1e-12).any():
            continue
        if float(row.reindex(ETF_ASSETS).fillna(0.0).sum()) > ETF_GROUP_CAP + 1e-12:
            continue
        rows.append(row)
    if len(rows) < 1_000:
        raise RuntimeError(f"Too few feasible weight rows generated: {len(rows)}")
    grid = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    return grid


def monthly_returns_for_grid(returns: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    assets = grid.columns.tolist()
    weights_np = grid.to_numpy(dtype=float)
    blocks = []
    for _period, block in returns.groupby(returns.index.to_period("M")):
        block = block.reindex(columns=assets).fillna(0.0)
        asset_growth = (1.0 + block).cumprod().to_numpy(dtype=float)
        values = asset_growth @ weights_np.T
        previous = np.vstack([np.ones((1, values.shape[1])), values[:-1]])
        block_returns = values / np.maximum(previous, 1e-12) - 1.0
        blocks.append(pd.DataFrame(block_returns, index=block.index, columns=grid.index))
    return pd.concat(blocks, axis=0).sort_index()


def metrics_from_returns(returns: pd.Series, strategy: str, weights: pd.Series) -> dict[str, object]:
    curve = curve_from_returns(returns.fillna(0.0), initial=INITIAL_VALUE)
    row = compute_port_opt_style_metrics(curve, risk_free_rate=RISK_FREE_RATE).to_dict()
    row.update({"Strategy": strategy, "Start": curve.index.min().date().isoformat(), "End": curve.index.max().date().isoformat()})
    for asset, weight in weights.items():
        row[f"{asset} Weight"] = float(weight)
    row["ETF Group Weight"] = float(weights.reindex(ETF_ASSETS).fillna(0.0).sum())
    return row


def label_weights(prefix: str, weights: pd.Series) -> str:
    active = [(asset, weight) for asset, weight in weights.items() if weight > 0.004]
    asset_label = "/".join(asset for asset, _weight in active)
    weight_label = "/".join(str(int(round(weight * 100))) for _asset, weight in active)
    return f"{prefix} {asset_label} {weight_label}"


def run_search(returns: pd.DataFrame, grid: pd.DataFrame, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    assets = grid.columns.tolist()
    portfolio_returns = monthly_returns_for_grid(returns.reindex(columns=assets).dropna(), grid)
    rows = []
    curves = {}
    for idx in portfolio_returns.columns:
        weights = grid.loc[idx]
        strategy = label_weights(prefix, weights)
        row = metrics_from_returns(portfolio_returns[idx], strategy, weights)
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["Sharpe", "CAGR"], ascending=False).reset_index(drop=True)
    for strategy in summary.head(5)["Strategy"]:
        idx = summary.index[summary["Strategy"].eq(strategy)][0]
        # Match back by exact strategy label; labels are unique enough after drop_duplicates for top rows.
        original_idx = next(i for i, row in grid.iterrows() if label_weights(prefix, row) == strategy)
        curves[strategy] = curve_from_returns(portfolio_returns[original_idx], initial=INITIAL_VALUE)
    return summary, pd.DataFrame(curves)


def coverage_frame(prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in PRICE_TICKERS:
        series = prices[ticker].dropna() if ticker in prices else pd.Series(dtype=float)
        rows.append(
            {
                "Ticker": ticker,
                "Asset": TICKER_TO_ASSET.get(ticker, ticker),
                "Available": not series.empty,
                "First Date": series.index.min().date().isoformat() if not series.empty else "",
                "Last Date": series.index.max().date().isoformat() if not series.empty else "",
                "Observations": int(series.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    paths = default_paths(ROOT)
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    ticker_prices = load_prices()
    asset_prices = asset_prices_from_tickers(ticker_prices)
    usable_assets = [asset for asset in CORE_ASSETS + ETF_ASSETS if asset in asset_prices.columns and asset_prices[asset].dropna().shape[0] >= 1260]
    if "SPY" not in usable_assets or "Gold" not in usable_assets or "BTC" not in usable_assets:
        raise RuntimeError("Core assets missing usable data.")
    grid = random_weight_grid(usable_assets, N_CANDIDATES, RANDOM_SEED)
    raw_returns = asset_prices[usable_assets].pct_change(fill_method=None).where(asset_prices[usable_assets].notna()).fillna(0.0)
    daily_returns = exposed_returns(asset_prices[usable_assets])

    raw_summary, raw_curves = run_search(raw_returns, grid, "Core allocation")
    exposed_summary, exposed_curves = run_search(daily_returns, grid, "Daily-exposure core allocation")
    raw_summary.insert(0, "Mode", "Monthly allocation")
    exposed_summary.insert(0, "Mode", "Daily exposure allocation")
    combined = pd.concat([raw_summary.head(50), exposed_summary.head(50)], ignore_index=True).sort_values(["Sharpe", "CAGR"], ascending=False)
    curves = pd.concat([raw_curves, exposed_curves], axis=1)

    coverage_frame(ticker_prices).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_coverage.csv", index=False)
    raw_summary.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_monthly_allocation_summary.csv", index=False)
    exposed_summary.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_daily_exposure_summary.csv", index=False)
    combined.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_combined_top_summary.csv", index=False)
    curves.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_top_curves.csv")

    display_cols = ["Mode", "Strategy", "CAGR", "Annual Vol", "Sharpe", "Max Drawdown", "ETF Group Weight"]
    print(combined[display_cols].head(15).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()