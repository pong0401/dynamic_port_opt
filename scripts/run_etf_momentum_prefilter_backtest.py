from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Callable

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
    EPSILON,
    compute_port_opt_style_metrics,
    curve_from_returns,
    default_paths,
    monthly_rebalance_dates,
    optimize_portfolio,
)


OUTPUT_PREFIX = "etf_momentum_prefilter"
DEFAULT_ETF_UNIVERSE = ["SPMO", "MTUM", "SCHG", "XLK", "EWT", "EWY", "EWJ", "INDA", "MCHI"]
DEFAULT_START_DATE = "2012-01-01"
DEFAULT_END_DATE = "2026-06-18"
INITIAL_VALUE = 10_000.0
RISK_FREE_RATE = 0.03


@dataclass(frozen=True)
class ETFMomentumConfig:
    universe: tuple[str, ...] = tuple(DEFAULT_ETF_UNIVERSE)
    rebalance_frequency: str = "ME"
    top_n: int = 5
    min_n: int = 3
    use_sma_filter: bool = True
    sma_window: int = 200
    require_positive_3m: bool = True
    require_positive_6m: bool = True
    momentum_weights: dict[str, float] = field(
        default_factory=lambda: {"ret_1m": 0.20, "ret_3m": 0.30, "ret_6m": 0.40, "ret_12m": 0.10}
    )
    max_weight_per_asset: float = 0.35
    long_only: bool = True
    fallback_asset: str = "CASH"
    lookback_days: int = 252
    risk_aversion: float = 8.0
    concentration_penalty: float = 0.02
    momentum_strength: float = 1.0
    risk_free_rate: float = RISK_FREE_RATE


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


def load_etf_adjusted_close(
    tickers: list[str] | tuple[str, ...] = DEFAULT_ETF_UNIVERSE,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    cache_name: str = OUTPUT_PREFIX,
) -> pd.DataFrame:
    paths = default_paths(ROOT)
    paths.local_cache_root.mkdir(parents=True, exist_ok=True)
    tickers = list(dict.fromkeys(tickers))
    cache_file = paths.local_cache_root / f"{cache_name}_prices.parquet"

    cached = pd.DataFrame()
    if cache_file.exists():
        cached = pd.read_parquet(cache_file)
        cached.index = pd.to_datetime(cached.index)
    legacy_cache = paths.local_cache_root / "etf_universe_ex_thailand_raw_prices.parquet"
    if legacy_cache.exists():
        legacy = pd.read_parquet(legacy_cache)
        legacy.index = pd.to_datetime(legacy.index)
        cached = cached.combine_first(legacy.reindex(columns=tickers)) if not cached.empty else legacy.reindex(columns=tickers)

    if not cached.empty:
        cached = cached.sort_index().loc[:, [ticker for ticker in tickers if ticker in cached.columns]]
    missing = [ticker for ticker in tickers if ticker not in cached.columns or cached[ticker].dropna().empty]

    if missing:
        if yf is None:
            raise RuntimeError(f"Missing ETF prices for {missing} and yfinance is not available.")
        yf.set_tz_cache_location(str(paths.local_cache_root / ".yfinance"))
        raw = yf.download(
            missing,
            start=start_date,
            end=(pd.Timestamp(end_date) + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=False,
        )
        downloaded = _extract_close(raw, missing)
        cached = pd.concat([cached, downloaded], axis=1)
        cached = cached.loc[:, ~cached.columns.duplicated(keep="last")]

    cached.index = pd.to_datetime(cached.index, errors="coerce")
    cached = cached.loc[cached.index.notna()].sort_index()
    prices = cached.reindex(columns=tickers).sort_index().loc[start_date:end_date].ffill()
    prices = prices.dropna(axis=1, thresh=260)
    if prices.empty:
        raise RuntimeError("No usable ETF prices loaded.")
    prices.to_parquet(cache_file)
    return prices


def calculate_momentum_features(price_df: pd.DataFrame, rebalance_date: pd.Timestamp, config: ETFMomentumConfig | None = None) -> pd.DataFrame:
    config = config or ETFMomentumConfig()
    prices = price_df.sort_index().astype(float)
    rebalance_date = pd.Timestamp(rebalance_date)
    history = prices.loc[:rebalance_date]
    if history.empty:
        return pd.DataFrame(columns=["ETF", "price", "ret_1m", "ret_3m", "ret_6m", "ret_12m", "sma200"])

    as_of = history.index[-1]
    rows = []
    for ticker in prices.columns:
        series = history[ticker].dropna()
        row = {"rebalance_date": as_of, "ETF": ticker}
        if len(series) < max(252, config.sma_window) + 1:
            row.update({"price": series.iloc[-1] if len(series) else np.nan, "has_required_history": False})
        else:
            latest = float(series.iloc[-1])
            row.update(
                {
                    "price": latest,
                    "ret_1m": latest / float(series.iloc[-22]) - 1.0 if len(series) >= 22 else np.nan,
                    "ret_3m": latest / float(series.iloc[-64]) - 1.0 if len(series) >= 64 else np.nan,
                    "ret_6m": latest / float(series.iloc[-127]) - 1.0 if len(series) >= 127 else np.nan,
                    "ret_12m": latest / float(series.iloc[-253]) - 1.0 if len(series) >= 253 else np.nan,
                    "sma200": float(series.iloc[-config.sma_window :].mean()) if len(series) >= config.sma_window else np.nan,
                    "realized_vol_3m": float(series.pct_change().iloc[-63:].std(ddof=0) * np.sqrt(252)) if len(series) >= 64 else np.nan,
                    "sharpe_6m": _rolling_sharpe(series.pct_change().iloc[-126:]),
                    "has_required_history": True,
                }
            )
        rows.append(row)

    features = pd.DataFrame(rows)
    for column in ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "sma200", "price"]:
        if column not in features:
            features[column] = np.nan
    features["pass_ma200"] = features["price"].gt(features["sma200"]) if config.use_sma_filter else True
    features["pass_momentum"] = features["has_required_history"].fillna(False)
    if config.use_sma_filter:
        features["pass_momentum"] &= features["pass_ma200"].fillna(False)
    if config.require_positive_3m:
        features["pass_momentum"] &= features["ret_3m"].gt(0.0)
    if config.require_positive_6m:
        features["pass_momentum"] &= features["ret_6m"].gt(0.0)

    score = pd.Series(0.0, index=features.index, dtype=float)
    for column, weight in config.momentum_weights.items():
        rank = features[column].rank(pct=True, method="average")
        score = score.add(float(weight) * rank.fillna(0.0), fill_value=0.0)
    features["momentum_score"] = score
    return features.sort_values(["pass_momentum", "momentum_score"], ascending=[False, False]).reset_index(drop=True)


def _rolling_sharpe(returns: pd.Series) -> float:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan
    vol = clean.std(ddof=0) * np.sqrt(252)
    return float((clean.mean() * 252) / max(vol, EPSILON))


def apply_momentum_filter(
    momentum_df: pd.DataFrame,
    top_n: int = 5,
    min_n: int = 3,
) -> tuple[list[str], pd.DataFrame]:
    ranked = momentum_df.sort_values("momentum_score", ascending=False).reset_index(drop=True).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    passing = ranked.loc[ranked["pass_momentum"].fillna(False)].copy()
    selected = passing.head(top_n)["ETF"].astype(str).tolist()
    if len(selected) < min_n:
        selected = passing["ETF"].astype(str).tolist()
    ranked["selected"] = ranked["ETF"].isin(selected)
    return selected, ranked


def run_optimizer_on_selected_universe(
    returns_df: pd.DataFrame,
    selected_etfs: list[str],
    rebalance_date: pd.Timestamp,
    params: ETFMomentumConfig | dict | None = None,
    optimizer_fn: Callable[..., pd.Series] | None = None,
) -> pd.Series:
    config = _coerce_config(params)
    if not selected_etfs:
        return pd.Series({config.fallback_asset: 1.0}, dtype=float)
    train = returns_df.loc[:pd.Timestamp(rebalance_date), selected_etfs].dropna(axis=1, thresh=max(60, int(0.70 * config.lookback_days)))
    train = train.tail(config.lookback_days)
    selected = [ticker for ticker in selected_etfs if ticker in train.columns]
    if not selected:
        return pd.Series({config.fallback_asset: 1.0}, dtype=float)

    optimizer_assets = selected.copy()
    train_for_optimizer = train[selected].copy()
    asset_caps = {ticker: config.max_weight_per_asset for ticker in selected}
    if sum(asset_caps.values()) < 1.0 - EPSILON:
        train_for_optimizer[config.fallback_asset] = 0.0
        optimizer_assets.append(config.fallback_asset)
        asset_caps[config.fallback_asset] = 1.0

    cov = train_for_optimizer[optimizer_assets].cov().reindex(index=optimizer_assets, columns=optimizer_assets).fillna(0.0)
    momentum_signal = train_for_optimizer[optimizer_assets].tail(63).mean().mul(252).fillna(0.0)
    if config.fallback_asset in momentum_signal.index:
        momentum_signal.loc[config.fallback_asset] = 0.0
    optimizer = optimizer_fn or optimize_portfolio
    weights = optimizer(
        cov,
        momentum_signal,
        max_weight=config.max_weight_per_asset,
        risk_aversion=config.risk_aversion,
        objective_mode="mean_variance",
        asset_caps=asset_caps,
        concentration_penalty=config.concentration_penalty,
        momentum_strength=config.momentum_strength,
    )
    return weights.reindex(optimizer_assets).fillna(0.0).pipe(lambda s: s / s.sum() if s.sum() > EPSILON else s)


def run_monthly_rebalance_backtest(
    price_df: pd.DataFrame,
    params: ETFMomentumConfig | dict | None = None,
    optimizer_fn: Callable[..., pd.Series] | None = None,
) -> dict[str, pd.DataFrame | pd.Series]:
    config = _coerce_config(params)
    prices = price_df.reindex(columns=list(config.universe)).sort_index().ffill().dropna(how="all")
    prices = prices.loc[:, prices.notna().sum().ge(max(260, config.lookback_days))]
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    schedule = monthly_rebalance_dates(prices.index, lookback_days=max(config.lookback_days, 252), freq=config.rebalance_frequency)
    if len(schedule) < 2:
        raise RuntimeError("Not enough monthly rebalance dates after lookback requirements.")

    rebalance_rows = []
    momentum_frames = []
    weight_rows = []
    selected_rows = []
    period_returns = []
    prev_weights = pd.Series(dtype=float)

    for idx, rebalance_date in enumerate(schedule[:-1]):
        next_date = schedule[idx + 1]
        test_index = prices.index[(prices.index > rebalance_date) & (prices.index <= next_date)]
        if len(test_index) == 0:
            continue

        momentum = calculate_momentum_features(prices, rebalance_date, config)
        selected, ranked = apply_momentum_filter(momentum, config.top_n, config.min_n)
        weights = run_optimizer_on_selected_universe(
            returns,
            selected,
            rebalance_date,
            config,
            optimizer_fn=optimizer_fn,
        )
        if weights.empty or weights.sum() <= EPSILON:
            weights = pd.Series({config.fallback_asset: 1.0})

        active_assets = [asset for asset in weights.index if asset in returns.columns]
        if active_assets:
            next_returns = returns.reindex(test_index)[active_assets].fillna(0.0).mul(weights.reindex(active_assets), axis=1).sum(axis=1)
        else:
            next_returns = pd.Series(0.0, index=test_index, name="portfolio_return")

        turnover = float((weights.reindex(prev_weights.index.union(weights.index)).fillna(0.0) - prev_weights.reindex(prev_weights.index.union(weights.index)).fillna(0.0)).abs().sum() / 2.0) if not prev_weights.empty else float(weights.drop(labels=[config.fallback_asset], errors="ignore").sum())
        prev_weights = weights.copy()

        ranked["rebalance_date"] = rebalance_date
        ranked["selected_etfs"] = ",".join(selected)
        momentum_frames.append(ranked)
        selected_rows.append({"rebalance_date": rebalance_date, "selected_etfs": ",".join(selected), "selected_count": len(selected)})
        for asset, weight in weights.items():
            weight_rows.append({"rebalance_date": rebalance_date, "ETF": asset, "weight": float(weight)})
        rebalance_rows.append(
            {
                "rebalance_date": rebalance_date,
                "next_rebalance_date": next_date,
                "full_universe": ",".join(prices.columns),
                "selected_etfs": ",".join(selected),
                "selected_count": len(selected),
                "cash_weight": float(weights.get(config.fallback_asset, 0.0)),
                "portfolio_return_next_period": float((1.0 + next_returns).prod() - 1.0),
                "turnover": turnover,
            }
        )
        period_returns.append(next_returns.rename("portfolio_return"))

    daily_returns = pd.concat(period_returns).sort_index() if period_returns else pd.Series(dtype=float, name="portfolio_return")
    daily_returns = daily_returns.loc[~daily_returns.index.duplicated(keep="last")]
    equity_curve = curve_from_returns(daily_returns, initial=INITIAL_VALUE).rename("ETF momentum prefilter")
    metrics = compute_port_opt_style_metrics(equity_curve, risk_free_rate=config.risk_free_rate)
    metrics["Calmar"] = metrics["CAGR"] / abs(metrics["Max Drawdown"]) if abs(metrics.get("Max Drawdown", 0.0)) > EPSILON else np.nan
    metrics["Monthly Win Rate"] = float((daily_returns.resample("ME").apply(lambda x: (1.0 + x).prod() - 1.0) > 0.0).mean())
    metrics["Turnover"] = float(pd.DataFrame(rebalance_rows)["turnover"].mean()) if rebalance_rows else np.nan
    metrics["Average Selected ETFs"] = float(pd.DataFrame(selected_rows)["selected_count"].mean()) if selected_rows else np.nan
    metrics["Exposure Percentage"] = 1.0 - float(pd.DataFrame(rebalance_rows)["cash_weight"].mean()) if rebalance_rows else np.nan

    return {
        "rebalance_table": pd.DataFrame(rebalance_rows),
        "momentum_history": pd.concat(momentum_frames, ignore_index=True) if momentum_frames else pd.DataFrame(),
        "weight_history": pd.DataFrame(weight_rows),
        "selected_history": pd.DataFrame(selected_rows),
        "portfolio_returns": daily_returns.to_frame(),
        "equity_curve": equity_curve.to_frame(),
        "performance_metrics": metrics.rename("ETF momentum prefilter").to_frame().T,
    }


def _coerce_config(params: ETFMomentumConfig | dict | None) -> ETFMomentumConfig:
    if params is None:
        return ETFMomentumConfig()
    if isinstance(params, ETFMomentumConfig):
        return params
    return ETFMomentumConfig(**params)


def _write_outputs(results: dict[str, pd.DataFrame | pd.Series], prefix: str = OUTPUT_PREFIX) -> None:
    paths = default_paths(ROOT)
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    for key, value in results.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(paths.result_dir / f"{prefix}_{key}.csv", index=key in {"portfolio_returns", "equity_curve"})


def main() -> None:
    config = ETFMomentumConfig()
    prices = load_etf_adjusted_close(config.universe, start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE)
    results = run_monthly_rebalance_backtest(prices, config)
    _write_outputs(results)
    print(results["performance_metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
