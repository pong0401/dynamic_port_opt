from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import csv
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.cluster.vq import kmeans2
from scipy.optimize import minimize
from scipy.stats import norm

try:
    import yfinance as yf
except Exception:  # pragma: no cover - fallback if yfinance is unavailable locally
    yf = None


EPSILON = 1e-8
TRADING_DAYS = 252
DEFAULT_OVERLAY_PROFILE = {
    "use_daily_trend": True,
    "trend_cap": 0.65,
    "use_daily_drawdown": True,
    "drawdown_warn": -0.08,
    "drawdown_warn_cap": 0.50,
    "drawdown_crash": -0.15,
    "drawdown_crash_cap": 0.25,
    "use_daily_vix": True,
    "vix_warn": 28.0,
    "vix_warn_cap": 0.50,
    "vix_crash": 35.0,
    "vix_crash_cap": 0.25,
}
DEFAULT_UNIVERSE = [
    "NVDA",
    "AAPL",
    "AMZN",
    "MSFT",
    "AMD",
    "META",
    "GOOGL",
    "AVGO",
    "NFLX",
    "UNH",
    "JPM",
    "LLY",
    "V",
    "XOM",
    "ORCL",
    "CRM",
    "BAC",
    "BRK-B",
    "ADBE",
    "COST",
]
THAI_SYMBOL_EXCLUDE = {
    "BANKING.BK",
    "COMMERCE.BK",
    "FASHION.BK",
    "FIN.BK",
    "HOSPITAL.BK",
    "MINING.BK",
    "STEEL.BK",
    "NEWENTRY.BK",
    "NOTE.BK",
    "NO..BK",
}


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    source_cache_root: Path
    local_cache_root: Path
    notebook_dir: Path
    result_dir: Path
    sp500_repo_dir: Path
    port_opt_advance_dir: Path
    thai_stock_dir: Path


def default_paths(root: Optional[Path] = None) -> ProjectPaths:
    project_root = Path(root or Path.cwd()).resolve()
    return ProjectPaths(
        root=project_root,
        source_cache_root=project_root.parent / "port_opt_advance" / "data" / "cache" / "portopt_optimizer_proof" / "20Y",
        local_cache_root=project_root / "data" / "cache" / "dynamic_factor_copula",
        notebook_dir=project_root / "notebook",
        result_dir=project_root / "result",
        sp500_repo_dir=project_root.parent / "sp500",
        port_opt_advance_dir=project_root.parent / "port_opt_advance",
        thai_stock_dir=project_root.parent / "port_opt_advance" / "data" / "thai_stock",
    )


def _ensure_dirs(paths: ProjectPaths) -> None:
    paths.local_cache_root.mkdir(parents=True, exist_ok=True)
    paths.notebook_dir.mkdir(parents=True, exist_ok=True)
    paths.result_dir.mkdir(parents=True, exist_ok=True)


def _ensure_yfinance_cache(paths: ProjectPaths) -> None:
    if yf is None or not hasattr(yf, "set_tz_cache_location"):
        return
    tz_cache_dir = paths.local_cache_root / ".yfinance"
    tz_cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        yf.set_tz_cache_location(str(tz_cache_dir))
    except Exception:
        pass


def _normalize_ticker_symbol(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".", "-")


def load_sp500_membership_intervals(paths: Optional[ProjectPaths] = None) -> pd.DataFrame:
    paths = paths or default_paths()
    interval_file = paths.sp500_repo_dir / "sp500_ticker_start_end.csv"
    if not interval_file.exists():
        return pd.DataFrame(columns=["ticker", "start_date", "end_date"])
    intervals = pd.read_csv(interval_file)
    if intervals.empty:
        return pd.DataFrame(columns=["ticker", "start_date", "end_date"])
    intervals = intervals.rename(columns=str.lower)
    intervals["ticker"] = intervals["ticker"].map(_normalize_ticker_symbol)
    intervals["start_date"] = pd.to_datetime(intervals["start_date"], errors="coerce")
    intervals["end_date"] = pd.to_datetime(intervals["end_date"], errors="coerce")
    return intervals.dropna(subset=["ticker", "start_date"])


def _normalize_set_symbol(symbol: str) -> str:
    value = str(symbol).strip().upper()
    if value.endswith(".BK"):
        return value
    return f"{value}.BK" if value else ""


def load_set100_membership_intervals(paths: Optional[ProjectPaths] = None) -> pd.DataFrame:
    paths = paths or default_paths()
    interval_file = paths.thai_stock_dir / "set100_ticker_start_end.csv"
    if not interval_file.exists():
        return pd.DataFrame(columns=["ticker", "start_date", "end_date"])
    intervals = pd.read_csv(interval_file)
    if intervals.empty:
        return pd.DataFrame(columns=["ticker", "start_date", "end_date"])
    intervals = intervals.rename(columns=str.lower)
    intervals["ticker"] = intervals["ticker"].map(_normalize_set_symbol)
    intervals = intervals.loc[~intervals["ticker"].isin(THAI_SYMBOL_EXCLUDE)].copy()
    intervals["start_date"] = pd.to_datetime(intervals["start_date"], errors="coerce")
    intervals["end_date"] = pd.to_datetime(intervals["end_date"], errors="coerce")
    return intervals.dropna(subset=["ticker", "start_date", "end_date"])


def get_set100_members_as_of(as_of_date: pd.Timestamp, paths: Optional[ProjectPaths] = None) -> List[str]:
    intervals = load_set100_membership_intervals(paths)
    if intervals.empty:
        return []
    timestamp = pd.Timestamp(as_of_date).normalize()
    active = intervals.loc[
        (intervals["start_date"] <= timestamp)
        & (intervals["end_date"] >= timestamp)
    ]
    return active["ticker"].drop_duplicates().tolist()


def get_sp500_members_as_of(as_of_date: pd.Timestamp, paths: Optional[ProjectPaths] = None) -> List[str]:
    intervals = load_sp500_membership_intervals(paths)
    if intervals.empty:
        return []
    timestamp = pd.Timestamp(as_of_date).normalize()
    active = intervals.loc[
        (intervals["start_date"] <= timestamp)
        & (intervals["end_date"].isna() | (intervals["end_date"] >= timestamp))
    ]
    return active["ticker"].drop_duplicates().tolist()


def _read_parquet_columns(path: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if columns is None:
        return pd.read_parquet(path)
    return pd.read_parquet(path, columns=list(columns))


@lru_cache(maxsize=8)
def _parquet_column_names(path_str: str) -> Tuple[str, ...]:
    return tuple(pq.ParquetFile(path_str).schema.names)


def load_cached_market_data(
    paths: Optional[ProjectPaths] = None,
    tickers: Optional[Sequence[str]] = None,
) -> Dict[str, pd.DataFrame]:
    paths = paths or default_paths()
    _ensure_dirs(paths)
    _ensure_yfinance_cache(paths)
    requested = list(dict.fromkeys(tickers or []))
    source_price_cols = set(_parquet_column_names(str(paths.source_cache_root / "prices.parquet")))
    source_volume_cols = set(_parquet_column_names(str(paths.source_cache_root / "volumes.parquet")))
    requested_price_cols = [ticker for ticker in requested if ticker in source_price_cols]
    requested_volume_cols = [ticker for ticker in requested if ticker in source_volume_cols]
    frames = {
        "prices": _read_parquet_columns(paths.source_cache_root / "prices.parquet", requested_price_cols if requested else None),
        "volumes": _read_parquet_columns(paths.source_cache_root / "volumes.parquet", requested_volume_cols if requested else None),
        "benchmark": pd.read_parquet(paths.source_cache_root / "benchmark.parquet").rename(columns={"value": "benchmark"}),
        "vol_proxy": pd.read_parquet(paths.source_cache_root / "vol_proxy.parquet").rename(columns={"value": "vol_proxy"}),
    }

    local_prices = paths.local_cache_root / "extra_prices.parquet"
    local_volumes = paths.local_cache_root / "extra_volumes.parquet"
    if local_prices.exists():
        local_price_cols = set(_parquet_column_names(str(local_prices)))
        extra_prices = _read_parquet_columns(local_prices, [ticker for ticker in requested if ticker in local_price_cols] if requested else None)
        frames["prices"] = frames["prices"].combine_first(extra_prices)
    if local_volumes.exists():
        local_volume_cols = set(_parquet_column_names(str(local_volumes)))
        extra_volumes = _read_parquet_columns(local_volumes, [ticker for ticker in requested if ticker in local_volume_cols] if requested else None)
        frames["volumes"] = frames["volumes"].combine_first(extra_volumes)
    return frames


def _persist_extra_cache(
    paths: ProjectPaths,
    extra_prices: pd.DataFrame,
    extra_volumes: pd.DataFrame,
    downloaded: Sequence[str],
    missing: Sequence[str],
) -> None:
    if not extra_prices.empty:
        price_file = paths.local_cache_root / "extra_prices.parquet"
        combined = extra_prices
        if price_file.exists():
            combined = pd.read_parquet(price_file).combine_first(extra_prices)
        combined.sort_index().to_parquet(price_file)
    if not extra_volumes.empty:
        volume_file = paths.local_cache_root / "extra_volumes.parquet"
        combined = extra_volumes
        if volume_file.exists():
            combined = pd.read_parquet(volume_file).combine_first(extra_volumes)
        combined.sort_index().to_parquet(volume_file)

    log_file = paths.local_cache_root / "download_log.csv"
    write_header = not log_file.exists()
    with log_file.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(["saved_at", "downloaded_tickers", "missing_tickers"])
        writer.writerow(
            [
                pd.Timestamp.utcnow().isoformat(),
                ",".join(downloaded),
                ",".join(missing),
            ]
        )


def ensure_assets_available(
    tickers: Sequence[str],
    start_date: str,
    end_date: str,
    paths: Optional[ProjectPaths] = None,
) -> Dict[str, pd.DataFrame]:
    paths = paths or default_paths()
    _ensure_yfinance_cache(paths)
    target = list(dict.fromkeys(tickers))
    frames = load_cached_market_data(paths, tickers=target)

    prices = frames["prices"]
    volumes = frames["volumes"]
    missing = [ticker for ticker in target if ticker not in prices.columns]
    weak_history = [
        ticker
        for ticker in target
        if ticker in prices.columns
        and (
            prices.loc[prices.index >= pd.Timestamp(start_date), ticker].dropna().empty
            or prices.loc[prices.index <= pd.Timestamp(end_date), ticker].dropna().empty
        )
    ]
    missing = list(dict.fromkeys(missing + weak_history))

    if not missing:
        return frames
    if yf is None:
        warnings.warn("yfinance is unavailable; proceeding with cached assets only.")
        return frames

    try:
        raw = yf.download(
            missing,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=False,
        )
    except Exception as exc:  # pragma: no cover - external I/O path
        warnings.warn(f"Asset download failed: {exc}")
        return frames

    extra_price_frames: List[pd.Series] = []
    extra_volume_frames: List[pd.Series] = []
    downloaded: List[str] = []
    unresolved: List[str] = []

    def _extract_ticker_frame(raw_data: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
        if isinstance(raw_data.columns, pd.MultiIndex):
            if ticker not in raw_data.columns.get_level_values(0):
                return None
            return raw_data[ticker]
        if "Close" in raw_data.columns:
            return raw_data
        return None

    for ticker in missing:
        sub = _extract_ticker_frame(raw, ticker)
        if sub is None or "Close" not in sub.columns:
            unresolved.append(ticker)
            continue
        close = sub["Close"].dropna().rename(ticker)
        if close.empty:
            unresolved.append(ticker)
            continue
        volume = sub.get("Volume", pd.Series(index=close.index, dtype=float)).reindex(close.index).fillna(0.0).rename(ticker)
        extra_price_frames.append(close)
        extra_volume_frames.append(volume)
        downloaded.append(ticker)

    if extra_price_frames:
        extra_prices = pd.concat(extra_price_frames, axis=1).sort_index()
        extra_volumes = pd.concat(extra_volume_frames, axis=1).sort_index()
        _persist_extra_cache(paths, extra_prices, extra_volumes, downloaded, unresolved)
        frames["prices"] = frames["prices"].combine_first(extra_prices)
        frames["volumes"] = frames["volumes"].combine_first(extra_volumes)

    if unresolved:
        warnings.warn(f"Some assets could not be loaded and remain missing: {unresolved}")
    return frames


def select_universe(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    preferred: Optional[Sequence[str]] = None,
    n_assets: int = 16,
    start_date: str = "2012-01-01",
) -> List[str]:
    preferred = list(preferred or DEFAULT_UNIVERSE)
    start_ts = pd.Timestamp(start_date)
    candidate_pool = [ticker for ticker in preferred if ticker in prices.columns]
    if not candidate_pool:
        candidate_pool = [ticker for ticker in prices.columns if ticker not in {"benchmark", "vol_proxy"}]

    availability = prices.loc[prices.index >= start_ts, candidate_pool].notna().mean()
    liquid_proxy = (
        prices.loc[prices.index >= start_ts, candidate_pool].ffill()
        * volumes.loc[volumes.index >= start_ts, candidate_pool].reindex(prices.loc[prices.index >= start_ts].index).fillna(0.0)
    ).median()
    ranked = (
        pd.DataFrame({"availability": availability, "liquidity": liquid_proxy})
        .fillna(0.0)
        .query("availability >= 0.90")
        .sort_values(["liquidity", "availability"], ascending=False)
    )
    if ranked.empty:
        ranked = pd.DataFrame({"availability": availability, "liquidity": liquid_proxy}).sort_values(
            ["liquidity", "availability"], ascending=False
    )
    return ranked.head(n_assets).index.tolist()


def select_point_in_time_universe(
    prices_window: pd.DataFrame,
    volumes_window: pd.DataFrame,
    candidate_tickers: Sequence[str],
    n_assets: int,
    min_history_ratio: float = 0.90,
) -> List[str]:
    candidates = [ticker for ticker in candidate_tickers if ticker in prices_window.columns]
    if not candidates:
        return []
    availability = prices_window[candidates].notna().mean()
    liquidity = (
        prices_window[candidates].ffill()
        * volumes_window.reindex(prices_window.index).reindex(columns=candidates).fillna(0.0)
    ).median()
    ranked = (
        pd.DataFrame({"availability": availability, "liquidity": liquidity})
        .fillna(0.0)
        .query("availability >= @min_history_ratio")
        .sort_values(["liquidity", "availability"], ascending=False)
    )
    if ranked.empty:
        ranked = pd.DataFrame({"availability": availability, "liquidity": liquidity}).sort_values(
            ["liquidity", "availability"], ascending=False
        )
    return ranked.head(n_assets).index.tolist()


def prepare_panel(
    start_date: str = "2012-01-01",
    end_date: str = "2026-04-30",
    tickers: Optional[Sequence[str]] = None,
    n_assets: int = 16,
    preselect_universe: bool = True,
    universe_mode: str = "fixed",
    benchmark_ticker: Optional[str] = None,
    vol_proxy_ticker: Optional[str] = None,
    paths: Optional[ProjectPaths] = None,
) -> Dict[str, pd.DataFrame | List[str]]:
    paths = paths or default_paths()
    requested = list(dict.fromkeys(tickers or DEFAULT_UNIVERSE))
    if universe_mode == "sp500_pit" and tickers is None:
        available_cache_cols = set(_parquet_column_names(str(paths.source_cache_root / "prices.parquet")))
        requested = [
            ticker
            for ticker in load_sp500_membership_intervals(paths)["ticker"].drop_duplicates().tolist()
            if ticker in available_cache_cols
        ]
    elif universe_mode == "set100_pit" and tickers is None:
        available_cache_cols = set(_parquet_column_names(str(paths.source_cache_root / "prices.parquet")))
        local_prices = paths.local_cache_root / "extra_prices.parquet"
        if local_prices.exists():
            available_cache_cols |= set(_parquet_column_names(str(local_prices)))
        requested = [
            ticker
            for ticker in load_set100_membership_intervals(paths)["ticker"].drop_duplicates().tolist()
            if ticker in available_cache_cols
        ]
    base = ensure_assets_available(requested, start_date, end_date, paths)
    prices = base["prices"].loc[start_date:end_date].sort_index().ffill()
    volumes = base["volumes"].loc[start_date:end_date].sort_index().fillna(0.0)

    def _load_reference_series(ticker: str, series_name: str) -> pd.Series:
        reference_frames = ensure_assets_available([ticker], start_date, end_date, paths)
        frame = reference_frames["prices"]
        if ticker not in frame.columns:
            raise ValueError(
                f"{series_name} ticker '{ticker}' is not available in the current cache or local downloads. "
                f"For Thailand PIT runs, build a Thai cache first."
            )
        series = frame.loc[start_date:end_date, ticker].sort_index().ffill().rename(series_name)
        if series.dropna().empty:
            raise ValueError(
                f"{series_name} ticker '{ticker}' has no usable history in the requested window. "
                f"For Thailand PIT runs, build a Thai cache first."
            )
        return series

    if benchmark_ticker:
        benchmark = _load_reference_series(benchmark_ticker, "benchmark")
    else:
        benchmark = base["benchmark"].loc[start_date:end_date, "benchmark"].sort_index().ffill()
    if vol_proxy_ticker:
        vol_proxy = _load_reference_series(vol_proxy_ticker, "vol_proxy")
    elif vol_proxy_ticker == "":
        vol_proxy = pd.Series(0.0, index=benchmark.index, dtype=float, name="vol_proxy")
    else:
        vol_proxy = base["vol_proxy"].loc[start_date:end_date, "vol_proxy"].sort_index().ffill()

    selected = [ticker for ticker in requested if ticker in prices.columns]
    if preselect_universe:
        selected = select_universe(prices, volumes, tickers, n_assets=n_assets, start_date=start_date)
    selected = [ticker for ticker in selected if ticker in prices.columns]
    prices = prices[selected].dropna(how="all")
    volumes = volumes.reindex(prices.index).reindex(columns=selected).fillna(0.0)

    common_index = prices.index.intersection(benchmark.index)
    common_index = common_index.intersection(vol_proxy.index)
    prices = prices.reindex(common_index).ffill().dropna(how="all")
    volumes = volumes.reindex(common_index).fillna(0.0)
    benchmark = benchmark.reindex(common_index).ffill()
    vol_proxy = vol_proxy.reindex(common_index).ffill()

    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan)
    benchmark_ret = benchmark.pct_change().rename("benchmark")
    if vol_proxy_ticker == "":
        vol_proxy_ret = pd.Series(0.0, index=vol_proxy.index, dtype=float, name="vol_proxy")
    else:
        vol_proxy_ret = vol_proxy.pct_change().rename("vol_proxy")
    panel = {
        "prices": prices,
        "volumes": volumes,
        "returns": returns,
        "benchmark": benchmark,
        "benchmark_ret": benchmark_ret,
        "vol_proxy": vol_proxy,
        "vol_proxy_ret": vol_proxy_ret,
        "tickers": selected,
        "benchmark_ticker": benchmark_ticker or "SPY",
        "vol_proxy_ticker": vol_proxy_ticker if vol_proxy_ticker is not None else "^VIX",
    }
    return panel


def compute_feature_table(
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    vol_proxy_returns: pd.Series,
    price_window: pd.DataFrame,
    include_momentum_features: bool = True,
    feature_flags: Optional[Dict[str, bool]] = None,
) -> pd.DataFrame:
    feature_flags = feature_flags or {}
    market_var = float(benchmark_returns.var(ddof=1))
    vol_var = float(vol_proxy_returns.var(ddof=1))
    downside_mask = benchmark_returns < 0
    market_down = benchmark_returns.loc[downside_mask]
    down_var = float(market_down.var(ddof=1)) if len(market_down) > 5 else market_var

    rows = []
    for ticker in asset_returns.columns:
        series = asset_returns[ticker].dropna()
        joined = pd.concat(
            [series, benchmark_returns, vol_proxy_returns],
            axis=1,
            join="inner",
        ).dropna()
        if len(joined) < 40:
            continue
        joined.columns = ["asset", "market", "vol_proxy"]
        beta_mkt = float(np.cov(joined["asset"], joined["market"], ddof=1)[0, 1] / max(market_var, EPSILON))
        beta_vol = float(np.cov(joined["asset"], joined["vol_proxy"], ddof=1)[0, 1] / max(vol_var, EPSILON))
        down_join = joined.loc[joined["market"] < 0]
        if len(down_join) >= 10:
            downside_beta = float(np.cov(down_join["asset"], down_join["market"], ddof=1)[0, 1] / max(down_var, EPSILON))
        else:
            downside_beta = beta_mkt
        resid = joined["asset"] - beta_mkt * joined["market"]
        resid_vol = float(resid.std(ddof=1))
        momentum_63 = float((1.0 + series.tail(63)).prod() - 1.0) if len(series) >= 63 else float(series.mean() * 63.0)
        momentum_21 = float((1.0 + series.tail(21)).prod() - 1.0) if len(series) >= 21 else float(series.mean() * 21.0)
        if not include_momentum_features:
            momentum_63 = 0.0
            momentum_21 = 0.0
        price_series = price_window[ticker].dropna()
        drawdown = 0.0
        if not price_series.empty:
            running_max = price_series.cummax()
            drawdown = float((price_series / running_max - 1.0).min())
        rows.append(
            {
                "ticker": ticker,
                "beta_mkt": beta_mkt,
                "beta_vol": beta_vol,
                "downside_beta": downside_beta,
                "resid_vol": resid_vol,
                "mom_63": momentum_63,
                "mom_21": momentum_21,
                "drawdown": drawdown,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["beta_mkt", "beta_vol", "downside_beta", "resid_vol", "mom_63", "mom_21", "drawdown"], dtype=float)
    features = pd.DataFrame(rows).set_index("ticker").sort_index()
    features = features.apply(pd.to_numeric, errors="coerce").astype(float)
    for column, enabled in feature_flags.items():
        if column in features.columns and not enabled:
            features = features.drop(columns=[column])
    return features


def _standardize_features(features: pd.DataFrame) -> Tuple[np.ndarray, pd.Series, pd.Series]:
    mean = features.mean()
    std = features.std(ddof=0).replace(0.0, 1.0)
    return ((features - mean) / std).to_numpy(), mean, std


def initialize_static_clusters(features: pd.DataFrame, n_clusters: int, seed: int = 7) -> Dict[str, object]:
    matrix, mean, std = _standardize_features(features)
    centroids, labels = kmeans2(matrix, k=n_clusters, minit="points", seed=seed)
    label_series = pd.Series(labels, index=features.index, name="cluster")
    centroid_df = pd.DataFrame(centroids, columns=features.columns)
    centroid_unscaled = centroid_df.mul(std.values, axis=1).add(mean.values, axis=1)
    return {
        "labels": label_series,
        "centroids_scaled": centroid_df,
        "centroids_unscaled": centroid_unscaled,
        "feature_mean": mean,
        "feature_std": std,
    }


def _distance_emission(features: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame:
    diff = features.to_numpy()[:, None, :] - centroids.to_numpy()[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    scores = np.exp(-0.5 * dist2)
    scores = scores / np.clip(scores.sum(axis=1, keepdims=True), EPSILON, None)
    return pd.DataFrame(scores, index=features.index, columns=centroids.index)


def _transition_matrix(cluster_betas: pd.Series, stay_bias: float = 2.0, distance_penalty: float = 3.0) -> pd.DataFrame:
    beta_diff = np.abs(cluster_betas.to_numpy()[:, None] - cluster_betas.to_numpy()[None, :])
    mat = np.exp(-distance_penalty * beta_diff)
    np.fill_diagonal(mat, np.diag(mat) + stay_bias)
    mat = mat / np.clip(mat.sum(axis=1, keepdims=True), EPSILON, None)
    return pd.DataFrame(mat, index=cluster_betas.index, columns=cluster_betas.index)


def compute_market_stress_signal(
    benchmark_returns: pd.Series,
    vol_proxy_returns: pd.Series,
) -> float:
    joined = pd.concat([benchmark_returns.rename("benchmark"), vol_proxy_returns.rename("vol_proxy")], axis=1).dropna()
    if joined.empty:
        return 0.0
    recent = joined.tail(min(63, len(joined)))
    bench_momentum = float(recent["benchmark"].mean())
    bench_vol = float(recent["benchmark"].std(ddof=1)) if len(recent) > 1 else 0.0
    vol_spike = float(recent["vol_proxy"].mean())
    stress = (-4.0 * bench_momentum) + (3.0 * vol_spike) + (2.0 * bench_vol)
    return float(np.tanh(stress))


def run_dynamic_hmm(
    feature_history: Dict[pd.Timestamp, pd.DataFrame],
    initial_state: Dict[str, object],
    gas_alpha: float = 0.25,
    gas_beta: float = 0.65,
    market_stress_history: Optional[Dict[pd.Timestamp, float]] = None,
    posterior_power: float = 1.0,
) -> Dict[str, Dict[pd.Timestamp, pd.DataFrame | pd.Series]]:
    cluster_ids = pd.Index(range(len(initial_state["centroids_unscaled"])), name="cluster")
    prev_centroids = initial_state["centroids_unscaled"].copy()
    prev_posteriors: Dict[str, pd.Series] = {}

    posterior_history: Dict[pd.Timestamp, pd.DataFrame] = {}
    assignment_history: Dict[pd.Timestamp, pd.Series] = {}
    centroid_history: Dict[pd.Timestamp, pd.DataFrame] = {}
    transition_history: Dict[pd.Timestamp, pd.DataFrame] = {}

    for date in sorted(feature_history):
        features = feature_history[date].copy().sort_index()
        if features.empty:
            continue

        emission = _distance_emission(features, prev_centroids)
        if posterior_power != 1.0:
            emission = emission.pow(posterior_power)
            emission = emission.div(emission.sum(axis=1), axis=0).fillna(1.0 / len(emission.columns))
        cluster_betas = prev_centroids["beta_mkt"].copy()
        cluster_betas.index = cluster_ids
        stress = 0.0 if market_stress_history is None else float(market_stress_history.get(date, 0.0))
        stay_bias = 2.2 - 1.4 * max(stress, 0.0)
        distance_penalty = 2.2 + 2.0 * max(stress, 0.0)
        transition = _transition_matrix(cluster_betas, stay_bias=stay_bias, distance_penalty=distance_penalty)

        posteriors = pd.DataFrame(index=features.index, columns=cluster_ids, dtype=float)
        for ticker in features.index:
            prior = prev_posteriors.get(ticker)
            if prior is None:
                prior = pd.Series(1.0 / len(cluster_ids), index=cluster_ids)
            else:
                prior = prior.reindex(cluster_ids).fillna(1.0 / len(cluster_ids))
            filtered = emission.loc[ticker].reindex(cluster_ids) * (prior.to_numpy() @ transition.to_numpy())
            filtered = filtered / np.clip(filtered.sum(), EPSILON, None)
            posteriors.loc[ticker] = filtered
            prev_posteriors[ticker] = pd.Series(filtered, index=cluster_ids)

        assignments = posteriors.idxmax(axis=1).astype(int)
        new_centroids = prev_centroids.copy()
        for cluster in cluster_ids:
            weights = posteriors[cluster].astype(float)
            weight_sum = float(weights.sum())
            if weight_sum <= EPSILON:
                continue
            weighted_mean = features.mul(weights, axis=0).sum(axis=0) / weight_sum
            score = weighted_mean - prev_centroids.loc[cluster]
            stress_boost = 1.0 + 0.75 * max(stress, 0.0)
            new_centroids.loc[cluster] = (
                prev_centroids.loc[cluster] * gas_beta
                + weighted_mean * (1.0 - gas_beta)
                + (gas_alpha * stress_boost) * score
            )

        posterior_history[date] = posteriors
        assignment_history[date] = assignments
        centroid_history[date] = new_centroids.copy()
        transition_history[date] = transition.copy()
        prev_centroids = new_centroids

    return {
        "posterior_history": posterior_history,
        "assignment_history": assignment_history,
        "centroid_history": centroid_history,
        "transition_history": transition_history,
    }


def gaussian_copula_covariance(factor_returns: pd.DataFrame) -> pd.DataFrame:
    ranks = factor_returns.rank(pct=True).clip(1e-4, 1.0 - 1e-4)
    normals = ranks.apply(norm.ppf)
    corr = normals.corr().fillna(0.0)
    std = factor_returns.std(ddof=1).replace(0.0, EPSILON)
    cov = corr.mul(std, axis=0).mul(std, axis=1)
    return cov


def build_factor_covariance(
    train_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    posteriors: pd.DataFrame,
    feature_table: pd.DataFrame,
    dynamic: bool,
    centroid_snapshot: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    assets = train_returns.columns.intersection(posteriors.index).intersection(feature_table.index)
    train = train_returns[assets].dropna(how="all")
    market = benchmark_returns.reindex(train.index).dropna()
    train = train.reindex(market.index)
    post = posteriors.reindex(assets).fillna(0.0)
    feature_table = feature_table.reindex(assets)

    cluster_factor_returns: Dict[str, pd.Series] = {}
    for cluster in post.columns:
        weights = post[cluster].astype(float)
        if not dynamic:
            weights = (weights == weights.max()).astype(float)
        else:
            weights = weights.pow(2.0)
        if float(weights.sum()) <= EPSILON:
            cluster_factor_returns[f"cluster_{cluster}"] = pd.Series(0.0, index=train.index)
            continue
        normalized = weights / weights.sum()
        cluster_factor_returns[f"cluster_{cluster}"] = train.mul(normalized, axis=1).sum(axis=1)

    factor_df = pd.concat([market.rename("market"), pd.DataFrame(cluster_factor_returns)], axis=1).dropna()
    factor_cov = gaussian_copula_covariance(factor_df)

    market_loading = feature_table["beta_mkt"].astype(float).fillna(0.0)
    cluster_loading = post.astype(float)
    if not dynamic:
        cluster_loading = pd.get_dummies(post.idxmax(axis=1)).reindex(index=assets, columns=post.columns, fill_value=0.0)
    elif centroid_snapshot is not None and "beta_mkt" in centroid_snapshot.columns:
        dynamic_market = cluster_loading.mul(centroid_snapshot["beta_mkt"].reindex(cluster_loading.columns), axis=1).sum(axis=1)
        market_loading = 0.35 * market_loading + 0.65 * dynamic_market.reindex(assets).fillna(market_loading)
        if "downside_beta" in centroid_snapshot.columns:
            downside_scale = cluster_loading.mul(centroid_snapshot["downside_beta"].reindex(cluster_loading.columns), axis=1).sum(axis=1)
            market_loading = market_loading * (1.0 + 0.15 * np.sign(downside_scale.reindex(assets).fillna(0.0)))

    loadings = pd.concat([market_loading.rename("market"), cluster_loading], axis=1).reindex(index=assets)
    loadings.columns = factor_cov.columns
    factor_component = loadings.to_numpy() @ factor_cov.to_numpy() @ loadings.to_numpy().T
    factor_component = pd.DataFrame(factor_component, index=assets, columns=assets)

    fitted = pd.DataFrame(index=train.index, columns=assets, dtype=float)
    for asset in assets:
        fitted[asset] = market_loading[asset] * factor_df["market"]
        for cluster in post.columns:
            factor_name = f"cluster_{cluster}"
            fitted[asset] += cluster_loading.loc[asset, cluster] * factor_df[factor_name]
    residual_var = (train.reindex(fitted.index)[assets] - fitted).var(ddof=1).fillna(train.var(ddof=1)).clip(lower=EPSILON)

    cov = factor_component + np.diag(residual_var.reindex(assets).to_numpy())
    cov = 0.80 * cov + 0.20 * np.diag(np.diag(cov))
    cov = pd.DataFrame(cov, index=assets, columns=assets)
    return cov, factor_df.mean().reindex(factor_cov.columns).fillna(0.0)


def optimize_portfolio(
    cov: pd.DataFrame,
    momentum_signal: pd.Series,
    max_weight: float = 0.18,
    risk_aversion: float = 8.0,
    objective_mode: str = "mean_variance",
    asset_caps: Optional[dict[str, float]] = None,
    concentration_penalty: float = 0.0,
    momentum_strength: float = 1.0,
) -> pd.Series:
    assets = cov.index
    cov = cov.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    cov_matrix = cov.to_numpy(dtype=float)
    mu = momentum_signal.reindex(assets).fillna(momentum_signal.median() if not momentum_signal.empty else 0.0).to_numpy()
    mu = np.clip(mu, np.nanpercentile(mu, 10), np.nanpercentile(mu, 90)) if len(mu) else mu
    n_assets = len(assets)
    caps = pd.Series(max_weight, index=assets, dtype=float)
    if asset_caps:
        caps.update(pd.Series(asset_caps, dtype=float).reindex(assets).dropna())
    if float(caps.sum()) < 1.0 - EPSILON:
        raise ValueError("Portfolio caps must sum to at least 100%.")
    x0 = caps / caps.sum()
    bounds = [(0.0, float(caps.loc[asset])) for asset in assets]
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]

    if objective_mode == "risk_parity_mom_tilt":
        base = optimize_risk_parity(cov, max_weight=max_weight, asset_caps=asset_caps)
        mu_series = pd.Series(mu, index=assets)
        mu_std = float(mu_series.std(ddof=0))
        if mu_std <= EPSILON:
            tilt = pd.Series(1.0, index=assets)
        else:
            tilt = 1.0 + 0.35 * ((mu_series - mu_series.mean()) / mu_std).clip(-1.0, 1.0)
        weights = (base * tilt).clip(lower=0.0)
        weights = weights / weights.sum()
        for _ in range(len(weights) * 2):
            over_mask = weights > caps
            if not over_mask.any():
                break
            excess = float((weights[over_mask] - caps[over_mask]).sum())
            weights.loc[over_mask] = caps[over_mask]
            under_mask = weights < caps - 1e-12
            if not under_mask.any():
                break
            capacity = (caps - weights).clip(lower=0.0)
            weights.loc[under_mask] = weights.loc[under_mask] + excess * (capacity.loc[under_mask] / capacity.loc[under_mask].sum())
            weights = weights / weights.sum()
        return weights / weights.sum()

    def objective(x: np.ndarray) -> float:
        variance = float(x @ cov_matrix @ x)
        expected = float(momentum_strength * (mu @ x))
        concentration = float(np.sum(np.square(x)))
        if objective_mode == "mean_variance":
            return 0.5 * risk_aversion * variance - expected + concentration_penalty * concentration
        if objective_mode == "max_sharpe_mom":
            return -expected / max(np.sqrt(max(variance, EPSILON)), EPSILON) + concentration_penalty * concentration
        if objective_mode == "min_vol_mom_tilt":
            return variance - 0.20 * expected + concentration_penalty * concentration
        raise ValueError(f"Unsupported optimizer objective_mode: {objective_mode}")

    result = minimize(objective, x0=x0, bounds=bounds, constraints=constraints, method="SLSQP")
    if not result.success:
        return pd.Series(x0, index=assets)
    weights = pd.Series(result.x, index=assets)
    weights = weights.clip(lower=0.0)
    return weights / weights.sum()


def build_momentum_signal(
    feature_table: pd.DataFrame,
    mode: str = "mom_63",
) -> pd.Series:
    if feature_table.empty:
        return pd.Series(dtype=float)

    index = feature_table.index

    def _safe_col(name: str) -> pd.Series:
        if name in feature_table.columns:
            return feature_table[name].astype(float)
        return pd.Series(0.0, index=index, dtype=float)

    mom_63 = _safe_col("mom_63")
    mom_21 = _safe_col("mom_21")

    if mode == "mom_63":
        signal = mom_63
    elif mode == "mom_21":
        signal = mom_21
    elif mode == "blend_21_63":
        signal = 0.5 * mom_21 + 0.5 * mom_63
    elif mode == "zscore_63":
        std = float(mom_63.std(ddof=0))
        std = std if std > EPSILON else 1.0
        signal = (mom_63 - mom_63.mean()) / std
    elif mode == "rank_63":
        n = max(len(mom_63), 1)
        signal = mom_63.rank(method="average", pct=True).mul(2.0).sub(1.0)
        if n == 1:
            signal = pd.Series(0.0, index=index, dtype=float)
    else:
        raise ValueError(f"Unsupported momentum signal mode: {mode}")

    return signal.replace([np.inf, -np.inf], np.nan).fillna(signal.median() if not signal.dropna().empty else 0.0)


def optimize_risk_parity(
    cov: pd.DataFrame,
    max_weight: float = 0.18,
    asset_caps: Optional[dict[str, float]] = None,
) -> pd.Series:
    assets = cov.index
    cov = cov.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    caps = pd.Series(max_weight, index=assets, dtype=float)
    if asset_caps:
        caps.update(pd.Series(asset_caps, dtype=float).reindex(assets).dropna())
    if float(caps.sum()) < 1.0 - EPSILON:
        raise ValueError("Portfolio caps must sum to at least 100%.")
    vol = pd.Series(np.sqrt(np.clip(np.diag(cov.to_numpy(dtype=float)), EPSILON, None)), index=assets)
    inv_vol = 1.0 / vol.replace(0.0, np.nan)
    inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).fillna(inv_vol.median())
    weights = inv_vol / inv_vol.sum()

    # Cap oversized positions and redistribute remaining weight proportionally.
    for _ in range(len(weights) * 2):
        over_mask = weights > caps
        if not over_mask.any():
            break
        capped_weight = float((weights[over_mask] - caps[over_mask]).sum())
        weights.loc[over_mask] = caps[over_mask]
        under_mask = weights < caps - 1e-12
        if not under_mask.any():
            break
        capacity = (caps - weights).clip(lower=0.0)
        weights.loc[under_mask] = weights.loc[under_mask] + capped_weight * (capacity.loc[under_mask] / capacity.loc[under_mask].sum())
        weights = weights / weights.sum()
    return weights / weights.sum()


def compute_metrics(nav: pd.Series, benchmark_nav: Optional[pd.Series] = None) -> pd.Series:
    returns = nav.pct_change().dropna()
    if returns.empty:
        return pd.Series(dtype=float)
    total_return = nav.iloc[-1] / nav.iloc[0] - 1.0
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, EPSILON)
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0
    vol = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    downside = returns.where(returns < 0, 0.0)
    downside_vol = downside.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = returns.mean() / max(returns.std(ddof=1), EPSILON) * np.sqrt(TRADING_DAYS)
    sortino = returns.mean() / max(downside.std(ddof=1), EPSILON) * np.sqrt(TRADING_DAYS)
    drawdown = nav / nav.cummax() - 1.0
    turnover = np.nan
    relative_total = np.nan
    if benchmark_nav is not None and not benchmark_nav.empty:
        relative_total = total_return - (benchmark_nav.iloc[-1] / benchmark_nav.iloc[0] - 1.0)
    return pd.Series(
        {
            "Total Return": total_return,
            "CAGR": cagr,
            "Annual Vol": vol,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Max Drawdown": drawdown.min(),
            "Benchmark Relative Return": relative_total,
            "Turnover": turnover,
        }
    )


def compute_port_opt_style_metrics(nav: pd.Series, risk_free_rate: float = 0.03) -> pd.Series:
    values = nav.dropna().astype(float)
    if len(values) < 2:
        return pd.Series(dtype=float)
    returns = values.pct_change().dropna()
    total_return = values.iloc[-1] / values.iloc[0] - 1.0
    years = max(len(values) / TRADING_DAYS, 1.0 / TRADING_DAYS)
    cagr = values.iloc[-1] ** (1.0 / years) / values.iloc[0] ** (1.0 / years) - 1.0
    annual_vol = returns.std(ddof=0) * np.sqrt(TRADING_DAYS)
    sharpe = ((returns.mean() * TRADING_DAYS) - risk_free_rate) / np.maximum(annual_vol, EPSILON)
    downside = returns[returns < 0].std(ddof=0) * np.sqrt(TRADING_DAYS)
    sortino = ((returns.mean() * TRADING_DAYS) - risk_free_rate) / np.maximum(downside, EPSILON)
    drawdown = values / values.cummax() - 1.0
    hit_rate = float((returns > 0).mean())
    return pd.Series(
        {
            "Total Return": float(total_return),
            "CAGR": float(cagr),
            "Annual Vol": float(annual_vol),
            "Sharpe": float(sharpe),
            "Sortino": float(sortino),
            "Max Drawdown": float(drawdown.min()),
            "Hit Rate": hit_rate,
        }
    )


def curve_from_returns(returns: pd.Series, initial: float = 10_000.0) -> pd.Series:
    clean = returns.fillna(0.0).sort_index()
    return (initial * (1.0 + clean).cumprod()).rename("PortValue")


def daily_overlay_cap(
    date: pd.Timestamp,
    benchmark: pd.Series,
    vol_proxy: Optional[pd.Series],
    profile: Optional[dict] = None,
) -> Tuple[float, Dict[str, float | bool]]:
    profile = profile or DEFAULT_OVERLAY_PROFILE
    cap = 1.0
    details: Dict[str, float | bool] = {}
    bench_hist = benchmark.loc[:date].dropna()
    if bench_hist.empty:
        return cap, details

    last_bench = float(bench_hist.iloc[-1])
    ma200 = float(bench_hist.rolling(200, min_periods=40).mean().iloc[-1]) if len(bench_hist) >= 40 else np.nan
    if profile.get("use_daily_trend", False) and not pd.isna(ma200):
        below_ma200 = last_bench < ma200
        details["below_ma200"] = below_ma200
        if below_ma200:
            cap = min(cap, float(profile.get("trend_cap", 0.65)))

    if profile.get("use_daily_drawdown", False):
        drawdown = last_bench / float(bench_hist.cummax().iloc[-1]) - 1.0
        details["benchmark_drawdown"] = drawdown
        if drawdown <= float(profile.get("drawdown_crash", -0.15)):
            cap = min(cap, float(profile.get("drawdown_crash_cap", 0.25)))
        elif drawdown <= float(profile.get("drawdown_warn", -0.08)):
            cap = min(cap, float(profile.get("drawdown_warn_cap", 0.50)))

    if profile.get("use_daily_vix", False) and vol_proxy is not None and not vol_proxy.empty:
        vol_hist = vol_proxy.loc[:date].dropna()
        if not vol_hist.empty:
            vix_now = float(vol_hist.iloc[-1])
            details["vix"] = vix_now
            if vix_now >= float(profile.get("vix_crash", 35.0)):
                cap = min(cap, float(profile.get("vix_crash_cap", 0.25)))
            elif vix_now >= float(profile.get("vix_warn", 28.0)):
                cap = min(cap, float(profile.get("vix_warn_cap", 0.50)))

    return float(np.clip(cap, 0.0, 1.0)), details


def lag_close_signal_to_next_session(signal: pd.Series, initial: float = 1.0) -> pd.Series:
    effective = signal.sort_index().shift(1)
    if not effective.empty:
        effective.iloc[0] = initial
    return effective.ffill().fillna(initial).astype(float)


def apply_daily_exposure_overlay(
    returns: pd.Series,
    benchmark: pd.Series,
    vol_proxy: Optional[pd.Series],
    profile: Optional[dict] = None,
) -> Tuple[pd.Series, pd.DataFrame]:
    profile = profile or DEFAULT_OVERLAY_PROFILE
    aligned = pd.concat(
        [
            returns.rename("asset_return"),
            benchmark.rename("benchmark"),
            vol_proxy.rename("vol_proxy") if vol_proxy is not None else pd.Series(dtype=float, name="vol_proxy"),
        ],
        axis=1,
    ).dropna(subset=["asset_return", "benchmark"])
    exposure_rows = []
    vol_series = aligned["vol_proxy"] if "vol_proxy" in aligned.columns else pd.Series(dtype=float)
    for dt, _row in aligned.iterrows():
        cap, details = daily_overlay_cap(dt, aligned["benchmark"], vol_series, profile)
        exposure_rows.append({"Date": dt, "Signal Exposure": cap, **details})
    exposure = pd.DataFrame(exposure_rows).set_index("Date").sort_index() if exposure_rows else pd.DataFrame()
    if exposure.empty:
        return pd.Series(dtype=float, name="Overlay Return"), exposure
    exposure["Daily Exposure"] = lag_close_signal_to_next_session(exposure["Signal Exposure"])
    overlay = (aligned["asset_return"] * exposure["Daily Exposure"]).sort_index().rename("Overlay Return")
    return overlay, exposure


def load_reference_overlay_curve(
    paths: ProjectPaths,
    strategy_name: str = "Fixed 70/20/10",
) -> pd.Series:
    curve_file = (
        paths.root.parent
        / "port_opt_advance"
        / "result"
        / "gold_btc_sp500_overlay"
        / "long_history_test"
        / "long_history_walk_forward_curves.csv"
    )
    if not curve_file.exists():
        raise FileNotFoundError(f"Reference overlay curve file not found: {curve_file}")
    curves = pd.read_csv(curve_file)
    if "Unnamed: 0" in curves.columns:
        curves = curves.rename(columns={"Unnamed: 0": "Date"})
    if "Date" not in curves.columns:
        raise KeyError("Reference overlay curve file is missing a Date column.")
    if strategy_name not in curves.columns:
        raise KeyError(f"Reference overlay curve file is missing strategy column: {strategy_name}")
    series = pd.Series(curves[strategy_name].values, index=pd.to_datetime(curves["Date"]), name=strategy_name)
    return series.sort_index().dropna()


def load_overlay_compare_prices(
    paths: ProjectPaths,
    start_date: str = "2016-01-01",
    end_date: Optional[str] = None,
    tickers: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    paths = paths or default_paths()
    _ensure_dirs(paths)
    _ensure_yfinance_cache(paths)
    end_date = end_date or pd.Timestamp.today().date().isoformat()
    tickers = list(tickers or ["SPY", "GLD", "GC=F", "BTC-USD", "^VIX", "USDTHB=X"])
    cache_file = paths.local_cache_root / "overlay_compare_prices.parquet"

    if cache_file.exists():
        try:
            cached = pd.read_parquet(cache_file)
        except Exception:
            table = pq.ParquetFile(cache_file).read()
            cached = table.to_pandas()
        if "Date" in cached.columns:
            cached = cached.set_index("Date")
        cached = cached.sort_index()
        cached.index = pd.to_datetime(cached.index)
        has_cols = all(ticker in cached.columns for ticker in tickers)
        end_ts = pd.Timestamp(end_date)
        cache_end_ok = not cached.empty and cached.index.max() >= (end_ts - pd.Timedelta(days=3))
        has_range = not cached.empty and cached.index.min() <= pd.Timestamp(start_date) and cache_end_ok
        if has_cols and has_range:
            return cached.loc[start_date:min(end_ts, cached.index.max()), tickers].copy()
    if yf is None:
        raise RuntimeError("yfinance is required to load overlay compare prices.")

    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=False,
    )
    if raw.empty:
        raise RuntimeError("Overlay compare price download returned no rows.")
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"]
        else:
            close = raw.xs("Close", axis=1, level=1)
    else:
        close = raw
    close = close.reindex(columns=tickers).sort_index().ffill()
    close.index = pd.to_datetime(close.index)
    close.to_parquet(cache_file)
    return close.loc[start_date:end_date, tickers].copy()


def compare_trend_exposure(price: pd.Series, below: float) -> pd.Series:
    ma = price.rolling(200, min_periods=40).mean()
    signal = pd.Series(1.0, index=price.index, dtype=float)
    signal.loc[price < ma] = below
    signal.loc[ma.isna()] = 1.0
    return lag_close_signal_to_next_session(signal)


def compare_sp_exposure(
    price: pd.Series,
    vix: pd.Series,
    trend_cap: float = 0.65,
    warn_cap: float = 0.50,
    crash_cap: float = 0.25,
) -> pd.Series:
    ma = price.rolling(200, min_periods=40).mean()
    drawdown = price / price.cummax() - 1.0
    candidates = pd.concat(
        [
            pd.Series(1.0, index=price.index, dtype=float),
            pd.Series(np.where(price < ma, trend_cap, 1.0), index=price.index, dtype=float),
            pd.Series(np.where(drawdown <= -0.08, warn_cap, 1.0), index=price.index, dtype=float),
            pd.Series(np.where(drawdown <= -0.15, crash_cap, 1.0), index=price.index, dtype=float),
            pd.Series(np.where(vix >= 28.0, warn_cap, 1.0), index=price.index, dtype=float),
            pd.Series(np.where(vix >= 35.0, crash_cap, 1.0), index=price.index, dtype=float),
        ],
        axis=1,
    )
    return lag_close_signal_to_next_session(candidates.min(axis=1))


def compare_cash_returns(mode: str, fx_returns: pd.Series, index: pd.Index) -> pd.Series:
    if mode == "USD":
        return fx_returns.reindex(index).fillna(0.0)
    if mode in {"THB", "USD_STATIC"}:
        return pd.Series(0.0, index=index, dtype=float)
    raise ValueError(mode)


def compare_apply_returns(asset_returns: pd.Series, exposure: pd.Series, cash_mode: str, fx_returns: pd.Series) -> pd.Series:
    cash_ret = compare_cash_returns(cash_mode, fx_returns, asset_returns.index)
    return asset_returns * exposure + cash_ret * (1.0 - exposure)


def convert_usd_returns_to_local(asset_returns: pd.Series, fx_returns: pd.Series) -> pd.Series:
    aligned_fx = fx_returns.reindex(asset_returns.index).fillna(0.0)
    local = (1.0 + asset_returns.fillna(0.0)) * (1.0 + aligned_fx) - 1.0
    return local.astype(float)


def compare_rebalanced_portfolio(
    sleeve_returns: pd.DataFrame,
    weights: Optional[pd.Series] = None,
    rebalance_months: int = 3,
) -> pd.Series:
    sleeve_returns = sleeve_returns.copy()
    sleeve_returns.index = pd.to_datetime(sleeve_returns.index)
    weights = (weights if weights is not None else pd.Series({"SP500_OVERLAY": 0.70, "GOLD": 0.20, "BTC": 0.10}, dtype=float)).copy()
    weights = weights.reindex(sleeve_returns.columns).fillna(0.0)
    if weights.sum() <= 0:
        raise ValueError("Portfolio weights must sum to a positive value.")
    weights = weights / weights.sum()
    values = weights * 10_000.0
    month_ends = sleeve_returns.groupby(sleeve_returns.index.to_period("M")).tail(1).index
    if rebalance_months <= 1:
        rebalance_dates = set(month_ends)
    else:
        rebalance_dates = set(month_ends[::rebalance_months])
    rows = []
    for dt, row in sleeve_returns.iterrows():
        total_before = float(values.sum())
        values = values * (1.0 + row.fillna(0.0))
        total_after = float(values.sum())
        rows.append((dt, total_after / total_before - 1.0 if total_before > 0 else 0.0))
        if dt in rebalance_dates and total_after > 0:
            values = weights * total_after
    return pd.Series(dict(rows), name="Portfolio").sort_index()


def build_overlay_comparison(
    results: Dict[str, object],
    paths: Optional[ProjectPaths] = None,
    gold_ticker: str = "GLD",
    btc_ticker: str = "BTC-USD",
    mix_weights: Tuple[float, float, float] = (0.70, 0.20, 0.10),
    profile: Optional[dict] = None,
    strategic_rebalance_months: int = 3,
    report_currency: str = "USD",
) -> Dict[str, object]:
    paths = paths or default_paths()
    profile = profile or DEFAULT_OVERLAY_PROFILE
    sample_index = results["nav"]["Static Copula"].index.sort_values()
    sample_index = sample_index[sample_index >= sample_index.min()]
    compare_prices = load_overlay_compare_prices(
        paths,
        start_date="2016-01-01",
        end_date=str(sample_index.max().date()),
        tickers=["SPY", "GC=F", "BTC-USD", "^VIX", "USDTHB=X"],
    ).dropna(subset=["SPY", "GC=F", "BTC-USD", "^VIX", "USDTHB=X"])
    compare_prices = compare_prices.sort_index()
    compare_fx = compare_prices["USDTHB=X"].dropna()
    compare_fx_returns = compare_fx.pct_change(fill_method=None).fillna(0.0)

    spy = compare_prices["SPY"]
    gold = compare_prices["GC=F"]
    btc = compare_prices["BTC-USD"]
    vix = compare_prices["^VIX"]
    spy_returns = spy.pct_change(fill_method=None).fillna(0.0)
    gold_returns = gold.pct_change(fill_method=None).fillna(0.0)
    btc_returns = btc.pct_change(fill_method=None).fillna(0.0)

    sp500_exposure_series = compare_sp_exposure(spy, vix)
    gold_exposure_series = compare_trend_exposure(gold, 0.50)
    btc_exposure_series = compare_trend_exposure(btc, 0.00)

    report_currency = report_currency.upper()
    if report_currency == "THB":
        spy_base_returns = convert_usd_returns_to_local(spy_returns, compare_fx_returns)
        gold_base_returns = convert_usd_returns_to_local(gold_returns, compare_fx_returns)
        btc_base_returns = convert_usd_returns_to_local(btc_returns, compare_fx_returns)
    elif report_currency == "USD":
        spy_base_returns = spy_returns
        gold_base_returns = gold_returns
        btc_base_returns = btc_returns
    else:
        raise ValueError(f"Unsupported report currency: {report_currency}")

    sp500_overlay_returns = compare_apply_returns(spy_base_returns, sp500_exposure_series, "USD_STATIC", compare_fx_returns).rename("SP500 Overlay Return")
    gold_overlay_returns = compare_apply_returns(gold_base_returns, gold_exposure_series, "USD_STATIC", compare_fx_returns).rename("Gold Overlay Return")
    btc_overlay_returns = compare_apply_returns(btc_base_returns, btc_exposure_series, "USD_STATIC", compare_fx_returns).rename("BTC Overlay Return")

    sp_w, gold_w, btc_w = mix_weights
    sleeve_returns = pd.concat(
        {
            "SP500_OVERLAY": sp500_overlay_returns,
            "GOLD": gold_overlay_returns,
            "BTC": btc_overlay_returns,
        },
        axis=1,
    ).dropna()
    mix_returns = compare_rebalanced_portfolio(
        sleeve_returns,
        weights=pd.Series({"SP500_OVERLAY": sp_w, "GOLD": gold_w, "BTC": btc_w}, dtype=float),
        rebalance_months=strategic_rebalance_months,
    ).rename("SP500_Gold_BTC_Overlay_Return")

    mix_curve_name = f"S&P/Gold/BTC {int(sp_w*100)}/{int(gold_w*100)}/{int(btc_w*100)} daily exposure"

    benchmark = spy.reindex(sample_index).ffill()
    vol_proxy = vix.reindex(sample_index).ffill()
    static_returns = results["nav"]["Static Copula"].reindex(sample_index).pct_change(fill_method=None).fillna(0.0)
    if report_currency == "THB":
        static_returns = convert_usd_returns_to_local(static_returns, compare_fx_returns.reindex(sample_index).fillna(0.0))
    static_overlay_returns, static_exposure = apply_daily_exposure_overlay(
        static_returns,
        benchmark,
        vol_proxy,
        profile=profile,
    )
    static_hmm_mix_returns = compare_rebalanced_portfolio(
        pd.concat(
            {
                "STATIC_HMM_OVERLAY": static_overlay_returns.reindex(sleeve_returns.index).fillna(0.0),
                "GOLD": gold_overlay_returns.reindex(sleeve_returns.index).fillna(0.0),
                "BTC": btc_overlay_returns.reindex(sleeve_returns.index).fillna(0.0),
            },
            axis=1,
        ).dropna(),
        weights=pd.Series({"STATIC_HMM_OVERLAY": sp_w, "GOLD": gold_w, "BTC": btc_w}, dtype=float),
        rebalance_months=strategic_rebalance_months,
    ).rename("STATIC_HMM_Gold_BTC_Overlay_Return")
    static_hmm_mix_curve_name = f"Static HMM/Gold/BTC {int(sp_w*100)}/{int(gold_w*100)}/{int(btc_w*100)} daily exposure"

    curves = {
        "S&P 500": curve_from_returns(spy_returns),
        "S&P 500 daily exposure": curve_from_returns(sp500_overlay_returns),
        mix_curve_name: curve_from_returns(mix_returns),
        "Static HMM daily exposure": curve_from_returns(static_overlay_returns.reindex(sample_index).fillna(0.0)),
        static_hmm_mix_curve_name: curve_from_returns(static_hmm_mix_returns),
    }
    common_start = max(series.dropna().index.min() for series in curves.values())
    common_end = min(series.dropna().index.max() for series in curves.values())
    trimmed_curves = {name: series.loc[common_start:common_end] for name, series in curves.items()}
    overlap_curves = pd.concat(trimmed_curves, axis=1).sort_index().ffill().dropna()
    rebased_curves = overlap_curves / overlap_curves.iloc[0] * 10_000.0

    summary = pd.DataFrame(
        {
            name: compute_port_opt_style_metrics(rebased_curves[name], risk_free_rate=0.03)
            for name, curve in curves.items()
            if name != "S&P 500" and name in rebased_curves
        }
    ).T
    exposure_compare = pd.concat(
        [
            sp500_exposure_series.rename("S&P 500 overlay exposure"),
            (sp500_exposure_series * sp_w).rename(f"S&P/Gold/BTC {int(sp_w*100)}/{int(gold_w*100)}/{int(btc_w*100)} SP sleeve exposure"),
            (gold_exposure_series * gold_w).rename(f"S&P/Gold/BTC {int(sp_w*100)}/{int(gold_w*100)}/{int(btc_w*100)} Gold sleeve exposure"),
            (btc_exposure_series * btc_w).rename(f"S&P/Gold/BTC {int(sp_w*100)}/{int(gold_w*100)}/{int(btc_w*100)} BTC sleeve exposure"),
            static_exposure["Daily Exposure"].rename("Static HMM overlay exposure"),
            (static_exposure["Daily Exposure"] * sp_w).rename(f"Static HMM/Gold/BTC {int(sp_w*100)}/{int(gold_w*100)}/{int(btc_w*100)} HMM sleeve exposure"),
            (gold_exposure_series * gold_w).rename(f"Static HMM/Gold/BTC {int(sp_w*100)}/{int(gold_w*100)}/{int(btc_w*100)} Gold sleeve exposure"),
            (btc_exposure_series * btc_w).rename(f"Static HMM/Gold/BTC {int(sp_w*100)}/{int(gold_w*100)}/{int(btc_w*100)} BTC sleeve exposure"),
        ],
        axis=1,
    ).reindex(rebased_curves.index).ffill().bfill()
    return {
        "curves": rebased_curves.to_dict(orient="series"),
        "summary": summary,
        "sp500_exposure": sp500_exposure_series.rename("Daily Exposure").to_frame(),
        "static_exposure": static_exposure,
        "exposure_compare": exposure_compare,
        "gold_returns": gold_returns,
        "btc_returns": btc_returns,
        "mix_returns": mix_returns,
        "sp500_overlay_returns": sp500_overlay_returns,
        "static_overlay_returns": static_overlay_returns,
        "static_hmm_mix_returns": static_hmm_mix_returns,
        "report_currency": report_currency,
    }


def monthly_rebalance_dates(index: pd.DatetimeIndex, lookback_days: int = 756, freq: str = "ME") -> List[pd.Timestamp]:
    month_ends = pd.Series(index, index=index).resample(freq).last().dropna().tolist()
    return [date for date in month_ends if date >= index[min(len(index) - 1, lookback_days)]]


def backtest_dynamic_factor_copula(
    start_date: str = "2012-01-01",
    end_date: str = "2026-04-30",
    n_assets: int = 16,
    n_clusters: int = 3,
    lookback_days: int = 756,
    rebalance_freq: str = "ME",
    max_weight: float = 0.18,
    tickers: Optional[Sequence[str]] = None,
    point_in_time_liquid: bool = True,
    universe_mode: str = "fixed",
    benchmark_ticker: Optional[str] = None,
    vol_proxy_ticker: Optional[str] = None,
    include_momentum: bool = True,
    include_momentum_features: Optional[bool] = None,
    include_momentum_signal: Optional[bool] = None,
    momentum_signal_mode: str = "mom_63",
    optimizer_objective: str = "mean_variance",
    feature_flags: Optional[Dict[str, bool]] = None,
    paths: Optional[ProjectPaths] = None,
) -> Dict[str, object]:
    paths = paths or default_paths()
    include_momentum_features = include_momentum if include_momentum_features is None else include_momentum_features
    include_momentum_signal = include_momentum if include_momentum_signal is None else include_momentum_signal
    requested_tickers = list(dict.fromkeys(tickers or DEFAULT_UNIVERSE))
    if universe_mode == "sp500_pit" and tickers is None:
        available_cache_cols = set(_parquet_column_names(str(paths.source_cache_root / "prices.parquet")))
        requested_tickers = [
            ticker
            for ticker in load_sp500_membership_intervals(paths)["ticker"].drop_duplicates().tolist()
            if ticker in available_cache_cols
        ]
    elif universe_mode == "set100_pit" and tickers is None:
        available_cache_cols = set(_parquet_column_names(str(paths.source_cache_root / "prices.parquet")))
        local_prices = paths.local_cache_root / "extra_prices.parquet"
        if local_prices.exists():
            available_cache_cols |= set(_parquet_column_names(str(local_prices)))
        requested_tickers = [
            ticker
            for ticker in load_set100_membership_intervals(paths)["ticker"].drop_duplicates().tolist()
            if ticker in available_cache_cols
        ]
    panel = prepare_panel(
        start_date=start_date,
        end_date=end_date,
        tickers=requested_tickers,
        n_assets=len(requested_tickers),
        preselect_universe=False,
        universe_mode=universe_mode,
        benchmark_ticker=benchmark_ticker,
        vol_proxy_ticker=vol_proxy_ticker,
        paths=paths,
    )
    prices = panel["prices"]
    returns = panel["returns"]
    benchmark = panel["benchmark"]
    benchmark_ret = panel["benchmark_ret"]
    vol_proxy_ret = panel["vol_proxy_ret"]
    assets = list(panel["tickers"])

    schedule = monthly_rebalance_dates(prices.index, lookback_days=lookback_days, freq=rebalance_freq)
    if len(schedule) < 3:
        raise ValueError("Not enough rebalance dates to run the backtest.")

    feature_history: Dict[pd.Timestamp, pd.DataFrame] = {}
    universe_history: Dict[pd.Timestamp, List[str]] = {}
    market_stress_history: Dict[pd.Timestamp, float] = {}
    for rebalance_date in schedule:
        loc = prices.index.get_loc(rebalance_date)
        train_index = prices.index[max(0, loc - lookback_days + 1) : loc + 1]
        pit_assets = assets
        if universe_mode == "sp500_pit":
            sp500_members = set(get_sp500_members_as_of(rebalance_date, paths))
            if tickers is None:
                pit_assets = [ticker for ticker in assets if ticker in sp500_members]
            else:
                pit_assets = [ticker for ticker in assets if ticker in sp500_members and ticker in requested_tickers]
        elif universe_mode == "set100_pit":
            set100_members = set(get_set100_members_as_of(rebalance_date, paths))
            if tickers is None:
                pit_assets = [ticker for ticker in assets if ticker in set100_members]
            else:
                pit_assets = [ticker for ticker in assets if ticker in set100_members and ticker in requested_tickers]
        if point_in_time_liquid:
            pit_assets = select_point_in_time_universe(
                prices.reindex(train_index),
                panel["volumes"].reindex(train_index),
                pit_assets,
                n_assets=n_assets,
            )
        else:
            pit_assets = select_universe(
                prices.reindex(train_index),
                panel["volumes"].reindex(train_index),
                pit_assets,
                n_assets=n_assets,
                start_date=str(train_index[0].date()),
            )
        if not pit_assets:
            continue
        universe_history[rebalance_date] = pit_assets
        train_returns = returns.reindex(train_index)[pit_assets].dropna(axis=1, thresh=max(int(0.85 * len(train_index)), 60))
        if train_returns.shape[1] < max(n_clusters + 2, 6):
            continue
        universe_history[rebalance_date] = train_returns.columns.tolist()
        market_stress_history[rebalance_date] = compute_market_stress_signal(
            benchmark_ret.reindex(train_index),
            vol_proxy_ret.reindex(train_index),
        )
        feature_table = compute_feature_table(
            train_returns,
            benchmark_ret.reindex(train_index),
            vol_proxy_ret.reindex(train_index),
            prices.reindex(train_index)[train_returns.columns],
            include_momentum_features=include_momentum_features,
            feature_flags=feature_flags,
        )
        if feature_table.empty:
            continue
        feature_history[rebalance_date] = feature_table

    first_date = min(feature_history)
    initial = initialize_static_clusters(feature_history[first_date], n_clusters=n_clusters)
    dynamic_state = run_dynamic_hmm(
        feature_history,
        initial_state=initial,
        gas_alpha=0.40,
        gas_beta=0.45,
        market_stress_history=market_stress_history,
        posterior_power=2.25,
    )

    strategy_names = ["Equal Weight", "Risk Parity", "Static Copula", "Dynamic HMM Copula"]
    nav = {name: pd.Series(1.0, index=[schedule[0]]) for name in strategy_names}
    weights_history: Dict[str, Dict[pd.Timestamp, pd.Series]] = {name: {} for name in strategy_names}
    cluster_snapshots: Dict[str, Dict[pd.Timestamp, pd.Series | pd.DataFrame]] = {
        "static_assignment": {},
        "dynamic_assignment": {},
        "dynamic_posterior": {},
    }

    static_post = pd.get_dummies(initial["labels"]).reindex(columns=range(n_clusters), fill_value=0.0)

    for idx, rebalance_date in enumerate(schedule[:-1]):
        next_date = schedule[idx + 1]
        if rebalance_date not in feature_history:
            continue
        loc = prices.index.get_loc(rebalance_date)
        train_index = prices.index[max(0, loc - lookback_days + 1) : loc + 1]
        test_index = prices.index[(prices.index > rebalance_date) & (prices.index <= next_date)]
        if len(test_index) == 0:
            continue

        current_features = feature_history[rebalance_date]
        current_assets = current_features.index.tolist()
        train_returns = returns.loc[train_index, current_assets].dropna(how="all")
        bench_train = benchmark_ret.loc[train_index]

        cluster_snapshots["static_assignment"][rebalance_date] = initial["labels"].reindex(current_assets)
        cluster_snapshots["dynamic_assignment"][rebalance_date] = dynamic_state["assignment_history"][rebalance_date].reindex(current_assets)
        cluster_snapshots["dynamic_posterior"][rebalance_date] = dynamic_state["posterior_history"][rebalance_date].reindex(current_assets)

        if include_momentum_signal:
            momentum_signal = build_momentum_signal(current_features, mode=momentum_signal_mode)
        else:
            momentum_signal = pd.Series(0.0, index=current_features.index, dtype=float)
        eq_weights = pd.Series(1.0 / len(current_assets), index=current_assets)
        risk_parity_cov = train_returns.cov().reindex(index=current_assets, columns=current_assets).fillna(0.0)
        static_cov, _ = build_factor_covariance(
            train_returns,
            bench_train,
            static_post.reindex(current_assets).fillna(0.0),
            current_features,
            dynamic=False,
        )
        dyn_cov, _ = build_factor_covariance(
            train_returns,
            bench_train,
            dynamic_state["posterior_history"][rebalance_date].reindex(current_assets).fillna(0.0),
            current_features,
            dynamic=True,
            centroid_snapshot=dynamic_state["centroid_history"][rebalance_date],
        )

        risk_parity_weights = optimize_risk_parity(risk_parity_cov, max_weight=max_weight)
        static_weights = optimize_portfolio(
            static_cov,
            momentum_signal,
            max_weight=max_weight,
            objective_mode=optimizer_objective,
        )
        dynamic_weights = optimize_portfolio(
            dyn_cov,
            momentum_signal,
            max_weight=max_weight,
            objective_mode=optimizer_objective,
        )
        weights_history["Equal Weight"][rebalance_date] = eq_weights
        weights_history["Risk Parity"][rebalance_date] = risk_parity_weights
        weights_history["Static Copula"][rebalance_date] = static_weights
        weights_history["Dynamic HMM Copula"][rebalance_date] = dynamic_weights

        period_returns = returns.reindex(test_index)[current_assets].fillna(0.0)
        period_benchmark = benchmark.reindex(test_index).ffill()

        for strategy, weights in [
            ("Equal Weight", eq_weights),
            ("Risk Parity", risk_parity_weights),
            ("Static Copula", static_weights),
            ("Dynamic HMM Copula", dynamic_weights),
        ]:
            weighted = period_returns.mul(weights, axis=1).sum(axis=1)
            starting_value = float(nav[strategy].iloc[-1])
            period_nav = starting_value * (1.0 + weighted).cumprod()
            nav[strategy] = pd.concat([nav[strategy], period_nav])
        benchmark_start = 1.0 if "Benchmark" not in nav else float(nav["Benchmark"].iloc[-1])
        benchmark_nav = benchmark_start * period_benchmark.div(period_benchmark.iloc[0]).ffill()
        if "Benchmark" in nav:
            nav["Benchmark"] = pd.concat([nav["Benchmark"], benchmark_nav])
        else:
            nav["Benchmark"] = benchmark_nav

    nav = {name: series[~series.index.duplicated(keep="last")].sort_index() for name, series in nav.items()}
    benchmark_nav = nav["Benchmark"]
    metrics = pd.DataFrame(
        {
            name: compute_metrics(series, benchmark_nav=benchmark_nav)
            for name, series in nav.items()
            if name != "Benchmark"
        }
    ).T

    turnover_rows = {}
    for strategy, history in weights_history.items():
        ordered_dates = sorted(history)
        if len(ordered_dates) < 2:
            turnover_rows[strategy] = np.nan
            continue
        turns = []
        for prev_date, curr_date in zip(ordered_dates[:-1], ordered_dates[1:]):
            prev_w = history[prev_date]
            curr_w = history[curr_date].reindex(prev_w.index.union(history[curr_date].index), fill_value=0.0)
            prev_w = prev_w.reindex(curr_w.index, fill_value=0.0)
            turns.append(0.5 * np.abs(curr_w - prev_w).sum())
        turnover_rows[strategy] = float(np.mean(turns))
    metrics["Turnover"] = pd.Series(turnover_rows)

    return {
        "panel": panel,
        "schedule": schedule,
        "feature_history": feature_history,
        "universe_history": universe_history,
        "market_stress_history": market_stress_history,
        "initial_clusters": initial,
        "dynamic_state": dynamic_state,
        "nav": nav,
        "metrics": metrics,
        "weights_history": weights_history,
        "cluster_snapshots": cluster_snapshots,
        "benchmark_ticker": benchmark_ticker or "SPY",
        "vol_proxy_ticker": vol_proxy_ticker if vol_proxy_ticker is not None else "^VIX",
        "include_momentum": include_momentum,
        "include_momentum_features": include_momentum_features,
        "include_momentum_signal": include_momentum_signal,
        "momentum_signal_mode": momentum_signal_mode,
        "optimizer_objective": optimizer_objective,
        "feature_flags": feature_flags or {},
    }
