from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dynamic_factor_copula import default_paths, optimize_portfolio  # noqa: E402
from run_us_th_jp_allocation_models import (  # noqa: E402
    ALLOCATION_PROFILES,
    _asset_exposure_variants,
    _evaluate_index_signal_models,
    _load_japan_prices,
    _load_japan_signal_price,
    _load_jpy_thb_fx,
    _load_overlay_assets,
    _read_us_th_returns,
)


PREFIX = "us_th_jp_optimized_sleeve_sweep"
INITIAL_TRAIN_DAYS = 40
LOOKBACK_DAYS = 120
JP_OPTIMIZER_GRID = [
    {"jp_assets": 10, "jp_cap": 0.15, "objective": "mean_variance"},
    {"jp_assets": 10, "jp_cap": 0.20, "objective": "mean_variance"},
    {"jp_assets": 15, "jp_cap": 0.10, "objective": "mean_variance"},
    {"jp_assets": 15, "jp_cap": 0.12, "objective": "mean_variance"},
    {"jp_assets": 15, "jp_cap": 0.15, "objective": "mean_variance"},
    {"jp_assets": 20, "jp_cap": 0.08, "objective": "mean_variance"},
    {"jp_assets": 20, "jp_cap": 0.10, "objective": "mean_variance"},
    {"jp_assets": 20, "jp_cap": 0.12, "objective": "mean_variance"},
    {"jp_assets": 10, "jp_cap": 0.15, "objective": "min_vol_mom_tilt"},
    {"jp_assets": 10, "jp_cap": 0.20, "objective": "min_vol_mom_tilt"},
    {"jp_assets": 15, "jp_cap": 0.12, "objective": "min_vol_mom_tilt"},
    {"jp_assets": 20, "jp_cap": 0.10, "objective": "min_vol_mom_tilt"},
]
FOCUS_EXPOSURES = [
    "daily exposure all assets + gold drawdown 252d warn10 crash20",
    "weekly exposure all assets + gold drawdown 252d warn10 crash20",
]
FOCUS_PROFILE = "Stock60/Gold30/BTC10"


def _momentum_signal(train_returns: pd.DataFrame) -> pd.Series:
    if train_returns.empty:
        return pd.Series(dtype=float)
    lookback = train_returns.tail(min(63, len(train_returns)))
    signal = (1.0 + lookback).prod() - 1.0
    return signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _fallback_equal(assets: list[str]) -> pd.Series:
    if not assets:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(assets), index=assets, dtype=float)


def _optimized_weights(
    train_returns: pd.DataFrame,
    assets: list[str],
    jp_cap: float,
    objective: str,
) -> pd.Series:
    usable = train_returns.reindex(columns=assets).dropna(axis=1, thresh=max(INITIAL_TRAIN_DAYS, int(len(train_returns) * 0.75)))
    usable = usable.dropna(how="all")
    if usable.shape[1] < 2 or usable.shape[0] < INITIAL_TRAIN_DAYS or usable.shape[1] * jp_cap < 1.0 - 1e-9:
        return _fallback_equal(assets)
    cov = usable.cov().fillna(0.0)
    cov = 0.80 * cov + 0.20 * pd.DataFrame(np.diag(np.diag(cov)), index=cov.index, columns=cov.columns)
    momentum = _momentum_signal(usable)
    try:
        weights = optimize_portfolio(
            cov,
            momentum,
            max_weight=jp_cap,
            risk_aversion=8.0,
            objective_mode=objective,
            concentration_penalty=0.01,
            momentum_strength=1.0,
        )
    except Exception:
        return _fallback_equal(assets)
    return weights.reindex(assets).fillna(0.0)


def _build_japan_optimized_sleeve(
    paths,
    index: pd.DatetimeIndex,
    jp_assets: int,
    jp_cap: float,
    objective: str,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.Timestamp]:
    universe = pd.read_parquet(paths.local_cache_root / "japan_pit_universe_history.parquet")
    universe["signal_date"] = pd.to_datetime(universe["signal_date"], errors="coerce")
    universe["entry_date"] = pd.to_datetime(universe["entry_date"], errors="coerce")
    universe["Code"] = universe["Code"].astype(str).str.strip()
    prices = _load_japan_prices(paths)
    last_japan_price_date = pd.Timestamp(prices.dropna(how="all").index.max())
    prices = prices.reindex(index.union(prices.index).sort_values()).ffill()
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    all_codes = sorted(universe["Code"].dropna().unique())
    weights = pd.DataFrame(0.0, index=prices.index, columns=all_codes)
    entry_groups = [
        (pd.Timestamp(entry_date), month_rows.copy())
        for entry_date, month_rows in universe.dropna(subset=["entry_date"]).groupby("entry_date")
    ]
    entry_groups.sort(key=lambda item: item[0])
    for idx, (entry, month_rows) in enumerate(entry_groups):
        next_entry = entry_groups[idx + 1][0] if idx + 1 < len(entry_groups) else None
        assets = [ticker for ticker in month_rows.sort_values("rank")["Code"].tolist() if ticker in weights.columns]
        assets = assets[:jp_assets]
        if not assets:
            continue
        train_end_pos = returns.index.searchsorted(entry, side="left")
        train_start_pos = max(0, train_end_pos - LOOKBACK_DAYS)
        train_returns = returns.iloc[train_start_pos:train_end_pos].reindex(columns=assets)
        month_weights = _optimized_weights(train_returns, assets, jp_cap=jp_cap, objective=objective)
        period_mask = weights.index >= entry
        if next_entry is not None:
            period_mask &= weights.index < next_entry
        if period_mask.any():
            weights.loc[period_mask, month_weights.index] = month_weights.to_numpy()

    sleeve_returns_jpy = returns.reindex(columns=weights.columns).mul(weights.shift(1).fillna(0.0), axis=0).sum(axis=1)
    sleeve_returns_jpy = sleeve_returns_jpy.reindex(index).fillna(0.0)
    fx = _load_jpy_thb_fx(paths, index)
    sleeve_returns_thb = ((1.0 + sleeve_returns_jpy) * (1.0 + fx.pct_change(fill_method=None).fillna(0.0)) - 1.0).rename(
        f"JP optimized sleeve THB {objective} top{jp_assets} cap{jp_cap:.0%}"
    )
    signal_proxy = prices.reindex(columns=weights.columns).mul(weights.replace(0.0, np.nan), axis=0).sum(axis=1)
    signal_proxy = signal_proxy.replace(0.0, np.nan).ffill().rename("JP optimized signal proxy")
    return sleeve_returns_thb, signal_proxy, weights.reindex(index).fillna(0.0), last_japan_price_date


def _asset_returns_for_case(paths, us_th_returns: pd.DataFrame, case: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    jp_returns, jp_index, jp_weights, jp_price_end = _build_japan_optimized_sleeve(
        paths,
        us_th_returns.index,
        jp_assets=int(case["jp_assets"]),
        jp_cap=float(case["jp_cap"]),
        objective=str(case["objective"]),
    )
    common_index = us_th_returns.index.intersection(jp_returns.dropna().index).sort_values()
    active_mask = jp_weights.sum(axis=1).gt(0.0)
    common_index = common_index[(common_index >= active_mask.idxmax()) & (common_index <= jp_price_end)]
    us_th = us_th_returns.reindex(common_index).fillna(0.0)
    jp = jp_returns.reindex(common_index).fillna(0.0)
    overlay_prices, signal_prices = _load_overlay_assets(paths, common_index)
    signal_prices["JP Equity"] = _load_japan_signal_price(paths, common_index, jp_index)
    asset_returns = pd.DataFrame(
        {
            "US Equity": us_th["US PIT optimized sleeve THB"],
            "TH Equity": us_th["TH PIT optimized sleeve THB"],
            "JP Equity": jp,
            "Gold": overlay_prices["Gold"].pct_change(fill_method=None).fillna(0.0),
            "BTC": overlay_prices["BTC"].pct_change(fill_method=None).fillna(0.0),
        },
        index=common_index,
    ).fillna(0.0)
    return asset_returns, signal_prices, jp_weights.reindex(common_index).fillna(0.0)


def main() -> None:
    paths = default_paths(ROOT)
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    us_th_returns = _read_us_th_returns(paths)
    profile = next(profile for profile in ALLOCATION_PROFILES if profile["profile"] == FOCUS_PROFILE)

    rows: list[dict[str, object]] = []
    curves: dict[str, pd.Series] = {}
    weight_frames: list[pd.DataFrame] = []
    latest_rows: list[dict[str, object]] = []
    jp_internal_frames: list[pd.DataFrame] = []

    for case in JP_OPTIMIZER_GRID:
        asset_returns, signal_prices, jp_weights = _asset_returns_for_case(paths, us_th_returns, case)
        exposure_variants = {
            name: frame.reindex(asset_returns.index).ffill().fillna(1.0).clip(0.0, 1.0)
            for name, frame in _asset_exposure_variants(signal_prices).items()
            if name in FOCUS_EXPOSURES
        }
        new_rows, new_curves, new_weights = _evaluate_index_signal_models(
            asset_returns,
            signal_prices,
            profile,
            exposure_variants,
        )
        case_label = f"JP optimized {case['objective']} top{case['jp_assets']} cap{float(case['jp_cap']):.0%}"
        latest_internal_max = float(jp_weights.iloc[-1].max()) if not jp_weights.empty else 0.0
        avg_internal_max = float(jp_weights.max(axis=1).mean()) if not jp_weights.empty else 0.0
        for row in new_rows:
            row["JP Sleeve Mode"] = case_label
            row["JP Assets"] = int(case["jp_assets"])
            row["JP Internal Cap"] = float(case["jp_cap"])
            row["JP Objective"] = str(case["objective"])
            row["Latest JP Internal Max Weight"] = latest_internal_max
            row["Average JP Internal Max Weight"] = avg_internal_max
        rows.extend(new_rows)
        for name, curve in new_curves.items():
            curves[f"{case_label} {name}"] = curve
        for frame in new_weights:
            frame = frame.copy()
            frame["JP Sleeve Mode"] = case_label
            frame["JP Assets"] = int(case["jp_assets"])
            frame["JP Internal Cap"] = float(case["jp_cap"])
            weight_frames.append(frame)
            latest = frame.sort_values("Date").tail(1)
            if not latest.empty:
                latest_row = latest.iloc[0]
                latest_rows.append(
                    {
                        "Strategy": latest_row["Strategy"],
                        "JP Sleeve Mode": case_label,
                        "Date": latest_row["Date"],
                        "Latest US Weight": latest_row.get("US Equity", 0.0),
                        "Latest TH Weight": latest_row.get("TH Equity", 0.0),
                        "Latest JP Weight": latest_row.get("JP Equity", 0.0),
                        "Latest Gold Weight": latest_row.get("Gold", 0.0),
                        "Latest BTC Weight": latest_row.get("BTC", 0.0),
                        "Latest Cash Weight": latest_row.get("Cash / Reduced Exposure", 0.0),
                        "Latest JP Internal Max Weight": latest_internal_max,
                        "Latest JP Stock Max Effective Weight": latest_row.get("JP Equity", 0.0) * latest_internal_max,
                    }
                )
        jp_internal = jp_weights.copy()
        jp_internal["JP Sleeve Mode"] = case_label
        jp_internal_frames.append(jp_internal.reset_index(names="Date"))

    summary = pd.DataFrame(rows)
    focus = summary.loc[
        summary["Strategy"].str.contains("Index signal leaves inactive equity in cash", regex=False)
        & ~summary["Strategy"].str.contains("no JP", regex=False)
        & summary["Strategy"].str.contains("gold drawdown 252d", regex=False)
    ].sort_values(["Sharpe", "CAGR"], ascending=False)

    summary.sort_values(["Sharpe", "CAGR"], ascending=False).to_csv(paths.result_dir / f"{PREFIX}_summary_thb.csv", index=False)
    focus.to_csv(paths.result_dir / f"{PREFIX}_focus_summary_thb.csv", index=False)
    pd.DataFrame(curves).dropna(how="all").to_csv(paths.result_dir / f"{PREFIX}_curves_thb.csv")
    pd.concat(weight_frames, ignore_index=True).to_csv(paths.result_dir / f"{PREFIX}_weight_history_thb.csv", index=False)
    pd.DataFrame(latest_rows).to_csv(paths.result_dir / f"{PREFIX}_latest_weights_thb.csv", index=False)
    pd.concat(jp_internal_frames, ignore_index=True).to_csv(paths.result_dir / f"{PREFIX}_jp_internal_weight_history.csv", index=False)

    print(focus.head(30).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
