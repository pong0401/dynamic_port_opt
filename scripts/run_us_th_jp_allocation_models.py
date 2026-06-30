from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dynamic_factor_copula import (  # noqa: E402
    compute_port_opt_style_metrics,
    curve_from_returns,
    default_paths,
    load_overlay_compare_prices,
)
from run_us_th_tactical_gold_crash_protection_sweep import _gold_crash_exposure  # noqa: E402
from run_us_th_tactical_perf_momentum import _close_trend_exposure  # noqa: E402


PREFIX = "us_th_jp_allocation_models"
US_TH_PREFIX = "us_th_tactical_perf_momentum"
INITIAL_VALUE = 10_000.0
RISK_FREE_RATE = 0.03
BTC_WEIGHT = 0.10
ALLOCATION_PROFILES = [
    {"profile": "Stock60/Gold30/BTC10", "equity_budget": 0.60, "gold_weight": 0.30, "btc_weight": BTC_WEIGHT},
    {"profile": "Stock65/Gold25/BTC10", "equity_budget": 0.65, "gold_weight": 0.25, "btc_weight": BTC_WEIGHT},
    {"profile": "Stock70/Gold20/BTC10", "equity_budget": 0.70, "gold_weight": 0.20, "btc_weight": BTC_WEIGHT},
]
JP_INDEX_PROXY = "13060"  # TOPIX ETF-style proxy in J-Quants code format when available.
JP_SIGNAL_TICKER = "^N225"
MIN_JP_HISTORY_DAYS = 40


def _read_us_th_returns(paths) -> pd.DataFrame:
    curves = pd.read_csv(
        paths.result_dir / f"{US_TH_PREFIX}_comparison_curves_thb.csv",
        index_col=0,
        parse_dates=True,
    )
    return curves[["US PIT optimized sleeve THB", "TH PIT optimized sleeve THB"]].pct_change(fill_method=None).fillna(0.0)


def _load_japan_prices(paths) -> pd.DataFrame:
    universe_file = paths.local_cache_root / "japan_pit_universe_history.parquet"
    bars_file = paths.local_cache_root / "japan_daily_bars.parquet"
    if not universe_file.exists():
        raise FileNotFoundError(f"Missing Japan PIT universe file: {universe_file}")
    if not bars_file.exists():
        raise FileNotFoundError(f"Missing Japan daily bars cache: {bars_file}")
    universe = pd.read_parquet(universe_file)
    tickers = sorted(universe["Code"].dropna().astype(str).str.strip().unique().tolist())
    wanted = set(tickers)
    wanted.add(JP_INDEX_PROXY)

    available_cols = set(pq.ParquetFile(bars_file).schema.names)
    close_col = next((col for col in ["AdjC", "Close", "C"] if col in available_cols), None)
    volume_col = next((col for col in ["AdjVo", "Volume", "Vo"] if col in available_cols), None)
    if close_col is None:
        raise ValueError(f"No close column found in {bars_file}")
    read_cols = ["Date", "Code", close_col]
    if volume_col is not None:
        read_cols.append(volume_col)
    bars = pd.read_parquet(bars_file, columns=read_cols)
    bars["Date"] = pd.to_datetime(bars["Date"], errors="coerce")
    bars["Code"] = bars["Code"].astype(str).str.strip()
    bars = bars.loc[bars["Code"].isin(wanted)].dropna(subset=["Date", "Code"])
    prices = bars.pivot_table(index="Date", columns="Code", values=close_col, aggfunc="last").sort_index().ffill()
    return prices


def _build_japan_pit_sleeve(
    paths,
    index: pd.DatetimeIndex,
    selected_count: int | None = None,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.Timestamp]:
    universe = pd.read_parquet(paths.local_cache_root / "japan_pit_universe_history.parquet")
    universe["signal_date"] = pd.to_datetime(universe["signal_date"], errors="coerce")
    universe["entry_date"] = pd.to_datetime(universe["entry_date"], errors="coerce")
    universe["Code"] = universe["Code"].astype(str).str.strip()
    prices = _load_japan_prices(paths)
    last_japan_price_date = prices.dropna(how="all").index.max()
    prices = prices.reindex(index.union(prices.index).sort_values()).ffill()
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    weights = pd.DataFrame(0.0, index=prices.index, columns=sorted(universe["Code"].dropna().unique()))
    entry_groups = [
        (pd.Timestamp(entry_date), month_rows.copy())
        for entry_date, month_rows in universe.dropna(subset=["entry_date"]).groupby("entry_date")
    ]
    entry_groups.sort(key=lambda item: item[0])
    for idx, (entry, month_rows) in enumerate(entry_groups):
        next_entry = entry_groups[idx + 1][0] if idx + 1 < len(entry_groups) else None
        eligible = [ticker for ticker in month_rows.sort_values("rank")["Code"].tolist() if ticker in weights.columns]
        if selected_count is not None:
            eligible = eligible[:selected_count]
        if not eligible:
            continue
        period_mask = weights.index >= entry
        if next_entry is not None:
            period_mask &= weights.index < next_entry
        if not period_mask.any():
            continue
        weights.loc[period_mask, eligible] = 1.0 / len(eligible)
    # Use yesterday's selected universe weights for today's return.
    sleeve_returns_jpy = returns.reindex(columns=weights.columns).mul(weights.shift(1).fillna(0.0), axis=0).sum(axis=1)
    sleeve_returns_jpy = sleeve_returns_jpy.reindex(index).fillna(0.0).rename("JP PIT equal selected sleeve JPY")

    fx = _load_jpy_thb_fx(paths, index)
    sleeve_returns_thb = ((1.0 + sleeve_returns_jpy) * (1.0 + fx.pct_change(fill_method=None).fillna(0.0)) - 1.0).rename(
        "JP PIT equal selected sleeve THB"
    )
    index_proxy = _load_japan_index_proxy(prices, universe).reindex(index.union(prices.index).sort_values()).ffill().reindex(index)
    return sleeve_returns_thb, index_proxy, weights.reindex(index).fillna(0.0), pd.Timestamp(last_japan_price_date)


def _load_jpy_thb_fx(paths, index: pd.DatetimeIndex) -> pd.Series:
    try:
        fx = load_overlay_compare_prices(paths, start_date=str(index.min().date()), end_date=str(index.max().date()), tickers=["JPYTHB=X"])
        if "JPYTHB=X" in fx.columns and not fx["JPYTHB=X"].dropna().empty:
            return fx["JPYTHB=X"].reindex(index).ffill().bfill().rename("JPYTHB=X")
    except Exception as exc:
        print(f"Warning: could not load JPYTHB=X ({exc}); Japan sleeve will use JPY returns without FX conversion.")
    return pd.Series(1.0, index=index, name="JPYTHB_FALLBACK")


def _load_japan_index_proxy(prices: pd.DataFrame, universe: pd.DataFrame) -> pd.Series:
    if JP_INDEX_PROXY in prices.columns and prices[JP_INDEX_PROXY].dropna().shape[0] >= MIN_JP_HISTORY_DAYS:
        return prices[JP_INDEX_PROXY].rename("JP index proxy")
    selected = sorted(universe["Code"].dropna().astype(str).str.strip().unique().tolist())
    proxy = prices.reindex(columns=[ticker for ticker in selected if ticker in prices.columns]).mean(axis=1)
    return proxy.rename("JP PIT equal-weight index proxy")


def _load_japan_signal_price(paths, index: pd.DatetimeIndex, fallback_proxy: pd.Series) -> pd.Series:
    try:
        nikkei = load_overlay_compare_prices(
            paths,
            start_date=str(index.min().date()),
            end_date=str(index.max().date()),
            tickers=[JP_SIGNAL_TICKER],
        ).sort_index().ffill()
        if JP_SIGNAL_TICKER in nikkei.columns and nikkei[JP_SIGNAL_TICKER].dropna().shape[0] >= MIN_JP_HISTORY_DAYS:
            return nikkei[JP_SIGNAL_TICKER].reindex(index).ffill().rename("Nikkei 225 signal")
    except Exception as exc:
        print(f"Warning: could not load {JP_SIGNAL_TICKER} for Japan signal ({exc}); using Japan PIT proxy signal.")
    return fallback_proxy.reindex(index).ffill().rename("Japan PIT proxy signal")


def _load_overlay_assets(paths, index: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    overlay = load_overlay_compare_prices(
        paths,
        start_date=str(index.min().date()),
        end_date=str(index.max().date()),
        tickers=["SPY", "GC=F", "BTC-USD", "USDTHB=X"],
    ).sort_index().ffill()
    set_index = pd.read_parquet(paths.local_cache_root / "extra_prices.parquet", columns=["^SET.BK"])["^SET.BK"].sort_index().ffill()
    fx = overlay["USDTHB=X"].reindex(index).ffill().bfill()
    prices_thb = pd.DataFrame(
        {
            "Gold": overlay["GC=F"].reindex(index).ffill().mul(fx),
            "BTC": overlay["BTC-USD"].reindex(index).ffill().mul(fx),
        },
        index=index,
    )
    signal_prices = pd.DataFrame(
        {
            "US Equity": overlay["SPY"].reindex(index).ffill(),
            "TH Equity": set_index.reindex(index).ffill(),
            "Gold": overlay["GC=F"].reindex(index).ffill(),
            "BTC": overlay["BTC-USD"].reindex(index).ffill(),
        },
        index=index,
    )
    return prices_thb, signal_prices


def _metrics_row(
    curve: pd.Series,
    strategy: str,
    weights: pd.DataFrame,
    model_type: str,
    profile: dict[str, object],
) -> dict[str, object]:
    clean = curve.dropna()
    row = compute_port_opt_style_metrics(clean, risk_free_rate=RISK_FREE_RATE).to_dict()
    row.update(
        {
            "Strategy": strategy,
            "Model Type": model_type,
            "Allocation Profile": profile["profile"],
            "Equity Budget": profile["equity_budget"],
            "Gold Budget": profile["gold_weight"],
            "BTC Budget": profile["btc_weight"],
            "Start": clean.index.min().date().isoformat() if not clean.empty else "",
            "End": clean.index.max().date().isoformat() if not clean.empty else "",
            "Average US Weight": weights.get("US Equity", pd.Series(0.0, index=weights.index)).mean(),
            "Average TH Weight": weights.get("TH Equity", pd.Series(0.0, index=weights.index)).mean(),
            "Average JP Weight": weights.get("JP Equity", pd.Series(0.0, index=weights.index)).mean(),
            "Average Gold Weight": weights.get("Gold", pd.Series(0.0, index=weights.index)).mean(),
            "Average BTC Weight": weights.get("BTC", pd.Series(0.0, index=weights.index)).mean(),
            "Average Cash Weight": weights.get("Cash / Reduced Exposure", pd.Series(0.0, index=weights.index)).mean(),
        }
    )
    return row


def _evaluate_fixed_weight_grid(
    asset_returns: pd.DataFrame,
    profile: dict[str, object],
    exposure_variants: dict[str, pd.DataFrame],
) -> tuple[list[dict[str, object]], dict[str, pd.Series], list[pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    curves: dict[str, pd.Series] = {}
    weight_frames: list[pd.DataFrame] = []
    index = asset_returns.index
    equity_budget = float(profile["equity_budget"])
    gold_weight = float(profile["gold_weight"])
    btc_weight = float(profile["btc_weight"])
    for th_weight in [0.00, 0.05, 0.10, 0.15, 0.20]:
        for jp_weight in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]:
            us_weight = equity_budget - th_weight - jp_weight
            if us_weight < -1e-9:
                continue
            strategy = (
                f"{profile['profile']} fixed US/TH/JP/Gold/BTC "
                f"{us_weight:.0%}/{th_weight:.0%}/{jp_weight:.0%}/{gold_weight:.0%}/{btc_weight:.0%}"
            )
            weights = pd.DataFrame(
                {
                    "US Equity": us_weight,
                    "TH Equity": th_weight,
                    "JP Equity": jp_weight,
                    "Gold": gold_weight,
                    "BTC": btc_weight,
                    "Cash / Reduced Exposure": 0.0,
                },
                index=index,
            )
            _append_result(rows, curves, weight_frames, asset_returns, weights, strategy, "Fixed weight grid", profile)
            for exposure_name, exposure in exposure_variants.items():
                exposed_strategy = f"{strategy} + {exposure_name}"
                exposed_weights = _apply_daily_exposure(weights, exposure)
                _append_result(
                    rows,
                    curves,
                    weight_frames,
                    asset_returns,
                    exposed_weights,
                    exposed_strategy,
                    "Fixed weight grid + exposure",
                    profile,
                )
    return rows, curves, weight_frames


def _signal_scores(signal_prices: pd.DataFrame) -> pd.DataFrame:
    scores = pd.DataFrame(index=signal_prices.index)
    rules = {
        "US Equity": 300,
        "TH Equity": 200,
        "JP Equity": 120,
    }
    for asset, lookback in rules.items():
        price = signal_prices[asset].ffill()
        trend = (price > price.rolling(lookback, min_periods=max(40, lookback // 3)).mean()).astype(float)
        momentum = (price.pct_change(63, fill_method=None) > 0.0).astype(float)
        scores[asset] = ((trend + momentum) / 2.0).shift(1).fillna(0.0)
    return scores


def _asset_daily_exposure(signal_prices: pd.DataFrame) -> pd.DataFrame:
    exposure = pd.DataFrame(
        {
            "US Equity": _close_trend_exposure(signal_prices["US Equity"], 300, 0.50),
            "TH Equity": _close_trend_exposure(signal_prices["TH Equity"], 200, 0.00),
            "JP Equity": _close_trend_exposure(signal_prices["JP Equity"], 120, 0.00),
            "Gold": _close_trend_exposure(signal_prices["Gold"], 80, 0.50),
            "BTC": _close_trend_exposure(signal_prices["BTC"], 50, 0.00),
        },
        index=signal_prices.index,
    )
    return exposure.ffill().fillna(1.0).clip(0.0, 1.0)


def _to_weekly_exposure(exposure: pd.DataFrame) -> pd.DataFrame:
    weekly = exposure.resample("W-FRI").last()
    return weekly.reindex(exposure.index).ffill().fillna(1.0).clip(0.0, 1.0)


def _asset_exposure_variants(signal_prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    daily = _asset_daily_exposure(signal_prices)
    variants: dict[str, pd.DataFrame] = {
        "daily exposure all assets": daily,
        "weekly exposure all assets": _to_weekly_exposure(daily),
    }
    gold_configs = [
        {
            "label": "gold drawdown 126d warn8 crash15",
            "dd_window": 126,
            "warn_dd": -0.08,
            "crash_dd": -0.15,
            "recovery_dd": -0.05,
        },
        {
            "label": "gold drawdown 252d warn10 crash20",
            "dd_window": 252,
            "warn_dd": -0.10,
            "crash_dd": -0.20,
            "recovery_dd": -0.05,
        },
    ]
    for config in gold_configs:
        gold_dd = _gold_crash_exposure(
            signal_prices["Gold"],
            dd_window=int(config["dd_window"]),
            warn_dd=float(config["warn_dd"]),
            crash_dd=float(config["crash_dd"]),
            warn_exposure=0.75,
            crash_exposure=0.50,
            recovery_dd=float(config["recovery_dd"]),
            panic_dd=-0.30,
            panic_ma_period=200,
            panic_mom_period=63,
        )
        daily_dd = daily.copy()
        daily_dd["Gold"] = gold_dd.reindex(daily.index).ffill().fillna(1.0).clip(0.0, 1.0)
        variants[f"daily exposure all assets + {config['label']}"] = daily_dd
        variants[f"weekly exposure all assets + {config['label']}"] = _to_weekly_exposure(daily_dd)
    return variants


def _apply_daily_exposure(weights: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    asset_cols = ["US Equity", "TH Equity", "JP Equity", "Gold", "BTC"]
    effective = weights.copy()
    aligned_exposure = exposure.reindex(weights.index).ffill().fillna(1.0)
    effective[asset_cols] = weights[asset_cols].mul(aligned_exposure[asset_cols], axis=0)
    effective["Cash / Reduced Exposure"] = (1.0 - effective[asset_cols].sum(axis=1)).clip(lower=0.0)
    return effective


def _append_result(
    rows: list[dict[str, object]],
    curves: dict[str, pd.Series],
    weight_frames: list[pd.DataFrame],
    asset_returns: pd.DataFrame,
    weights: pd.DataFrame,
    strategy: str,
    model_type: str,
    profile: dict[str, object],
) -> None:
    asset_cols = ["US Equity", "TH Equity", "JP Equity", "Gold", "BTC"]
    returns = asset_returns.mul(weights[asset_cols], axis=0).sum(axis=1)
    curve = curve_from_returns(returns, initial=INITIAL_VALUE)
    rows.append(_metrics_row(curve, strategy, weights, model_type, profile))
    curves[strategy] = curve
    weight_frames.append(weights.assign(Strategy=strategy).reset_index(names="Date"))


def _evaluate_index_signal_models(
    asset_returns: pd.DataFrame,
    signal_prices: pd.DataFrame,
    profile: dict[str, object],
    exposure_variants: dict[str, pd.DataFrame],
) -> tuple[list[dict[str, object]], dict[str, pd.Series], list[pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    curves: dict[str, pd.Series] = {}
    weight_frames: list[pd.DataFrame] = []
    index = asset_returns.index
    equity_budget = float(profile["equity_budget"])
    gold_weight = float(profile["gold_weight"])
    btc_weight = float(profile["btc_weight"])
    scores = _signal_scores(signal_prices).reindex(index).fillna(0.0)
    variants = [
        ("with_jp", "redistribute_active", ["US Equity", "TH Equity", "JP Equity"]),
        ("with_jp", "cash_inactive", ["US Equity", "TH Equity", "JP Equity"]),
        ("no_jp", "redistribute_active", ["US Equity", "TH Equity"]),
        ("no_jp", "cash_inactive", ["US Equity", "TH Equity"]),
    ]
    for jp_mode, mode, active_assets in variants:
        equity_weights = scores[active_assets].copy()
        if mode == "redistribute_active":
            denom = equity_weights.sum(axis=1).replace(0.0, np.nan)
            fallback = {asset: 0.0 for asset in active_assets}
            fallback["US Equity"] = 1.0
            equity_weights = equity_weights.div(denom, axis=0).fillna(fallback)
            equity_weights = equity_weights.mul(equity_budget)
            label = "Index signal assigns equity to active markets"
        else:
            equity_weights = equity_weights.div(float(len(active_assets))).mul(equity_budget)
            label = "Index signal leaves inactive equity in cash"
        if jp_mode == "no_jp":
            label = f"{label} no JP"
        label = f"{profile['profile']} {label}"
        weights = pd.DataFrame(
            {
                "US Equity": equity_weights.get("US Equity", pd.Series(0.0, index=index)),
                "TH Equity": equity_weights.get("TH Equity", pd.Series(0.0, index=index)),
                "JP Equity": equity_weights.get("JP Equity", pd.Series(0.0, index=index)),
                "Gold": gold_weight,
                "BTC": btc_weight,
            },
            index=index,
        ).fillna(0.0)
        weights["Cash / Reduced Exposure"] = (1.0 - weights.sum(axis=1)).clip(lower=0.0)
        _append_result(rows, curves, weight_frames, asset_returns, weights, label, "Index signal assignment", profile)
        for exposure_name, exposure in exposure_variants.items():
            exposed_label = f"{label} + {exposure_name}"
            exposed_weights = _apply_daily_exposure(weights, exposure)
            _append_result(
                rows,
                curves,
                weight_frames,
                asset_returns,
                exposed_weights,
                exposed_label,
                "Index signal assignment + exposure",
                profile,
            )
    return rows, curves, weight_frames


def _evaluate_stock_bucket_cutout_models(
    asset_returns: pd.DataFrame,
    profile: dict[str, object],
    exposure_variants: dict[str, pd.DataFrame],
) -> tuple[list[dict[str, object]], dict[str, pd.Series], list[pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    curves: dict[str, pd.Series] = {}
    weight_frames: list[pd.DataFrame] = []
    index = asset_returns.index
    equity_budget = float(profile["equity_budget"])
    gold_weight = float(profile["gold_weight"])
    btc_weight = float(profile["btc_weight"])
    variants = [
        ("with JP", ["US Equity", "TH Equity", "JP Equity"]),
        ("no JP", ["US Equity", "TH Equity"]),
    ]
    for jp_label, stock_assets in variants:
        for exposure_name, exposure in exposure_variants.items():
            aligned_exposure = exposure.reindex(index).ffill().fillna(1.0).clip(0.0, 1.0)
            active = aligned_exposure[stock_assets].ge(0.999).astype(float)
            active_count = active.sum(axis=1).replace(0.0, np.nan)
            stock_weights = active.div(active_count, axis=0).fillna(0.0).mul(equity_budget)
            weights = pd.DataFrame(
                {
                    "US Equity": stock_weights.get("US Equity", pd.Series(0.0, index=index)),
                    "TH Equity": stock_weights.get("TH Equity", pd.Series(0.0, index=index)),
                    "JP Equity": stock_weights.get("JP Equity", pd.Series(0.0, index=index)),
                    "Gold": gold_weight * aligned_exposure["Gold"],
                    "BTC": btc_weight * aligned_exposure["BTC"],
                },
                index=index,
            ).fillna(0.0)
            weights["Cash / Reduced Exposure"] = (1.0 - weights.sum(axis=1)).clip(lower=0.0)
            label = f"{profile['profile']} stock bucket country cutout {jp_label} + {exposure_name}"
            _append_result(
                rows,
                curves,
                weight_frames,
                asset_returns,
                weights,
                label,
                "Stock bucket country cutout",
                profile,
            )
    return rows, curves, weight_frames


def main() -> None:
    paths = default_paths(ROOT)
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    us_th_returns = _read_us_th_returns(paths)
    jp_returns, jp_index, jp_internal_weights, jp_price_end = _build_japan_pit_sleeve(paths, us_th_returns.index)
    common_index = us_th_returns.index.intersection(jp_returns.dropna().index).sort_values()
    jp_holding_mask = jp_internal_weights.sum(axis=1).gt(0.0)
    if not jp_holding_mask.any():
        raise RuntimeError("Japan PIT sleeve has no active holdings.")
    jp_return_mask = jp_returns.ne(0.0)
    if not jp_return_mask.any():
        raise RuntimeError("Japan PIT sleeve has no non-zero returns.")
    common_index = common_index[
        (common_index >= jp_holding_mask.idxmax())
        & (common_index <= jp_price_end)
    ]
    us_th_returns = us_th_returns.reindex(common_index).fillna(0.0)
    jp_returns = jp_returns.reindex(common_index).fillna(0.0)
    overlay_prices, signal_prices = _load_overlay_assets(paths, common_index)
    signal_prices["JP Equity"] = _load_japan_signal_price(paths, common_index, jp_index)
    asset_returns = pd.DataFrame(
        {
            "US Equity": us_th_returns["US PIT optimized sleeve THB"],
            "TH Equity": us_th_returns["TH PIT optimized sleeve THB"],
            "JP Equity": jp_returns,
            "Gold": overlay_prices["Gold"].pct_change(fill_method=None).fillna(0.0),
            "BTC": overlay_prices["BTC"].pct_change(fill_method=None).fillna(0.0),
        },
        index=common_index,
    ).fillna(0.0)
    exposure_variants = {
        name: exposure.reindex(common_index).ffill().fillna(1.0).clip(0.0, 1.0)
        for name, exposure in _asset_exposure_variants(signal_prices).items()
    }

    rows: list[dict[str, object]] = []
    curves: dict[str, pd.Series] = {}
    weights: list[pd.DataFrame] = []
    for profile in ALLOCATION_PROFILES:
        for new_rows, new_curves, new_weights in [
            _evaluate_fixed_weight_grid(asset_returns, profile, exposure_variants),
            _evaluate_index_signal_models(asset_returns, signal_prices, profile, exposure_variants),
            _evaluate_stock_bucket_cutout_models(asset_returns, profile, exposure_variants),
        ]:
            rows.extend(new_rows)
            curves.update(new_curves)
            weights.extend(new_weights)

    summary = pd.DataFrame(rows).sort_values(["Sharpe", "CAGR"], ascending=False)
    curves_df = pd.DataFrame(curves).dropna(how="all")
    weights_df = pd.concat(weights, ignore_index=True)

    summary.to_csv(paths.result_dir / f"{PREFIX}_summary_thb.csv", index=False)
    curves_df.to_csv(paths.result_dir / f"{PREFIX}_curves_thb.csv")
    weights_df.to_csv(paths.result_dir / f"{PREFIX}_weight_history_thb.csv", index=False)
    exposure_variants["daily exposure all assets"].to_csv(paths.result_dir / f"{PREFIX}_daily_exposure_history.csv")
    exposure_history = pd.concat(exposure_variants, names=["Exposure Variant", "Date"]).reset_index()
    exposure_history.to_csv(paths.result_dir / f"{PREFIX}_exposure_variant_history.csv", index=False)
    jp_internal_weights.to_csv(paths.result_dir / f"{PREFIX}_jp_internal_weight_history.csv")
    _write_jp_comparison(summary, paths.result_dir / f"{PREFIX}_jp_comparison_thb.csv")

    print(summary.head(20).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def _write_jp_comparison(summary: pd.DataFrame, path: Path) -> None:
    rows = []
    for profile, profile_summary in summary.groupby("Allocation Profile"):
        fixed = profile_summary.loc[profile_summary["Model Type"].eq("Fixed weight grid")].copy()
        best_fixed_with_jp = fixed.loc[fixed["Average JP Weight"].gt(0.0)].sort_values(["Sharpe", "CAGR"], ascending=False).head(1)
        best_fixed_no_jp = fixed.loc[fixed["Average JP Weight"].eq(0.0)].sort_values(["Sharpe", "CAGR"], ascending=False).head(1)
        if not best_fixed_with_jp.empty and not best_fixed_no_jp.empty:
            rows.append(_comparison_row(f"{profile} fixed weight grid", best_fixed_with_jp.iloc[0], best_fixed_no_jp.iloc[0]))

        signal = profile_summary.loc[profile_summary["Model Type"].eq("Index signal assignment")].copy()
        for label in ["assigns equity to active markets", "leaves inactive equity in cash"]:
            with_jp = signal.loc[signal["Strategy"].str.contains(label, regex=False) & ~signal["Strategy"].str.contains("no JP", regex=False)]
            no_jp = signal.loc[signal["Strategy"].str.contains(label, regex=False) & signal["Strategy"].str.contains("no JP", regex=False)]
            if not with_jp.empty and not no_jp.empty:
                rows.append(_comparison_row(f"{profile} index signal {label}", with_jp.iloc[0], no_jp.iloc[0]))
    pd.DataFrame(rows).to_csv(path, index=False)


def _comparison_row(model: str, with_jp: pd.Series, no_jp: pd.Series) -> dict[str, object]:
    return {
        "Comparison": model,
        "With JP Strategy": with_jp["Strategy"],
        "No JP Strategy": no_jp["Strategy"],
        "With JP Sharpe": with_jp["Sharpe"],
        "No JP Sharpe": no_jp["Sharpe"],
        "Delta Sharpe": with_jp["Sharpe"] - no_jp["Sharpe"],
        "With JP CAGR": with_jp["CAGR"],
        "No JP CAGR": no_jp["CAGR"],
        "Delta CAGR": with_jp["CAGR"] - no_jp["CAGR"],
        "With JP Max Drawdown": with_jp["Max Drawdown"],
        "No JP Max Drawdown": no_jp["Max Drawdown"],
        "Delta Max Drawdown": with_jp["Max Drawdown"] - no_jp["Max Drawdown"],
    }


if __name__ == "__main__":
    main()
