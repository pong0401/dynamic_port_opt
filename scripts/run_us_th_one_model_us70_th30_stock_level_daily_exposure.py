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

from dynamic_factor_copula import compute_port_opt_style_metrics, curve_from_returns, default_paths  # noqa: E402
from dynamic_factor_copula import load_cached_market_data, load_set100_membership_intervals, load_sp500_membership_intervals  # noqa: E402
from run_us_th_tactical_gold_crash_protection_sweep import _gold_crash_exposure  # noqa: E402
from run_us_th_tactical_one_model import BTC_CAP, GOLD_CAP, INITIAL_VALUE, OVERLAY_ASSETS, STOCK_CAP  # noqa: E402
from run_us_th_tactical_one_model import END_DATE, START_DATE  # noqa: E402
import run_us_th_tactical_one_model_asym_group_caps as asym_caps  # noqa: E402
from run_us_th_tactical_one_model_asym_group_caps import CASH_ASSET  # noqa: E402
from run_us_th_tactical_perf_momentum import RISK_FREE_RATE, _close_trend_exposure  # noqa: E402
from us_th_pit_reselect_utils import available_cached_columns  # noqa: E402


OUTPUT_PREFIX = "us_th_one_model_us70_th30_stock_level_daily_exposure"
CASE_LABEL = "US cap 70% / TH cap 30%"
US_GROUP_CAP = 0.70
TH_GROUP_CAP = 0.30

US_STOCK_MA = 300
US_STOCK_BELOW_EXPOSURE = 0.50
TH_STOCK_MA = 200
TH_STOCK_BELOW_EXPOSURE = 0.00
BTC_MA = 50
BTC_BELOW_EXPOSURE = 0.00

VARIANTS = [
    {
        "name": "Current proxy daily exposure",
        "us_stock_level": False,
        "th_stock_level": False,
    },
    {
        "name": "US stock-level daily exposure only",
        "us_stock_level": True,
        "th_stock_level": False,
    },
    {
        "name": "TH stock-level daily exposure only",
        "us_stock_level": False,
        "th_stock_level": True,
    },
    {
        "name": "All stock-level daily exposure",
        "us_stock_level": True,
        "th_stock_level": True,
    },
]


def _stock_columns(weights: pd.DataFrame) -> list[str]:
    return [
        column
        for column in weights.columns
        if column not in OVERLAY_ASSETS and column != CASH_ASSET
    ]


def _metrics_row(curve: pd.Series, strategy: str, weights: pd.DataFrame, variant: str) -> dict[str, object]:
    clean = curve.dropna()
    us_cols = [
        column
        for column in weights.columns
        if column not in OVERLAY_ASSETS and column != CASH_ASSET and not column.endswith(".BK")
    ]
    th_cols = [column for column in weights.columns if column.endswith(".BK")]
    row = compute_port_opt_style_metrics(clean, risk_free_rate=RISK_FREE_RATE).to_dict()
    row.update(
        {
            "Strategy": strategy,
            "Variant": variant,
            "Start": clean.index.min().date().isoformat(),
            "End": clean.index.max().date().isoformat(),
            "US Group Cap": US_GROUP_CAP,
            "TH Group Cap": TH_GROUP_CAP,
            "Stock Cap": STOCK_CAP,
            "Gold Cap": GOLD_CAP,
            "BTC Cap": BTC_CAP,
            "Average US Stock Weight": float(weights[us_cols].sum(axis=1).mean()) if us_cols else 0.0,
            "Average TH Stock Weight": float(weights[th_cols].sum(axis=1).mean()) if th_cols else 0.0,
            "Average Gold Weight": float(weights["GC=F"].mean()) if "GC=F" in weights else 0.0,
            "Average BTC Weight": float(weights["BTC-USD"].mean()) if "BTC-USD" in weights else 0.0,
            "Average Cash / Reduced Exposure Weight": float(weights[CASH_ASSET].mean()) if CASH_ASSET in weights else 0.0,
            "Latest US Stock Weight": float(weights[us_cols].sum(axis=1).iloc[-1]) if us_cols else 0.0,
            "Latest TH Stock Weight": float(weights[th_cols].sum(axis=1).iloc[-1]) if th_cols else 0.0,
            "Latest Gold Weight": float(weights["GC=F"].iloc[-1]) if "GC=F" in weights else 0.0,
            "Latest BTC Weight": float(weights["BTC-USD"].iloc[-1]) if "BTC-USD" in weights else 0.0,
            "Latest Cash / Reduced Exposure Weight": float(weights[CASH_ASSET].iloc[-1]) if CASH_ASSET in weights else 0.0,
        }
    )
    return row


def _period_compare(curves: pd.DataFrame) -> pd.DataFrame:
    end = curves.dropna(how="all").index.max()
    rows = []
    for period_name, years in [("Full period", None), ("10Y", 10), ("5Y", 5), ("3Y", 3), ("1Y", 1)]:
        start = curves.index.min() if years is None else end - pd.DateOffset(years=years)
        for strategy in curves.columns:
            sample = curves[strategy].dropna()
            sample = sample.loc[sample.index >= start]
            if sample.shape[0] < 2:
                continue
            row = compute_port_opt_style_metrics(sample, risk_free_rate=RISK_FREE_RATE).to_dict()
            row.update(
                {
                    "Period": period_name,
                    "Strategy": strategy,
                    "Start": sample.index.min().date().isoformat(),
                    "End": sample.index.max().date().isoformat(),
                    "Observations": int(sample.shape[0]),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)




def _load_overlay_inputs_from_cache(index: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = default_paths(ROOT)
    extra = pd.read_parquet(paths.local_cache_root / "extra_prices.parquet").sort_index().ffill()
    overlay = pd.read_parquet(paths.local_cache_root / "stock_level_overlay_prices_yf.parquet").sort_index().ffill()
    full_index = index.union(extra.index).union(overlay.index).sort_values()
    extra = extra.reindex(full_index).ffill()
    overlay = overlay.reindex(full_index).ffill()
    fx = overlay["USDTHB=X"].ffill()
    thb_prices = pd.DataFrame(
        {
            "Gold": overlay["GC=F"].mul(fx),
            "BTC": overlay["BTC-USD"].mul(fx),
        },
        index=full_index,
    ).reindex(index).ffill()
    signal_prices = pd.DataFrame(
        {
            "US Equity": overlay["SPY"],
            "TH Equity": extra["^SET.BK"],
            "Gold": overlay["GC=F"],
            "BTC": overlay["BTC-USD"],
        },
        index=full_index,
    ).reindex(index).ffill()
    return thb_prices, signal_prices


def _load_full_us_th_overlay_panel_from_cache() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str], list[str]]:
    paths = default_paths(ROOT)
    source_cols = available_cached_columns(paths.source_cache_root / "prices.parquet")
    extra_cols = available_cached_columns(paths.local_cache_root / "extra_prices.parquet")
    all_us = [
        ticker
        for ticker in load_sp500_membership_intervals(paths)["ticker"].dropna().astype(str).drop_duplicates()
        if ticker in source_cols
    ]
    all_th = [
        ticker
        for ticker in load_set100_membership_intervals(paths)["ticker"].dropna().astype(str).drop_duplicates()
        if ticker in extra_cols
    ]
    stock_tickers = list(dict.fromkeys(all_us + all_th))
    cached_panel = load_cached_market_data(paths, tickers=stock_tickers + ["^SET.BK"])
    prices = cached_panel["prices"].loc[START_DATE:END_DATE].sort_index().ffill()
    volumes = cached_panel["volumes"].loc[START_DATE:END_DATE].sort_index().fillna(0.0)
    overlay = pd.read_parquet(paths.local_cache_root / "stock_level_overlay_prices_yf.parquet").loc[START_DATE:END_DATE].sort_index().ffill()
    full_index = prices.index
    overlay = overlay.reindex(full_index).ffill()
    fx = overlay["USDTHB=X"].ffill()

    us_cols = [ticker for ticker in all_us if ticker in prices.columns]
    th_cols = [ticker for ticker in all_th if ticker in prices.columns]
    us_price_df = prices.reindex(columns=us_cols).mul(fx, axis=0)
    th_price_df = prices.reindex(columns=th_cols)
    overlay_asset_df = pd.DataFrame(
        {
            "GC=F": overlay["GC=F"].ffill().mul(fx),
            "BTC-USD": overlay["BTC-USD"].ffill().mul(fx),
        },
        index=full_index,
    )
    thb_prices = pd.concat([us_price_df, th_price_df, overlay_asset_df], axis=1)
    benchmark = overlay["SPY"].ffill().mul(fx).rename("benchmark")
    vol_proxy = overlay["^VIX"].ffill().rename("vol_proxy")
    common_index = thb_prices.index.intersection(benchmark.dropna().index).intersection(vol_proxy.dropna().index)
    thb_prices = thb_prices.reindex(common_index).ffill().dropna(how="all")
    volumes = volumes.reindex(thb_prices.index).reindex(columns=thb_prices.columns).fillna(0.0)
    benchmark = benchmark.reindex(thb_prices.index).ffill()
    vol_proxy = vol_proxy.reindex(thb_prices.index).ffill()
    return thb_prices, volumes, benchmark, vol_proxy, all_us, all_th

def _base_exposure(index: pd.DatetimeIndex) -> pd.DataFrame:
    _, signal_prices = _load_overlay_inputs_from_cache(index)
    return pd.DataFrame(
        {
            "US": _close_trend_exposure(signal_prices["US Equity"], US_STOCK_MA, US_STOCK_BELOW_EXPOSURE),
            "TH": _close_trend_exposure(signal_prices["TH Equity"], TH_STOCK_MA, TH_STOCK_BELOW_EXPOSURE),
            "GC=F": _gold_crash_exposure(
                signal_prices["Gold"],
                dd_window=252,
                warn_dd=-0.08,
                crash_dd=-0.20,
                warn_exposure=0.50,
                crash_exposure=0.50,
                recovery_dd=-0.05,
                panic_dd=-0.30,
                panic_ma_period=200,
                panic_mom_period=63,
            ),
            "BTC-USD": _close_trend_exposure(signal_prices["BTC"], BTC_MA, BTC_BELOW_EXPOSURE),
        },
        index=index,
    ).ffill().fillna(1.0).clip(0.0, 1.0)


def _stock_level_exposure(price: pd.Series, ma_period: int, below_exposure: float, index: pd.DatetimeIndex) -> pd.Series:
    return _close_trend_exposure(price.reindex(index).ffill(), ma_period, below_exposure).reindex(index).ffill().fillna(1.0).clip(0.0, 1.0)


def _build_variant_exposure(
    raw_weights: pd.DataFrame,
    prices: pd.DataFrame,
    base_exposure: pd.DataFrame,
    us_stock_level: bool,
    th_stock_level: bool,
) -> pd.DataFrame:
    index = raw_weights.index
    exposures = pd.DataFrame(1.0, index=index, columns=raw_weights.columns, dtype=float)
    for column in raw_weights.columns:
        if column == CASH_ASSET:
            exposures[column] = 1.0
        elif column == "GC=F":
            exposures[column] = base_exposure["GC=F"]
        elif column == "BTC-USD":
            exposures[column] = base_exposure["BTC-USD"]
        elif column.endswith(".BK"):
            exposures[column] = (
                _stock_level_exposure(prices[column], TH_STOCK_MA, TH_STOCK_BELOW_EXPOSURE, index)
                if th_stock_level and column in prices
                else base_exposure["TH"]
            )
        else:
            exposures[column] = (
                _stock_level_exposure(prices[column], US_STOCK_MA, US_STOCK_BELOW_EXPOSURE, index)
                if us_stock_level and column in prices
                else base_exposure["US"]
            )
    return exposures.ffill().fillna(1.0).clip(0.0, 1.0)


def _apply_exposure(raw_weights: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    effective = raw_weights.mul(exposure.reindex(columns=raw_weights.columns).fillna(1.0), axis=0)
    noncash = effective.drop(columns=[CASH_ASSET], errors="ignore").sum(axis=1)
    raw_cash = raw_weights[CASH_ASSET] if CASH_ASSET in raw_weights else 0.0
    effective[CASH_ASSET] = raw_cash + (1.0 - raw_cash - noncash).clip(lower=0.0)
    return effective.clip(lower=0.0)


def _grouped_history(effective_history: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for strategy, frame in effective_history.groupby("Strategy"):
        weights = frame.drop(columns=["Strategy"]).set_index("Date")
        weights.index = pd.to_datetime(weights.index)
        grouped = pd.DataFrame(
            {
                "US Stocks": weights[
                    [
                        column
                        for column in weights.columns
                        if column not in OVERLAY_ASSETS and column != CASH_ASSET and not column.endswith(".BK")
                    ]
                ].sum(axis=1),
                "TH Stocks": weights[[column for column in weights.columns if column.endswith(".BK")]].sum(axis=1),
                "Gold": weights["GC=F"] if "GC=F" in weights else 0.0,
                "BTC": weights["BTC-USD"] if "BTC-USD" in weights else 0.0,
                CASH_ASSET: weights[CASH_ASSET] if CASH_ASSET in weights else 0.0,
            }
        )
        grouped.insert(0, "Strategy", strategy)
        frames.append(grouped.reset_index(names="Date"))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    paths = default_paths(ROOT)
    asym_caps._load_full_us_th_overlay_panel_from_cache = _load_full_us_th_overlay_panel_from_cache
    asym_caps._load_overlay_inputs_from_cache = _load_overlay_inputs_from_cache
    raw_curve, _, raw_weights, _, selected = asym_caps._run_case(CASE_LABEL, US_GROUP_CAP, TH_GROUP_CAP)
    prices, _, _, _, _, _ = _load_full_us_th_overlay_panel_from_cache()
    raw_curve = raw_curve.sort_index()
    raw_weights = raw_weights.reindex(raw_curve.index).ffill().fillna(0.0)
    raw_weights = raw_weights.loc[:, raw_weights.abs().sum(axis=0).gt(0.0)]
    if CASH_ASSET not in raw_weights:
        raw_weights[CASH_ASSET] = 0.0

    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).reindex(raw_curve.index).fillna(0.0)
    base_exposure = _base_exposure(raw_curve.index)

    curves = {f"One-model {CASE_LABEL} raw": raw_curve}
    rows = [_metrics_row(raw_curve, f"One-model {CASE_LABEL} raw", raw_weights, "Raw")]
    latest_frames = []
    exposure_frames = []
    effective_frames = []

    raw_latest = raw_weights.iloc[-1].rename("Effective Weight").reset_index().rename(columns={"index": "Asset"})
    raw_latest = raw_latest.loc[raw_latest["Effective Weight"].abs().gt(1e-12)].sort_values("Effective Weight", ascending=False)
    raw_latest["Date"] = raw_weights.index.max().date().isoformat()
    raw_latest["Strategy"] = f"One-model {CASE_LABEL} raw"
    latest_frames.append(raw_latest)

    for variant in VARIANTS:
        exposure = _build_variant_exposure(
            raw_weights,
            prices,
            base_exposure,
            us_stock_level=bool(variant["us_stock_level"]),
            th_stock_level=bool(variant["th_stock_level"]),
        )
        effective = _apply_exposure(raw_weights, exposure)
        asset_cols = [column for column in effective.columns if column in returns.columns]
        variant_returns = returns[asset_cols].mul(effective[asset_cols], axis=0).sum(axis=1)
        strategy = f"One-model {CASE_LABEL} + {variant['name']}"
        curve = curve_from_returns(variant_returns, initial=INITIAL_VALUE).rename(strategy)
        curves[strategy] = curve
        rows.append(_metrics_row(curve, strategy, effective, str(variant["name"])))

        latest = effective.iloc[-1].rename("Effective Weight").reset_index().rename(columns={"index": "Asset"})
        latest = latest.loc[latest["Effective Weight"].abs().gt(1e-12)].sort_values("Effective Weight", ascending=False)
        latest["Date"] = effective.index.max().date().isoformat()
        latest["Strategy"] = strategy
        latest_frames.append(latest)

        exposed = exposure.copy()
        exposed.insert(0, "Strategy", strategy)
        exposure_frames.append(exposed.reset_index(names="Date"))

        out = effective.copy()
        out.insert(0, "Strategy", strategy)
        effective_frames.append(out.reset_index(names="Date"))

    curves_df = pd.DataFrame(curves).dropna(how="all")
    summary = pd.DataFrame(rows).sort_values(["Sharpe", "CAGR"], ascending=False)
    latest_weights = pd.concat(latest_frames, ignore_index=True)
    exposure_history = pd.concat(exposure_frames, ignore_index=True)
    effective_history = pd.concat(effective_frames, ignore_index=True)
    grouped_history = _grouped_history(effective_history)

    summary.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_summary_thb.csv", index=False)
    curves_df.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_curves_thb.csv")
    selected.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_universe_history_thb.csv", index=False)
    latest_weights.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_latest_weights_thb.csv", index=False)
    exposure_history.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_exposure_history_thb.csv", index=False)
    effective_history.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_effective_weights_thb.csv", index=False)
    grouped_history.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_grouped_weight_history_thb.csv", index=False)
    _period_compare(curves_df).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_period_compare_thb.csv", index=False)

    cols = [
        "Strategy",
        "CAGR",
        "Annual Vol",
        "Sharpe",
        "Max Drawdown",
        "Average US Stock Weight",
        "Average TH Stock Weight",
        "Average Gold Weight",
        "Average BTC Weight",
        "Average Cash / Reduced Exposure Weight",
    ]
    print(summary[cols].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(latest_weights.loc[latest_weights["Strategy"].eq(summary.iloc[0]["Strategy"])].head(40).to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
