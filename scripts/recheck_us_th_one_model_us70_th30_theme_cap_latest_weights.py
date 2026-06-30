from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dynamic_factor_copula import (  # noqa: E402
    build_momentum_signal,
    compute_feature_table,
    default_paths,
    get_set100_members_as_of,
    get_sp500_members_as_of,
    select_point_in_time_universe,
)
from recheck_us_th_tactical_gold_btc_latest_weights import (  # noqa: E402
    FRESH_START_DATE,
    _fresh_us_th_panel,
    _gold_crash_protection_exposure,
)
from run_us_th_joint_model import FEATURE_FLAGS, LOOKBACK_DAYS  # noqa: E402
from run_us_th_tactical_perf_momentum import _close_trend_exposure  # noqa: E402
from us_th_pit_reselect_utils import drop_duplicate_share_classes  # noqa: E402


OUTPUT_PREFIX = "us_th_one_model_us70_th30_theme_cap_latest"
ALIAS_OUTPUT_PREFIX = "us_th_one_model_us70_th30_stockcap5_penalty002_assets50_ai_tech_cap25"
STRATEGY = "One-model US cap 70% / TH cap 30% stockcap5 penalty0.02 assets50 AI-tech cap 25% + daily exposure"
CASE = "US cap 70% / TH cap 30% stockcap5 penalty0.02 assets50 AI-tech cap 25%"
START_DATE = "2016-01-01"
US_GROUP_CAP = 0.70
TH_GROUP_CAP = 0.30
STOCK_CAP = 0.05
GOLD_CAP = 0.30
BTC_CAP = 0.10
US_ASSETS = 50
TH_ASSETS = 50
RISK_AVERSION = 8.0
CONCENTRATION_PENALTY = 0.02
THEME_CAP = 0.25
SEGMENT_FILE = ROOT / "data" / "us_segment.csv"
CAP_SEGMENT = "Information Technology"


def _segment_cap_bucket() -> set[str]:
    if not SEGMENT_FILE.exists():
        return set()
    segment = pd.read_csv(SEGMENT_FILE)
    required = {"ticker", "segment"}
    if not required.issubset(segment.columns):
        raise RuntimeError(f"{SEGMENT_FILE} must contain columns: {sorted(required)}")
    mask = segment["segment"].astype(str).str.casefold().eq(CAP_SEGMENT.casefold())
    return set(segment.loc[mask, "ticker"].dropna().astype(str).str.upper())


def _active_members(kind: str, as_of: pd.Timestamp, available: list[str]) -> list[str]:
    paths = default_paths(ROOT)
    if kind == "us":
        active = get_sp500_members_as_of(as_of, paths)
    else:
        active = get_set100_members_as_of(as_of, paths)
    active = [ticker for ticker in active if ticker in available]
    return drop_duplicate_share_classes(active) if kind == "us" else active


def _select_stock_group(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    candidates: list[str],
    n_assets: int,
    as_of: pd.Timestamp,
) -> tuple[list[str], pd.Timestamp]:
    stock_dates = prices.dropna(how="all").index
    stock_dates = stock_dates[stock_dates <= as_of]
    if stock_dates.empty:
        return [], as_of
    stock_as_of = pd.Timestamp(stock_dates.max())
    loc = prices.index.get_loc(stock_as_of)
    train_index = prices.index[max(0, loc - LOOKBACK_DAYS + 1) : loc + 1]
    selected = select_point_in_time_universe(
        prices.reindex(train_index),
        volumes.reindex(train_index),
        candidates,
        n_assets=n_assets,
        min_history_ratio=0.75,
    )
    return selected, stock_as_of


def _month_end(series: pd.Series) -> pd.Series:
    return series.groupby(series.index.to_period("M")).last()


def _th_tactical_weight(overlay: pd.DataFrame, as_of: pd.Timestamp) -> float:
    fx = overlay["USDTHB=X"].ffill()
    us_month = _month_end((overlay["SPY"] * fx).loc[:as_of].dropna())
    th_month = _month_end(overlay["^SET.BK"].loc[:as_of].dropna())
    monthly = pd.concat({"US": us_month, "TH": th_month}, axis=1).dropna()
    if len(monthly) < 2:
        return 0.0
    rel = monthly["TH"].pct_change(1) - monthly["US"].pct_change(1)
    signal = (rel > 0.0).astype(float).shift(1).ffill().fillna(0.0) * TH_GROUP_CAP
    return float(signal.iloc[-1])


def _source_close_date(index: pd.Index, effective_date: pd.Timestamp) -> str:
    dates = pd.DatetimeIndex(index[index < effective_date])
    source = dates.max() if len(dates) else effective_date
    return pd.Timestamp(source).date().isoformat()


def _optimize_one_model_with_theme_cap(
    train_returns: pd.DataFrame,
    benchmark: pd.Series,
    vol_proxy: pd.Series,
    prices: pd.DataFrame,
    us_assets: set[str],
    th_assets: set[str],
) -> pd.Series:
    selected = train_returns.dropna(axis=1, thresh=max(int(0.75 * len(train_returns)), 60)).columns.tolist()
    selected = drop_duplicate_share_classes(selected)
    train_returns = train_returns.reindex(columns=selected)
    if train_returns.empty or not selected:
        raise RuntimeError("No selected assets survived the combined one-model training window.")

    features = compute_feature_table(
        train_returns,
        benchmark.pct_change(fill_method=None).reindex(train_returns.index),
        vol_proxy.pct_change(fill_method=None).reindex(train_returns.index),
        prices.reindex(train_returns.index)[selected],
        include_momentum_features=True,
        feature_flags=FEATURE_FLAGS,
    )
    momentum = build_momentum_signal(features, mode="mom_63").reindex(selected)
    mu = momentum.fillna(momentum.median() if momentum.notna().any() else 0.0).to_numpy(dtype=float)
    if len(mu):
        mu = np.clip(mu, np.nanpercentile(mu, 10), np.nanpercentile(mu, 90))

    cov = train_returns.cov().reindex(index=selected, columns=selected).fillna(0.0)
    cov_matrix = cov.to_numpy(dtype=float)
    caps = pd.Series(STOCK_CAP, index=selected, dtype=float)
    caps.loc[[asset for asset in selected if asset == "GC=F"]] = GOLD_CAP
    caps.loc[[asset for asset in selected if asset == "BTC-USD"]] = BTC_CAP
    if float(caps.sum()) < 1.0 - 1e-12:
        raise RuntimeError("One-model caps are infeasible; caps sum below 100%.")

    x0 = caps / caps.sum()
    bounds = [(0.0, float(caps.loc[asset])) for asset in selected]
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
    us_idx = [i for i, asset in enumerate(selected) if asset in us_assets]
    th_idx = [i for i, asset in enumerate(selected) if asset in th_assets]
    cap_bucket = _segment_cap_bucket()
    theme_idx = [i for i, asset in enumerate(selected) if asset in cap_bucket]
    if us_idx:
        constraints.append({"type": "ineq", "fun": lambda x, idx=us_idx: US_GROUP_CAP - float(np.sum(x[idx]))})
    if th_idx:
        constraints.append({"type": "ineq", "fun": lambda x, idx=th_idx: TH_GROUP_CAP - float(np.sum(x[idx]))})
    if theme_idx:
        constraints.append({"type": "ineq", "fun": lambda x, idx=theme_idx: THEME_CAP - float(np.sum(x[idx]))})

    def objective(x: np.ndarray) -> float:
        variance = float(x @ cov_matrix @ x)
        expected = float(mu @ x)
        concentration = float(np.sum(np.square(x)))
        return 0.5 * RISK_AVERSION * variance - expected + CONCENTRATION_PENALTY * concentration

    result = minimize(objective, x0=x0.to_numpy(dtype=float), bounds=bounds, constraints=constraints, method="SLSQP")
    weights = pd.Series(result.x, index=selected).clip(lower=0.0) if result.success else x0.copy()
    return (weights / weights.sum()).sort_values(ascending=False)


def _sleeve(asset: str, us_assets: set[str], th_assets: set[str]) -> str:
    if asset in us_assets:
        return "US Equity"
    if asset in th_assets:
        return "TH Equity"
    if asset == "GC=F":
        return "Gold"
    if asset == "BTC-USD":
        return "BTC"
    if asset == "Cash / Reduced Exposure":
        return "Cash / Reduced Exposure"
    return "Other"


def _write_outputs(security: pd.DataFrame, sleeve: pd.DataFrame, meta: pd.DataFrame) -> None:
    for output_dir in [ROOT / "result", ROOT / "data" / "precomputed"]:
        output_dir.mkdir(parents=True, exist_ok=True)
        for prefix in [OUTPUT_PREFIX, ALIAS_OUTPUT_PREFIX]:
            security.to_csv(output_dir / f"{prefix}_latest_effective_weights_thb.csv", index=False)
            security.to_csv(output_dir / f"{prefix}_latest_raw_weights_thb.csv", index=False)
            sleeve.to_csv(output_dir / f"{prefix}_latest_sleeve_weights_thb.csv", index=False)
            meta.to_csv(output_dir / f"{prefix}_latest_meta.csv", index=False)
            payload = meta.iloc[0].to_dict() if not meta.empty else {}
            payload["calculated_at"] = pd.Timestamp.now(tz="Asia/Bangkok").isoformat()
            (output_dir / f"{prefix}_latest_meta.json").write_text(
                json.dumps(payload, indent=2, default=str),
                encoding="utf-8",
            )


def main() -> None:
    paths = default_paths(ROOT)
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    fresh_prices, fresh_volumes, benchmark, vol_proxy, us_all, th_all, overlay, as_of = _fresh_us_th_panel()
    as_of = pd.Timestamp(as_of)

    us_active = _active_members("us", as_of, us_all)
    th_active = _active_members("th", as_of, th_all)
    us_prices = fresh_prices.reindex(columns=us_active).loc[START_DATE:as_of].ffill()
    th_prices = fresh_prices.reindex(columns=th_active).loc[START_DATE:as_of].ffill()
    us_volumes = fresh_volumes.reindex(us_prices.index).reindex(columns=us_active).fillna(0.0)
    th_volumes = fresh_volumes.reindex(th_prices.index).reindex(columns=th_active).fillna(0.0)

    us_selected, us_internal_date = _select_stock_group(us_prices, us_volumes, us_active, US_ASSETS, as_of)
    th_signal_weight = _th_tactical_weight(overlay, as_of)
    th_selected: list[str] = []
    th_internal_date = as_of
    if th_signal_weight > 1e-12:
        th_selected, th_internal_date = _select_stock_group(th_prices, th_volumes, th_active, TH_ASSETS, as_of)

    combined_index = us_prices.index.union(th_prices.index).union(overlay.index).sort_values()
    prices = pd.DataFrame(index=combined_index)
    for asset in us_selected:
        prices[asset] = us_prices[asset].reindex(combined_index)
    for asset in th_selected:
        prices[asset] = th_prices[asset].reindex(combined_index)
    prices["GC=F"] = (overlay["GC=F"] * overlay["USDTHB=X"]).reindex(combined_index)
    prices["BTC-USD"] = (overlay["BTC-USD"] * overlay["USDTHB=X"]).reindex(combined_index)
    prices = prices.loc[START_DATE:as_of].ffill()

    train_dates = prices.dropna(how="all").index
    loc = train_dates.get_loc(train_dates.max())
    train_index = train_dates[max(0, loc - LOOKBACK_DAYS + 1) : loc + 1]
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    train_returns = returns.reindex(train_index)
    benchmark_train = benchmark.reindex(train_index).ffill().rename("benchmark")
    vol_proxy_train = vol_proxy.reindex(train_index).ffill().rename("vol_proxy")

    raw_weights = _optimize_one_model_with_theme_cap(
        train_returns,
        benchmark_train,
        vol_proxy_train,
        prices,
        set(us_selected),
        set(th_selected),
    )
    sleeve_map = raw_weights.index.to_series().map(lambda asset: _sleeve(str(asset), set(us_selected), set(th_selected)))
    exposures = pd.Series(
        {
            "US Equity": float(_close_trend_exposure(overlay["SPY"], 300, 0.50).loc[:as_of].iloc[-1]),
            "TH Equity": float(_close_trend_exposure(overlay["^SET.BK"], 200, 0.00).loc[:as_of].iloc[-1]),
            "Gold": float(_gold_crash_protection_exposure(overlay["GC=F"]).loc[:as_of].iloc[-1]),
            "BTC": float(_close_trend_exposure(overlay["BTC-USD"], 50, 0.00).loc[:as_of].iloc[-1]),
        },
        dtype=float,
    )
    asset_exposure = sleeve_map.map(exposures).fillna(1.0).astype(float)
    effective = raw_weights.mul(asset_exposure)
    cash_weight = max(0.0, 1.0 - float(effective.sum()))

    security = pd.DataFrame(
        {
            "Asset": raw_weights.index,
            "Sleeve": sleeve_map.to_numpy(),
            "Effective Weight": effective.to_numpy(dtype=float),
            "Raw Optimizer Weight": raw_weights.to_numpy(dtype=float),
            "Daily Exposure": asset_exposure.to_numpy(dtype=float),
        }
    )
    if cash_weight > 1e-12:
        security = pd.concat(
            [
                security,
                pd.DataFrame(
                    [
                        {
                            "Asset": "Cash / Reduced Exposure",
                            "Sleeve": "Cash / Reduced Exposure",
                            "Effective Weight": cash_weight,
                            "Raw Optimizer Weight": 0.0,
                            "Daily Exposure": 1.0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    security["Effective Weight %"] = security["Effective Weight"].mul(100.0)
    security["Date"] = as_of.date().isoformat()
    security["Strategy"] = STRATEGY
    security["Case"] = CASE
    security["Last Exposure Date"] = as_of.date().isoformat()
    security["Signal Source Close Date"] = _source_close_date(overlay.index, as_of)
    security = security.loc[security["Effective Weight"].abs().gt(1e-12)].sort_values("Effective Weight", ascending=False)

    sleeve = (
        security.groupby("Sleeve", as_index=False)
        .agg(
            **{
                "Effective Weight": ("Effective Weight", "sum"),
                "Raw Optimizer Weight": ("Raw Optimizer Weight", "sum"),
                "Daily Exposure": ("Daily Exposure", "max"),
            }
        )
        .sort_values("Effective Weight", ascending=False)
    )
    sleeve["Effective Weight %"] = sleeve["Effective Weight"].mul(100.0)
    sleeve["Date"] = as_of.date().isoformat()
    sleeve["Strategy"] = STRATEGY
    sleeve["Case"] = CASE

    cap_bucket = _segment_cap_bucket()
    theme_mask = security["Asset"].astype(str).isin(cap_bucket)
    meta = pd.DataFrame(
        [
            {
                "Date": as_of.date().isoformat(),
                "Strategy": STRATEGY,
                "Case": CASE,
                "Model": "one combined mean-covariance optimizer",
                "Objective": "mean_variance + mom_63 + concentration penalty",
                "Universe": f"PIT S&P 500 top{US_ASSETS}, PIT SET100 top{TH_ASSETS} when TH tactical signal is active, Gold, BTC",
                "Caps": f"stock {STOCK_CAP:.0%}; US group {US_GROUP_CAP:.0%}; TH group {TH_GROUP_CAP:.0%}; Gold {GOLD_CAP:.0%}; BTC {BTC_CAP:.0%}; {CAP_SEGMENT} segment {THEME_CAP:.0%}",
                "Daily Exposure": "US SPY MA300 below50%; TH SET MA200 below0%; Gold crash protection; BTC MA50 below0%; reduced exposure to cash",
                "TH Tactical Rule": "monthly SET-vs-SPY THB relative-return binary lb1 entry0 exit0 hold0 confirm1",
                "TH Tactical Active Weight": th_signal_weight,
                "Selected US Assets": len(us_selected),
                "Selected TH Assets": len(th_selected),
                "US Internal Weight Date": us_internal_date.date().isoformat(),
                "TH Internal Weight Date": th_internal_date.date().isoformat(),
                "Train Start": pd.Timestamp(train_index.min()).date().isoformat(),
                "Train End": pd.Timestamp(train_index.max()).date().isoformat(),
                "US Daily Exposure": float(exposures["US Equity"]),
                "TH Daily Exposure": float(exposures["TH Equity"]),
                "Gold Daily Exposure": float(exposures["Gold"]),
                "BTC Daily Exposure": float(exposures["BTC"]),
                "Segment Cap Name": CAP_SEGMENT,
                "Segment Cap": THEME_CAP,
                "Segment Cap Source": "data/us_segment.csv",
                "Segment Cap Bucket": ", ".join(sorted(cap_bucket)),
                "Latest Segment-Capped Effective Weight": float(security.loc[theme_mask, "Effective Weight"].sum()),
                "Latest Segment-Capped Raw Optimizer Weight": float(security.loc[theme_mask, "Raw Optimizer Weight"].sum()),
                "Latest Weight Source": "Standalone refresh from a fresh yfinance panel at run time; no static latest-weight file is read.",
                "Fresh Start Date": FRESH_START_DATE,
            }
        ]
    )

    _write_outputs(security, sleeve, meta)
    print(meta.to_string(index=False))
    print(sleeve.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(security[["Asset", "Sleeve", "Effective Weight", "Raw Optimizer Weight", "Daily Exposure"]].head(50).to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
