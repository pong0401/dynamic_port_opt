from __future__ import annotations

from pathlib import Path
import sys
from dataclasses import dataclass

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
    _parquet_column_names,
    build_momentum_signal,
    compute_feature_table,
    compute_port_opt_style_metrics,
    curve_from_returns,
    default_paths,
    get_set100_members_as_of,
    get_sp500_members_as_of,
    load_cached_market_data,
    load_overlay_compare_prices,
    load_set100_membership_intervals,
    load_sp500_membership_intervals,
    monthly_rebalance_dates,
    select_point_in_time_universe,
)
from recheck_us_th_tactical_gold_btc_latest_weights import _gold_crash_protection_exposure  # noqa: E402
from run_us_th_joint_model import FEATURE_FLAGS, LOOKBACK_DAYS  # noqa: E402
from run_us_th_tactical_perf_momentum import RISK_FREE_RATE, _close_trend_exposure  # noqa: E402
from us_th_pit_reselect_utils import drop_duplicate_share_classes  # noqa: E402


PREFIX = "us_th_one_model_us70_th30_segment_cap25_backtest"
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
SEGMENT_CAP = 0.25
US_SEGMENT_FILE = ROOT / "data" / "us_segment.csv"
TH_SEGMENT_FILE = ROOT / "data" / "set100_segment.xls"
TH_SEGMENT_COLUMN = "Industry"
TH_TO_US_SEGMENT = {
    "Agro & Food Industry": "Consumer Staples",
    "Consumer Products": "Consumer Discretionary",
    "Financials": "Financials",
    "Industrial": "Industrials",
    "Industrials": "Industrials",
    "Property & Construction": "Real Estate",
    "Resources": "Energy",
    "Services": "Consumer Discretionary",
    "Technology": "Information Technology",
}
OVERLAY_ASSETS = ["GC=F", "BTC-USD"]
CASH_ASSET = "Cash / Reduced Exposure"
INITIAL_VALUE = 10_000.0


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    cap_mode: str

    @property
    def strategy(self) -> str:
        return f"US/TH one-model stockcap5 penalty0.02 assets50 {self.label} + daily exposure"


VARIANTS = [
    Variant("it_only", "IT segment cap 25%", "it_only"),
    Variant("all_segments", "all US segments cap 25% each", "all_segments"),
    Variant("it_plus_th_segments", "IT and TH segments cap 25% each", "it_plus_th_segments"),
    Variant("all_us_plus_th_segments", "all US and TH segments cap 25% each", "all_us_plus_th_segments"),
]


def _us_segment_table() -> pd.DataFrame:
    segment = pd.read_csv(US_SEGMENT_FILE)
    required = {"ticker", "segment"}
    if not required.issubset(segment.columns):
        raise RuntimeError(f"{US_SEGMENT_FILE} must contain columns: {sorted(required)}")
    segment = segment.copy()
    segment["ticker"] = segment["ticker"].astype(str).str.upper()
    segment["segment"] = segment["segment"].astype(str)
    return segment.dropna(subset=["ticker", "segment"])


def _th_segment_table() -> pd.DataFrame:
    tables = pd.read_html(TH_SEGMENT_FILE)
    if not tables:
        raise RuntimeError(f"No tables found in {TH_SEGMENT_FILE}")
    table = tables[0].copy()
    header_row = table.index[table.iloc[:, 0].astype(str).str.strip().eq("Symbol")]
    if header_row.empty:
        raise RuntimeError(f"{TH_SEGMENT_FILE} must contain a Symbol header row")
    table.columns = table.loc[header_row[0]].astype(str).str.strip()
    table = table.loc[header_row[0] + 1 :].copy()
    required = {"Symbol", TH_SEGMENT_COLUMN}
    if not required.issubset(table.columns):
        raise RuntimeError(f"{TH_SEGMENT_FILE} must contain columns: {sorted(required)}")
    table = table.rename(columns={"Symbol": "ticker", TH_SEGMENT_COLUMN: "segment"})
    table["ticker"] = table["ticker"].astype(str).str.strip().str.upper() + ".BK"
    table["segment"] = table["segment"].astype(str).str.strip()
    table = table.loc[table["segment"].ne("") & table["segment"].ne("-")]
    return table[["ticker", "segment"]].dropna(subset=["ticker", "segment"])


def _segment_map(market: str | None = None) -> dict[str, str]:
    if market == "us":
        table = _us_segment_table()
    elif market == "th":
        table = _th_segment_table()
    else:
        table = pd.concat([_us_segment_table(), _th_segment_table()], ignore_index=True)
    return dict(zip(table["ticker"], table["segment"]))


def _normalized_segment(segment: str | None, market: str) -> str | None:
    if segment is None or pd.isna(segment):
        return None
    value = str(segment).strip()
    if not value or value.lower() == "nan" or value == "-":
        return None
    return TH_TO_US_SEGMENT.get(value, value) if market == "th" else value


def _segment_groups_for_variant(selected: list[str], variant: Variant, us_assets: set[str], th_assets: set[str]) -> dict[str, list[int]]:
    us_segment_by_ticker = _segment_map("us")
    th_segment_by_ticker = _segment_map("th")
    groups: dict[str, list[int]] = {}
    for idx, asset in enumerate(selected):
        ticker = str(asset).upper()
        if ticker in us_assets:
            segment = _normalized_segment(us_segment_by_ticker.get(ticker), "us")
            if not segment:
                continue
            if variant.cap_mode in {"it_only", "it_plus_th_segments"} and segment.casefold() != "Information Technology".casefold():
                continue
            groups.setdefault(segment, []).append(idx)
        elif ticker in th_assets and variant.cap_mode in {"it_plus_th_segments", "all_us_plus_th_segments"}:
            segment = _normalized_segment(th_segment_by_ticker.get(ticker), "th")
            if not segment:
                continue
            if variant.cap_mode == "it_plus_th_segments" and segment.casefold() != "Information Technology".casefold():
                continue
            groups.setdefault(segment, []).append(idx)
    return groups


def _available_cached_columns(path: Path) -> set[str]:
    return set(_parquet_column_names(str(path))) if path.exists() else set()


def _active_members(kind: str, as_of: pd.Timestamp, available: list[str]) -> list[str]:
    paths = default_paths(ROOT)
    active = get_sp500_members_as_of(as_of, paths) if kind == "us" else get_set100_members_as_of(as_of, paths)
    active = [ticker for ticker in active if ticker in available]
    return drop_duplicate_share_classes(active) if kind == "us" else active


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


def _select_group(prices: pd.DataFrame, volumes: pd.DataFrame, candidates: list[str], train_index: pd.DatetimeIndex, n_assets: int) -> list[str]:
    if not candidates:
        return []
    return select_point_in_time_universe(
        prices.reindex(train_index),
        volumes.reindex(train_index),
        candidates,
        n_assets=n_assets,
        min_history_ratio=0.75,
    )


def _load_backtest_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, list[str], list[str]]:
    paths = default_paths(ROOT)
    source_cols = _available_cached_columns(paths.source_cache_root / "prices.parquet")
    extra_cols = _available_cached_columns(paths.local_cache_root / "extra_prices.parquet")
    available = source_cols | extra_cols
    us_all = [
        ticker
        for ticker in load_sp500_membership_intervals(paths)["ticker"].dropna().astype(str).drop_duplicates()
        if ticker in available
    ]
    us_all = drop_duplicate_share_classes(us_all)
    th_all = [
        ticker
        for ticker in load_set100_membership_intervals(paths)["ticker"].dropna().astype(str).drop_duplicates()
        if ticker in available
    ]
    stock_tickers = list(dict.fromkeys(us_all + th_all))
    cached = load_cached_market_data(paths, tickers=stock_tickers + ["SPY", "^VIX", "^SET.BK"])
    stock_prices = cached["prices"].sort_index().ffill()
    stock_volumes = cached["volumes"].sort_index().fillna(0.0)
    overlay = load_overlay_compare_prices(
        paths,
        start_date=START_DATE,
        tickers=["SPY", "^VIX", "GC=F", "BTC-USD", "USDTHB=X"],
    ).sort_index().ffill()
    set_index = pd.read_parquet(paths.local_cache_root / "extra_prices.parquet", columns=["^SET.BK"]).sort_index().ffill()["^SET.BK"]
    full_index = stock_prices.index.union(overlay.index).union(set_index.index).sort_values()
    stock_prices = stock_prices.reindex(full_index).ffill()
    stock_volumes = stock_volumes.reindex(full_index).fillna(0.0)
    overlay = overlay.reindex(full_index).ffill()
    set_index = set_index.reindex(full_index).ffill()
    fx = overlay["USDTHB=X"].ffill()

    us_cols = [ticker for ticker in us_all if ticker in stock_prices.columns]
    th_cols = [ticker for ticker in th_all if ticker in stock_prices.columns]
    us_prices_thb = stock_prices[us_cols].mul(fx, axis=0)
    th_prices = stock_prices[th_cols]
    overlay_assets = pd.DataFrame(
        {
            "GC=F": overlay["GC=F"].mul(fx),
            "BTC-USD": overlay["BTC-USD"].mul(fx),
        },
        index=full_index,
    )
    prices = pd.concat([us_prices_thb, th_prices, overlay_assets], axis=1).sort_index().ffill()
    volumes = stock_volumes.reindex(columns=prices.columns).fillna(0.0)
    volumes.loc[:, [asset for asset in OVERLAY_ASSETS if asset in volumes.columns]] = 1.0
    benchmark = overlay["SPY"].mul(fx).rename("benchmark")
    vol_proxy = overlay["^VIX"].rename("vol_proxy")
    signal_overlay = pd.DataFrame(
        {
            "SPY": overlay["SPY"],
            "^VIX": overlay["^VIX"],
            "GC=F": overlay["GC=F"],
            "BTC-USD": overlay["BTC-USD"],
            "USDTHB=X": overlay["USDTHB=X"],
            "^SET.BK": set_index,
        },
        index=full_index,
    ).ffill()
    common = prices.index.intersection(benchmark.dropna().index).intersection(vol_proxy.dropna().index)
    prices = prices.reindex(common).ffill().loc[START_DATE:]
    volumes = volumes.reindex(prices.index).fillna(0.0)
    benchmark = benchmark.reindex(prices.index).ffill()
    vol_proxy = vol_proxy.reindex(prices.index).ffill()
    signal_overlay = signal_overlay.reindex(prices.index).ffill()
    return prices, volumes, benchmark, vol_proxy, signal_overlay, us_cols, th_cols


def _optimize(
    train_returns: pd.DataFrame,
    benchmark: pd.Series,
    vol_proxy: pd.Series,
    prices: pd.DataFrame,
    us_assets: set[str],
    th_assets: set[str],
    variant: Variant,
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
        raise RuntimeError(f"Caps infeasible for {variant.label}: {caps.sum():.4f}")
    x0 = caps / caps.sum()
    bounds = [(0.0, float(caps.loc[asset])) for asset in selected]
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
    us_idx = [i for i, asset in enumerate(selected) if asset in us_assets]
    th_idx = [i for i, asset in enumerate(selected) if asset in th_assets]
    if us_idx:
        constraints.append({"type": "ineq", "fun": lambda x, idx=us_idx: US_GROUP_CAP - float(np.sum(x[idx]))})
    if th_idx:
        constraints.append({"type": "ineq", "fun": lambda x, idx=th_idx: TH_GROUP_CAP - float(np.sum(x[idx]))})
    for segment, idx in _segment_groups_for_variant(selected, variant, us_assets, th_assets).items():
        constraints.append({"type": "ineq", "fun": lambda x, idx=idx: SEGMENT_CAP - float(np.sum(x[idx]))})

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
    if asset == CASH_ASSET:
        return CASH_ASSET
    return "Other"


def _segment_weights(weights: pd.Series, us_assets: set[str], th_assets: set[str]) -> dict[str, float]:
    us_segmap = _segment_map("us")
    th_segmap = _segment_map("th")
    out: dict[str, float] = {}
    for asset, weight in weights.items():
        ticker = str(asset).upper()
        if ticker in us_assets:
            label = _normalized_segment(us_segmap.get(ticker), "us")
        elif ticker in th_assets:
            label = _normalized_segment(th_segmap.get(ticker), "th")
        else:
            label = None
        if label:
            out[label] = out.get(label, 0.0) + float(weight)
    return out


def _metrics_row(strategy: str, curve: pd.Series, effective_history: pd.DataFrame, raw_rebalance: pd.DataFrame) -> dict[str, object]:
    row = compute_port_opt_style_metrics(curve.dropna(), risk_free_rate=RISK_FREE_RATE).to_dict()
    latest = effective_history.loc[effective_history["Date"].eq(effective_history["Date"].max())]
    row.update(
        {
            "Strategy": strategy,
            "Start": curve.dropna().index.min().date().isoformat(),
            "End": curve.dropna().index.max().date().isoformat(),
            "Average US Weight": float(effective_history.loc[effective_history["Sleeve"].eq("US Equity")].groupby("Date")["Effective Weight"].sum().mean()),
            "Average TH Weight": float(effective_history.loc[effective_history["Sleeve"].eq("TH Equity")].groupby("Date")["Effective Weight"].sum().mean()) if (effective_history["Sleeve"].eq("TH Equity")).any() else 0.0,
            "Average Cash Weight": float(effective_history.loc[effective_history["Sleeve"].eq(CASH_ASSET)].groupby("Date")["Effective Weight"].sum().mean()) if (effective_history["Sleeve"].eq(CASH_ASSET)).any() else 0.0,
            "Latest US Weight": float(latest.loc[latest["Sleeve"].eq("US Equity"), "Effective Weight"].sum()),
            "Latest TH Weight": float(latest.loc[latest["Sleeve"].eq("TH Equity"), "Effective Weight"].sum()),
            "Latest Cash Weight": float(latest.loc[latest["Sleeve"].eq(CASH_ASSET), "Effective Weight"].sum()),
            "Rebalances": int(raw_rebalance["Date"].nunique()),
        }
    )
    return row


def main() -> None:
    paths = default_paths(ROOT)
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    prices, volumes, benchmark, vol_proxy, overlay, us_all, th_all = _load_backtest_panel()
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    benchmark_ret = benchmark.pct_change(fill_method=None).rename("benchmark")
    vol_proxy_ret = vol_proxy.pct_change(fill_method=None).rename("vol_proxy")
    schedule = monthly_rebalance_dates(prices.index, lookback_days=LOOKBACK_DAYS, freq="ME")
    exposure = pd.DataFrame(
        {
            "US Equity": _close_trend_exposure(overlay["SPY"], 300, 0.50),
            "TH Equity": _close_trend_exposure(overlay["^SET.BK"], 200, 0.00),
            "Gold": _gold_crash_protection_exposure(overlay["GC=F"]),
            "BTC": _close_trend_exposure(overlay["BTC-USD"], 50, 0.00),
        },
        index=prices.index,
    ).ffill().fillna(1.0).clip(0.0, 1.0)

    nav = {v.key: pd.Series(INITIAL_VALUE, index=[schedule[0]], name=v.strategy) for v in VARIANTS}
    raw_rows: list[pd.DataFrame] = []
    effective_rows: list[pd.DataFrame] = []
    universe_rows: list[dict[str, object]] = []
    exposure_rows: list[pd.DataFrame] = []
    segment_rows: list[dict[str, object]] = []

    for idx, rebalance_date in enumerate(schedule[:-1]):
        next_date = schedule[idx + 1]
        loc = prices.index.get_loc(rebalance_date)
        train_index = prices.index[max(0, loc - LOOKBACK_DAYS + 1) : loc + 1]
        test_index = prices.index[(prices.index > rebalance_date) & (prices.index <= next_date)]
        if len(test_index) == 0:
            continue
        us_active = _active_members("us", rebalance_date, us_all)
        th_active = _active_members("th", rebalance_date, th_all)
        us_selected = _select_group(prices, volumes, us_active, train_index, US_ASSETS)
        th_weight = _th_tactical_weight(overlay, rebalance_date)
        th_selected = _select_group(prices, volumes, th_active, train_index, TH_ASSETS) if th_weight > 1e-12 else []
        current_assets = list(dict.fromkeys(us_selected + th_selected + [asset for asset in OVERLAY_ASSETS if asset in prices.columns]))
        if len(current_assets) < 6:
            continue
        train_returns = returns.reindex(train_index)[current_assets].replace([np.inf, -np.inf], np.nan)
        current_us = set([asset for asset in current_assets if asset in us_selected])
        current_th = set([asset for asset in current_assets if asset in th_selected])
        universe_rows.append(
            {
                "Date": rebalance_date.date().isoformat(),
                "TH Tactical Weight": th_weight,
                "US Count": len(current_us),
                "TH Count": len(current_th),
                "US Selected": ",".join(sorted(current_us)),
                "TH Selected": ",".join(sorted(current_th)),
            }
        )
        period_returns = returns.reindex(test_index).fillna(0.0)
        period_exposure = exposure.reindex(test_index).ffill().fillna(1.0)
        exposure_out = period_exposure.reset_index(names="Date")
        exposure_out.insert(1, "Rebalance Date", rebalance_date.date().isoformat())
        exposure_rows.append(exposure_out)

        for variant in VARIANTS:
            weights = _optimize(train_returns, benchmark.reindex(train_index), vol_proxy.reindex(train_index), prices, current_us, current_th, variant)
            sleeve_map = weights.index.to_series().map(lambda asset: _sleeve(str(asset), current_us, current_th))
            raw = pd.DataFrame(
                {
                    "Date": rebalance_date.date().isoformat(),
                    "Strategy": variant.strategy,
                    "Variant": variant.key,
                    "Asset": weights.index,
                    "Sleeve": sleeve_map.to_numpy(),
                    "Raw Optimizer Weight": weights.to_numpy(dtype=float),
                }
            )
            raw_rows.append(raw)
            seg_weights = _segment_weights(weights, current_us, current_th)
            for segment, weight in seg_weights.items():
                segment_rows.append(
                    {
                        "Date": rebalance_date.date().isoformat(),
                        "Strategy": variant.strategy,
                        "Variant": variant.key,
                        "Segment": segment,
                        "Raw Segment Weight": weight,
                    }
                )
            daily = pd.DataFrame(index=test_index)
            for asset, weight in weights.items():
                sleeve = _sleeve(str(asset), current_us, current_th)
                daily[asset] = float(weight) * period_exposure[sleeve].to_numpy(dtype=float)
            noncash = daily.sum(axis=1)
            daily[CASH_ASSET] = (1.0 - noncash).clip(lower=0.0)
            asset_cols = [asset for asset in daily.columns if asset in period_returns.columns]
            strat_returns = period_returns[asset_cols].mul(daily[asset_cols], axis=0).sum(axis=1)
            nav[variant.key] = pd.concat([nav[variant.key], (1.0 + strat_returns).cumprod().mul(float(nav[variant.key].iloc[-1]))])
            eff_long = daily.reset_index(names="Date").melt(id_vars="Date", var_name="Asset", value_name="Effective Weight")
            eff_long = eff_long.loc[eff_long["Effective Weight"].abs().gt(1e-12)]
            eff_long["Strategy"] = variant.strategy
            eff_long["Variant"] = variant.key
            eff_long["Rebalance Date"] = rebalance_date.date().isoformat()
            eff_long["Sleeve"] = eff_long["Asset"].map(lambda asset: _sleeve(str(asset), current_us, current_th))
            effective_rows.append(eff_long)
        print(f"{idx + 1}/{len(schedule)-1} {rebalance_date.date()} US={len(us_selected)} TH={len(th_selected)}")

    curves = pd.DataFrame({v.strategy: nav[v.key][~nav[v.key].index.duplicated(keep="last")].sort_index() for v in VARIANTS}).dropna(how="all")
    effective = pd.concat(effective_rows, ignore_index=True)
    raw_weights = pd.concat(raw_rows, ignore_index=True)
    universe = pd.DataFrame(universe_rows)
    exposure_history = pd.concat(exposure_rows, ignore_index=True).drop_duplicates()
    segment_history = pd.DataFrame(segment_rows)
    summary_rows = []
    for variant in VARIANTS:
        curve = curves[variant.strategy].dropna()
        eff = effective.loc[effective["Variant"].eq(variant.key)]
        raw = raw_weights.loc[raw_weights["Variant"].eq(variant.key)]
        row = _metrics_row(variant.strategy, curve, eff, raw)
        row["Variant"] = variant.key
        row["Segment Cap Mode"] = variant.label
        row["Segment Cap"] = SEGMENT_CAP
        if not segment_history.empty:
            seg = segment_history.loc[segment_history["Variant"].eq(variant.key)]
            row["Max Raw Combined Segment Weight"] = float(seg["Raw Segment Weight"].max()) if not seg.empty else 0.0
            row["Average Max Raw Combined Segment Weight"] = float(seg.groupby("Date")["Raw Segment Weight"].max().mean()) if not seg.empty else 0.0
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["Sharpe", "CAGR"], ascending=False)

    latest_date = effective["Date"].max()
    latest = effective.loc[effective["Date"].eq(latest_date)].copy().sort_values(["Variant", "Effective Weight"], ascending=[True, False])
    summary.to_csv(paths.result_dir / f"{PREFIX}_summary_thb.csv", index=False)
    curves.to_csv(paths.result_dir / f"{PREFIX}_curves_thb.csv")
    raw_weights.to_csv(paths.result_dir / f"{PREFIX}_raw_weight_history_thb.csv", index=False)
    effective.to_csv(paths.result_dir / f"{PREFIX}_effective_weight_history_thb.csv", index=False)
    latest.to_csv(paths.result_dir / f"{PREFIX}_latest_effective_weights_thb.csv", index=False)
    universe.to_csv(paths.result_dir / f"{PREFIX}_universe_history_thb.csv", index=False)
    exposure_history.to_csv(paths.result_dir / f"{PREFIX}_exposure_history_thb.csv", index=False)
    segment_history.to_csv(paths.result_dir / f"{PREFIX}_segment_weight_history_thb.csv", index=False)

    cols = ["Variant", "Strategy", "CAGR", "Sharpe", "Max Drawdown", "Average US Weight", "Average TH Weight", "Average Cash Weight", "Max Raw Combined Segment Weight"]
    print(summary.reindex(columns=cols).to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()


