from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in [SRC, SCRIPTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dynamic_factor_copula import (  # noqa: E402
    build_factor_covariance,
    build_momentum_signal,
    compute_feature_table,
    compute_market_stress_signal,
    compute_port_opt_style_metrics,
    curve_from_returns,
    default_paths,
    get_set100_members_as_of,
    get_sp500_members_as_of,
    initialize_static_clusters,
    load_overlay_compare_prices,
    monthly_rebalance_dates,
    optimize_portfolio,
    optimize_risk_parity,
    run_dynamic_hmm,
    select_point_in_time_universe,
)
from run_us_th_joint_model import FEATURE_FLAGS, N_CLUSTERS  # noqa: E402
from run_us_th_tactical_perf_momentum import RISK_FREE_RATE  # noqa: E402
from us_th_pit_reselect_utils import drop_duplicate_share_classes, load_full_us_th_thb_panel, weights_history_to_frame  # noqa: E402


OUTPUT_PREFIX = "us_th_jp_all_asset_model"
START_DATE = "2024-03-01"
END_DATE = "2026-01-30"
LOOKBACK_DAYS = 120
US_ASSETS = 30
TH_ASSETS = 30
JP_ASSETS = 20
STOCK_CAP = 0.08
GOLD_CAP = 0.30
BTC_CAP = 0.10
OVERLAY_ASSETS = ["GC=F", "BTC-USD"]
OBJECTIVE_MODES = ["mean_variance", "min_vol_mom_tilt", "max_sharpe_mom", "risk_parity_mom_tilt"]
MODEL_FAMILIES = ["Equal Weight", "Risk Parity", "Static Copula", "Dynamic HMM Copula"]


def _sleeve(asset: str) -> str:
    if asset == "GC=F":
        return "Gold"
    if asset == "BTC-USD":
        return "BTC"
    if asset.endswith(".BK"):
        return "TH Equity"
    if asset[:1].isdigit() or asset.endswith("0"):
        return "JP Equity"
    return "US Equity"


def _load_japan_prices_volumes(paths, index: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe = pd.read_parquet(paths.local_cache_root / "japan_pit_universe_history.parquet")
    universe["entry_date"] = pd.to_datetime(universe["entry_date"], errors="coerce")
    universe["Code"] = universe["Code"].astype(str).str.strip()
    tickers = sorted(universe["Code"].dropna().unique().tolist())

    bars_file = paths.local_cache_root / "japan_daily_bars.parquet"
    available_cols = set(pq.ParquetFile(bars_file).schema.names)
    close_col = next((col for col in ["AdjC", "Close", "C"] if col in available_cols), None)
    volume_col = next((col for col in ["AdjVo", "Volume", "Vo"] if col in available_cols), None)
    if close_col is None or volume_col is None:
        raise ValueError(f"Japan daily bars cache lacks close/volume columns: {bars_file}")
    bars = pd.read_parquet(bars_file, columns=["Date", "Code", close_col, volume_col])
    bars["Date"] = pd.to_datetime(bars["Date"], errors="coerce")
    bars["Code"] = bars["Code"].astype(str).str.strip()
    bars = bars.loc[bars["Code"].isin(tickers)].dropna(subset=["Date", "Code"])
    prices_jpy = bars.pivot_table(index="Date", columns="Code", values=close_col, aggfunc="last").sort_index().ffill()
    volumes = bars.pivot_table(index="Date", columns="Code", values=volume_col, aggfunc="last").sort_index().fillna(0.0)

    fx = _load_jpy_thb_fx(paths, index.union(prices_jpy.index).sort_values())
    prices_thb = prices_jpy.mul(fx.reindex(prices_jpy.index).ffill().bfill(), axis=0)
    return prices_thb.reindex(index).ffill(), volumes.reindex(index).fillna(0.0), universe


def _load_jpy_thb_fx(paths, index: pd.DatetimeIndex) -> pd.Series:
    try:
        fx = load_overlay_compare_prices(
            paths,
            start_date=str(index.min().date()),
            end_date=str(index.max().date()),
            tickers=["JPYTHB=X"],
        ).sort_index()
        if "JPYTHB=X" in fx.columns and not fx["JPYTHB=X"].dropna().empty:
            return fx["JPYTHB=X"].reindex(index).ffill().bfill().rename("JPYTHB=X")
    except Exception as exc:
        print(f"Warning: could not load JPYTHB=X ({exc}); JP stocks remain JPY-denominated.")
    return pd.Series(1.0, index=index, name="JPYTHB_FALLBACK")


def _get_japan_members_as_of(universe: pd.DataFrame, as_of_date: pd.Timestamp) -> list[str]:
    eligible = universe.loc[universe["entry_date"].le(pd.Timestamp(as_of_date))].dropna(subset=["entry_date"])
    if eligible.empty:
        return []
    latest_entry = eligible["entry_date"].max()
    rows = eligible.loc[eligible["entry_date"].eq(latest_entry)].sort_values("rank")
    return rows["Code"].drop_duplicates().head(JP_ASSETS).tolist()


def _load_all_asset_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str], list[str], pd.DataFrame]:
    paths = default_paths(ROOT)
    us_th_prices, us_th_volumes, benchmark, vol_proxy, us_all, th_all = load_full_us_th_thb_panel(
        include_overlay_assets=True,
        overlay_asset_tickers=OVERLAY_ASSETS,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    jp_prices, jp_volumes, jp_universe = _load_japan_prices_volumes(paths, us_th_prices.index)
    prices = pd.concat([us_th_prices, jp_prices], axis=1).sort_index().ffill()
    volumes = pd.concat([us_th_volumes, jp_volumes], axis=1).reindex(prices.index).fillna(0.0)
    jp_active_start = jp_universe["entry_date"].dropna().min()
    common_index = prices.index[
        (prices.index >= jp_active_start)
        & (prices.index <= jp_prices.dropna(how="all").index.max())
    ]
    prices = prices.reindex(common_index).ffill().dropna(how="all")
    volumes = volumes.reindex(prices.index).fillna(0.0)
    benchmark = benchmark.reindex(prices.index).ffill()
    vol_proxy = vol_proxy.reindex(prices.index).ffill()
    return prices, volumes, benchmark, vol_proxy, us_all, th_all, jp_universe


def run_all_asset_model(objective_mode: str) -> dict[str, object]:
    prices, volumes, benchmark, vol_proxy, us_all, th_all, jp_universe = _load_all_asset_panel()
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    benchmark_ret = benchmark.pct_change(fill_method=None).rename("benchmark")
    vol_proxy_ret = vol_proxy.pct_change(fill_method=None).rename("vol_proxy")
    schedule = monthly_rebalance_dates(prices.index, lookback_days=LOOKBACK_DAYS, freq="ME")
    if len(schedule) < 3:
        raise RuntimeError("Not enough schedule points for JP all-asset model.")

    feature_history: dict[pd.Timestamp, pd.DataFrame] = {}
    market_stress_history: dict[pd.Timestamp, float] = {}
    selected_split_history: dict[pd.Timestamp, dict[str, list[str]]] = {}

    paths = default_paths(ROOT)
    for rebalance_date in schedule:
        loc = prices.index.get_loc(rebalance_date)
        train_index = prices.index[max(0, loc - LOOKBACK_DAYS + 1) : loc + 1]
        us_pool = [
            ticker
            for ticker in get_sp500_members_as_of(rebalance_date, paths)
            if ticker in us_all and ticker in prices.columns
        ]
        us_pool = drop_duplicate_share_classes(us_pool)
        th_pool = [ticker for ticker in get_set100_members_as_of(rebalance_date, paths) if ticker in th_all and ticker in prices.columns]
        jp_pool = [ticker for ticker in _get_japan_members_as_of(jp_universe, rebalance_date) if ticker in prices.columns]
        us_selected = select_point_in_time_universe(prices.loc[train_index], volumes.loc[train_index], us_pool, n_assets=US_ASSETS)
        th_selected = select_point_in_time_universe(prices.loc[train_index], volumes.loc[train_index], th_pool, n_assets=TH_ASSETS)
        jp_selected = [ticker for ticker in jp_pool if prices.loc[train_index, ticker].dropna().shape[0] >= max(40, int(0.5 * len(train_index)))]
        current_assets = list(dict.fromkeys(us_selected + th_selected + jp_selected + OVERLAY_ASSETS))
        current_assets = [asset for asset in current_assets if asset in prices.columns]
        if len(current_assets) < max(N_CLUSTERS + 2, 8):
            continue
        train_returns = returns.reindex(train_index)[current_assets].dropna(axis=1, thresh=max(int(0.70 * len(train_index)), 40))
        current_assets = train_returns.columns.tolist()
        if len(current_assets) < max(N_CLUSTERS + 2, 8):
            continue
        selected_split_history[rebalance_date] = {
            "US": [ticker for ticker in current_assets if ticker in us_selected],
            "TH": [ticker for ticker in current_assets if ticker in th_selected],
            "JP": [ticker for ticker in current_assets if ticker in jp_selected],
            "Overlay": [ticker for ticker in current_assets if ticker in OVERLAY_ASSETS],
        }
        market_stress_history[rebalance_date] = compute_market_stress_signal(
            benchmark_ret.reindex(train_index),
            vol_proxy_ret.reindex(train_index),
        )
        feature_table = compute_feature_table(
            train_returns,
            benchmark_ret.reindex(train_index),
            vol_proxy_ret.reindex(train_index),
            prices.reindex(train_index)[current_assets],
            include_momentum_features=True,
            feature_flags=FEATURE_FLAGS,
        )
        if not feature_table.empty:
            feature_history[rebalance_date] = feature_table

    if not feature_history:
        raise RuntimeError("No feature history was available for US/TH/JP all-asset model.")

    first_date = min(feature_history)
    initial = initialize_static_clusters(feature_history[first_date], n_clusters=N_CLUSTERS)
    dynamic_state = run_dynamic_hmm(
        feature_history,
        initial_state=initial,
        gas_alpha=0.40,
        gas_beta=0.45,
        market_stress_history=market_stress_history,
        posterior_power=2.25,
    )
    static_post = pd.get_dummies(initial["labels"]).reindex(columns=range(N_CLUSTERS), fill_value=0.0)

    nav = {name: pd.Series(1.0, index=[schedule[0]]) for name in MODEL_FAMILIES}
    weights_history: dict[str, dict[pd.Timestamp, pd.Series]] = {name: {} for name in MODEL_FAMILIES}
    max_weight = max(STOCK_CAP, GOLD_CAP, BTC_CAP)

    for idx, rebalance_date in enumerate(schedule[:-1]):
        next_date = schedule[idx + 1]
        if rebalance_date not in feature_history:
            continue
        loc = prices.index.get_loc(rebalance_date)
        train_index = prices.index[max(0, loc - LOOKBACK_DAYS + 1) : loc + 1]
        test_index = prices.index[(prices.index > rebalance_date) & (prices.index <= next_date)]
        if len(test_index) == 0:
            continue

        current_features = feature_history[rebalance_date]
        current_assets = current_features.index.tolist()
        train_returns = returns.loc[train_index, current_assets].dropna(how="all")
        bench_train = benchmark_ret.loc[train_index]
        momentum_signal = build_momentum_signal(current_features, mode="mom_63")

        asset_caps = {asset: STOCK_CAP for asset in current_assets}
        if "GC=F" in current_assets:
            asset_caps["GC=F"] = GOLD_CAP
        if "BTC-USD" in current_assets:
            asset_caps["BTC-USD"] = BTC_CAP

        eq_weights = pd.Series(1.0 / len(current_assets), index=current_assets).clip(upper=pd.Series(asset_caps))
        eq_weights = eq_weights / eq_weights.sum()
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
        weights = {
            "Equal Weight": eq_weights,
            "Risk Parity": optimize_risk_parity(risk_parity_cov, max_weight=max_weight, asset_caps=asset_caps),
            "Static Copula": optimize_portfolio(static_cov, momentum_signal, max_weight=max_weight, objective_mode=objective_mode, asset_caps=asset_caps),
            "Dynamic HMM Copula": optimize_portfolio(dyn_cov, momentum_signal, max_weight=max_weight, objective_mode=objective_mode, asset_caps=asset_caps),
        }
        period_returns = returns.reindex(test_index)[current_assets].fillna(0.0)
        for strategy, strategy_weights in weights.items():
            weights_history[strategy][rebalance_date] = strategy_weights
            weighted = period_returns.mul(strategy_weights, axis=1).sum(axis=1)
            starting_value = float(nav[strategy].iloc[-1])
            nav[strategy] = pd.concat([nav[strategy], starting_value * (1.0 + weighted).cumprod()])

    nav = {name: series[~series.index.duplicated(keep="last")].sort_index() for name, series in nav.items()}
    return {
        "nav": nav,
        "weights_history": weights_history,
        "selected_split_history": selected_split_history,
        "objective": objective_mode,
        "lookback_days": LOOKBACK_DAYS,
    }


def _summary_row(curve: pd.Series, strategy: str, model_family: str, objective: str) -> dict[str, object]:
    sample = curve.dropna()
    row = compute_port_opt_style_metrics(sample, risk_free_rate=RISK_FREE_RATE).to_dict()
    row.update(
        {
            "Strategy": strategy,
            "Model Family": model_family,
            "Objective": objective,
            "Start": sample.index.min().date().isoformat(),
            "End": sample.index.max().date().isoformat(),
            "Lookback Days": LOOKBACK_DAYS,
            "US Assets": US_ASSETS,
            "TH Assets": TH_ASSETS,
            "JP Assets": JP_ASSETS,
            "Stock Cap": STOCK_CAP,
            "Gold Cap": GOLD_CAP,
            "BTC Cap": BTC_CAP,
            "Selection Rule": "US+TH+JP PIT reselect with Gold/BTC in one optimizer",
        }
    )
    return row


def main() -> None:
    paths = default_paths(ROOT)
    rows: list[dict[str, object]] = []
    curves: dict[str, pd.Series] = {}
    latest_rows: list[pd.DataFrame] = []

    for objective in OBJECTIVE_MODES:
        print(f"Running US/TH/JP all-asset model objective={objective}", flush=True)
        results = run_all_asset_model(objective)
        for model_family in MODEL_FAMILIES:
            curve = results["nav"][model_family].dropna().mul(10_000.0)
            strategy = f"US/TH/JP all assets {model_family} [{objective}] caps stock8 gold30 btc10"
            rows.append(_summary_row(curve, strategy, model_family, objective))
            curves[strategy] = curve.rename(strategy)
            weights = weights_history_to_frame(results["weights_history"][model_family])
            if weights.empty:
                continue
            latest_date = weights.index.max()
            latest = weights.loc[latest_date].rename("Portfolio Weight").reset_index()
            latest.columns = ["Asset", "Portfolio Weight"]
            latest = latest.loc[latest["Portfolio Weight"].abs().gt(1e-12)].copy()
            latest["Sleeve"] = latest["Asset"].map(_sleeve)
            latest["Date"] = pd.Timestamp(latest_date).date().isoformat()
            latest["Strategy"] = strategy
            latest["Model Family"] = model_family
            latest["Objective"] = objective
            latest_rows.append(latest.sort_values("Portfolio Weight", ascending=False))

    summary = pd.DataFrame(rows).sort_values(["Sharpe", "CAGR"], ascending=False)
    curves_df = pd.DataFrame(curves).dropna(how="all")
    latest_df = pd.concat(latest_rows, ignore_index=True) if latest_rows else pd.DataFrame()
    summary.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_summary_thb.csv", index=False)
    curves_df.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_curves_thb.csv")
    latest_df.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_latest_weights_thb.csv", index=False)
    summary.head(1).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_best_thb.csv", index=False)
    print(summary.head(20).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
