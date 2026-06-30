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

from dynamic_factor_copula import (  # noqa: E402
    build_momentum_signal,
    compute_feature_table,
    compute_market_stress_signal,
    compute_port_opt_style_metrics,
    curve_from_returns,
    default_paths,
    get_set100_members_as_of,
    get_sp500_members_as_of,
    load_cached_market_data,
    load_set100_membership_intervals,
    load_sp500_membership_intervals,
    monthly_rebalance_dates,
    optimize_portfolio,
    select_point_in_time_universe,
)
from run_us_th_tactical_gold_crash_protection_sweep import _gold_crash_exposure  # noqa: E402
from run_us_th_tactical_gold_exposure_sweep import _load_overlay_inputs_from_cache  # noqa: E402
from run_us_th_tactical_perf_momentum import (  # noqa: E402
    FEATURE_FLAGS,
    PRIMARY_MODEL,
    RESULT_PREFIX,
    RISK_FREE_RATE,
    START_DATE,
    _best_tactical_daily_weight,
    _close_trend_exposure,
)
from run_us_th_joint_model import END_DATE, LOOKBACK_DAYS, N_CLUSTERS  # noqa: E402
from us_th_pit_reselect_utils import available_cached_columns, drop_duplicate_share_classes  # noqa: E402


OUTPUT_PREFIX = f"{RESULT_PREFIX}_one_model_gold30_btc10_th_signal"
US_ASSETS = 30
TH_ASSETS = 30
STOCK_CAP = 0.08
GOLD_CAP = 0.30
BTC_CAP = 0.10
OVERLAY_ASSETS = ["GC=F", "BTC-USD"]
INITIAL_VALUE = 10_000.0


def _load_tactical_th_signal(index: pd.DatetimeIndex) -> tuple[str, pd.Series]:
    paths = default_paths(ROOT)
    monthly = pd.read_csv(paths.result_dir / f"{RESULT_PREFIX}_monthly_returns_thb.csv", index_col=0, parse_dates=True)
    tactical_summary = pd.read_csv(paths.result_dir / f"{RESULT_PREFIX}_tactical_exit_summary_thb.csv")
    tactical_weights = pd.read_csv(paths.result_dir / f"{RESULT_PREFIX}_tactical_exit_weight_history_thb.csv", index_col=0, parse_dates=True)
    return _best_tactical_daily_weight(tactical_summary, tactical_weights, monthly, index)


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
    cached_panel = load_cached_market_data(paths, tickers=stock_tickers + ["SPY", "^VIX"])
    prices = cached_panel["prices"].loc[START_DATE:END_DATE].sort_index().ffill()
    volumes = cached_panel["volumes"].loc[START_DATE:END_DATE].sort_index().fillna(0.0)
    overlay = pd.read_parquet(paths.local_cache_root / "overlay_compare_prices.parquet")
    if "Date" in overlay.columns:
        overlay = overlay.set_index("Date")
    overlay.index = pd.to_datetime(overlay.index)
    overlay = overlay.loc[START_DATE:END_DATE].sort_index().ffill()
    fx = overlay["USDTHB=X"].reindex(prices.index).ffill()

    us_cols = [ticker for ticker in all_us if ticker in prices.columns]
    th_cols = [ticker for ticker in all_th if ticker in prices.columns]
    us_price_df = prices.reindex(columns=us_cols).mul(fx, axis=0)
    th_price_df = prices.reindex(columns=th_cols)
    overlay_asset_df = pd.DataFrame(
        {
            "GC=F": overlay["GC=F"].reindex(prices.index).ffill().mul(fx),
            "BTC-USD": overlay["BTC-USD"].reindex(prices.index).ffill().mul(fx),
        },
        index=prices.index,
    )
    thb_prices = pd.concat([us_price_df, th_price_df, overlay_asset_df], axis=1)
    benchmark = overlay["SPY"].reindex(prices.index).ffill().mul(fx).rename("benchmark")
    vol_proxy = overlay["^VIX"].reindex(prices.index).ffill().rename("vol_proxy")
    common_index = thb_prices.index.intersection(benchmark.dropna().index).intersection(vol_proxy.dropna().index)
    thb_prices = thb_prices.reindex(common_index).ffill().dropna(how="all")
    volumes = volumes.reindex(thb_prices.index).reindex(columns=thb_prices.columns).fillna(0.0)
    benchmark = benchmark.reindex(thb_prices.index).ffill()
    vol_proxy = vol_proxy.reindex(thb_prices.index).ffill()
    return thb_prices, volumes, benchmark, vol_proxy, all_us, all_th


def _metrics_row(curve: pd.Series, strategy: str, weights: pd.DataFrame) -> dict[str, object]:
    row = compute_port_opt_style_metrics(curve.dropna(), risk_free_rate=RISK_FREE_RATE).to_dict()
    row.update(
        {
            "Strategy": strategy,
            "Start": curve.dropna().index.min().date().isoformat(),
            "End": curve.dropna().index.max().date().isoformat(),
            "Average US Stock Weight": float(weights[[c for c in weights.columns if c not in OVERLAY_ASSETS and not c.endswith(".BK")]].sum(axis=1).mean()),
            "Average TH Stock Weight": float(weights[[c for c in weights.columns if c.endswith(".BK")]].sum(axis=1).mean()),
            "Average Gold Weight": float(weights["GC=F"].mean() if "GC=F" in weights else 0.0),
            "Average BTC Weight": float(weights["BTC-USD"].mean() if "BTC-USD" in weights else 0.0),
            "Average Cash / Reduced Exposure Weight": float(weights["Cash / Reduced Exposure"].mean() if "Cash / Reduced Exposure" in weights else 0.0),
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


def main() -> None:
    paths = default_paths(ROOT)
    prices, volumes, benchmark, vol_proxy, us_all, th_all = _load_full_us_th_overlay_panel_from_cache()
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    benchmark_ret = benchmark.pct_change(fill_method=None).rename("benchmark")
    vol_proxy_ret = vol_proxy.pct_change(fill_method=None).rename("vol_proxy")
    schedule = monthly_rebalance_dates(prices.index, lookback_days=LOOKBACK_DAYS, freq="ME")
    best_signal_name, th_signal = _load_tactical_th_signal(prices.index)

    raw_weight_history: dict[pd.Timestamp, pd.Series] = {}
    selected_rows = []
    nav = pd.Series(INITIAL_VALUE, index=[schedule[0]], dtype=float)
    daily_weights = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    for idx, rebalance_date in enumerate(schedule[:-1]):
        next_date = schedule[idx + 1]
        loc = prices.index.get_loc(rebalance_date)
        train_index = prices.index[max(0, loc - LOOKBACK_DAYS + 1) : loc + 1]
        test_index = prices.index[(prices.index > rebalance_date) & (prices.index <= next_date)]
        if len(test_index) == 0:
            continue

        us_pool = [
            ticker
            for ticker in get_sp500_members_as_of(rebalance_date, paths)
            if ticker in us_all and ticker in prices.columns
        ]
        us_pool = drop_duplicate_share_classes(us_pool)
        th_is_on = float(th_signal.loc[:rebalance_date].iloc[-1]) > 1e-12
        th_pool = [
            ticker
            for ticker in get_set100_members_as_of(rebalance_date, paths)
            if ticker in th_all and ticker in prices.columns
        ] if th_is_on else []
        us_selected = select_point_in_time_universe(prices.loc[train_index], volumes.loc[train_index], us_pool, n_assets=US_ASSETS)
        th_selected = select_point_in_time_universe(prices.loc[train_index], volumes.loc[train_index], th_pool, n_assets=TH_ASSETS) if th_is_on else []
        current_assets = list(dict.fromkeys(us_selected + th_selected + [asset for asset in OVERLAY_ASSETS if asset in prices.columns]))
        train_returns = returns.reindex(train_index)[current_assets].dropna(axis=1, thresh=max(int(0.85 * len(train_index)), 60))
        if train_returns.shape[1] < max(N_CLUSTERS + 2, 6):
            continue
        current_assets = train_returns.columns.tolist()
        feature_table = compute_feature_table(
            train_returns,
            benchmark_ret.reindex(train_index),
            vol_proxy_ret.reindex(train_index),
            prices.reindex(train_index)[current_assets],
            include_momentum_features=True,
            feature_flags=FEATURE_FLAGS,
        )
        if feature_table.empty:
            continue
        momentum_signal = build_momentum_signal(feature_table, mode="mom_63")
        cov = train_returns.cov().reindex(index=current_assets, columns=current_assets).fillna(0.0)
        caps = {
            asset: (GOLD_CAP if asset == "GC=F" else BTC_CAP if asset == "BTC-USD" else STOCK_CAP)
            for asset in current_assets
        }
        weights = optimize_portfolio(
            cov,
            momentum_signal,
            max_weight=max(GOLD_CAP, BTC_CAP, STOCK_CAP),
            objective_mode="mean_variance",
            asset_caps=caps,
            concentration_penalty=0.02,
            momentum_strength=1.0,
        ).reindex(current_assets).fillna(0.0)
        raw_weight_history[rebalance_date] = weights
        daily_weights.loc[test_index, :] = 0.0
        daily_weights.loc[test_index, weights.index] = pd.DataFrame(
            np.tile(weights.to_numpy(dtype=float), (len(test_index), 1)),
            index=test_index,
            columns=weights.index,
        )
        selected_rows.append(
            {
                "Date": rebalance_date.date().isoformat(),
                "TH Signal On": th_is_on,
                "US Count": len([asset for asset in current_assets if asset in us_selected]),
                "TH Count": len([asset for asset in current_assets if asset in th_selected]),
                "Overlay Count": len([asset for asset in current_assets if asset in OVERLAY_ASSETS]),
                "Assets": ",".join(current_assets),
            }
        )
        period_returns = returns.reindex(test_index)[weights.index].fillna(0.0).mul(weights, axis=1).sum(axis=1)
        period_nav = (1.0 + period_returns).cumprod().mul(float(nav.iloc[-1]))
        nav = pd.concat([nav, period_nav])

    raw_weights = daily_weights.ffill().fillna(0.0)
    raw_weights = raw_weights.loc[nav.index.intersection(raw_weights.index)]
    raw_curve = nav[~nav.index.duplicated(keep="last")].sort_index().rename("One-model US+TH signal stocks + Gold/BTC caps")
    raw_returns = raw_curve.pct_change(fill_method=None).fillna(0.0)

    _, signal_prices = _load_overlay_inputs_from_cache(raw_curve.index)
    exposure = pd.DataFrame(
        {
            "US": _close_trend_exposure(signal_prices["US Equity"], 300, 0.50),
            "TH": _close_trend_exposure(signal_prices["TH Equity"], 200, 0.00),
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
            "BTC-USD": _close_trend_exposure(signal_prices["BTC"], 50, 0.00),
        },
        index=raw_curve.index,
    ).ffill().fillna(1.0).clip(0.0, 1.0)

    aligned_returns = returns.reindex(raw_curve.index).fillna(0.0)
    effective_weights = raw_weights.reindex(raw_curve.index).ffill().fillna(0.0)
    for column in effective_weights.columns:
        if column == "GC=F":
            effective_weights[column] = effective_weights[column] * exposure["GC=F"]
        elif column == "BTC-USD":
            effective_weights[column] = effective_weights[column] * exposure["BTC-USD"]
        elif column.endswith(".BK"):
            effective_weights[column] = effective_weights[column] * exposure["TH"]
        else:
            effective_weights[column] = effective_weights[column] * exposure["US"]
    effective_weights["Cash / Reduced Exposure"] = (1.0 - effective_weights.sum(axis=1)).clip(lower=0.0)
    daily_exposed_returns = aligned_returns.reindex(columns=[c for c in effective_weights.columns if c in aligned_returns.columns]).mul(
        effective_weights[[c for c in effective_weights.columns if c in aligned_returns.columns]],
        axis=1,
    ).sum(axis=1)
    exposed_curve = curve_from_returns(daily_exposed_returns, initial=INITIAL_VALUE).rename("One-model + asset-level daily exposure")

    curves = pd.concat([raw_curve, exposed_curve], axis=1).dropna(how="all")
    raw_weights_out = raw_weights.copy()
    raw_weights_out["Cash / Reduced Exposure"] = 0.0
    summary = pd.DataFrame(
        [
            _metrics_row(raw_curve, raw_curve.name, raw_weights_out.reindex(raw_curve.index).ffill().fillna(0.0)),
            _metrics_row(exposed_curve, exposed_curve.name, effective_weights.reindex(exposed_curve.index).ffill().fillna(0.0)),
        ]
    ).sort_values(["Sharpe", "CAGR"], ascending=False)
    latest_weights = effective_weights.iloc[-1].rename("Effective Weight").reset_index().rename(columns={"index": "Asset"})
    latest_weights = latest_weights.loc[latest_weights["Effective Weight"].abs().gt(1e-12)].sort_values("Effective Weight", ascending=False)
    latest_weights["Date"] = effective_weights.index.max().date().isoformat()

    summary["Tactical Signal"] = best_signal_name
    summary["US Assets"] = US_ASSETS
    summary["TH Assets When Signal On"] = TH_ASSETS
    summary["Stock Cap"] = STOCK_CAP
    summary["Gold Cap"] = GOLD_CAP
    summary["BTC Cap"] = BTC_CAP
    summary.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_summary_thb.csv", index=False)
    curves.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_curves_thb.csv")
    pd.DataFrame(selected_rows).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_universe_history_thb.csv", index=False)
    effective_weights.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_effective_weights_thb.csv")
    latest_weights.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_latest_weights_thb.csv", index=False)
    _period_compare(curves).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_period_compare_thb.csv", index=False)

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
    print(latest_weights.head(40).to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
