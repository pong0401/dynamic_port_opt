from __future__ import annotations

from pathlib import Path
import os
import sys
import warnings

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynamic_factor_copula import (  # noqa: E402
    backtest_dynamic_factor_copula,
    compute_port_opt_style_metrics,
    curve_from_returns,
    default_paths,
    load_overlay_compare_prices,
)
from us_th_pit_reselect_utils import load_full_us_th_thb_panel, run_joint_pit_reselect_model  # noqa: E402


START_DATE = "2005-01-01"
END_DATE = "2026-04-29"
RISK_FREE_RATE = 0.03
FEATURE_FLAGS = {"resid_vol": False, "drawdown": False, "downside_beta": False}
PRIMARY_MODEL = "Dynamic HMM Copula"
RESULT_PREFIX = "us_th_tactical_perf_momentum"
OVERLAY_MIX = {"Equity": 0.60, "Gold": 0.30, "BTC": 0.10}

LOOKBACK_MONTHS = [1, 2, 3, 6, 12]
FORWARD_MONTHS = [1, 2, 3, 6]
TH_WEIGHTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
ENTRY_THRESHOLDS = [0.00, 0.01, 0.02]
EXIT_THRESHOLDS = [-0.02, -0.01, 0.00]
MIN_HOLD_MONTHS = [0, 1, 2, 3]
EXIT_CONFIRM_MONTHS = [1, 2]
SIGNAL_MODES = [
    "relative_return",
    "relative_sharpe",
    "relative_return_and_sharpe",
    "relative_return_or_sharpe",
    "relative_return_positive_th",
    "positive_th_return",
    "score_3_of_5",
]
ALLOCATION_METHODS = [
    "binary",
    "return_spread_scaled",
    "sharpe_spread_scaled",
    "inverse_vol",
    "return_spread_inverse_vol",
    "score_scaled",
]
RETURN_SPREAD_SCALE = 0.05
SHARPE_SPREAD_SCALE = 1.0


def _run_th_sleeve() -> dict[str, object]:
    return backtest_dynamic_factor_copula(
        start_date=START_DATE,
        end_date=END_DATE,
        n_assets=30,
        n_clusters=4,
        lookback_days=504,
        rebalance_freq="ME",
        max_weight=0.08,
        point_in_time_liquid=True,
        universe_mode="set100_pit",
        benchmark_ticker="^SET.BK",
        vol_proxy_ticker="",
        include_momentum=True,
        include_momentum_features=True,
        include_momentum_signal=True,
        momentum_signal_mode="mom_63",
        optimizer_objective="mean_variance",
        feature_flags=FEATURE_FLAGS,
        paths=default_paths(ROOT),
    )


def _run_us_sleeve() -> dict[str, object]:
    prices, volumes, benchmark, vol_proxy, us_all, _ = load_full_us_th_thb_panel(
        include_overlay_assets=False,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    return run_joint_pit_reselect_model(
        prices=prices,
        volumes=volumes,
        benchmark=benchmark,
        vol_proxy=vol_proxy,
        us_all=us_all,
        th_all=[],
        us_assets=30,
        th_assets=0,
        objective_mode="mean_variance",
        max_weight=0.08,
        include_overlay_assets=False,
        include_momentum=True,
        momentum_signal_mode="mom_63",
    )


def _nav_to_returns(nav: pd.Series) -> pd.Series:
    return nav.dropna().pct_change(fill_method=None).fillna(0.0)


def _load_fx(index: pd.DatetimeIndex) -> pd.Series:
    paths = default_paths(ROOT)
    fx = load_overlay_compare_prices(paths, start_date=START_DATE, end_date=END_DATE, tickers=["USDTHB=X"])["USDTHB=X"]
    return fx.reindex(index).ffill()


def _load_benchmarks(index: pd.DatetimeIndex) -> pd.DataFrame:
    paths = default_paths(ROOT)
    overlay = load_overlay_compare_prices(paths, start_date=START_DATE, end_date=END_DATE, tickers=["SPY", "USDTHB=X"]).sort_index().ffill()
    spy_source = overlay["SPY"].mul(overlay["USDTHB=X"])
    spy_thb = spy_source.reindex(spy_source.index.union(index)).sort_index().ffill().reindex(index)
    set_source = pd.read_parquet(paths.local_cache_root / "extra_prices.parquet")["^SET.BK"].loc[START_DATE:END_DATE]
    set_index = set_source.reindex(set_source.index.union(index)).sort_index().ffill().reindex(index)
    return pd.DataFrame(
        {
            "S&P 500 ETF THB": spy_thb,
            "SET Index THB proxy": set_index,
        }
    ).dropna(how="all")


def _monthly_returns(curves: pd.DataFrame) -> pd.DataFrame:
    month_end = curves.resample("ME").last().dropna(how="all")
    return month_end.pct_change(fill_method=None).dropna(how="all")


def _rolling_monthly_sharpe(monthly_returns: pd.DataFrame, months: int) -> pd.DataFrame:
    mean = monthly_returns.rolling(months, min_periods=months).mean()
    vol = monthly_returns.rolling(months, min_periods=months).std(ddof=0)
    return (mean / vol.replace(0.0, np.nan)) * np.sqrt(12.0)


def _forward_compound_return(monthly_returns: pd.DataFrame, months: int) -> pd.DataFrame:
    values = {}
    for column in monthly_returns:
        compounded = (1.0 + monthly_returns[column]).shift(-1).rolling(months, min_periods=months).apply(np.prod, raw=True) - 1.0
        values[column] = compounded.shift(-(months - 1))
    return pd.DataFrame(values, index=monthly_returns.index)


def _monthly_performance_table(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rolling_returns = {
        months: (1.0 + monthly).rolling(months, min_periods=months).apply(np.prod, raw=True) - 1.0
        for months in LOOKBACK_MONTHS
    }
    rolling_sharpes = {months: _rolling_monthly_sharpe(monthly, months) for months in LOOKBACK_MONTHS}
    for date, row in monthly.iterrows():
        for sleeve, value in row.dropna().items():
            out = {
                "Month": date.date().isoformat(),
                "Series": sleeve,
                "Monthly Return": float(value),
            }
            for months in LOOKBACK_MONTHS:
                out[f"Rolling {months}M Return"] = float(rolling_returns[months].loc[date, sleeve]) if sleeve in rolling_returns[months] and pd.notna(rolling_returns[months].loc[date, sleeve]) else np.nan
                out[f"Rolling {months}M Sharpe"] = float(rolling_sharpes[months].loc[date, sleeve]) if sleeve in rolling_sharpes[months] and pd.notna(rolling_sharpes[months].loc[date, sleeve]) else np.nan
            rows.append(out)
    return pd.DataFrame(rows)


def _persistence_table(
    monthly: pd.DataFrame,
    us_col: str,
    th_col: str,
    signal_prefix: str,
) -> pd.DataFrame:
    rows = []
    pair = monthly[[us_col, th_col]].dropna()
    for lookback in LOOKBACK_MONTHS:
        trailing_return = (1.0 + pair).rolling(lookback, min_periods=lookback).apply(np.prod, raw=True) - 1.0
        trailing_sharpe = _rolling_monthly_sharpe(pair, lookback)
        signals = {
            f"{signal_prefix} trailing return > US": trailing_return[th_col] > trailing_return[us_col],
            f"{signal_prefix} trailing Sharpe > US": trailing_sharpe[th_col] > trailing_sharpe[us_col],
            f"{signal_prefix} trailing return and Sharpe > US": (
                (trailing_return[th_col] > trailing_return[us_col])
                & (trailing_sharpe[th_col] > trailing_sharpe[us_col])
            ),
        }
        for horizon in FORWARD_MONTHS:
            forward = _forward_compound_return(pair, horizon)
            forward_spread = forward[th_col] - forward[us_col]
            for signal_name, signal in signals.items():
                signal_mask = signal.shift(1).fillna(False).astype(bool)
                sample = forward_spread.loc[signal_mask].dropna()
                all_sample = forward_spread.dropna()
                rows.append(
                    {
                        "Signal": signal_name,
                        "Signal Source": signal_prefix,
                        "Lookback Months": lookback,
                        "Forward Months": horizon,
                        "Signal Count": int(len(sample)),
                        "Hit Rate": float((sample > 0.0).mean()) if len(sample) else np.nan,
                        "Average Forward TH-US Return": float(sample.mean()) if len(sample) else np.nan,
                        "Median Forward TH-US Return": float(sample.median()) if len(sample) else np.nan,
                        "All Months Average TH-US Return": float(all_sample.mean()) if len(all_sample) else np.nan,
                    }
                )
    return pd.DataFrame(rows).sort_values(["Forward Months", "Average Forward TH-US Return"], ascending=[True, False])


def _signal_inputs(pair: pd.DataFrame, us_col: str, th_col: str, lookback: int) -> dict[str, pd.Series]:
    trailing_return = (1.0 + pair).rolling(lookback, min_periods=lookback).apply(np.prod, raw=True) - 1.0
    trailing_sharpe = _rolling_monthly_sharpe(pair, lookback)
    trailing_vol = pair.rolling(lookback, min_periods=lookback).std(ddof=0) * np.sqrt(12.0)
    curves = (1.0 + pair).cumprod()
    ma_window = max(3, min(12, lookback * 2))
    th_curve = curves[th_col]
    th_drawdown = th_curve.div(th_curve.cummax()).sub(1.0)
    return_spread = trailing_return[th_col] - trailing_return[us_col]
    sharpe_spread = trailing_sharpe[th_col] - trailing_sharpe[us_col]
    th_above_ma = th_curve > th_curve.rolling(ma_window, min_periods=ma_window).mean()
    drawdown_improving = th_drawdown > th_drawdown.shift(1)
    score = (
        (return_spread > 0.0).astype(float)
        + (sharpe_spread > 0.0).astype(float)
        + (trailing_return[th_col] > 0.0).astype(float)
        + th_above_ma.astype(float)
        + drawdown_improving.astype(float)
    )
    return {
        "return_spread": return_spread,
        "sharpe_spread": sharpe_spread,
        "th_return": trailing_return[th_col],
        "us_vol": trailing_vol[us_col],
        "th_vol": trailing_vol[th_col],
        "score": score,
    }


def _raw_signal(inputs: dict[str, pd.Series], mode: str, entry: float) -> pd.Series:
    return_spread = inputs["return_spread"]
    sharpe_spread = inputs["sharpe_spread"]
    th_return = inputs["th_return"]
    score = inputs["score"]
    if mode == "relative_return":
        return return_spread > entry
    if mode == "relative_sharpe":
        return sharpe_spread > entry
    if mode == "relative_return_and_sharpe":
        return (return_spread > entry) & (sharpe_spread > entry)
    if mode == "relative_return_or_sharpe":
        return (return_spread > entry) | (sharpe_spread > entry)
    if mode == "relative_return_positive_th":
        return (return_spread > entry) & (th_return > 0.0)
    if mode == "positive_th_return":
        return th_return > entry
    if mode == "score_3_of_5":
        return score >= 3.0
    raise ValueError(f"Unknown signal mode: {mode}")


def _exit_value(inputs: dict[str, pd.Series], mode: str) -> pd.Series:
    if mode in {
        "relative_return",
        "relative_return_and_sharpe",
        "relative_return_or_sharpe",
        "relative_return_positive_th",
    }:
        return inputs["return_spread"]
    if mode == "positive_th_return":
        return inputs["th_return"]
    if mode == "score_3_of_5":
        return inputs["score"] - 3.0
    return inputs["sharpe_spread"]


def _stateful_gate(
    inputs: dict[str, pd.Series],
    mode: str,
    entry: float,
    exit_threshold: float,
    min_hold: int,
    exit_confirm: int,
) -> pd.Series:
    raw_entry = _raw_signal(inputs, mode, entry).fillna(False)
    exit_metric = _exit_value(inputs, mode)
    active = False
    hold = 0
    bad_count = 0
    values = []
    for date in raw_entry.index:
        if active:
            hold += 1
            if bool(exit_metric.loc[date] < exit_threshold):
                bad_count += 1
            else:
                bad_count = 0
            if hold >= min_hold and bad_count >= exit_confirm:
                active = False
                hold = 0
                bad_count = 0
        elif bool(raw_entry.loc[date]):
            active = True
            hold = 1
            bad_count = 0
        values.append(1.0 if active else 0.0)
    return pd.Series(values, index=raw_entry.index, name="TH Gate")


def _monthly_weight_signal(
    monthly: pd.DataFrame,
    mode: str,
    allocation_method: str,
    lookback: int,
    th_weight: float,
    entry: float,
    exit_threshold: float,
    min_hold: int,
    exit_confirm: int,
    us_col: str = "US PIT optimized sleeve THB",
    th_col: str = "TH PIT optimized sleeve THB",
    inputs: dict[str, pd.Series] | None = None,
) -> pd.Series:
    if inputs is None:
        pair = monthly[[us_col, th_col]].dropna()
        inputs = _signal_inputs(pair, us_col, th_col, lookback)
    gate = _stateful_gate(inputs, mode, entry, exit_threshold, min_hold, exit_confirm)
    gate = gate.shift(1).fillna(0.0)
    return_spread = inputs["return_spread"].shift(1)
    sharpe_spread = inputs["sharpe_spread"].shift(1)
    score = inputs["score"].shift(1)
    vol_ratio = inputs["us_vol"].shift(1).div(inputs["th_vol"].shift(1).replace(0.0, np.nan)).clip(upper=1.0)

    if allocation_method == "binary":
        scale = pd.Series(1.0, index=gate.index)
    elif allocation_method == "return_spread_scaled":
        scale = return_spread.clip(lower=0.0).div(RETURN_SPREAD_SCALE).clip(upper=1.0)
    elif allocation_method == "sharpe_spread_scaled":
        scale = sharpe_spread.clip(lower=0.0).div(SHARPE_SPREAD_SCALE).clip(upper=1.0)
    elif allocation_method == "inverse_vol":
        scale = vol_ratio
    elif allocation_method == "return_spread_inverse_vol":
        scale = return_spread.clip(lower=0.0).div(RETURN_SPREAD_SCALE).clip(upper=1.0).mul(vol_ratio)
    elif allocation_method == "score_scaled":
        scale = score.clip(lower=0.0, upper=5.0).div(5.0)
    else:
        raise ValueError(f"Unknown allocation method: {allocation_method}")
    return (th_weight * gate * scale.fillna(0.0)).rename("TH Weight")


def _daily_weight_from_monthly(monthly_weight: pd.Series, daily_index: pd.DatetimeIndex) -> pd.Series:
    month_key = daily_index.to_period("M").to_timestamp("M")
    mapped = pd.Series(month_key, index=daily_index).map(monthly_weight)
    return mapped.ffill().fillna(0.0).clip(0.0, 1.0)


def _close_trend_exposure(price: pd.Series, ma_period: int, below_exposure: float, initial: float = 1.0) -> pd.Series:
    price = price.astype(float).sort_index().ffill()
    min_periods = max(20, int(ma_period * 0.20))
    ma = price.rolling(ma_period, min_periods=min_periods).mean()
    signal = pd.Series(1.0, index=price.index, dtype=float)
    signal.loc[price < ma] = below_exposure
    signal.loc[ma.isna()] = initial
    return signal.shift(1).fillna(initial).rename("Daily Exposure")


def _metrics_row(curve: pd.Series, strategy: str, weight_history: pd.DataFrame | None = None) -> dict[str, object]:
    clean = curve.dropna()
    row = compute_port_opt_style_metrics(clean, risk_free_rate=RISK_FREE_RATE).to_dict()
    row["Strategy"] = strategy
    row["Start"] = clean.index.min().date().isoformat() if not clean.empty else ""
    row["End"] = clean.index.max().date().isoformat() if not clean.empty else ""
    if weight_history is not None and not weight_history.empty:
        for column in ["US Equity", "TH Equity", "Gold", "BTC", "Cash / Reduced Exposure"]:
            row[f"Average {column} Weight"] = float(weight_history.get(column, pd.Series(0.0, index=weight_history.index)).mean())
    return row


def _evaluate_tactical(
    daily_returns: pd.DataFrame,
    monthly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    baseline_curves: dict[str, pd.Series] = {
        "US PIT optimized sleeve THB": curve_from_returns(daily_returns["US PIT optimized sleeve THB"]),
        "TH PIT optimized sleeve THB": curve_from_returns(daily_returns["TH PIT optimized sleeve THB"]),
    }
    baseline_weights: dict[str, pd.Series] = {}
    top_items: list[tuple[float, float, str, pd.Series, pd.Series]] = []

    def add_top_candidate(strategy: str, curve: pd.Series, daily_weight: pd.Series, row: dict[str, object]) -> None:
        sharpe = float(row.get("Sharpe", np.nan))
        cagr = float(row.get("CAGR", np.nan))
        if np.isnan(sharpe):
            return
        top_items.append((sharpe, cagr, strategy, curve, daily_weight.rename(strategy)))
        top_items.sort(key=lambda item: (item[0], item[1]), reverse=True)
        del top_items[40:]

    for baseline_strategy, column, th_weight_value in [
        ("US PIT optimized sleeve THB", "US PIT optimized sleeve THB", 0.0),
        ("TH PIT optimized sleeve THB", "TH PIT optimized sleeve THB", 1.0),
    ]:
        curve = baseline_curves[baseline_strategy]
        row = compute_port_opt_style_metrics(curve, risk_free_rate=RISK_FREE_RATE).to_dict()
        row.update(
            {
                "Strategy": baseline_strategy,
                "Signal Mode": "baseline",
                "Allocation Method": "baseline",
                "Signal Source": "baseline",
                "Lookback Months": 0,
                "TH Weight Cap": th_weight_value,
                "Entry Threshold": np.nan,
                "Exit Threshold": np.nan,
                "Min Hold Months": 0,
                "Exit Confirm Months": 0,
                "Average TH Weight": th_weight_value,
                "TH On Months": int(monthly.shape[0] if th_weight_value > 0 else 0),
            }
        )
        rows.append(row)
        baseline_weights[baseline_strategy] = pd.Series(th_weight_value, index=daily_returns.index, name=column)

    for fixed_weight in [0.00, 0.05, 0.10, 0.15, 0.20]:
        returns = (1.0 - fixed_weight) * daily_returns["US PIT optimized sleeve THB"] + fixed_weight * daily_returns["TH PIT optimized sleeve THB"]
        strategy = f"Fixed TH sleeve {int(fixed_weight * 100)}"
        curve = curve_from_returns(returns)
        row = compute_port_opt_style_metrics(curve, risk_free_rate=RISK_FREE_RATE).to_dict()
        row.update(
            {
                "Strategy": strategy,
                "Signal Mode": "fixed",
                "Allocation Method": "fixed",
                "Signal Source": "fixed",
                "Lookback Months": 0,
                "TH Weight Cap": fixed_weight,
                "Entry Threshold": np.nan,
                "Exit Threshold": np.nan,
                "Min Hold Months": 0,
                "Exit Confirm Months": 0,
                "Average TH Weight": fixed_weight,
                "TH On Months": int(monthly.shape[0]),
            }
        )
        rows.append(row)
        add_top_candidate(strategy, curve, pd.Series(fixed_weight, index=daily_returns.index, name=strategy), row)

    signal_sources = [
        ("sleeve_performance", "US PIT optimized sleeve THB", "TH PIT optimized sleeve THB"),
        ("proxy_regime", "S&P 500 ETF THB", "SET Index THB proxy"),
    ]
    for signal_source, us_col, th_col in signal_sources:
        if us_col not in monthly.columns or th_col not in monthly.columns:
            continue
        pair = monthly[[us_col, th_col]].dropna()
        input_by_lookback = {
            lookback: _signal_inputs(pair, us_col, th_col, lookback)
            for lookback in LOOKBACK_MONTHS
        }
        for lookback in LOOKBACK_MONTHS:
            inputs = input_by_lookback[lookback]
            for mode in SIGNAL_MODES:
                for allocation_method in ALLOCATION_METHODS:
                    for th_weight in TH_WEIGHTS:
                        for entry in ENTRY_THRESHOLDS:
                            for exit_threshold in EXIT_THRESHOLDS:
                                if mode == "score_3_of_5" and exit_threshold > 0.0:
                                    continue
                                if mode != "score_3_of_5" and exit_threshold > entry:
                                    continue
                                for min_hold in MIN_HOLD_MONTHS:
                                    for exit_confirm in EXIT_CONFIRM_MONTHS:
                                        monthly_weight = _monthly_weight_signal(
                                            monthly,
                                            mode,
                                            allocation_method,
                                            lookback,
                                            th_weight,
                                            entry,
                                            exit_threshold,
                                            min_hold,
                                            exit_confirm,
                                            us_col=us_col,
                                            th_col=th_col,
                                            inputs=inputs,
                                        )
                                        daily_weight = _daily_weight_from_monthly(monthly_weight, daily_returns.index)
                                        returns = (1.0 - daily_weight) * daily_returns["US PIT optimized sleeve THB"] + daily_weight * daily_returns["TH PIT optimized sleeve THB"]
                                        strategy = (
                                            f"Tactical TH {signal_source} {mode} {allocation_method} lb{lookback} "
                                            f"cap{int(th_weight * 100)} entry{entry:.0%} exit{exit_threshold:.0%} "
                                            f"hold{min_hold} confirm{exit_confirm}"
                                        )
                                        curve = curve_from_returns(returns)
                                        row = compute_port_opt_style_metrics(curve, risk_free_rate=RISK_FREE_RATE).to_dict()
                                        row.update(
                                            {
                                                "Strategy": strategy,
                                                "Signal Mode": mode,
                                                "Allocation Method": allocation_method,
                                                "Signal Source": signal_source,
                                                "Lookback Months": lookback,
                                                "TH Weight Cap": th_weight,
                                                "Entry Threshold": entry,
                                                "Exit Threshold": exit_threshold,
                                                "Min Hold Months": min_hold,
                                                "Exit Confirm Months": exit_confirm,
                                                "Average TH Weight": float(monthly_weight.mean()),
                                                "TH On Months": int((monthly_weight > 0.0).sum()),
                                            }
                                        )
                                        rows.append(row)
                                        add_top_candidate(strategy, curve, daily_weight, row)

    summary = pd.DataFrame(rows).sort_values(["Sharpe", "CAGR"], ascending=False)
    top = summary.head(30)["Strategy"].tolist()
    top_curve_map = {strategy: curve for _, _, strategy, curve, _ in top_items}
    top_weight_map = {strategy: weight for _, _, strategy, _, weight in top_items}
    keep_curves = ["US PIT optimized sleeve THB"] + [name for name in top if name in top_curve_map]
    curves_df = pd.DataFrame(
        {"US PIT optimized sleeve THB": baseline_curves["US PIT optimized sleeve THB"]}
        | {name: top_curve_map[name] for name in keep_curves if name in top_curve_map}
    ).dropna(how="all")
    weight_df = pd.DataFrame({name: top_weight_map[name] for name in top if name in top_weight_map}).dropna(how="all")
    return summary, curves_df, weight_df


def _best_tactical_daily_weight(
    summary: pd.DataFrame,
    tactical_weight_history: pd.DataFrame,
    monthly: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
) -> tuple[str, pd.Series]:
    best_row = summary.sort_values(["Sharpe", "CAGR"], ascending=False).iloc[0]
    best_strategy = str(best_row["Strategy"])
    if best_strategy in tactical_weight_history.columns:
        return best_strategy, tactical_weight_history[best_strategy].reindex(daily_index).ffill().fillna(0.0).clip(0.0, 1.0)

    source_map = {
        "sleeve_performance": ("US PIT optimized sleeve THB", "TH PIT optimized sleeve THB"),
        "proxy_regime": ("S&P 500 ETF THB", "SET Index THB proxy"),
    }
    us_col, th_col = source_map[str(best_row["Signal Source"])]
    monthly_weight = _monthly_weight_signal(
        monthly,
        str(best_row["Signal Mode"]),
        str(best_row["Allocation Method"]),
        int(best_row["Lookback Months"]),
        float(best_row["TH Weight Cap"]),
        float(best_row["Entry Threshold"]),
        float(best_row["Exit Threshold"]),
        int(best_row["Min Hold Months"]),
        int(best_row["Exit Confirm Months"]),
        us_col=us_col,
        th_col=th_col,
    )
    return best_strategy, _daily_weight_from_monthly(monthly_weight, daily_index)


def _load_gold_btc_overlay_inputs(index: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = default_paths(ROOT)
    overlay = load_overlay_compare_prices(
        paths,
        start_date=START_DATE,
        end_date=END_DATE,
        tickers=["SPY", "GC=F", "BTC-USD", "USDTHB=X"],
    ).sort_index().ffill()
    fx = overlay["USDTHB=X"].ffill()
    set_index = pd.read_parquet(paths.local_cache_root / "extra_prices.parquet")["^SET.BK"].loc[START_DATE:END_DATE].sort_index().ffill()
    thb_prices = pd.DataFrame(
        {
            "Gold": overlay["GC=F"].mul(fx),
            "BTC": overlay["BTC-USD"].mul(fx),
        }
    )
    signal_prices = pd.DataFrame(
        {
            "US Equity": overlay["SPY"],
            "TH Equity": set_index,
            "Gold": overlay["GC=F"],
            "BTC": overlay["BTC-USD"],
        }
    )
    full_index = index.union(thb_prices.index).union(signal_prices.index).sort_values()
    thb_prices = thb_prices.reindex(full_index).ffill().reindex(index)
    signal_prices = signal_prices.reindex(full_index).ffill().reindex(index)
    return thb_prices, signal_prices


def _evaluate_gold_btc_overlay(
    daily_returns: pd.DataFrame,
    monthly: pd.DataFrame,
    tactical_summary: pd.DataFrame,
    tactical_weight_history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = daily_returns.index
    best_strategy, th_tactical_weight = _best_tactical_daily_weight(tactical_summary, tactical_weight_history, monthly, index)
    overlay_prices, signal_prices = _load_gold_btc_overlay_inputs(index)
    asset_returns = pd.DataFrame(
        {
            "US Equity": daily_returns["US PIT optimized sleeve THB"],
            "TH Equity": daily_returns["TH PIT optimized sleeve THB"],
            "Gold": overlay_prices["Gold"].pct_change(fill_method=None).fillna(0.0),
            "BTC": overlay_prices["BTC"].pct_change(fill_method=None).fillna(0.0),
        },
        index=index,
    ).fillna(0.0)

    raw_weights = {
        "US/Gold/BTC 60/30/10 no daily exposure": pd.DataFrame(
            {
                "US Equity": OVERLAY_MIX["Equity"],
                "TH Equity": 0.0,
                "Gold": OVERLAY_MIX["Gold"],
                "BTC": OVERLAY_MIX["BTC"],
            },
            index=index,
        ),
        f"Tactical TH/Gold/BTC 60/30/10 no daily exposure ({best_strategy})": pd.DataFrame(
            {
                "US Equity": OVERLAY_MIX["Equity"] * (1.0 - th_tactical_weight),
                "TH Equity": OVERLAY_MIX["Equity"] * th_tactical_weight,
                "Gold": OVERLAY_MIX["Gold"],
                "BTC": OVERLAY_MIX["BTC"],
            },
            index=index,
        ),
    }

    exposure = pd.DataFrame(
        {
            "US Equity": _close_trend_exposure(signal_prices["US Equity"], 300, 0.50),
            "TH Equity": _close_trend_exposure(signal_prices["TH Equity"], 200, 0.00),
            "Gold": _close_trend_exposure(signal_prices["Gold"], 50, 1.00),
            "BTC": _close_trend_exposure(signal_prices["BTC"], 50, 0.00),
        },
        index=index,
    ).ffill().fillna(1.0).clip(0.0, 1.0)

    rows = []
    curves: dict[str, pd.Series] = {}
    weight_frames: dict[str, pd.DataFrame] = {}
    for strategy, weights in raw_weights.items():
        weights = weights.reindex(index).ffill().fillna(0.0)
        returns = asset_returns.mul(weights, axis=0).sum(axis=1)
        curve = curve_from_returns(returns)
        rows.append(_metrics_row(curve, strategy, weights))
        curves[strategy] = curve
        weight_frames[strategy] = weights.assign(**{"Cash / Reduced Exposure": 0.0})

        daily_strategy = strategy.replace("no daily exposure", "asset-level daily exposure")
        effective = weights.mul(exposure, axis=0)
        effective["Cash / Reduced Exposure"] = (1.0 - effective[["US Equity", "TH Equity", "Gold", "BTC"]].sum(axis=1)).clip(lower=0.0)
        exposed_returns = asset_returns.mul(effective[["US Equity", "TH Equity", "Gold", "BTC"]], axis=0).sum(axis=1)
        exposed_curve = curve_from_returns(exposed_returns)
        rows.append(_metrics_row(exposed_curve, daily_strategy, effective))
        curves[daily_strategy] = exposed_curve
        weight_frames[daily_strategy] = effective

    summary = pd.DataFrame(rows).sort_values(["Sharpe", "CAGR"], ascending=False)
    curves_df = pd.DataFrame(curves).dropna(how="all")
    stacked_weights = []
    for strategy, frame in weight_frames.items():
        out = frame.copy()
        out.insert(0, "Strategy", strategy)
        stacked_weights.append(out.reset_index(names="Date"))
    weights_df = pd.concat(stacked_weights, ignore_index=True)
    period_compare = _period_compare(curves_df)
    return summary, curves_df, weights_df, period_compare


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
    return pd.DataFrame(rows).sort_values(["Period", "Sharpe"], ascending=[True, False])


def _build_daily_series() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = default_paths(ROOT)
    us_results = _run_us_sleeve()
    th_results = _run_th_sleeve()

    us_nav_thb = us_results["nav"][PRIMARY_MODEL].dropna()
    th_nav_thb = th_results["nav"][PRIMARY_MODEL].dropna()
    common_index = us_nav_thb.index.union(th_nav_thb.index).sort_values()
    us_returns_thb = _nav_to_returns(us_nav_thb).reindex(common_index).fillna(0.0).rename("US PIT optimized sleeve THB")
    th_returns = _nav_to_returns(th_nav_thb).reindex(common_index).fillna(0.0).rename("TH PIT optimized sleeve THB")
    daily_returns = pd.concat([us_returns_thb, th_returns], axis=1).loc[START_DATE:END_DATE].dropna(how="all").fillna(0.0)

    benchmark_prices = _load_benchmarks(daily_returns.index)
    benchmark_curves = benchmark_prices.div(benchmark_prices.iloc[0]).mul(10_000.0)
    sleeve_curves = pd.DataFrame(
        {
            "US PIT optimized sleeve THB": curve_from_returns(daily_returns["US PIT optimized sleeve THB"]),
            "TH PIT optimized sleeve THB": curve_from_returns(daily_returns["TH PIT optimized sleeve THB"]),
        }
    )
    curves = pd.concat([sleeve_curves, benchmark_curves], axis=1).dropna(how="all")
    monthly = _monthly_returns(curves)

    latest_rows = []
    for side, results in [("US", us_results), ("TH", th_results)]:
        weights_history = results["weights_history"][PRIMARY_MODEL]
        latest_date = max(weights_history)
        latest = weights_history[latest_date].rename("Internal Weight").reset_index()
        latest.columns = ["Asset", "Internal Weight"]
        latest["Sleeve"] = side
        latest["Date"] = pd.Timestamp(latest_date).date().isoformat()
        latest_rows.append(latest)
    pd.concat(latest_rows, ignore_index=True).sort_values(["Sleeve", "Internal Weight"], ascending=[True, False]).to_csv(
        paths.result_dir / f"{RESULT_PREFIX}_latest_internal_weights.csv",
        index=False,
    )
    return daily_returns, curves, monthly


def _load_existing_daily_series() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = default_paths(ROOT)
    curves_path = paths.result_dir / f"{RESULT_PREFIX}_comparison_curves_thb.csv"
    monthly_path = paths.result_dir / f"{RESULT_PREFIX}_monthly_returns_thb.csv"
    if not curves_path.exists() or not monthly_path.exists():
        raise FileNotFoundError("Existing comparison curves/monthly returns are required for reuse mode.")
    curves = pd.read_csv(curves_path, index_col=0, parse_dates=True)
    monthly = pd.read_csv(monthly_path, index_col=0, parse_dates=True)
    daily_returns = curves[["US PIT optimized sleeve THB", "TH PIT optimized sleeve THB"]].pct_change(fill_method=None).fillna(0.0)
    return daily_returns, curves, monthly


def _data_audit(daily_returns: pd.DataFrame, curves: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for frame_name, frame in [("Daily Returns", daily_returns), ("Curves", curves), ("Monthly Returns", monthly)]:
        for column in frame.columns:
            clean = frame[column].dropna()
            rows.append(
                {
                    "Frame": frame_name,
                    "Series": column,
                    "Start": clean.index.min().date().isoformat() if not clean.empty else "",
                    "End": clean.index.max().date().isoformat() if not clean.empty else "",
                    "Observations": int(clean.shape[0]),
                }
            )
    common = pd.concat(
        [
            daily_returns[["US PIT optimized sleeve THB", "TH PIT optimized sleeve THB"]],
            curves[["S&P 500 ETF THB", "SET Index THB proxy"]],
        ],
        axis=1,
    ).dropna()
    rows.append(
        {
            "Frame": "Common Window",
            "Series": "US sleeve + TH sleeve + S&P proxy + SET proxy",
            "Start": common.index.min().date().isoformat() if not common.empty else "",
            "End": common.index.max().date().isoformat() if not common.empty else "",
            "Observations": int(common.shape[0]),
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    paths = default_paths(ROOT)
    paths.result_dir.mkdir(parents=True, exist_ok=True)

    if os.environ.get("TACTICAL_REUSE_SERIES_ONLY") == "1":
        daily_returns, comparison_curves, monthly = _load_existing_daily_series()
    else:
        daily_returns, comparison_curves, monthly = _build_daily_series()
    overlay_only = os.environ.get("TACTICAL_OVERLAY_ONLY") == "1"
    monthly_table = pd.DataFrame()
    data_audit = pd.DataFrame()
    persistence = pd.DataFrame()
    tactical_curves = pd.DataFrame()
    if overlay_only:
        tactical_summary = pd.read_csv(paths.result_dir / f"{RESULT_PREFIX}_tactical_exit_summary_thb.csv")
        tactical_weight_history = pd.read_csv(
            paths.result_dir / f"{RESULT_PREFIX}_tactical_exit_weight_history_thb.csv",
            index_col=0,
            parse_dates=True,
        )
    else:
        monthly_table = _monthly_performance_table(monthly)
        data_audit = _data_audit(daily_returns, comparison_curves, monthly)
        persistence = pd.concat(
            [
                _persistence_table(
                    monthly,
                    "US PIT optimized sleeve THB",
                    "TH PIT optimized sleeve THB",
                    "sleeve_performance",
                ),
                _persistence_table(
                    monthly,
                    "S&P 500 ETF THB",
                    "SET Index THB proxy",
                    "proxy_regime",
                ),
            ],
            ignore_index=True,
        ).sort_values(["Forward Months", "Average Forward TH-US Return"], ascending=[True, False])
        tactical_summary, tactical_curves, tactical_weight_history = _evaluate_tactical(daily_returns, monthly)
    overlay_summary, overlay_curves, overlay_weight_history, overlay_period_compare = _evaluate_gold_btc_overlay(
        daily_returns,
        monthly,
        tactical_summary,
        tactical_weight_history,
    )

    if not overlay_only:
        data_audit.to_csv(paths.result_dir / f"{RESULT_PREFIX}_data_audit_thb.csv", index=False)
        comparison_curves.to_csv(paths.result_dir / f"{RESULT_PREFIX}_comparison_curves_thb.csv")
        monthly.to_csv(paths.result_dir / f"{RESULT_PREFIX}_monthly_returns_thb.csv")
        monthly_table.to_csv(paths.result_dir / f"{RESULT_PREFIX}_monthly_performance_table_thb.csv", index=False)
        persistence.to_csv(paths.result_dir / f"{RESULT_PREFIX}_persistence_thb.csv", index=False)
        tactical_summary.to_csv(paths.result_dir / f"{RESULT_PREFIX}_tactical_exit_summary_thb.csv", index=False)
        tactical_curves.to_csv(paths.result_dir / f"{RESULT_PREFIX}_tactical_exit_curves_thb.csv")
        tactical_weight_history.to_csv(paths.result_dir / f"{RESULT_PREFIX}_tactical_exit_weight_history_thb.csv")
    overlay_summary.to_csv(paths.result_dir / f"{RESULT_PREFIX}_gold_btc_overlay_summary_thb.csv", index=False)
    overlay_curves.to_csv(paths.result_dir / f"{RESULT_PREFIX}_gold_btc_overlay_curves_thb.csv")
    overlay_weight_history.to_csv(paths.result_dir / f"{RESULT_PREFIX}_gold_btc_overlay_weight_history_thb.csv", index=False)
    overlay_period_compare.to_csv(paths.result_dir / f"{RESULT_PREFIX}_gold_btc_overlay_period_compare_thb.csv", index=False)

    cols = [
        "Strategy",
        "Signal Mode",
        "Allocation Method",
        "Signal Source",
        "Lookback Months",
        "TH Weight Cap",
        "Entry Threshold",
        "Exit Threshold",
        "Min Hold Months",
        "Exit Confirm Months",
        "CAGR",
        "Annual Vol",
        "Sharpe",
        "Max Drawdown",
        "Average TH Weight",
        "TH On Months",
    ]
    if not overlay_only:
        print("\nMonthly comparison columns")
        print(monthly.tail(12).to_string(float_format=lambda value: f"{value:.4f}"))
        print("\nPersistence")
        print(persistence.head(20).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        print("\nBest tactical exit strategies")
        print(tactical_summary.reindex(columns=cols).head(20).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    overlay_cols = [
        "Strategy",
        "Start",
        "End",
        "CAGR",
        "Annual Vol",
        "Sharpe",
        "Max Drawdown",
        "Average US Equity Weight",
        "Average TH Equity Weight",
        "Average Gold Weight",
        "Average BTC Weight",
        "Average Cash / Reduced Exposure Weight",
    ]
    print("\nGold/BTC overlay comparison")
    print(overlay_summary.reindex(columns=overlay_cols).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
