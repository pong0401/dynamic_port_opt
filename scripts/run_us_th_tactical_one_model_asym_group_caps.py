from __future__ import annotations

from pathlib import Path
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
    EPSILON,
    build_momentum_signal,
    compute_feature_table,
    compute_port_opt_style_metrics,
    curve_from_returns,
    default_paths,
    get_set100_members_as_of,
    get_sp500_members_as_of,
    monthly_rebalance_dates,
    select_point_in_time_universe,
)
from run_us_th_joint_model import END_DATE, FEATURE_FLAGS, LOOKBACK_DAYS, N_CLUSTERS  # noqa: E402
from run_us_th_tactical_gold_crash_protection_sweep import _gold_crash_exposure  # noqa: E402
from run_us_th_tactical_gold_exposure_sweep import _load_overlay_inputs_from_cache  # noqa: E402
from run_us_th_tactical_one_model import (  # noqa: E402
    BTC_CAP,
    GOLD_CAP,
    INITIAL_VALUE,
    OUTPUT_PREFIX,
    OVERLAY_ASSETS,
    STOCK_CAP,
    TH_ASSETS,
    US_ASSETS,
    _load_full_us_th_overlay_panel_from_cache,
    _load_tactical_th_signal,
    _metrics_row,
    _period_compare,
)
from run_us_th_tactical_perf_momentum import _close_trend_exposure  # noqa: E402
from us_th_pit_reselect_utils import drop_duplicate_share_classes  # noqa: E402


ASYM_OUTPUT_PREFIX = f"{OUTPUT_PREFIX}_asym_group_cap_grid_us70_80_th30_40"
CASH_ASSET = "Cash / Reduced Exposure"
CASES = [
    {"label": "US cap 70% / TH cap 30%", "us_cap": 0.70, "th_cap": 0.30},
    {"label": "US cap 70% / TH cap 40%", "us_cap": 0.70, "th_cap": 0.40},
    {"label": "US cap 80% / TH cap 30%", "us_cap": 0.80, "th_cap": 0.30},
    {"label": "US cap 80% / TH cap 40%", "us_cap": 0.80, "th_cap": 0.40},
]


def _optimize_with_asym_caps(
    cov: pd.DataFrame,
    momentum_signal: pd.Series,
    us_assets: list[str],
    th_assets: list[str],
    us_cap: float,
    th_cap: float,
    th_is_on: bool,
) -> pd.Series:
    base_assets = cov.index.tolist()
    cash_cap = max(0.0, 1.0 - (us_cap + (th_cap if th_is_on else 0.0) + GOLD_CAP + BTC_CAP))
    assets = base_assets + ([CASH_ASSET] if cash_cap > EPSILON else [])
    cov2 = cov.reindex(index=assets, columns=assets).fillna(0.0)
    if CASH_ASSET in cov2.index:
        cov2.loc[CASH_ASSET, CASH_ASSET] = EPSILON
    mu = momentum_signal.reindex(assets).fillna(0.0)
    mu = mu.clip(mu.quantile(0.10), mu.quantile(0.90))

    caps = pd.Series(STOCK_CAP, index=assets, dtype=float)
    if "GC=F" in caps.index:
        caps.loc["GC=F"] = GOLD_CAP
    if "BTC-USD" in caps.index:
        caps.loc["BTC-USD"] = BTC_CAP
    if CASH_ASSET in caps.index:
        caps.loc[CASH_ASSET] = cash_cap
    if float(caps.sum()) < 1.0 - EPSILON:
        raise RuntimeError(f"Infeasible caps: {caps.sum():.4f}")

    cov_matrix = cov2.to_numpy(dtype=float)
    mu_vec = mu.to_numpy(dtype=float)
    x0 = (caps / caps.sum()).to_numpy(dtype=float)
    bounds = [(0.0, float(caps.loc[asset])) for asset in assets]
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
    if us_assets:
        us_idx = [assets.index(asset) for asset in us_assets if asset in assets]
        constraints.append({"type": "ineq", "fun": lambda x, idx=us_idx: us_cap - float(np.sum(x[idx]))})
    if th_assets:
        th_idx = [assets.index(asset) for asset in th_assets if asset in assets]
        constraints.append({"type": "ineq", "fun": lambda x, idx=th_idx: th_cap - float(np.sum(x[idx]))})

    def objective(x: np.ndarray) -> float:
        variance = float(x @ cov_matrix @ x)
        expected = float(mu_vec @ x)
        concentration = float(np.sum(np.square(x)))
        cash_penalty = 0.0
        if CASH_ASSET in assets:
            cash_penalty = 0.01 * float(x[assets.index(CASH_ASSET)])
        return 0.5 * 8.0 * variance - expected + 0.02 * concentration + cash_penalty

    result = minimize(objective, x0=x0, bounds=bounds, constraints=constraints, method="SLSQP")
    weights = pd.Series(result.x if result.success else x0, index=assets).clip(lower=0.0)
    return weights / weights.sum()


def _run_case(label: str, us_cap: float, th_cap: float) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = default_paths(ROOT)
    prices, volumes, benchmark, vol_proxy, us_all, th_all = _load_full_us_th_overlay_panel_from_cache()
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    benchmark_ret = benchmark.pct_change(fill_method=None).rename("benchmark")
    vol_proxy_ret = vol_proxy.pct_change(fill_method=None).rename("vol_proxy")
    schedule = monthly_rebalance_dates(prices.index, lookback_days=LOOKBACK_DAYS, freq="ME")
    _, th_signal = _load_tactical_th_signal(prices.index)

    nav = pd.Series(INITIAL_VALUE, index=[schedule[0]], dtype=float)
    daily_weights = pd.DataFrame(index=prices.index, columns=list(prices.columns) + [CASH_ASSET], dtype=float)
    selected_rows = []

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
        current_us = [asset for asset in current_assets if asset in us_selected]
        current_th = [asset for asset in current_assets if asset in th_selected]
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
        weights = _optimize_with_asym_caps(cov, momentum_signal, current_us, current_th, us_cap, th_cap, th_is_on)
        daily_weights.loc[test_index, :] = 0.0
        daily_weights.loc[test_index, weights.index] = pd.DataFrame(
            np.tile(weights.to_numpy(dtype=float), (len(test_index), 1)),
            index=test_index,
            columns=weights.index,
        )
        period_assets = [asset for asset in weights.index if asset in returns.columns]
        period_returns = returns.reindex(test_index)[period_assets].fillna(0.0).mul(weights.reindex(period_assets), axis=1).sum(axis=1)
        nav = pd.concat([nav, (1.0 + period_returns).cumprod().mul(float(nav.iloc[-1]))])
        selected_rows.append(
            {
                "Case": label,
                "Date": rebalance_date.date().isoformat(),
                "TH Signal On": th_is_on,
                "US Count": len(current_us),
                "TH Count": len(current_th),
                "US Group Cap": us_cap,
                "TH Group Cap": th_cap,
                "US Weight": float(weights.reindex(current_us).fillna(0.0).sum()),
                "TH Weight": float(weights.reindex(current_th).fillna(0.0).sum()),
                "Gold Weight": float(weights.get("GC=F", 0.0)),
                "BTC Weight": float(weights.get("BTC-USD", 0.0)),
                "Cash Weight": float(weights.get(CASH_ASSET, 0.0)),
            }
        )

    raw_weights = daily_weights.ffill().fillna(0.0)
    raw_curve = nav[~nav.index.duplicated(keep="last")].sort_index().rename(f"One-model {label}")

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
    effective_weights = raw_weights.reindex(raw_curve.index).ffill().fillna(0.0)
    for column in effective_weights.columns:
        if column == CASH_ASSET:
            continue
        if column == "GC=F":
            effective_weights[column] = effective_weights[column] * exposure["GC=F"]
        elif column == "BTC-USD":
            effective_weights[column] = effective_weights[column] * exposure["BTC-USD"]
        elif column.endswith(".BK"):
            effective_weights[column] = effective_weights[column] * exposure["TH"]
        else:
            effective_weights[column] = effective_weights[column] * exposure["US"]
    noncash_sum = effective_weights.drop(columns=[CASH_ASSET], errors="ignore").sum(axis=1)
    effective_weights[CASH_ASSET] = effective_weights.get(CASH_ASSET, 0.0) + (1.0 - effective_weights.get(CASH_ASSET, 0.0) - noncash_sum).clip(lower=0.0)
    asset_cols = [column for column in effective_weights.columns if column in returns.columns]
    exposed_returns = returns.reindex(raw_curve.index)[asset_cols].fillna(0.0).mul(effective_weights[asset_cols], axis=1).sum(axis=1)
    exposed_curve = curve_from_returns(exposed_returns, initial=INITIAL_VALUE).rename(f"One-model {label} + daily exposure")
    return raw_curve, exposed_curve, raw_weights.reindex(raw_curve.index).ffill().fillna(0.0), effective_weights, pd.DataFrame(selected_rows)


def _grouped_history(effective_history: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for case, frame in effective_history.groupby("Case"):
        weights = frame.drop(columns=["Case"]).set_index("Date")
        weights.index = pd.to_datetime(weights.index)
        monthly = weights.resample("ME").last().fillna(0.0)
        grouped = pd.DataFrame(
            {
                "US Stocks": monthly[
                    [
                        column
                        for column in monthly.columns
                        if column not in ["GC=F", "BTC-USD", CASH_ASSET] and not column.endswith(".BK")
                    ]
                ].sum(axis=1),
                "TH Stocks": monthly[[column for column in monthly.columns if column.endswith(".BK")]].sum(axis=1),
                "Gold": monthly["GC=F"] if "GC=F" in monthly else 0.0,
                "BTC": monthly["BTC-USD"] if "BTC-USD" in monthly else 0.0,
                CASH_ASSET: monthly[CASH_ASSET] if CASH_ASSET in monthly else 0.0,
            }
        ).clip(lower=0.0)
        grouped.insert(0, "Case", case)
        frames.append(grouped.reset_index(names="Date"))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    paths = default_paths(ROOT)
    curves = {}
    summaries = []
    selected_frames = []
    latest_frames = []
    effective_frames = []
    for case in CASES:
        raw_curve, exposed_curve, raw_weights, effective_weights, selected = _run_case(case["label"], case["us_cap"], case["th_cap"])
        curves[raw_curve.name] = raw_curve
        curves[exposed_curve.name] = exposed_curve
        raw_out = raw_weights.copy()
        if CASH_ASSET not in raw_out:
            raw_out[CASH_ASSET] = 0.0
        for strategy, curve, weights, mode in [
            (raw_curve.name, raw_curve, raw_out, "raw"),
            (exposed_curve.name, exposed_curve, effective_weights, "daily exposure"),
        ]:
            row = _metrics_row(curve, strategy, weights)
            row.update(
                {
                    "Case": case["label"],
                    "US Group Cap": case["us_cap"],
                    "TH Group Cap": case["th_cap"],
                    "Mode": mode,
                    "Stock Cap": STOCK_CAP,
                    "Gold Cap": GOLD_CAP,
                    "BTC Cap": BTC_CAP,
                }
            )
            summaries.append(row)
        selected_frames.append(selected)
        latest = effective_weights.iloc[-1].rename("Effective Weight").reset_index().rename(columns={"index": "Asset"})
        latest = latest.loc[latest["Effective Weight"].abs().gt(1e-12)].sort_values("Effective Weight", ascending=False)
        latest["Date"] = effective_weights.index.max().date().isoformat()
        latest["Case"] = case["label"]
        latest_frames.append(latest)
        out = effective_weights.copy()
        out.insert(0, "Case", case["label"])
        effective_frames.append(out.reset_index(names="Date"))

    curves_df = pd.DataFrame(curves).dropna(how="all")
    summary = pd.DataFrame(summaries).sort_values(["Sharpe", "CAGR"], ascending=False)
    selected_history = pd.concat(selected_frames, ignore_index=True)
    latest_weights = pd.concat(latest_frames, ignore_index=True)
    effective_history = pd.concat(effective_frames, ignore_index=True)
    grouped_history = _grouped_history(effective_history)

    summary.to_csv(paths.result_dir / f"{ASYM_OUTPUT_PREFIX}_summary_thb.csv", index=False)
    curves_df.to_csv(paths.result_dir / f"{ASYM_OUTPUT_PREFIX}_curves_thb.csv")
    selected_history.to_csv(paths.result_dir / f"{ASYM_OUTPUT_PREFIX}_universe_history_thb.csv", index=False)
    latest_weights.to_csv(paths.result_dir / f"{ASYM_OUTPUT_PREFIX}_latest_weights_thb.csv", index=False)
    effective_history.to_csv(paths.result_dir / f"{ASYM_OUTPUT_PREFIX}_effective_weights_thb.csv", index=False)
    grouped_history.to_csv(paths.result_dir / f"{ASYM_OUTPUT_PREFIX}_grouped_weight_history_thb.csv", index=False)
    _period_compare(curves_df).to_csv(paths.result_dir / f"{ASYM_OUTPUT_PREFIX}_period_compare_thb.csv", index=False)
    print(
        summary[
            [
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
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
