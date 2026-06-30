from __future__ import annotations

from dataclasses import dataclass
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
from run_us_th_joint_model import FEATURE_FLAGS, LOOKBACK_DAYS, N_CLUSTERS  # noqa: E402
from run_us_th_one_model_us70_th30_stock_level_daily_exposure import (  # noqa: E402
    _load_full_us_th_overlay_panel_from_cache,
    _load_overlay_inputs_from_cache,
)
from run_us_th_tactical_gold_crash_protection_sweep import _gold_crash_exposure  # noqa: E402
from run_us_th_tactical_one_model import BTC_CAP, GOLD_CAP, INITIAL_VALUE, OVERLAY_ASSETS, _load_tactical_th_signal  # noqa: E402
from run_us_th_tactical_one_model_asym_group_caps import CASH_ASSET  # noqa: E402
from run_us_th_tactical_perf_momentum import RISK_FREE_RATE, _close_trend_exposure  # noqa: E402
from us_th_pit_reselect_utils import drop_duplicate_share_classes  # noqa: E402


PREFIX = "us_th_one_model_us70_th30_concentration_sweep"
US_GROUP_CAP = 0.70
TH_GROUP_CAP = 0.30
BASE_STOCK_CAP = 0.08
BASE_CONCENTRATION_PENALTY = 0.02
BASE_ASSETS = 30


@dataclass(frozen=True)
class SweepCase:
    label: str
    stock_cap: float
    concentration_penalty: float
    us_assets: int
    th_assets: int

    @property
    def strategy(self) -> str:
        return (
            "One-model US cap 70% / TH cap 30% "
            f"stockcap{int(round(self.stock_cap * 100))} "
            f"penalty{self.concentration_penalty:g} "
            f"assets{self.us_assets}"
        )


def _build_cases() -> list[SweepCase]:
    specs = [
        ("Baseline", BASE_STOCK_CAP, BASE_CONCENTRATION_PENALTY, BASE_ASSETS, BASE_ASSETS),
        ("Stock cap 7%", 0.07, BASE_CONCENTRATION_PENALTY, BASE_ASSETS, BASE_ASSETS),
        ("Stock cap 6%", 0.06, BASE_CONCENTRATION_PENALTY, BASE_ASSETS, BASE_ASSETS),
        ("Stock cap 5%", 0.05, BASE_CONCENTRATION_PENALTY, BASE_ASSETS, BASE_ASSETS),
        ("Penalty 0.05", BASE_STOCK_CAP, 0.05, BASE_ASSETS, BASE_ASSETS),
        ("Penalty 0.10", BASE_STOCK_CAP, 0.10, BASE_ASSETS, BASE_ASSETS),
        ("Penalty 0.20", BASE_STOCK_CAP, 0.20, BASE_ASSETS, BASE_ASSETS),
        ("Assets 40", BASE_STOCK_CAP, BASE_CONCENTRATION_PENALTY, 40, 40),
        ("Assets 50", BASE_STOCK_CAP, BASE_CONCENTRATION_PENALTY, 50, 50),
        ("Stock cap 6% penalty 0.05", 0.06, 0.05, BASE_ASSETS, BASE_ASSETS),
        ("Stock cap 6% assets 40", 0.06, BASE_CONCENTRATION_PENALTY, 40, 40),
        ("Stock cap 5% assets 50", 0.05, BASE_CONCENTRATION_PENALTY, 50, 50),
        ("Stock cap 6% penalty 0.05 assets 40", 0.06, 0.05, 40, 40),
    ]
    return [SweepCase(*spec) for spec in specs]


def _stock_cols(weights: pd.DataFrame) -> list[str]:
    return [c for c in weights.columns if c not in OVERLAY_ASSETS and c != CASH_ASSET]


def _us_cols(weights: pd.DataFrame) -> list[str]:
    return [c for c in _stock_cols(weights) if not c.endswith(".BK")]


def _th_cols(weights: pd.DataFrame) -> list[str]:
    return [c for c in _stock_cols(weights) if c.endswith(".BK")]


def _concentration_frame(weights: pd.DataFrame, strategy: str, case: SweepCase) -> pd.DataFrame:
    stocks = weights[_stock_cols(weights)].clip(lower=0.0) if _stock_cols(weights) else pd.DataFrame(index=weights.index)
    us = weights[_us_cols(weights)].clip(lower=0.0) if _us_cols(weights) else pd.DataFrame(index=weights.index)
    th = weights[_th_cols(weights)].clip(lower=0.0) if _th_cols(weights) else pd.DataFrame(index=weights.index)

    def top_sum(frame: pd.DataFrame, n: int) -> pd.Series:
        if frame.empty:
            return pd.Series(0.0, index=weights.index)
        values = -np.sort(-frame.to_numpy(dtype=float), axis=1)[:, :n]
        return pd.Series(values.sum(axis=1), index=weights.index)

    def hhi(frame: pd.DataFrame) -> pd.Series:
        if frame.empty:
            return pd.Series(0.0, index=weights.index)
        return (frame * frame).sum(axis=1)

    stock_hhi = hhi(stocks)
    us_hhi = hhi(us)
    th_hhi = hhi(th)
    out = pd.DataFrame(
        {
            "Strategy": strategy,
            "Case": case.label,
            "Stock Cap": case.stock_cap,
            "Concentration Penalty": case.concentration_penalty,
            "US Assets": case.us_assets,
            "TH Assets": case.th_assets,
            "Max Single Stock Weight": stocks.max(axis=1) if not stocks.empty else 0.0,
            "Top 5 Stock Weight": top_sum(stocks, 5),
            "Top 10 Stock Weight": top_sum(stocks, 10),
            "Stock HHI": stock_hhi,
            "Effective Stock Count": 1.0 / stock_hhi.replace(0.0, np.nan),
            "US Max Single Stock Weight": us.max(axis=1) if not us.empty else 0.0,
            "US Top 5 Stock Weight": top_sum(us, 5),
            "US Stock HHI": us_hhi,
            "US Effective Stock Count": 1.0 / us_hhi.replace(0.0, np.nan),
            "TH Max Single Stock Weight": th.max(axis=1) if not th.empty else 0.0,
            "TH Top 5 Stock Weight": top_sum(th, 5),
            "TH Stock HHI": th_hhi,
            "TH Effective Stock Count": 1.0 / th_hhi.replace(0.0, np.nan),
        },
        index=weights.index,
    )
    return out.reset_index(names="Date")


def _metrics_row(curve: pd.Series, weights: pd.DataFrame, concentration: pd.DataFrame, case: SweepCase, strategy: str) -> dict[str, object]:
    clean = curve.dropna()
    us_cols = _us_cols(weights)
    th_cols = _th_cols(weights)
    row = compute_port_opt_style_metrics(clean, risk_free_rate=RISK_FREE_RATE).to_dict()
    latest = concentration.sort_values("Date").iloc[-1]
    row.update(
        {
            "Strategy": strategy,
            "Case": case.label,
            "Start": clean.index.min().date().isoformat(),
            "End": clean.index.max().date().isoformat(),
            "US Group Cap": US_GROUP_CAP,
            "TH Group Cap": TH_GROUP_CAP,
            "Stock Cap": case.stock_cap,
            "Concentration Penalty": case.concentration_penalty,
            "US Assets": case.us_assets,
            "TH Assets": case.th_assets,
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
            "Latest Max Single Stock Weight": float(latest["Max Single Stock Weight"]),
            "Latest Top 5 Stock Weight": float(latest["Top 5 Stock Weight"]),
            "Latest Top 10 Stock Weight": float(latest["Top 10 Stock Weight"]),
            "Latest Stock HHI": float(latest["Stock HHI"]),
            "Latest Effective Stock Count": float(latest["Effective Stock Count"]),
            "Average Max Single Stock Weight": float(concentration["Max Single Stock Weight"].mean()),
            "Average Top 5 Stock Weight": float(concentration["Top 5 Stock Weight"].mean()),
            "Average Stock HHI": float(concentration["Stock HHI"].mean()),
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


def _optimize_with_case(
    cov: pd.DataFrame,
    momentum_signal: pd.Series,
    us_assets: list[str],
    th_assets: list[str],
    case: SweepCase,
    th_is_on: bool,
) -> pd.Series:
    base_assets = cov.index.tolist()
    cash_cap = max(0.0, 1.0 - (US_GROUP_CAP + (TH_GROUP_CAP if th_is_on else 0.0) + GOLD_CAP + BTC_CAP))
    assets = base_assets + ([CASH_ASSET] if cash_cap > EPSILON else [])
    cov2 = cov.reindex(index=assets, columns=assets).fillna(0.0)
    if CASH_ASSET in cov2.index:
        cov2.loc[CASH_ASSET, CASH_ASSET] = EPSILON
    mu = momentum_signal.reindex(assets).fillna(0.0)
    mu = mu.clip(mu.quantile(0.10), mu.quantile(0.90))

    caps = pd.Series(case.stock_cap, index=assets, dtype=float)
    if "GC=F" in caps.index:
        caps.loc["GC=F"] = GOLD_CAP
    if "BTC-USD" in caps.index:
        caps.loc["BTC-USD"] = BTC_CAP
    if CASH_ASSET in caps.index:
        caps.loc[CASH_ASSET] = cash_cap
    if float(caps.sum()) < 1.0 - EPSILON:
        raise RuntimeError(f"Infeasible caps for {case.label}: {caps.sum():.4f}")

    cov_matrix = cov2.to_numpy(dtype=float)
    mu_vec = mu.to_numpy(dtype=float)
    x0 = (caps / caps.sum()).to_numpy(dtype=float)
    bounds = [(0.0, float(caps.loc[asset])) for asset in assets]
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
    if us_assets:
        us_idx = [assets.index(asset) for asset in us_assets if asset in assets]
        constraints.append({"type": "ineq", "fun": lambda x, idx=us_idx: US_GROUP_CAP - float(np.sum(x[idx]))})
    if th_assets:
        th_idx = [assets.index(asset) for asset in th_assets if asset in assets]
        constraints.append({"type": "ineq", "fun": lambda x, idx=th_idx: TH_GROUP_CAP - float(np.sum(x[idx]))})

    def objective(x: np.ndarray) -> float:
        variance = float(x @ cov_matrix @ x)
        expected = float(mu_vec @ x)
        concentration = float(np.sum(np.square(x)))
        cash_penalty = 0.01 * float(x[assets.index(CASH_ASSET)]) if CASH_ASSET in assets else 0.0
        return 0.5 * 8.0 * variance - expected + case.concentration_penalty * concentration + cash_penalty

    result = minimize(objective, x0=x0, bounds=bounds, constraints=constraints, method="SLSQP")
    weights = pd.Series(result.x if result.success else x0, index=assets).clip(lower=0.0)
    return weights / weights.sum()


def _proxy_exposure(index: pd.DatetimeIndex) -> pd.DataFrame:
    _, signal_prices = _load_overlay_inputs_from_cache(index)
    return pd.DataFrame(
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
        index=index,
    ).ffill().fillna(1.0).clip(0.0, 1.0)


def _apply_proxy_exposure(raw_weights: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    effective = raw_weights.copy()
    for column in effective.columns:
        if column == CASH_ASSET:
            continue
        if column == "GC=F":
            effective[column] = effective[column] * exposure["GC=F"]
        elif column == "BTC-USD":
            effective[column] = effective[column] * exposure["BTC-USD"]
        elif column.endswith(".BK"):
            effective[column] = effective[column] * exposure["TH"]
        else:
            effective[column] = effective[column] * exposure["US"]
    noncash_sum = effective.drop(columns=[CASH_ASSET], errors="ignore").sum(axis=1)
    raw_cash = raw_weights[CASH_ASSET] if CASH_ASSET in raw_weights else 0.0
    effective[CASH_ASSET] = raw_cash + (1.0 - raw_cash - noncash_sum).clip(lower=0.0)
    return effective.clip(lower=0.0)


def _run_case(
    case: SweepCase,
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    benchmark: pd.Series,
    vol_proxy: pd.Series,
    us_all: list[str],
    th_all: list[str],
    th_signal: pd.Series,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = default_paths(ROOT)
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    benchmark_ret = benchmark.pct_change(fill_method=None).rename("benchmark")
    vol_proxy_ret = vol_proxy.pct_change(fill_method=None).rename("vol_proxy")
    schedule = monthly_rebalance_dates(prices.index, lookback_days=LOOKBACK_DAYS, freq="ME")

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
        us_selected = select_point_in_time_universe(prices.loc[train_index], volumes.loc[train_index], us_pool, n_assets=case.us_assets)
        th_selected = select_point_in_time_universe(prices.loc[train_index], volumes.loc[train_index], th_pool, n_assets=case.th_assets) if th_is_on else []
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
        weights = _optimize_with_case(cov, momentum_signal, current_us, current_th, case, th_is_on)
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
                "Strategy": case.strategy,
                "Date": rebalance_date.date().isoformat(),
                "TH Signal On": th_is_on,
                "US Count": len(current_us),
                "TH Count": len(current_th),
                "US Group Cap": US_GROUP_CAP,
                "TH Group Cap": TH_GROUP_CAP,
                "Stock Cap": case.stock_cap,
                "Concentration Penalty": case.concentration_penalty,
                "US Weight": float(weights.reindex(current_us).fillna(0.0).sum()),
                "TH Weight": float(weights.reindex(current_th).fillna(0.0).sum()),
                "Gold Weight": float(weights.get("GC=F", 0.0)),
                "BTC Weight": float(weights.get("BTC-USD", 0.0)),
                "Cash Weight": float(weights.get(CASH_ASSET, 0.0)),
                "Raw Max Stock Weight": float(weights.reindex(current_us + current_th).fillna(0.0).max()),
            }
        )

    raw_weights = daily_weights.ffill().fillna(0.0).reindex(nav.index).ffill().fillna(0.0)
    raw_weights = raw_weights.loc[:, raw_weights.abs().sum(axis=0).gt(0.0)]
    if CASH_ASSET not in raw_weights:
        raw_weights[CASH_ASSET] = 0.0
    raw_curve = nav[~nav.index.duplicated(keep="last")].sort_index()
    raw_weights = raw_weights.reindex(raw_curve.index).ffill().fillna(0.0)
    return raw_curve, raw_weights, pd.DataFrame(selected_rows), returns


def main() -> None:
    paths = default_paths(ROOT)
    prices, volumes, benchmark, vol_proxy, us_all, th_all = _load_full_us_th_overlay_panel_from_cache()
    _, th_signal = _load_tactical_th_signal(prices.index)
    cases = _build_cases()

    summary_rows = []
    selected_frames = []
    latest_frames = []
    effective_frames = []
    concentration_frames = []
    curves = {}
    max_cap_violations = []
    group_cap_violations = []

    for i, case in enumerate(cases, start=1):
        print(f"[{i}/{len(cases)}] {case.label}")
        raw_curve, raw_weights, selected, returns = _run_case(case, prices, volumes, benchmark, vol_proxy, us_all, th_all, th_signal)
        exposure = _proxy_exposure(raw_curve.index)
        effective = _apply_proxy_exposure(raw_weights, exposure)
        asset_cols = [column for column in effective.columns if column in returns.columns]
        exposed_returns = returns.reindex(raw_curve.index)[asset_cols].fillna(0.0).mul(effective[asset_cols], axis=1).sum(axis=1)
        strategy = f"{case.strategy} + daily exposure"
        curve = curve_from_returns(exposed_returns, initial=INITIAL_VALUE).rename(strategy)
        concentration = _concentration_frame(effective, strategy, case)

        raw_stock = raw_weights[_stock_cols(raw_weights)] if _stock_cols(raw_weights) else pd.DataFrame(index=raw_weights.index)
        if not raw_stock.empty:
            max_cap_violations.append(float((raw_stock.max(axis=1) - case.stock_cap).max()))
        raw_us = raw_weights[_us_cols(raw_weights)].sum(axis=1) if _us_cols(raw_weights) else pd.Series(0.0, index=raw_weights.index)
        raw_th = raw_weights[_th_cols(raw_weights)].sum(axis=1) if _th_cols(raw_weights) else pd.Series(0.0, index=raw_weights.index)
        group_cap_violations.append(float(max((raw_us - US_GROUP_CAP).max(), (raw_th - TH_GROUP_CAP).max())))

        curves[strategy] = curve
        summary_rows.append(_metrics_row(curve, effective, concentration, case, strategy))
        selected_frames.append(selected)
        concentration_frames.append(concentration)

        latest = effective.iloc[-1].rename("Effective Weight").reset_index().rename(columns={"index": "Asset"})
        latest = latest.loc[latest["Effective Weight"].abs().gt(1e-12)].sort_values("Effective Weight", ascending=False)
        latest["Date"] = effective.index.max().date().isoformat()
        latest["Strategy"] = strategy
        latest["Case"] = case.label
        latest_frames.append(latest)

        out = effective.copy()
        out.insert(0, "Strategy", strategy)
        out.insert(1, "Case", case.label)
        effective_frames.append(out.reset_index(names="Date"))

    curves_df = pd.DataFrame(curves).dropna(how="all")
    summary = pd.DataFrame(summary_rows).sort_values(["Sharpe", "CAGR"], ascending=False)
    selected_history = pd.concat(selected_frames, ignore_index=True)
    latest_weights = pd.concat(latest_frames, ignore_index=True)
    effective_history = pd.concat(effective_frames, ignore_index=True)
    concentration_history = pd.concat(concentration_frames, ignore_index=True)
    period_compare = _period_compare(curves_df)

    baseline = summary.loc[summary["Case"].eq("Baseline")].iloc[0]
    summary["Sharpe Loss vs Baseline"] = float(baseline["Sharpe"]) - summary["Sharpe"]
    summary["Top 5 Reduction vs Baseline"] = float(baseline["Latest Top 5 Stock Weight"]) - summary["Latest Top 5 Stock Weight"]
    summary["HHI Reduction vs Baseline"] = float(baseline["Latest Stock HHI"]) - summary["Latest Stock HHI"]

    summary.to_csv(paths.result_dir / f"{PREFIX}_summary_thb.csv", index=False)
    curves_df.to_csv(paths.result_dir / f"{PREFIX}_curves_thb.csv")
    latest_weights.to_csv(paths.result_dir / f"{PREFIX}_latest_weights_thb.csv", index=False)
    effective_history.to_csv(paths.result_dir / f"{PREFIX}_effective_weights_thb.csv", index=False)
    concentration_history.to_csv(paths.result_dir / f"{PREFIX}_concentration_history_thb.csv", index=False)
    period_compare.to_csv(paths.result_dir / f"{PREFIX}_period_compare_thb.csv", index=False)
    selected_history.to_csv(paths.result_dir / f"{PREFIX}_universe_history_thb.csv", index=False)
    candidates = summary.loc[
        summary["Sharpe Loss vs Baseline"].le(0.10)
        & summary["Top 5 Reduction vs Baseline"].gt(0.0)
        & summary["HHI Reduction vs Baseline"].gt(0.0)
    ].copy()
    best_concentration = (
        candidates.sort_values(["HHI Reduction vs Baseline", "Sharpe"], ascending=[False, False]).head(1)
        if not candidates.empty
        else summary.sort_values(["Latest Stock HHI", "Sharpe"], ascending=[True, False]).head(1)
    )
    conservative = summary.sort_values(["Latest Max Single Stock Weight", "Sharpe"], ascending=[True, False]).head(1)

    display_cols = [
        "Case",
        "CAGR",
        "Sharpe",
        "Max Drawdown",
        "Latest Max Single Stock Weight",
        "Latest Top 5 Stock Weight",
        "Latest Top 10 Stock Weight",
        "Latest Effective Stock Count",
    ]
    print("\nTop by Sharpe")
    print(summary[display_cols].head(10).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nBest concentration candidate")
    print(best_concentration[display_cols].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nMost conservative")
    print(conservative[display_cols].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nMax raw stock-cap violation: {max(max_cap_violations):.12f}")
    print(f"Max raw group-cap violation: {max(group_cap_violations):.12f}")


if __name__ == "__main__":
    main()
