from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in [SRC, SCRIPTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dynamic_factor_copula import compute_port_opt_style_metrics, curve_from_returns, default_paths, lag_close_signal_to_next_session  # noqa: E402
from run_mean_covariance_th_gated_sleeve import (  # noqa: E402
    END_DATE,
    START_DATE,
    TH_WEIGHTS,
    _annual_returns,
    _dynamic_blend_returns,
    _load_core_returns_thb,
    _load_market_signals,
)
from run_mean_covariance_gold30_with_th_sleeve import _run_th_sleeve  # noqa: E402


MA_PERIOD = 200
SLOPE_LOOKBACKS = [20, 40, 63]
ENTRY_CONFIRMS = [1, 5, 10, 20]
EXIT_CONFIRMS = [5, 10, 20, 40]
ENTRY_BUFFERS = [0.0, 0.03, 0.05, 0.08]
BREADTH_FILTERS = {
    "none": {},
    "above_ma200_gt50": {"Above MA200": 0.50},
    "mom63_gt50": {"Positive Mom63": 0.50},
    "above_ma200_gt50_mom63_gt50": {"Above MA200": 0.50, "Positive Mom63": 0.50},
    "above_ma200_gt55_mom63_gt55": {"Above MA200": 0.55, "Positive Mom63": 0.55},
}
RESULT_PREFIX = "mean_covariance_th_ma200_regime_gate"


def _hysteresis_gate(
    entry_raw: pd.Series,
    exit_raw: pd.Series,
    entry_confirm: int,
    exit_confirm: int,
) -> pd.Series:
    entry_raw = entry_raw.fillna(False).astype(bool).sort_index()
    exit_raw = exit_raw.reindex(entry_raw.index).fillna(False).astype(bool)
    state = False
    entry_count = 0
    exit_count = 0
    values = []
    for entry_value, exit_value in zip(entry_raw, exit_raw):
        if state:
            if exit_value:
                exit_count += 1
                if exit_count >= exit_confirm:
                    state = False
                    entry_count = 0
                    exit_count = 0
            else:
                exit_count = 0
        else:
            if entry_value:
                entry_count += 1
                if entry_count >= entry_confirm:
                    state = True
                    entry_count = 0
                    exit_count = 0
            else:
                entry_count = 0
        values.append(float(state))
    return pd.Series(values, index=entry_raw.index, name="TH Gate")


def _regime_spans(gate: pd.Series) -> str:
    spans = []
    active = False
    prev = None
    for dt, value in gate.items():
        is_on = float(value) > 0.5
        if is_on and not active:
            start = dt
            active = True
        if active and not is_on:
            spans.append(f"{start.date().isoformat()} to {prev.date().isoformat()}")
            active = False
        prev = dt
    if active and prev is not None:
        spans.append(f"{start.date().isoformat()} to {prev.date().isoformat()}")
    return "; ".join(spans)


def _ma200_raw_gates(market: pd.DataFrame, slope_lookback: int, entry_buffer: float) -> tuple[pd.Series, pd.Series]:
    set_index = market["SET"]
    ma = set_index.rolling(MA_PERIOD, min_periods=MA_PERIOD).mean()
    slope_ok = ma.diff(slope_lookback) > 0.0
    entry_raw = (set_index > ma * (1.0 + entry_buffer)) & slope_ok
    exit_raw = (set_index < ma) | ~slope_ok
    return entry_raw.rename("Entry Raw Gate"), exit_raw.rename("Exit Raw Gate")


def _load_breadth(index: pd.DatetimeIndex) -> pd.DataFrame:
    paths = default_paths(ROOT)
    breadth_file = paths.result_dir / "set100_pit_breadth_history.csv"
    if not breadth_file.exists():
        return pd.DataFrame(index=index)
    return pd.read_csv(breadth_file, index_col=0, parse_dates=True).reindex(index).ffill()


def _breadth_mask(breadth: pd.DataFrame, thresholds: dict[str, float]) -> pd.Series:
    if not thresholds or breadth.empty:
        return pd.Series(True, index=breadth.index if not breadth.empty else pd.DatetimeIndex([]))
    mask = pd.Series(True, index=breadth.index)
    for column, threshold in thresholds.items():
        if column not in breadth.columns:
            mask &= False
        else:
            mask &= breadth[column] > threshold
    return mask


def main() -> None:
    paths = default_paths(ROOT)
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    core_returns = _load_core_returns_thb()
    th_results = _run_th_sleeve()
    th_returns = th_results["nav"]["Static Copula"].pct_change(fill_method=None).fillna(0.0).reindex(core_returns.index)
    market = _load_market_signals(core_returns.index)
    core_returns = core_returns.reindex(market.index).fillna(0.0)
    th_returns = th_returns.reindex(market.index).fillna(0.0)
    breadth = _load_breadth(market.index)

    rows = []
    curves = {"Core no-TH final": curve_from_returns(core_returns)}
    gate_history: dict[str, pd.Series] = {}
    weight_history: dict[str, pd.DataFrame] = {}
    for slope_lookback in SLOPE_LOOKBACKS:
        for entry_buffer in ENTRY_BUFFERS:
            base_entry_raw, base_exit_raw = _ma200_raw_gates(market, slope_lookback, entry_buffer)
            for breadth_name, thresholds in BREADTH_FILTERS.items():
                breadth_ok = _breadth_mask(breadth, thresholds).reindex(market.index).fillna(False)
                entry_raw = base_entry_raw & breadth_ok
                exit_raw = base_exit_raw | ~breadth_ok
                for entry_confirm in ENTRY_CONFIRMS:
                    for exit_confirm in EXIT_CONFIRMS:
                        gate = lag_close_signal_to_next_session(
                            _hysteresis_gate(entry_raw, exit_raw, entry_confirm, exit_confirm),
                            initial=0.0,
                        )
                        for th_weight in TH_WEIGHTS:
                            strategy = (
                                f"TH MA200 regime TH{int(th_weight * 100)} "
                                f"slope{slope_lookback} buffer{entry_buffer:.0%} {breadth_name} "
                                f"entry{entry_confirm} exit{exit_confirm}"
                            )
                            returns, weights = _dynamic_blend_returns(core_returns, th_returns, gate, th_weight)
                            curve = curve_from_returns(returns)
                            row = compute_port_opt_style_metrics(curve, risk_free_rate=0.03).to_dict()
                            row["Strategy"] = strategy
                            row["TH Weight When On"] = th_weight
                            row["MA Period"] = MA_PERIOD
                            row["Slope Lookback"] = slope_lookback
                            row["Entry Buffer"] = entry_buffer
                            row["Breadth Filter"] = breadth_name
                            row["Entry Confirm Days"] = entry_confirm
                            row["Exit Confirm Days"] = exit_confirm
                            row["Average TH Weight"] = float(weights["TH"].mean())
                            row["TH On Days"] = int((weights["TH"] > 0.0).sum())
                            row["Regime Spans"] = _regime_spans(gate.reindex(weights.index).ffill().fillna(0.0))
                            row["Start"] = curve.index.min().date().isoformat()
                            row["End"] = curve.index.max().date().isoformat()
                            rows.append(row)
                            curves[strategy] = curve
                            gate_history[strategy] = gate.reindex(weights.index).ffill().fillna(0.0)
                            weight_history[strategy] = weights

    summary = pd.DataFrame(rows).sort_values(["Sharpe", "CAGR"], ascending=False)
    best_strategy = str(summary.iloc[0]["Strategy"])
    curves_df = pd.DataFrame(curves).dropna(how="all")
    summary.to_csv(paths.result_dir / f"{RESULT_PREFIX}_summary_thb.csv", index=False)
    curves_df.to_csv(paths.result_dir / f"{RESULT_PREFIX}_curves_thb.csv")
    _annual_returns(curves_df[["Core no-TH final", best_strategy]]).to_csv(
        paths.result_dir / f"{RESULT_PREFIX}_annual_returns_thb.csv",
        index=False,
    )
    gate_history[best_strategy].rename("TH Gate").to_csv(paths.result_dir / f"{RESULT_PREFIX}_best_gate_history_thb.csv")
    weight_history[best_strategy].to_csv(paths.result_dir / f"{RESULT_PREFIX}_best_weight_history_thb.csv")

    cols = [
        "Strategy",
        "TH Weight When On",
        "Slope Lookback",
        "Entry Buffer",
        "Breadth Filter",
        "Entry Confirm Days",
        "Exit Confirm Days",
        "CAGR",
        "Annual Vol",
        "Sharpe",
        "Max Drawdown",
        "Average TH Weight",
        "TH On Days",
        "Regime Spans",
    ]
    print(summary.head(12).reindex(columns=cols).to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nBest annual returns")
    annual = pd.read_csv(paths.result_dir / f"{RESULT_PREFIX}_annual_returns_thb.csv")
    print(annual.pivot(index="Year", columns="Strategy", values="Annual Return").to_string(float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
