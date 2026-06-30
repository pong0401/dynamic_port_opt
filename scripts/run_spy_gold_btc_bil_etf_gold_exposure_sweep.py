from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in [SRC, SCRIPTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dynamic_factor_copula import compute_port_opt_style_metrics, curve_from_returns, default_paths, lag_close_signal_to_next_session, monthly_rebalance_dates  # noqa: E402
import run_spy_gold_btc_defensive_etf_expanded_sweep as expanded  # noqa: E402

OUTPUT_PREFIX = "spy_gold_btc_bil_etf_gold_exposure_sweep"
RISK_FREE_RATE = expanded.RISK_FREE_RATE
INITIAL_VALUE = expanded.INITIAL_VALUE
ETF_UNIVERSE = tuple(expanded.CORE_ETFS)


@dataclass(frozen=True)
class GoldRule:
    name: str
    mode: str
    ma_period: int = 50
    below_exposure: float = 1.0
    dd_window: int = 252
    warn_dd: float = -0.08
    crash_dd: float = -0.20
    warn_exposure: float = 0.50
    crash_exposure: float = 0.50
    recovery_dd: float = -0.05
    panic_dd: float = -0.30
    panic_ma_period: int = 200
    panic_mom_period: int = 63


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    gold_rule: GoldRule
    bucket: float = 0.20
    top_n: int = 1
    funding_mode: str = "spy2_def1"


def trend_exposure(price: pd.Series, ma_period: int, below_exposure: float) -> pd.Series:
    price = price.astype(float).sort_index().ffill()
    ma = price.rolling(ma_period, min_periods=max(20, int(ma_period * 0.20))).mean()
    signal = pd.Series(1.0, index=price.index, dtype=float)
    signal.loc[price < ma] = below_exposure
    signal.loc[ma.isna()] = 1.0
    return lag_close_signal_to_next_session(signal, initial=1.0)


def drawdown_exposure_simple(price: pd.Series, rule: GoldRule) -> pd.Series:
    price = price.astype(float).sort_index().ffill()
    dd = price / price.rolling(rule.dd_window, min_periods=max(60, rule.dd_window // 4)).max() - 1.0
    signal = pd.Series(1.0, index=price.index, dtype=float)
    signal.loc[dd <= rule.warn_dd] = rule.warn_exposure
    signal.loc[dd <= rule.crash_dd] = rule.crash_exposure
    return lag_close_signal_to_next_session(signal, initial=1.0)


def drawdown_exposure_recovery(price: pd.Series, rule: GoldRule) -> pd.Series:
    price = price.astype(float).sort_index().ffill()
    rolling_high = price.rolling(rule.dd_window, min_periods=max(60, rule.dd_window // 4)).max()
    dd = price / rolling_high - 1.0
    panic_ma = price.rolling(rule.panic_ma_period, min_periods=max(50, rule.panic_ma_period // 4)).mean()
    mom = price / price.shift(rule.panic_mom_period) - 1.0
    exposure = pd.Series(1.0, index=price.index, dtype=float)
    state = 1.0
    for dt in price.index:
        current_dd = float(dd.loc[dt]) if pd.notna(dd.loc[dt]) else 0.0
        panic = current_dd <= rule.panic_dd and pd.notna(panic_ma.loc[dt]) and price.loc[dt] < panic_ma.loc[dt] and pd.notna(mom.loc[dt]) and mom.loc[dt] < 0.0
        if panic:
            state = rule.crash_exposure
        elif current_dd <= rule.crash_dd:
            state = rule.crash_exposure
        elif current_dd <= rule.warn_dd:
            state = rule.warn_exposure
        elif current_dd >= rule.recovery_dd:
            state = 1.0
        exposure.loc[dt] = state
    return lag_close_signal_to_next_session(exposure, initial=1.0)


def gold_exposure(price: pd.Series, rule: GoldRule) -> pd.Series:
    if rule.mode == "hold":
        return pd.Series(1.0, index=price.index, dtype=float)
    if rule.mode == "trend":
        return trend_exposure(price, rule.ma_period, rule.below_exposure)
    if rule.mode == "dd_simple":
        return drawdown_exposure_simple(price, rule)
    if rule.mode == "dd_recovery":
        return drawdown_exposure_recovery(price, rule)
    raise ValueError(f"Unknown gold rule: {rule.mode}")


def momentum_rank(prices: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    hist = prices.loc[:date, list(ETF_UNIVERSE)].ffill()
    rows = []
    for asset in ETF_UNIVERSE:
        s = hist[asset].dropna() if asset in hist else pd.Series(dtype=float)
        if len(s) < 253:
            rows.append({"ETF": asset, "pass": False, "score": np.nan})
            continue
        latest = float(s.iloc[-1])
        ret_1m = latest / float(s.iloc[-22]) - 1.0
        ret_3m = latest / float(s.iloc[-64]) - 1.0
        ret_6m = latest / float(s.iloc[-127]) - 1.0
        ret_12m = latest / float(s.iloc[-253]) - 1.0
        sma200 = float(s.iloc[-200:].mean())
        rows.append({"ETF": asset, "ret_1m": ret_1m, "ret_3m": ret_3m, "ret_6m": ret_6m, "ret_12m": ret_12m, "pass": latest > sma200 and ret_3m > 0.0 and ret_6m > 0.0})
    df = pd.DataFrame(rows)
    score = pd.Series(0.0, index=df.index, dtype=float)
    for col, weight in {"ret_1m": 0.20, "ret_3m": 0.30, "ret_6m": 0.40, "ret_12m": 0.10}.items():
        score += weight * df[col].rank(pct=True).fillna(0.0)
    df["score"] = score
    return df.sort_values(["pass", "score"], ascending=[False, False]).reset_index(drop=True)


def monthly_weights(selected: list[str], config: StrategyConfig) -> pd.Series:
    w = pd.Series({"SPY": 0.45, "Gold": 0.30, "BTC": 0.10, "BIL": 0.15}, dtype=float)
    if not selected:
        return w
    if config.funding_mode == "spy2_def1":
        spy_funding = config.bucket * 2.0 / 3.0
        bil_funding = config.bucket - spy_funding
    elif config.funding_mode == "half":
        spy_funding = config.bucket * 0.50
        bil_funding = config.bucket * 0.50
    else:
        spy_funding = config.bucket
        bil_funding = 0.0
    w["SPY"] -= spy_funding
    w["BIL"] -= bil_funding
    if w["SPY"] < 0.30 - 1e-12 or w["BIL"] < -1e-12:
        return pd.Series(dtype=float)
    for asset in selected:
        w[asset] = config.bucket / len(selected)
    return w / w.sum()


def run_variant(prices: pd.DataFrame, returns: pd.DataFrame, config: StrategyConfig) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    schedule = monthly_rebalance_dates(prices.index, lookback_days=252, freq="ME")
    cols = ["SPY", "Gold", "BTC", "BIL"] + list(ETF_UNIVERSE)
    weights = pd.DataFrame(0.0, index=prices.index, columns=cols)
    selections = []
    for idx, date in enumerate(schedule[:-1]):
        next_date = schedule[idx + 1]
        test_idx = prices.index[(prices.index > date) & (prices.index <= next_date)]
        if len(test_idx) == 0:
            continue
        ranks = momentum_rank(prices, date)
        selected = ranks.loc[ranks["pass"].fillna(False), "ETF"].head(config.top_n).astype(str).tolist()
        w = monthly_weights(selected, config)
        if w.empty:
            continue
        weights.loc[test_idx, w.index] = w.to_numpy(dtype=float)
        selections.append({"Strategy": config.name, "rebalance_date": date, "next_rebalance_date": next_date, "selected_etfs": ",".join(selected), "selected_count": len(selected)})
    weights = weights.loc[weights.sum(axis=1).gt(0.0)].copy()
    exposure = pd.DataFrame(1.0, index=weights.index, columns=cols)
    exposure["SPY"] = trend_exposure(prices["SPY"], 300, 0.50).reindex(weights.index).ffill().fillna(1.0)
    exposure["Gold"] = gold_exposure(prices["Gold"], config.gold_rule).reindex(weights.index).ffill().fillna(1.0)
    exposure["BTC"] = trend_exposure(prices["BTC"], 50, 0.00).reindex(weights.index).ffill().fillna(1.0)
    effective = weights * exposure
    strat_ret = returns.reindex(effective.index).fillna(0.0).mul(effective, axis=1).sum(axis=1)
    curve = curve_from_returns(strat_ret, initial=INITIAL_VALUE).rename(config.name)
    latest = effective.iloc[-1].rename("Effective Weight").reset_index().rename(columns={"index": "Asset"})
    latest = latest.loc[latest["Effective Weight"].abs().gt(1e-12)].copy()
    latest.insert(0, "Strategy", config.name)
    latest.insert(1, "Date", effective.index.max())
    exposure_out = exposure.copy()
    exposure_out.insert(0, "Strategy", config.name)
    exposure_out.insert(1, "Date", exposure_out.index)
    return curve, pd.DataFrame(selections), latest, exposure_out.reset_index(drop=True)


def metrics_row(curve: pd.Series, config: StrategyConfig, selection: pd.DataFrame, latest: pd.DataFrame, exposure: pd.DataFrame) -> dict[str, object]:
    clean = curve.dropna()
    row = compute_port_opt_style_metrics(clean, risk_free_rate=RISK_FREE_RATE).to_dict()
    latest_weights = latest.set_index("Asset")["Effective Weight"] if not latest.empty else pd.Series(dtype=float)
    active_months = int(selection["selected_count"].gt(0).sum()) if not selection.empty else 0
    total_months = int(selection.shape[0]) if not selection.empty else 0
    row.update(
        {
            "Strategy": config.name,
            "Gold Rule": config.gold_rule.name,
            "Start": clean.index.min().date().isoformat(),
            "End": clean.index.max().date().isoformat(),
            "Active Rate": active_months / total_months if total_months else np.nan,
            "Average Gold Exposure": float(exposure["Gold"].mean()) if "Gold" in exposure else np.nan,
            "Latest SPY Weight": float(latest_weights.get("SPY", 0.0)),
            "Latest Gold Weight": float(latest_weights.get("Gold", 0.0)),
            "Latest BTC Weight": float(latest_weights.get("BTC", 0.0)),
            "Latest BIL Weight": float(latest_weights.get("BIL", 0.0)),
            "Latest ETF Weight": float(latest_weights.reindex(ETF_UNIVERSE).fillna(0.0).sum()),
            "Latest ETF Assets": ",".join(latest.loc[latest["Asset"].isin(ETF_UNIVERSE), "Asset"].astype(str).tolist()),
        }
    )
    return row


def main() -> None:
    paths = default_paths(ROOT)
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    raw = expanded.load_prices()
    prices = expanded.asset_prices(raw).ffill()
    prices = prices.reindex(columns=["SPY", "Gold", "BTC", "BIL"] + list(ETF_UNIVERSE)).ffill()
    returns = prices.pct_change(fill_method=None).where(prices.notna()).fillna(0.0)
    rules = [
        GoldRule("hold_100", "hold"),
        GoldRule("trend_ma50_below100", "trend", ma_period=50, below_exposure=1.0),
        GoldRule("trend_ma100_below50", "trend", ma_period=100, below_exposure=0.5),
        GoldRule("trend_ma200_below50", "trend", ma_period=200, below_exposure=0.5),
        GoldRule("dd252_warn8_crash20_half", "dd_simple", dd_window=252, warn_dd=-0.08, crash_dd=-0.20, warn_exposure=0.5, crash_exposure=0.5),
        GoldRule("dd252_warn10_crash20_half", "dd_simple", dd_window=252, warn_dd=-0.10, crash_dd=-0.20, warn_exposure=0.5, crash_exposure=0.5),
        GoldRule("dd252_warn8_crash15_50_25", "dd_simple", dd_window=252, warn_dd=-0.08, crash_dd=-0.15, warn_exposure=0.5, crash_exposure=0.25),
        GoldRule("dd504_warn10_crash25_half", "dd_simple", dd_window=504, warn_dd=-0.10, crash_dd=-0.25, warn_exposure=0.5, crash_exposure=0.5),
        GoldRule("dd252_recovery_panic", "dd_recovery", dd_window=252, warn_dd=-0.08, crash_dd=-0.20, warn_exposure=0.5, crash_exposure=0.5, recovery_dd=-0.05, panic_dd=-0.30),
    ]
    configs = [StrategyConfig(f"BIL core ETF top1 gold {rule.name}", rule) for rule in rules]
    curves = {}
    summaries = []
    selections = []
    latest_frames = []
    exposure_frames = []
    for config in configs:
        curve, selection, latest, exposure = run_variant(prices, returns, config)
        curves[config.name] = curve
        summaries.append(metrics_row(curve, config, selection, latest, exposure))
        selections.append(selection)
        latest_frames.append(latest)
        exposure_frames.append(exposure)
    summary = pd.DataFrame(summaries).sort_values(["Sharpe", "CAGR"], ascending=False)
    pd.DataFrame(curves).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_curves.csv")
    summary.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_summary.csv", index=False)
    pd.concat(selections, ignore_index=True).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_selection_history.csv", index=False)
    pd.concat(latest_frames, ignore_index=True).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_latest_weights.csv", index=False)
    pd.concat(exposure_frames, ignore_index=True).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_exposure_history.csv", index=False)
    cols = ["Strategy", "Gold Rule", "CAGR", "Annual Vol", "Sharpe", "Max Drawdown", "Average Gold Exposure", "Latest Gold Weight", "Latest ETF Assets", "Latest ETF Weight"]
    print(summary[cols].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()