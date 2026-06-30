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

from dynamic_factor_copula import (  # noqa: E402
    compute_port_opt_style_metrics,
    curve_from_returns,
    default_paths,
    lag_close_signal_to_next_session,
    monthly_rebalance_dates,
)
import run_spy_gold_btc_bil_country_etf_sweep as country  # noqa: E402
import run_spy_gold_btc_bil_leverage_etf_sweep as lev  # noqa: E402

OUTPUT_PREFIX = "spy_gold_btc_bil_adaptive_gold_country_winner"
COUNTRY_UNIVERSE = country.COUNTRY_ETFS


@dataclass(frozen=True)
class GoldRule:
    name: str
    mode: str
    ma_period: int = 100
    below_exposure: float = 0.50
    dd_window: int = 252
    warn_dd: float = -0.08
    crash_dd: float = -0.20
    warn_exposure: float = 0.50
    crash_exposure: float = 0.50
    recovery_dd: float = -0.05
    panic_dd: float = -0.30


@dataclass(frozen=True)
class Config:
    name: str
    gold_rule: GoldRule
    gold_weight_mode: str
    gold_boost: float
    boost_funding: str


def trend_exposure(price: pd.Series, ma_period: int, below_exposure: float) -> pd.Series:
    price = price.astype(float).sort_index().ffill()
    ma = price.rolling(ma_period, min_periods=max(20, int(ma_period * 0.20))).mean()
    signal = pd.Series(1.0, index=price.index, dtype=float)
    signal.loc[price < ma] = below_exposure
    signal.loc[ma.isna()] = 1.0
    return lag_close_signal_to_next_session(signal, initial=1.0)


def gold_exposure(price: pd.Series, rule: GoldRule) -> pd.Series:
    price = price.astype(float).sort_index().ffill()
    if rule.mode == "hold":
        return pd.Series(1.0, index=price.index, dtype=float)
    if rule.mode == "trend":
        return trend_exposure(price, rule.ma_period, rule.below_exposure)
    dd = price / price.rolling(rule.dd_window, min_periods=max(60, rule.dd_window // 4)).max() - 1.0
    signal = pd.Series(1.0, index=price.index, dtype=float)
    if rule.mode == "dd_simple":
        signal.loc[dd <= rule.warn_dd] = rule.warn_exposure
        signal.loc[dd <= rule.crash_dd] = rule.crash_exposure
        return lag_close_signal_to_next_session(signal, initial=1.0)
    if rule.mode == "dd_recovery":
        state = 1.0
        out = pd.Series(1.0, index=price.index, dtype=float)
        ma200 = price.rolling(200, min_periods=50).mean()
        mom63 = price / price.shift(63) - 1.0
        for dt in price.index:
            current_dd = float(dd.loc[dt]) if pd.notna(dd.loc[dt]) else 0.0
            panic = (
                current_dd <= rule.panic_dd
                and pd.notna(ma200.loc[dt])
                and price.loc[dt] < ma200.loc[dt]
                and pd.notna(mom63.loc[dt])
                and mom63.loc[dt] < 0.0
            )
            if panic or current_dd <= rule.crash_dd:
                state = rule.crash_exposure
            elif current_dd <= rule.warn_dd:
                state = rule.warn_exposure
            elif current_dd >= rule.recovery_dd:
                state = 1.0
            out.loc[dt] = state
        return lag_close_signal_to_next_session(out, initial=1.0)
    raise ValueError(f"Unknown gold rule: {rule.mode}")


def spy_risk_off(prices: pd.DataFrame, date: pd.Timestamp, mode: str) -> bool:
    if mode == "base30":
        return False
    spy = prices["SPY"].loc[:date].dropna()
    if len(spy) < 300:
        return False
    latest = float(spy.iloc[-1])
    ma200 = float(spy.iloc[-200:].mean())
    ma300 = float(spy.iloc[-300:].mean())
    dd252 = latest / float(spy.iloc[-252:].max()) - 1.0
    if mode == "spy_below_ma300":
        return latest < ma300
    if mode == "spy_below_ma200_or_dd8":
        return latest < ma200 or dd252 <= -0.08
    if mode == "spy_below_ma300_or_dd10":
        return latest < ma300 or dd252 <= -0.10
    return False


def apply_gold_boost(weights: pd.Series, amount: float, funding: str) -> pd.Series:
    out = weights.copy()
    if amount <= 0.0:
        return out
    if funding == "bil":
        actual = min(amount, float(out.get("BIL", 0.0)))
        out["BIL"] -= actual
    elif funding == "spy":
        actual = min(amount, max(0.0, float(out.get("SPY", 0.0)) - 0.20))
        out["SPY"] -= actual
    else:
        bil_part = min(amount * 0.5, float(out.get("BIL", 0.0)))
        spy_part = min(amount - bil_part, max(0.0, float(out.get("SPY", 0.0)) - 0.20))
        actual = bil_part + spy_part
        out["BIL"] -= bil_part
        out["SPY"] -= spy_part
    out["Gold"] = out.get("Gold", 0.0) + actual
    return out / out.sum()


def run_variant(prices: pd.DataFrame, returns: pd.DataFrame, config: Config, candidates: tuple[str, ...]):
    schedule = monthly_rebalance_dates(prices.index, lookback_days=252, freq="ME")
    cols = list(dict.fromkeys(["SPY", "Gold", "BTC", "BIL", *candidates]))
    weights = pd.DataFrame(0.0, index=prices.index, columns=cols)
    selections = []
    for idx, date in enumerate(schedule[:-1]):
        next_date = schedule[idx + 1]
        test_idx = prices.index[(prices.index > date) & (prices.index <= next_date)]
        if len(test_idx) == 0:
            continue
        base_w, base_selected = lev.baseline_weights(prices, date, "conservative_cash")
        ranks = lev.momentum_rank(prices, date, candidates)
        selected = ranks.loc[ranks["pass"].fillna(False), "ETF"].head(2).astype(str).tolist()
        w = lev.funded_weights(base_w, selected, 0.05, "spy")
        if w.empty:
            selected = []
            w = base_w
        boost_on = spy_risk_off(prices, date, config.gold_weight_mode)
        if boost_on:
            w = apply_gold_boost(w, config.gold_boost, config.boost_funding)
        weights.loc[test_idx, w.index] = w.to_numpy(dtype=float)
        selections.append(
            {
                "Strategy": config.name,
                "rebalance_date": date,
                "next_rebalance_date": next_date,
                "base_selected": ",".join(base_selected),
                "country_selected": ",".join(selected),
                "country_count": len(selected),
                "gold_boost_on": boost_on,
            }
        )
    weights = weights.loc[weights.sum(axis=1).gt(0.0)].copy()
    exposure = pd.DataFrame(1.0, index=weights.index, columns=weights.columns)
    exposure["SPY"] = trend_exposure(prices["SPY"], 300, 0.50).reindex(weights.index).ffill().fillna(1.0)
    exposure["Gold"] = gold_exposure(prices["Gold"], config.gold_rule).reindex(weights.index).ffill().fillna(1.0)
    exposure["BTC"] = trend_exposure(prices["BTC"], 50, 0.00).reindex(weights.index).ffill().fillna(1.0)
    effective = weights * exposure
    strat_ret = returns.reindex(effective.index).fillna(0.0).mul(effective, axis=1).sum(axis=1)
    curve = curve_from_returns(strat_ret, initial=lev.INITIAL_VALUE).rename(config.name)
    latest = effective.iloc[-1].rename("Effective Weight").reset_index().rename(columns={"index": "Asset"})
    latest = latest.loc[latest["Effective Weight"].abs().gt(1e-12)].copy()
    latest.insert(0, "Strategy", config.name)
    latest.insert(1, "Date", effective.index.max())
    return curve, pd.DataFrame(selections), latest, exposure


def metrics_row(curve: pd.Series, config: Config, sel: pd.DataFrame, latest: pd.DataFrame, exposure: pd.DataFrame, candidates: tuple[str, ...]) -> dict[str, object]:
    clean = curve.dropna()
    row = compute_port_opt_style_metrics(clean, risk_free_rate=lev.RISK_FREE_RATE).to_dict()
    latest_weights = latest.set_index("Asset")["Effective Weight"] if not latest.empty else pd.Series(dtype=float)
    total = int(sel.shape[0]) if not sel.empty else 0
    row.update(
        {
            "Strategy": config.name,
            "Gold Rule": config.gold_rule.name,
            "Gold Weight Mode": config.gold_weight_mode,
            "Gold Boost": config.gold_boost,
            "Boost Funding": config.boost_funding,
            "Start": clean.index.min().date().isoformat(),
            "End": clean.index.max().date().isoformat(),
            "Country Active Rate": float(sel["country_count"].gt(0).mean()) if total else np.nan,
            "Gold Boost Active Rate": float(sel["gold_boost_on"].mean()) if total else np.nan,
            "Average Gold Exposure": float(exposure["Gold"].mean()),
            "Latest SPY Weight": float(latest_weights.get("SPY", 0.0)),
            "Latest Gold Weight": float(latest_weights.get("Gold", 0.0)),
            "Latest BTC Weight": float(latest_weights.get("BTC", 0.0)),
            "Latest BIL Weight": float(latest_weights.get("BIL", 0.0)),
            "Latest Country Weight": float(latest_weights.reindex(candidates).fillna(0.0).sum()),
            "Latest Country Assets": ",".join(latest.loc[latest["Asset"].isin(candidates), "Asset"].astype(str).tolist()),
        }
    )
    return row


def main() -> None:
    paths = default_paths(ROOT)
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    raw = country.load_prices()
    prices = country.asset_prices(raw).ffill()
    returns = prices.pct_change(fill_method=None).where(prices.notna()).fillna(0.0)
    candidates = tuple(asset for asset in COUNTRY_UNIVERSE if asset in prices and prices[asset].dropna().shape[0] >= 2520)
    rules = [
        GoldRule("hold_100", "hold"),
        GoldRule("dd252_warn8_crash20_half", "dd_simple", warn_dd=-0.08, crash_dd=-0.20, warn_exposure=0.50, crash_exposure=0.50),
        GoldRule("dd252_warn8_crash15_50_25", "dd_simple", warn_dd=-0.08, crash_dd=-0.15, warn_exposure=0.50, crash_exposure=0.25),
        GoldRule("dd252_recovery_panic", "dd_recovery", warn_dd=-0.08, crash_dd=-0.20, warn_exposure=0.50, crash_exposure=0.50),
        GoldRule("trend_ma100_below50", "trend", ma_period=100, below_exposure=0.50),
    ]
    configs: list[Config] = []
    for rule in rules:
        configs.append(Config(f"country5_top2 gold_{rule.name} base30", rule, "base30", 0.0, "bil"))
        for mode in ["spy_below_ma300", "spy_below_ma200_or_dd8", "spy_below_ma300_or_dd10"]:
            for boost in [0.05, 0.10]:
                for funding in ["bil", "spy", "half"]:
                    configs.append(Config(f"country5_top2 gold_{rule.name} boost{boost:.0%}_{mode}_from_{funding}", rule, mode, boost, funding))

    curves = {}
    summaries = []
    selections = []
    latest_frames = []
    for config in configs:
        curve, sel, latest, exposure = run_variant(prices, returns, config, candidates)
        curves[config.name] = curve
        summaries.append(metrics_row(curve, config, sel, latest, exposure, candidates))
        selections.append(sel)
        latest_frames.append(latest)

    summary = pd.DataFrame(summaries).sort_values(["Sharpe", "CAGR"], ascending=False)
    summary.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_summary.csv", index=False)
    pd.DataFrame(curves).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_curves.csv")
    pd.concat(selections, ignore_index=True).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_selection_history.csv", index=False)
    pd.concat(latest_frames, ignore_index=True).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_latest_weights.csv", index=False)
    cols = [
        "Strategy",
        "CAGR",
        "Annual Vol",
        "Sharpe",
        "Max Drawdown",
        "Gold Boost Active Rate",
        "Average Gold Exposure",
        "Latest Gold Weight",
        "Latest BIL Weight",
        "Latest Country Assets",
        "Latest Country Weight",
    ]
    print(summary[cols].head(50).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
