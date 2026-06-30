from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for path in [SCRIPTS, SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dynamic_factor_copula import compute_port_opt_style_metrics, curve_from_returns, default_paths, monthly_rebalance_dates  # noqa: E402
import run_spy_gold_btc_bil_adaptive_gold_country_winner as adaptive  # noqa: E402
import run_spy_gold_btc_bil_country_etf_sweep as country  # noqa: E402
import run_spy_gold_btc_bil_leverage_etf_sweep as lev  # noqa: E402

OUTPUT_PREFIX = "spy_gold_btc_bil_step4_exposure_rebalance_frequency"


@dataclass(frozen=True)
class Config:
    exposure_freq: str
    rebalance_freq: str

    @property
    def name(self) -> str:
        return f"step4 exposure_{self.exposure_freq} rebalance_{self.rebalance_freq}"


def dd_rule() -> adaptive.GoldRule:
    return adaptive.GoldRule(
        "dd252_warn8_crash20_half",
        "dd_simple",
        warn_dd=-0.08,
        crash_dd=-0.20,
        warn_exposure=0.50,
        crash_exposure=0.50,
    )


def scheduled_exposure(raw: pd.Series, index: pd.DatetimeIndex, freq: str) -> pd.Series:
    raw = raw.reindex(index).ffill().fillna(1.0)
    if freq == "D":
        return raw
    update_dates = monthly_rebalance_dates(index, lookback_days=252, freq=freq)
    scheduled = pd.Series(np.nan, index=index, dtype=float)
    scheduled.loc[update_dates] = raw.reindex(update_dates).to_numpy(dtype=float)
    return scheduled.ffill().fillna(1.0)


def build_exposure(prices: pd.DataFrame, weights_index: pd.DatetimeIndex, freq: str) -> pd.DataFrame:
    raw = pd.DataFrame(1.0, index=prices.index, columns=["SPY", "Gold", "BTC"])
    raw["SPY"] = adaptive.trend_exposure(prices["SPY"], 300, 0.50)
    raw["Gold"] = adaptive.gold_exposure(prices["Gold"], dd_rule())
    raw["BTC"] = adaptive.trend_exposure(prices["BTC"], 50, 0.00)
    out = pd.DataFrame(1.0, index=weights_index, columns=["SPY", "Gold", "BTC"])
    for asset in ["SPY", "Gold", "BTC"]:
        out[asset] = scheduled_exposure(raw[asset], prices.index, freq).reindex(weights_index).ffill().fillna(1.0)
    return out


def run_variant(prices: pd.DataFrame, returns: pd.DataFrame, candidates: tuple[str, ...], config: Config):
    schedule = monthly_rebalance_dates(prices.index, lookback_days=252, freq=config.rebalance_freq)
    cols = list(dict.fromkeys(["SPY", "Gold", "BTC", "BIL", *candidates]))
    weights = pd.DataFrame(0.0, index=prices.index, columns=cols)
    selections = []
    for idx, date in enumerate(schedule[:-1]):
        next_date = schedule[idx + 1]
        test_idx = prices.index[(prices.index > date) & (prices.index <= next_date)]
        if len(test_idx) == 0:
            continue
        try:
            base_w, base_selected = lev.baseline_weights(prices, date, "conservative_cash")
        except KeyError:
            base_w = pd.Series({"SPY": 0.45, "Gold": 0.30, "BTC": 0.10, "BIL": 0.15}, dtype=float)
            base_selected = []
        try:
            ranks = lev.momentum_rank(prices, date, candidates)
            selected = ranks.loc[ranks["pass"].fillna(False), "ETF"].head(2).astype(str).tolist()
        except KeyError:
            selected = []
        w = lev.funded_weights(base_w, selected, 0.08, "spy")
        if w.empty:
            selected = []
            w = base_w
        boost_on = adaptive.spy_risk_off(prices, date, "spy_below_ma200_or_dd8")
        if boost_on:
            w = adaptive.apply_gold_boost(w, 0.16, "spy")
        weights.loc[test_idx, w.index] = w.to_numpy(dtype=float)
        selections.append(
            {
                "Strategy": config.name,
                "Exposure Freq": config.exposure_freq,
                "Rebalance Freq": config.rebalance_freq,
                "rebalance_date": date,
                "next_rebalance_date": next_date,
                "base_selected": ",".join(base_selected),
                "country_selected": ",".join(selected),
                "country_count": len(selected),
                "gold_boost_on": boost_on,
            }
        )
    weights = weights.loc[weights.sum(axis=1).gt(0.0)].copy()
    exposure = build_exposure(prices, weights.index, config.exposure_freq)
    effective = weights.copy()
    for asset in ["SPY", "Gold", "BTC"]:
        effective[asset] *= exposure[asset]
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
    row.update(
        {
            "Strategy": config.name,
            "Exposure Freq": config.exposure_freq,
            "Rebalance Freq": config.rebalance_freq,
            "Start": clean.index.min().date().isoformat(),
            "End": clean.index.max().date().isoformat(),
            "Country Active Rate": float(sel["country_count"].gt(0).mean()) if not sel.empty else np.nan,
            "Gold Boost Active Rate": float(sel["gold_boost_on"].mean()) if not sel.empty else np.nan,
            "Average SPY Exposure": float(exposure["SPY"].mean()),
            "Average Gold Exposure": float(exposure["Gold"].mean()),
            "Average BTC Exposure": float(exposure["BTC"].mean()),
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
    candidates = tuple(asset for asset in country.COUNTRY_ETFS if asset in prices and prices[asset].dropna().shape[0] >= 2520)
    configs = [
        Config(exposure_freq, rebalance_freq)
        for exposure_freq in ["D", "W-FRI", "ME"]
        for rebalance_freq in ["W-FRI", "2W-FRI", "ME", "QE"]
    ]
    curves = {}
    summaries = []
    selections = []
    latest_frames = []
    for config in configs:
        curve, sel, latest, exposure = run_variant(prices, returns, candidates, config)
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
        "Exposure Freq",
        "Rebalance Freq",
        "CAGR",
        "Annual Vol",
        "Sharpe",
        "Max Drawdown",
        "Country Active Rate",
        "Gold Boost Active Rate",
        "Latest Country Assets",
    ]
    print(summary[cols].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
