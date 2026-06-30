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

from dynamic_factor_copula import compute_port_opt_style_metrics, curve_from_returns, default_paths, monthly_rebalance_dates  # noqa: E402
import run_spy_gold_btc_bil_leverage_etf_sweep as lev  # noqa: E402
import run_spy_gold_btc_bil_country_etf_sweep as country  # noqa: E402

OUTPUT_PREFIX = "spy_gold_btc_bil_cash_deploy_vol_target_sweep"
COUNTRY_UNIVERSE = country.COUNTRY_ETFS


@dataclass(frozen=True)
class Config:
    name: str
    country_mode: str
    country_base_bucket: float
    country_max_bucket: float
    country_top_n: int
    country_funding: str
    vol_target: float
    cash_deploy_mode: str
    cash_deploy_bucket: float
    risk_mode: str


def strong_risk_on(prices: pd.DataFrame, date: pd.Timestamp, mode: str) -> bool:
    spy = prices["SPY"].loc[:date].dropna()
    if len(spy) < 300:
        return False
    latest = float(spy.iloc[-1])
    ma200 = float(spy.iloc[-200:].mean())
    ma300 = float(spy.iloc[-300:].mean())
    dd252 = latest / float(spy.iloc[-252:].max()) - 1.0
    ret63 = latest / float(spy.iloc[-64]) - 1.0
    if mode == "ma300_dd8_ret63":
        return latest > ma300 and dd252 > -0.08 and ret63 > 0.0
    if mode == "ma200_dd8_ret63":
        return latest > ma200 and dd252 > -0.08 and ret63 > 0.0
    if mode == "ma300_only":
        return latest > ma300
    return True


def basket_vol(returns: pd.DataFrame, date: pd.Timestamp, selected: list[str], window: int = 63) -> float:
    if not selected:
        return np.nan
    hist = returns.loc[:date, selected].tail(window).dropna(how="all").fillna(0.0)
    if hist.shape[0] < 20:
        return np.nan
    basket = hist.mean(axis=1)
    return float(basket.std() * np.sqrt(252))


def country_bucket(config: Config, returns: pd.DataFrame, date: pd.Timestamp, selected: list[str]) -> float:
    if not selected or config.country_base_bucket <= 0.0:
        return 0.0
    if config.country_mode == "fixed":
        return config.country_base_bucket
    vol = basket_vol(returns, date, selected)
    if not np.isfinite(vol) or vol <= 1e-12:
        return config.country_base_bucket
    scaled = config.country_base_bucket * config.vol_target / vol
    return float(np.clip(scaled, 0.0, config.country_max_bucket))


def fund(weights: pd.Series, selected: list[str], bucket: float, funding: str) -> pd.Series:
    out = weights.copy()
    if not selected or bucket <= 0.0:
        return out
    if funding == "spy":
        spy_funding, bil_funding = bucket, 0.0
    elif funding == "bil":
        spy_funding, bil_funding = 0.0, bucket
    else:
        spy_funding, bil_funding = bucket * 2.0 / 3.0, bucket / 3.0
    out["SPY"] = out.get("SPY", 0.0) - spy_funding
    out["BIL"] = out.get("BIL", 0.0) - bil_funding
    if out.get("SPY", 0.0) < 0.20 - 1e-12 or out.get("BIL", 0.0) < -1e-12:
        return pd.Series(dtype=float)
    for asset in selected:
        out[asset] = out.get(asset, 0.0) + bucket / len(selected)
    return out / out.sum()


def deploy_cash(weights: pd.Series, selected: list[str], bucket: float, mode: str) -> pd.Series:
    out = weights.copy()
    bucket = min(bucket, float(out.get("BIL", 0.0)))
    if bucket <= 0.0 or mode == "none":
        return out
    out["BIL"] -= bucket
    if mode == "spy":
        out["SPY"] = out.get("SPY", 0.0) + bucket
    elif mode == "country" and selected:
        for asset in selected:
            out[asset] = out.get(asset, 0.0) + bucket / len(selected)
    elif mode == "split" and selected:
        out["SPY"] = out.get("SPY", 0.0) + bucket * 0.5
        for asset in selected:
            out[asset] = out.get(asset, 0.0) + bucket * 0.5 / len(selected)
    else:
        out["SPY"] = out.get("SPY", 0.0) + bucket
    return out / out.sum()


def run_variant(prices: pd.DataFrame, returns: pd.DataFrame, config: Config, candidates: tuple[str, ...]):
    schedule = monthly_rebalance_dates(prices.index, lookback_days=252, freq="ME")
    cols = list(dict.fromkeys(["SPY", "Gold", "BTC", "BIL", "IEF", *candidates]))
    weights = pd.DataFrame(0.0, index=prices.index, columns=cols)
    selections = []
    for idx, date in enumerate(schedule[:-1]):
        next_date = schedule[idx + 1]
        test_idx = prices.index[(prices.index > date) & (prices.index <= next_date)]
        if len(test_idx) == 0:
            continue
        base_w, base_selected = lev.baseline_weights(prices, date, "conservative_cash")
        ranks = lev.momentum_rank(prices, date, candidates)
        selected = ranks.loc[ranks["pass"].fillna(False), "ETF"].head(config.country_top_n).astype(str).tolist()
        bucket = country_bucket(config, returns, date, selected)
        w = fund(base_w, selected, bucket, config.country_funding)
        if w.empty:
            selected = []
            bucket = 0.0
            w = base_w
        deploy_on = strong_risk_on(prices, date, config.risk_mode)
        if deploy_on:
            w = deploy_cash(w, selected, config.cash_deploy_bucket, config.cash_deploy_mode)
        weights.loc[test_idx, w.index] = w.to_numpy(dtype=float)
        selections.append({
            "Strategy": config.name,
            "rebalance_date": date,
            "next_rebalance_date": next_date,
            "base_selected": ",".join(base_selected),
            "country_selected": ",".join(selected),
            "country_count": len(selected),
            "country_bucket": bucket,
            "cash_deploy_on": deploy_applied,
        })
    weights = weights.loc[weights.sum(axis=1).gt(0.0)].copy()
    exposure = pd.DataFrame(1.0, index=weights.index, columns=weights.columns)
    exposure["SPY"] = lev.trend_exposure(prices["SPY"], 300, 0.50).reindex(weights.index).ffill().fillna(1.0)
    exposure["Gold"] = lev.gold_sweep.gold_exposure(prices["Gold"], lev.GOLD_RULE).reindex(weights.index).ffill().fillna(1.0)
    exposure["BTC"] = lev.trend_exposure(prices["BTC"], 50, 0.00).reindex(weights.index).ffill().fillna(1.0)
    effective = weights * exposure
    strat_ret = returns.reindex(effective.index).fillna(0.0).mul(effective, axis=1).sum(axis=1)
    curve = curve_from_returns(strat_ret, initial=lev.INITIAL_VALUE).rename(config.name)
    latest = effective.iloc[-1].rename("Effective Weight").reset_index().rename(columns={"index": "Asset"})
    latest = latest.loc[latest["Effective Weight"].abs().gt(1e-12)].copy()
    latest.insert(0, "Strategy", config.name)
    latest.insert(1, "Date", effective.index.max())
    return curve, pd.DataFrame(selections), latest


def metrics_row(curve: pd.Series, config: Config, sel: pd.DataFrame, latest: pd.DataFrame, candidates: tuple[str, ...]) -> dict[str, object]:
    clean = curve.dropna()
    row = compute_port_opt_style_metrics(clean, risk_free_rate=lev.RISK_FREE_RATE).to_dict()
    latest_weights = latest.set_index("Asset")["Effective Weight"] if not latest.empty else pd.Series(dtype=float)
    active = int(sel["country_count"].gt(0).sum()) if not sel.empty else 0
    total = int(sel.shape[0]) if not sel.empty else 0
    row.update({
        "Strategy": config.name,
        "Country Mode": config.country_mode,
        "Country Base Bucket": config.country_base_bucket,
        "Country Max Bucket": config.country_max_bucket,
        "Country Top N": config.country_top_n,
        "Country Funding": config.country_funding,
        "Vol Target": config.vol_target,
        "Cash Deploy Mode": config.cash_deploy_mode,
        "Cash Deploy Bucket": config.cash_deploy_bucket,
        "Risk Mode": config.risk_mode,
        "Start": clean.index.min().date().isoformat(),
        "End": clean.index.max().date().isoformat(),
        "Country Active Rate": active / total if total else np.nan,
        "Cash Deploy Active Rate": float(sel["cash_deploy_on"].mean()) if not sel.empty else np.nan,
        "Average Country Bucket": float(sel["country_bucket"].mean()) if not sel.empty else np.nan,
        "Latest SPY Weight": float(latest_weights.get("SPY", 0.0)),
        "Latest Gold Weight": float(latest_weights.get("Gold", 0.0)),
        "Latest BTC Weight": float(latest_weights.get("BTC", 0.0)),
        "Latest BIL Weight": float(latest_weights.get("BIL", 0.0)),
        "Latest Country Weight": float(latest_weights.reindex(candidates).fillna(0.0).sum()),
        "Latest Country Assets": ",".join(latest.loc[latest["Asset"].isin(candidates), "Asset"].astype(str).tolist()),
    })
    return row


def main() -> None:
    paths = default_paths(ROOT)
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    raw = country.load_prices()
    prices = country.asset_prices(raw).ffill()
    returns = prices.pct_change(fill_method=None).where(prices.notna()).fillna(0.0)
    candidates = tuple(asset for asset in COUNTRY_UNIVERSE if asset in prices and prices[asset].dropna().shape[0] >= 2520)
    configs = []
    for country_mode in ["fixed", "vol_target"]:
        for country_base_bucket, country_max_bucket in [(0.05, 0.05), (0.05, 0.10), (0.075, 0.10)]:
            if country_mode == "fixed" and country_base_bucket != country_max_bucket:
                continue
            for top_n in [1, 2]:
                for funding in ["spy", "bil", "spy2_bil1"]:
                    for vol_target in [0.10, 0.12, 0.15]:
                        if country_mode == "fixed" and vol_target != 0.10:
                            continue
                        for deploy_mode in ["none", "spy", "country", "split"]:
                            for deploy_bucket in [0.0, 0.05, 0.10]:
                                if deploy_mode == "none" and deploy_bucket != 0.0:
                                    continue
                                if deploy_mode != "none" and deploy_bucket == 0.0:
                                    continue
                                for risk_mode in ["ma300_dd8_ret63", "ma200_dd8_ret63"]:
                                    name = (
                                        f"cons country_{country_mode} base{country_base_bucket:.1%} max{country_max_bucket:.1%} "
                                        f"top{top_n} fund_{funding} vt{vol_target:.0%} deploy_{deploy_mode}{deploy_bucket:.0%} {risk_mode}"
                                    )
                                    configs.append(Config(name, country_mode, country_base_bucket, country_max_bucket, top_n, funding, vol_target, deploy_mode, deploy_bucket, risk_mode))
    curves = {}
    summaries = []
    selections = []
    latest_frames = []
    for config in configs:
        curve, sel, latest = run_variant(prices, returns, config, candidates)
        curves[config.name] = curve
        summaries.append(metrics_row(curve, config, sel, latest, candidates))
        selections.append(sel)
        latest_frames.append(latest)
    summary = pd.DataFrame(summaries).sort_values(["Sharpe", "CAGR"], ascending=False)
    summary.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_summary.csv", index=False)
    pd.DataFrame(curves).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_curves.csv")
    pd.concat(selections, ignore_index=True).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_selection_history.csv", index=False)
    pd.concat(latest_frames, ignore_index=True).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_latest_weights.csv", index=False)
    cols = ["Strategy", "CAGR", "Annual Vol", "Sharpe", "Max Drawdown", "Country Active Rate", "Cash Deploy Active Rate", "Average Country Bucket", "Latest Country Assets", "Latest Country Weight", "Latest BIL Weight"]
    print(summary[cols].head(50).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()

