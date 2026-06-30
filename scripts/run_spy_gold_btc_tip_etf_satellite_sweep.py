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
import run_spy_gold_btc_tip_etf_satellite as base  # noqa: E402

OUTPUT_PREFIX = "spy_gold_btc_tip_etf_satellite_sweep"
CORE_WEIGHTS = base.CORE_WEIGHTS
ALL_ETFS = ["SPMO", "MTUM", "SCHG", "XLK", "EWT", "EWY", "EWJ", "INDA", "MCHI"]
UNIVERSE_VARIANTS = {
    "all": ALL_ETFS,
    "no_china": ["SPMO", "MTUM", "SCHG", "XLK", "EWT", "EWY", "EWJ", "INDA"],
    "core_liquid": ["SPMO", "MTUM", "SCHG", "XLK", "EWY", "EWJ", "INDA"],
    "us_growth_asia": ["SPMO", "SCHG", "XLK", "EWT", "EWY", "EWJ", "INDA"],
}
BUCKETS = [0.125, 0.15, 0.175, 0.20]
FUNDING_MODES = ["half", "tip", "spy", "spy2_tip1"]
TOP_NS = [1, 2, 3]
RISK_FREE_RATE = base.RISK_FREE_RATE
INITIAL_VALUE = base.INITIAL_VALUE


@dataclass(frozen=True)
class SweepConfig:
    universe_name: str
    etfs: tuple[str, ...]
    bucket: float
    top_n: int
    funding_mode: str
    score_gap: float = 0.0

    @property
    def name(self) -> str:
        bps = int(round(self.bucket * 1000)) / 10
        gap = f" gap{self.score_gap:.2f}" if self.score_gap > 0 else ""
        return f"ETF sat {self.universe_name} bucket{bps:g} top{self.top_n} fund_{self.funding_mode}{gap}"


def momentum_rank(prices: pd.DataFrame, rebalance_date: pd.Timestamp, etfs: tuple[str, ...]) -> pd.DataFrame:
    history = prices.loc[:rebalance_date, list(etfs)].ffill()
    rows = []
    for asset in etfs:
        series = history[asset].dropna() if asset in history else pd.Series(dtype=float)
        if len(series) < 253:
            rows.append({"ETF": asset, "pass": False, "score": np.nan})
            continue
        latest = float(series.iloc[-1])
        ret_1m = latest / float(series.iloc[-22]) - 1.0
        ret_3m = latest / float(series.iloc[-64]) - 1.0
        ret_6m = latest / float(series.iloc[-127]) - 1.0
        ret_12m = latest / float(series.iloc[-253]) - 1.0
        sma200 = float(series.iloc[-200:].mean())
        rows.append(
            {
                "ETF": asset,
                "price": latest,
                "ret_1m": ret_1m,
                "ret_3m": ret_3m,
                "ret_6m": ret_6m,
                "ret_12m": ret_12m,
                "sma200": sma200,
                "pass": latest > sma200 and ret_3m > 0.0 and ret_6m > 0.0,
            }
        )
    frame = pd.DataFrame(rows)
    score = pd.Series(0.0, index=frame.index, dtype=float)
    for column, weight in {"ret_1m": 0.20, "ret_3m": 0.30, "ret_6m": 0.40, "ret_12m": 0.10}.items():
        score += weight * frame[column].rank(pct=True).fillna(0.0)
    frame["score"] = score
    return frame.sort_values(["pass", "score"], ascending=[False, False]).reset_index(drop=True)


def selected_from_rank(ranks: pd.DataFrame, top_n: int, score_gap: float) -> list[str]:
    passing = ranks.loc[ranks["pass"].fillna(False)].copy()
    if passing.empty:
        return []
    if score_gap > 0 and len(passing) > 1:
        gap = float(passing.iloc[0]["score"] - passing.iloc[1]["score"])
        if gap < score_gap:
            return []
    return passing.head(top_n)["ETF"].astype(str).tolist()


def fund_core(bucket: float, funding_mode: str) -> pd.Series:
    weights = pd.Series(CORE_WEIGHTS, dtype=float)
    if bucket <= 0:
        return weights
    if funding_mode == "tip":
        tip_funding = bucket
        spy_funding = 0.0
    elif funding_mode == "spy":
        tip_funding = 0.0
        spy_funding = bucket
    elif funding_mode == "spy2_tip1":
        spy_funding = bucket * 2.0 / 3.0
        tip_funding = bucket - spy_funding
    else:
        tip_funding = min(CORE_WEIGHTS["TIP"], bucket * 0.50)
        spy_funding = bucket - tip_funding
    weights["SPY"] -= spy_funding
    weights["TIP"] -= tip_funding
    if weights["SPY"] < 0.30 - 1e-12 or weights["TIP"] < -1e-12:
        return pd.Series(dtype=float)
    return weights


def weights_for(selected: list[str], config: SweepConfig) -> pd.Series:
    if not selected:
        return pd.Series(CORE_WEIGHTS, dtype=float)
    weights = fund_core(config.bucket, config.funding_mode)
    if weights.empty:
        return weights
    for asset in selected:
        weights.loc[asset] = config.bucket / len(selected)
    return weights / weights.sum()


def run_variant(prices: pd.DataFrame, returns: pd.DataFrame, exposure: pd.DataFrame, config: SweepConfig) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    schedule = monthly_rebalance_dates(prices.index, lookback_days=252, freq="ME")
    all_assets = ["SPY", "Gold", "BTC", "TIP"] + ALL_ETFS
    daily_weights = pd.DataFrame(0.0, index=prices.index, columns=all_assets)
    selection_rows = []
    for idx, rebalance_date in enumerate(schedule[:-1]):
        next_date = schedule[idx + 1]
        test_index = prices.index[(prices.index > rebalance_date) & (prices.index <= next_date)]
        if len(test_index) == 0:
            continue
        ranks = momentum_rank(prices, rebalance_date, config.etfs)
        selected = selected_from_rank(ranks, config.top_n, config.score_gap)
        monthly_weights = weights_for(selected, config)
        if monthly_weights.empty:
            continue
        daily_weights.loc[test_index, monthly_weights.index] = monthly_weights.to_numpy(dtype=float)
        selection_rows.append(
            {
                "Strategy": config.name,
                "rebalance_date": rebalance_date,
                "next_rebalance_date": next_date,
                "selected_etfs": ",".join(selected),
                "selected_count": len(selected),
                "bucket_used": config.bucket if selected else 0.0,
                "raw_spy_weight": float(monthly_weights.get("SPY", 0.0)),
                "raw_tip_weight": float(monthly_weights.get("TIP", 0.0)),
            }
        )
    daily_weights = daily_weights.loc[daily_weights.sum(axis=1).gt(0.0)].copy()
    effective = daily_weights.copy()
    for asset in ["SPY", "Gold", "BTC"]:
        effective[asset] *= exposure.reindex(effective.index)[asset].ffill().fillna(1.0)
    strategy_returns = returns.reindex(effective.index).fillna(0.0).mul(effective, axis=1).sum(axis=1)
    curve = curve_from_returns(strategy_returns, initial=INITIAL_VALUE).rename(config.name)
    latest = effective.iloc[-1].rename("Effective Weight").reset_index().rename(columns={"index": "Asset"})
    latest = latest.loc[latest["Effective Weight"].abs().gt(1e-12)].copy()
    latest.insert(0, "Strategy", config.name)
    latest.insert(1, "Date", effective.index.max())
    return curve, pd.DataFrame(selection_rows), latest


def metrics_row(curve: pd.Series, config: SweepConfig, selection: pd.DataFrame, latest: pd.DataFrame) -> dict[str, object]:
    clean = curve.dropna()
    row = compute_port_opt_style_metrics(clean, risk_free_rate=RISK_FREE_RATE).to_dict()
    selected_months = int(selection["selected_count"].gt(0).sum()) if not selection.empty else 0
    total_months = int(selection.shape[0]) if not selection.empty else 0
    latest_weights = latest.set_index("Asset")["Effective Weight"] if not latest.empty else pd.Series(dtype=float)
    row.update(
        {
            "Strategy": config.name,
            "Universe": config.universe_name,
            "ETF Bucket": config.bucket,
            "Top N": config.top_n,
            "Funding Mode": config.funding_mode,
            "Score Gap": config.score_gap,
            "Start": clean.index.min().date().isoformat(),
            "End": clean.index.max().date().isoformat(),
            "Active Months": selected_months,
            "Total Months": total_months,
            "Active Rate": selected_months / total_months if total_months else np.nan,
            "Latest SPY Weight": float(latest_weights.get("SPY", 0.0)),
            "Latest Gold Weight": float(latest_weights.get("Gold", 0.0)),
            "Latest BTC Weight": float(latest_weights.get("BTC", 0.0)),
            "Latest TIP Weight": float(latest_weights.get("TIP", 0.0)),
            "Latest ETF Weight": float(latest_weights.reindex(ALL_ETFS).fillna(0.0).sum()),
            "Latest ETF Assets": ",".join(latest.loc[latest["Asset"].isin(ALL_ETFS), "Asset"].astype(str).tolist()),
        }
    )
    return row


def main() -> None:
    paths = default_paths(ROOT)
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    prices = base.asset_prices_from_tickers(base.load_prices()).ffill()
    returns = prices.pct_change(fill_method=None).where(prices.notna()).fillna(0.0)
    exposure = base.build_exposures(prices)
    configs = []
    for universe_name, etfs in UNIVERSE_VARIANTS.items():
        for bucket in BUCKETS:
            for funding_mode in FUNDING_MODES:
                for top_n in TOP_NS:
                    configs.append(SweepConfig(universe_name, tuple(etfs), bucket, top_n, funding_mode))
    # Small confirm/hold proxy: only allocate when top score clearly beats runner-up.
    for gap in [0.03, 0.05, 0.08]:
        configs.append(SweepConfig("all", tuple(ALL_ETFS), 0.15, 1, "half", score_gap=gap))

    curves = {}
    summaries = []
    selections = []
    latest_frames = []
    for config in configs:
        curve, selection, latest = run_variant(prices, returns, exposure, config)
        curves[config.name] = curve
        summaries.append(metrics_row(curve, config, selection, latest))
        if not selection.empty:
            selections.append(selection)
        latest_frames.append(latest)

    summary = pd.DataFrame(summaries).sort_values(["Sharpe", "CAGR"], ascending=False)
    pd.DataFrame(curves).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_curves.csv")
    summary.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_summary.csv", index=False)
    pd.concat(selections, ignore_index=True).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_selection_history.csv", index=False)
    pd.concat(latest_frames, ignore_index=True).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_latest_weights.csv", index=False)
    cols = ["Strategy", "CAGR", "Annual Vol", "Sharpe", "Max Drawdown", "Active Rate", "Latest ETF Assets", "Latest ETF Weight"]
    print(summary[cols].head(25).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()