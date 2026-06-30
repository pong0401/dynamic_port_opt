from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_us_th_one_model_us70_th30_concentration_sweep as base  # noqa: E402


PREFIX = "us_th_one_model_us70_th30_theme_cap_sweep"
EVALUATION_START = pd.Timestamp('1900-01-01')
STRICT_AI_TECH_BUCKET = {
    "NVDA",
    "MU",
    "INTC",
    "TXN",
    "QCOM",
    "AMD",
    "AAPL",
    "GOOG",
    "GOOGL",
}
THEME_CASES = [
    ("Baseline", None),
    ("AI-tech cap 40%", 0.40),
    ("AI-tech cap 35%", 0.35),
    ("AI-tech cap 30%", 0.30),
    ("AI-tech cap 25%", 0.25),
    ("AI-tech cap 20%", 0.20),
]
CASE_THEME_CAP = dict(THEME_CASES)
_ORIGINAL_METRICS_ROW = None
_ORIGINAL_LOAD_PANEL = None

class ThemeCase(base.SweepCase):
    @property
    def strategy(self) -> str:
        return (
            "One-model US cap 70% / TH cap 30% "
            f"stockcap{int(round(self.stock_cap * 100))} "
            f"penalty{self.concentration_penalty:g} "
            f"assets{self.us_assets} {self.label}"
        )

def _build_cases() -> list[base.SweepCase]:
    return [
        ThemeCase(label, 0.05, 0.02, 50, 50)
        for label, _theme_cap in THEME_CASES
    ]


def _optimize_with_theme_cap(
    cov: pd.DataFrame,
    momentum_signal: pd.Series,
    us_assets: list[str],
    th_assets: list[str],
    case: base.SweepCase,
    th_is_on: bool,
) -> pd.Series:
    theme_cap = CASE_THEME_CAP.get(case.label)
    base_assets = cov.index.tolist()
    cash_cap = max(0.0, 1.0 - (base.US_GROUP_CAP + (base.TH_GROUP_CAP if th_is_on else 0.0) + base.GOLD_CAP + base.BTC_CAP))
    assets = base_assets + ([base.CASH_ASSET] if cash_cap > base.EPSILON else [])
    cov2 = cov.reindex(index=assets, columns=assets).fillna(0.0)
    if base.CASH_ASSET in cov2.index:
        cov2.loc[base.CASH_ASSET, base.CASH_ASSET] = base.EPSILON
    mu = momentum_signal.reindex(assets).fillna(0.0)
    mu = mu.clip(mu.quantile(0.10), mu.quantile(0.90))

    caps = pd.Series(case.stock_cap, index=assets, dtype=float)
    if "GC=F" in caps.index:
        caps.loc["GC=F"] = base.GOLD_CAP
    if "BTC-USD" in caps.index:
        caps.loc["BTC-USD"] = base.BTC_CAP
    if base.CASH_ASSET in caps.index:
        caps.loc[base.CASH_ASSET] = cash_cap
    if float(caps.sum()) < 1.0 - base.EPSILON:
        raise RuntimeError(f"Infeasible caps for {case.label}: {caps.sum():.4f}")

    cov_matrix = cov2.to_numpy(dtype=float)
    mu_vec = mu.to_numpy(dtype=float)
    x0 = (caps / caps.sum()).to_numpy(dtype=float)
    bounds = [(0.0, float(caps.loc[asset])) for asset in assets]
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
    if us_assets:
        us_idx = [assets.index(asset) for asset in us_assets if asset in assets]
        constraints.append({"type": "ineq", "fun": lambda x, idx=us_idx: base.US_GROUP_CAP - float(np.sum(x[idx]))})
    if th_assets:
        th_idx = [assets.index(asset) for asset in th_assets if asset in assets]
        constraints.append({"type": "ineq", "fun": lambda x, idx=th_idx: base.TH_GROUP_CAP - float(np.sum(x[idx]))})
    theme_idx = [assets.index(asset) for asset in STRICT_AI_TECH_BUCKET if asset in assets]
    if theme_cap is not None and theme_idx:
        constraints.append({"type": "ineq", "fun": lambda x, idx=theme_idx, cap=theme_cap: cap - float(np.sum(x[idx]))})

    def objective(x: np.ndarray) -> float:
        variance = float(x @ cov_matrix @ x)
        expected = float(mu_vec @ x)
        concentration = float(np.sum(np.square(x)))
        cash_penalty = 0.01 * float(x[assets.index(base.CASH_ASSET)]) if base.CASH_ASSET in assets else 0.0
        return 0.5 * 8.0 * variance - expected + case.concentration_penalty * concentration + cash_penalty

    result = minimize(objective, x0=x0, bounds=bounds, constraints=constraints, method="SLSQP")
    weights = pd.Series(result.x if result.success else x0, index=assets).clip(lower=0.0)
    return weights / weights.sum()


def _theme_cols(weights: pd.DataFrame) -> list[str]:
    return [column for column in weights.columns if column in STRICT_AI_TECH_BUCKET]


def _metrics_row_with_theme(curve: pd.Series, weights: pd.DataFrame, concentration: pd.DataFrame, case: base.SweepCase, strategy: str) -> dict[str, object]:
    if _ORIGINAL_METRICS_ROW is None:
        raise RuntimeError("Original metrics row function is not initialized.")
    row = _ORIGINAL_METRICS_ROW(curve, weights, concentration, case, strategy)
    theme_cols = _theme_cols(weights)
    theme_weight = weights[theme_cols].sum(axis=1) if theme_cols else pd.Series(0.0, index=weights.index)
    latest_theme = weights[theme_cols].iloc[-1].sort_values(ascending=False) if theme_cols else pd.Series(dtype=float)
    row.update(
        {
            "AI-Tech Theme Cap": CASE_THEME_CAP.get(case.label),
            "AI-Tech Bucket": ", ".join(sorted(STRICT_AI_TECH_BUCKET)),
            "Average AI-Tech Weight": float(theme_weight.mean()),
            "Latest AI-Tech Weight": float(theme_weight.iloc[-1]),
            "Latest AI-Tech Assets": ", ".join([f"{asset}:{weight:.2%}" for asset, weight in latest_theme.items() if weight > 1e-12]),
        }
    )
    return row


def _load_short_overlay_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str], list[str]]:
    prices, volumes, benchmark, vol_proxy, us_all, th_all = _ORIGINAL_LOAD_PANEL()
    prices = prices.loc[prices.index >= EVALUATION_START]
    volumes = volumes.reindex(prices.index)
    benchmark = benchmark.reindex(prices.index).ffill()
    vol_proxy = vol_proxy.reindex(prices.index).ffill()
    return prices, volumes, benchmark, vol_proxy, us_all, th_all

def _proxy_exposure_light(index: pd.DatetimeIndex) -> pd.DataFrame:
    paths = base.default_paths(base.ROOT)
    extra = pd.read_parquet(paths.local_cache_root / "extra_prices.parquet", columns=["^SET.BK"]).sort_index().ffill()
    overlay = pd.read_parquet(
        paths.local_cache_root / "stock_level_overlay_prices_yf.parquet",
        columns=["SPY", "GC=F", "BTC-USD", "USDTHB=X"],
    ).sort_index().ffill()
    full_index = index.union(extra.index).union(overlay.index).sort_values()
    extra = extra.reindex(full_index).ffill()
    overlay = overlay.reindex(full_index).ffill()
    signal_prices = pd.DataFrame(
        {
            "US Equity": overlay["SPY"],
            "TH Equity": extra["^SET.BK"],
            "Gold": overlay["GC=F"],
            "BTC": overlay["BTC-USD"],
        },
        index=full_index,
    ).reindex(index).ffill()
    return pd.DataFrame(
        {
            "US": base._close_trend_exposure(signal_prices["US Equity"], 300, 0.50),
            "TH": base._close_trend_exposure(signal_prices["TH Equity"], 200, 0.00),
            "GC=F": base._gold_crash_exposure(
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
            "BTC-USD": base._close_trend_exposure(signal_prices["BTC"], 50, 0.00),
        },
        index=index,
    ).ffill().fillna(1.0).clip(0.0, 1.0)

def main() -> None:
    original_prefix = base.PREFIX
    original_build_cases = base._build_cases
    original_optimize = base._optimize_with_case
    global _ORIGINAL_METRICS_ROW
    original_metrics_row = base._metrics_row
    _ORIGINAL_METRICS_ROW = original_metrics_row
    original_proxy_exposure = base._proxy_exposure
    original_load_panel = base._load_full_us_th_overlay_panel_from_cache
    global _ORIGINAL_LOAD_PANEL
    _ORIGINAL_LOAD_PANEL = original_load_panel
    try:
        base.PREFIX = PREFIX
        base._build_cases = _build_cases
        base._optimize_with_case = _optimize_with_theme_cap
        base._metrics_row = _metrics_row_with_theme
        base._proxy_exposure = _proxy_exposure_light
        base._load_full_us_th_overlay_panel_from_cache = _load_short_overlay_panel
        base.main()
    finally:
        base.PREFIX = original_prefix
        base._build_cases = original_build_cases
        base._optimize_with_case = original_optimize
        base._metrics_row = original_metrics_row
        base._proxy_exposure = original_proxy_exposure


if __name__ == "__main__":
    main()





