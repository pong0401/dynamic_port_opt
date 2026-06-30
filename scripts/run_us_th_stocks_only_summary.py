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

from dynamic_factor_copula import (  # noqa: E402
    backtest_dynamic_factor_copula,
    compare_rebalanced_portfolio,
    compute_port_opt_style_metrics,
    convert_usd_returns_to_local,
    curve_from_returns,
    default_paths,
    load_overlay_compare_prices,
)
from run_us_th_joint_model import (  # noqa: E402
    BEST_OBJECTIVE,
    _build_ranked_us_th_universe,
    _load_thb_panel,
    _run_model_on_prices,
)


FEATURE_FLAGS = {"resid_vol": False, "drawdown": False, "downside_beta": False}
START_DATE = "2012-01-01"
END_DATE = "2026-04-30"
BLEND_MIXES = {
    "US/TH stocks only 100/0": {"US_HMM": 1.00, "TH_HMM": 0.00},
    "US/TH stocks only 85/15": {"US_HMM": 0.85, "TH_HMM": 0.15},
    "US/TH stocks only 70/30": {"US_HMM": 0.70, "TH_HMM": 0.30},
    "US/TH stocks only 50/50": {"US_HMM": 0.50, "TH_HMM": 0.50},
    "US/TH stocks only 30/70": {"US_HMM": 0.30, "TH_HMM": 0.70},
    "US/TH stocks only 0/100": {"US_HMM": 0.00, "TH_HMM": 1.00},
}


def _run_equity_sleeve(universe_mode: str, benchmark_ticker: str | None, vol_proxy_ticker: str | None):
    paths = default_paths(ROOT)
    return backtest_dynamic_factor_copula(
        start_date=START_DATE,
        end_date=END_DATE,
        n_assets=30,
        n_clusters=4,
        lookback_days=504,
        rebalance_freq="ME",
        max_weight=0.08,
        point_in_time_liquid=True,
        universe_mode=universe_mode,
        benchmark_ticker=benchmark_ticker,
        vol_proxy_ticker=vol_proxy_ticker,
        include_momentum_features=True,
        include_momentum_signal=True,
        momentum_signal_mode="mom_63",
        feature_flags=FEATURE_FLAGS,
        paths=paths,
    )


def _summary_row(curve: pd.Series, strategy: str) -> dict[str, object]:
    row = compute_port_opt_style_metrics(curve, risk_free_rate=0.03).to_dict()
    row["Strategy"] = strategy
    row["Start"] = curve.dropna().index.min().date().isoformat()
    row["End"] = curve.dropna().index.max().date().isoformat()
    return row


def main() -> None:
    paths = default_paths(ROOT)

    us_results = _run_equity_sleeve("sp500_pit", None, None)
    th_results = _run_equity_sleeve("set100_pit", "^SET.BK", "")

    overlay_prices = load_overlay_compare_prices(
        paths,
        start_date="2016-01-01",
        end_date="2026-04-29",
        tickers=["USDTHB=X"],
    )
    fx_returns = overlay_prices["USDTHB=X"].pct_change(fill_method=None).fillna(0.0)

    sample_index = us_results["nav"]["Static Copula"].index.intersection(th_results["nav"]["Static Copula"].index).sort_values()
    us_static_returns_usd = us_results["nav"]["Static Copula"].reindex(sample_index).pct_change(fill_method=None).fillna(0.0)
    th_static_returns_thb = th_results["nav"]["Static Copula"].reindex(sample_index).pct_change(fill_method=None).fillna(0.0)
    us_static_returns_thb = convert_usd_returns_to_local(us_static_returns_usd, fx_returns.reindex(sample_index).fillna(0.0))

    sleeves_thb = pd.concat(
        {
            "US_HMM": us_static_returns_thb,
            "TH_HMM": th_static_returns_thb,
        },
        axis=1,
    ).dropna()

    blend_rows = []
    blend_curves = {}
    for strategy, weights in BLEND_MIXES.items():
        returns = compare_rebalanced_portfolio(
            sleeves_thb,
            weights=pd.Series(weights, dtype=float),
            rebalance_months=1,
        )
        curve = curve_from_returns(returns)
        blend_rows.append(_summary_row(curve, strategy))
        blend_curves[strategy] = curve

    blend_summary = pd.DataFrame(blend_rows).sort_values("Sharpe", ascending=False)
    blend_curve_df = pd.DataFrame(blend_curves).dropna(how="all")
    blend_summary.to_csv(paths.result_dir / "us_th_stocks_only_blended_summary_thb.csv", index=False)
    blend_curve_df.to_csv(paths.result_dir / "us_th_stocks_only_blended_curves_thb.csv")

    us_tickers, th_tickers = _build_ranked_us_th_universe(30, 30)
    us_joint_tickers = list(dict.fromkeys(us_tickers))
    joint_tickers = list(dict.fromkeys(us_tickers + th_tickers))
    us_only_prices, us_only_volumes, us_only_benchmark, us_only_vol_proxy, _fx_us_only = _load_thb_panel(us_joint_tickers)
    us_only_joint_results = _run_model_on_prices(
        us_only_prices,
        us_only_volumes,
        us_only_benchmark,
        us_only_vol_proxy,
        objective_mode=BEST_OBJECTIVE,
    )
    prices, volumes, benchmark, vol_proxy, _fx = _load_thb_panel(joint_tickers)
    joint_results = _run_model_on_prices(
        prices,
        volumes,
        benchmark,
        vol_proxy,
        objective_mode=BEST_OBJECTIVE,
    )

    joint_rows = []
    joint_curves = {}
    for strategy in ["Static Copula", "Dynamic HMM Copula"]:
        name = f"Joint US-only stocks {strategy}"
        curve = us_only_joint_results["nav"][strategy].loc["2017-12-29":].mul(10_000.0)
        joint_rows.append(_summary_row(curve, name))
        joint_curves[name] = curve
    for strategy in ["Static Copula", "Dynamic HMM Copula"]:
        name = f"Joint US+TH stocks only {strategy}"
        curve = joint_results["nav"][strategy].loc["2017-12-29":].mul(10_000.0)
        joint_rows.append(_summary_row(curve, name))
        joint_curves[name] = curve

    joint_summary = pd.DataFrame(joint_rows).sort_values("Sharpe", ascending=False)
    joint_curve_df = pd.DataFrame(joint_curves).dropna(how="all")
    joint_summary["Objective"] = BEST_OBJECTIVE
    joint_summary.to_csv(paths.result_dir / "us_th_joint_stocks_only_summary_thb.csv", index=False)
    joint_curve_df.to_csv(paths.result_dir / "us_th_joint_stocks_only_curves_thb.csv")

    print(blend_summary.to_string(index=False))
    print()
    print(joint_summary.to_string(index=False))


if __name__ == "__main__":
    main()
