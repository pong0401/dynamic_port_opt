from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynamic_factor_copula import (  # noqa: E402
    apply_daily_exposure_overlay,
    backtest_dynamic_factor_copula,
    compare_apply_returns,
    compare_rebalanced_portfolio,
    compare_sp_exposure,
    compare_trend_exposure,
    compute_port_opt_style_metrics,
    convert_usd_returns_to_local,
    curve_from_returns,
    default_paths,
    load_overlay_compare_prices,
)


START_DATE = "2012-01-01"
END_DATE = "2026-04-30"
FEATURE_FLAGS = {"resid_vol": False, "drawdown": False, "downside_beta": False}
MIXES = {
    "US HMM/Gold/BTC 60/30/10": {"US_HMM": 0.60, "TH_HMM": 0.00, "GOLD": 0.30, "BTC": 0.10},
    "US/TH/Gold/BTC 50/10/30/10": {"US_HMM": 0.50, "TH_HMM": 0.10, "GOLD": 0.30, "BTC": 0.10},
    "US/TH/Gold/BTC 45/15/30/10": {"US_HMM": 0.45, "TH_HMM": 0.15, "GOLD": 0.30, "BTC": 0.10},
    "US/TH/Gold/BTC 40/20/30/10": {"US_HMM": 0.40, "TH_HMM": 0.20, "GOLD": 0.30, "BTC": 0.10},
    "US/TH/Gold/BTC 30/30/30/10": {"US_HMM": 0.30, "TH_HMM": 0.30, "GOLD": 0.30, "BTC": 0.10},
}


def run_equity_sleeve(universe_mode: str, benchmark_ticker: str | None, vol_proxy_ticker: str | None):
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


def build_sleeve_returns(us_results: dict, th_results: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = default_paths(ROOT)
    sample_index = us_results["nav"]["Static Copula"].index.intersection(th_results["nav"]["Static Copula"].index).sort_values()
    compare_prices = load_overlay_compare_prices(
        paths,
        start_date="2016-01-01",
        end_date=str(sample_index.max().date()),
        tickers=["SPY", "GC=F", "BTC-USD", "^VIX", "USDTHB=X"],
    ).dropna(subset=["SPY", "GC=F", "BTC-USD", "^VIX", "USDTHB=X"])
    fx_returns = compare_prices["USDTHB=X"].pct_change(fill_method=None).fillna(0.0)

    spy = compare_prices["SPY"]
    gold = compare_prices["GC=F"]
    btc = compare_prices["BTC-USD"]
    vix = compare_prices["^VIX"]
    gold_returns_usd = gold.pct_change(fill_method=None).fillna(0.0)
    btc_returns_usd = btc.pct_change(fill_method=None).fillna(0.0)

    us_static_returns_usd = us_results["nav"]["Static Copula"].reindex(sample_index).pct_change(fill_method=None).fillna(0.0)
    th_static_returns_thb = th_results["nav"]["Static Copula"].reindex(sample_index).pct_change(fill_method=None).fillna(0.0)

    us_overlay_usd, us_exposure = apply_daily_exposure_overlay(
        us_static_returns_usd,
        spy.reindex(sample_index).ffill(),
        vix.reindex(sample_index).ffill(),
    )
    th_overlay_thb, th_exposure = apply_daily_exposure_overlay(
        th_static_returns_thb,
        th_results["panel"]["benchmark"].reindex(sample_index).ffill(),
        None,
    )

    gold_exposure = compare_trend_exposure(gold, 0.50)
    btc_exposure = compare_trend_exposure(btc, 0.00)
    gold_overlay_usd = compare_apply_returns(gold_returns_usd, gold_exposure, "USD_STATIC", fx_returns)
    btc_overlay_usd = compare_apply_returns(btc_returns_usd, btc_exposure, "USD_STATIC", fx_returns)

    us_overlay_thb = convert_usd_returns_to_local(us_overlay_usd, fx_returns.reindex(us_overlay_usd.index).fillna(0.0))
    gold_overlay_thb = convert_usd_returns_to_local(gold_overlay_usd, fx_returns.reindex(gold_overlay_usd.index).fillna(0.0))
    btc_overlay_thb = convert_usd_returns_to_local(btc_overlay_usd, fx_returns.reindex(btc_overlay_usd.index).fillna(0.0))
    th_overlay_usd = (1.0 + th_overlay_thb).div(1.0 + fx_returns.reindex(th_overlay_thb.index).fillna(0.0)).sub(1.0)

    sleeves_thb = pd.concat(
        {
            "US_HMM": us_overlay_thb,
            "TH_HMM": th_overlay_thb,
            "GOLD": gold_overlay_thb,
            "BTC": btc_overlay_thb,
        },
        axis=1,
    ).dropna()
    sleeves_usd = pd.concat(
        {
            "US_HMM": us_overlay_usd,
            "TH_HMM": th_overlay_usd,
            "GOLD": gold_overlay_usd,
            "BTC": btc_overlay_usd,
        },
        axis=1,
    ).dropna()
    exposure = pd.concat(
        {
            "US_HMM": us_exposure["Daily Exposure"],
            "TH_HMM": th_exposure["Daily Exposure"],
            "GOLD": gold_exposure,
            "BTC": btc_exposure,
        },
        axis=1,
    ).reindex(sleeves_thb.index).ffill().bfill()
    return sleeves_thb, sleeves_usd, exposure


def evaluate_mixes(sleeves: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves = {}
    rows = []
    for name, weights in MIXES.items():
        returns = compare_rebalanced_portfolio(
            sleeves,
            weights=pd.Series(weights, dtype=float),
            rebalance_months=1,
        )
        curve = curve_from_returns(returns)
        curves[name] = curve
        row = compute_port_opt_style_metrics(curve, risk_free_rate=0.03).to_dict()
        row["Start"] = curve.dropna().index.min().date().isoformat()
        row["End"] = curve.dropna().index.max().date().isoformat()
        row["Strategy"] = name
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("Strategy").sort_values("Sharpe", ascending=False)
    curve_df = pd.DataFrame(curves).dropna(how="all")
    return summary, curve_df


def main() -> None:
    paths = default_paths(ROOT)
    print("Running US S&P 500 PIT sleeve...")
    us_results = run_equity_sleeve("sp500_pit", None, None)
    print("Running Thailand SET100 PIT sleeve...")
    th_results = run_equity_sleeve("set100_pit", "^SET.BK", "")

    sleeves_thb, sleeves_usd, exposure = build_sleeve_returns(us_results, th_results)
    thb_summary, thb_curves = evaluate_mixes(sleeves_thb)
    usd_summary, usd_curves = evaluate_mixes(sleeves_usd)

    thb_summary.to_csv(paths.result_dir / "us_th_gold_btc_blended_summary_thb.csv")
    usd_summary.to_csv(paths.result_dir / "us_th_gold_btc_blended_summary_usd.csv")
    thb_curves.to_csv(paths.result_dir / "us_th_gold_btc_blended_curves_thb.csv")
    usd_curves.to_csv(paths.result_dir / "us_th_gold_btc_blended_curves_usd.csv")
    exposure.to_csv(paths.result_dir / "us_th_gold_btc_blended_exposure.csv")

    latest_us = max(us_results["universe_history"])
    latest_th = max(th_results["universe_history"])
    pd.Series(us_results["universe_history"][latest_us], name="ticker").to_csv(
        paths.result_dir / "latest_us_hmm_members.csv",
        index=False,
    )
    pd.Series(th_results["universe_history"][latest_th], name="ticker").to_csv(
        paths.result_dir / "latest_th_hmm_members.csv",
        index=False,
    )

    print("\nTHB summary")
    print(thb_summary.to_string())
    print("\nUSD summary")
    print(usd_summary.to_string())
    print("\nLatest US members:", ", ".join(us_results["universe_history"][latest_us]))
    print("\nLatest TH members:", ", ".join(th_results["universe_history"][latest_th]))


if __name__ == "__main__":
    main()
