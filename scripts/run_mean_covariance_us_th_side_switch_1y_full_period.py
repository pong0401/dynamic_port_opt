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

from dynamic_factor_copula import default_paths  # noqa: E402
from run_mean_covariance_us_th_gold30_1y_lookback_latest_year import (  # noqa: E402
    BIL_CAP,
    BTC_CAP,
    GOLD_CAP,
    LOOKBACK_DAYS,
    RISK_AVERSION,
    STOCK_CAP,
    TH_ASSETS,
    US_ASSETS,
    _sleeve_history,
    run_backtest,
)
from run_mean_covariance_us_th_side_switch_1y import (  # noqa: E402
    TH_BELOW_EXPOSURES,
    TH_MA_PERIODS,
    _apply_cash_drag,
    _apply_side_switch,
    _latest_frame,
    _trigger_exposures,
)


RESULT_PREFIX = "mean_covariance_us_th_gold30_stockcap8_1y_side_switch_full_period"
BASE_STRATEGY = "US+TH Mean Covariance Gold30 stock cap 8 mom_63 1Y lookback full-period rebalance"
COMPARISON_START = "2018-01-02"


def main() -> None:
    paths = default_paths(ROOT)
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    base = run_backtest(latest_year_only=False, strategy_name=BASE_STRATEGY)
    daily_weights: pd.DataFrame = base["daily_weights"].loc[COMPARISON_START:]
    prices: pd.DataFrame = base["prices"]
    benchmark: pd.Series = base["benchmark"]
    set_index: pd.Series = base["set_index"]

    param_rows = []
    param_curves = {}
    param_results = {}
    for ma_period in TH_MA_PERIODS:
        for below_exposure in TH_BELOW_EXPOSURES:
            strategy = f"Full-period TH SET daily exposure MA{ma_period} below{below_exposure:.0%} cash drag"
            exposures = _trigger_exposures(prices, benchmark, set_index, ma_period, below_exposure)
            result = _apply_cash_drag(daily_weights, prices, exposures, benchmark, strategy)
            row = result["metrics"].copy()
            row["Strategy"] = strategy
            row["TH MA Period"] = ma_period
            row["TH Below Exposure"] = below_exposure
            row["Start"] = result["nav"].index.min().date().isoformat()
            row["End"] = result["nav"].index.max().date().isoformat()
            param_rows.append(row)
            param_curves[strategy] = result["nav"]
            param_results[(ma_period, below_exposure)] = (exposures, result)

    param_summary = pd.DataFrame(param_rows).sort_values(["Sharpe", "CAGR"], ascending=False)
    best_param = param_summary.iloc[0]
    best_ma = int(best_param["TH MA Period"])
    best_below = float(best_param["TH Below Exposure"])
    best_exposures = param_results[(best_ma, best_below)][0]

    side_strategy = f"US+TH side-switch Mean Covariance Gold30 stock cap 8 1Y lookback full-period THSET_MA{best_ma}_below{best_below:.0%}"
    side = _apply_side_switch(daily_weights, prices, best_exposures, benchmark, side_strategy)
    base_cash_strategy = f"{BASE_STRATEGY} with optimized TH SET MA{best_ma}_below{best_below:.0%} cash drag"
    base_cash = param_results[(best_ma, best_below)][1]

    comparison_rows = []
    for strategy, result, variant in [
        (base_cash_strategy, base_cash, "Cash drag"),
        (side_strategy, side, "Side switch"),
    ]:
        row = result["metrics"].copy()
        row["Strategy"] = strategy
        row["Variant"] = variant
        row["TH MA Period"] = best_ma
        row["TH Below Exposure"] = best_below
        row["Start"] = result["nav"].index.min().date().isoformat()
        row["End"] = result["nav"].index.max().date().isoformat()
        row["US Assets"] = US_ASSETS
        row["TH Assets"] = TH_ASSETS
        row["Lookback Days"] = LOOKBACK_DAYS
        row["US Stock Cap"] = STOCK_CAP
        row["TH Stock Cap"] = STOCK_CAP
        row["Gold Cap"] = GOLD_CAP
        row["BTC Cap"] = BTC_CAP
        row["BIL Cap"] = BIL_CAP
        row["Risk Aversion"] = RISK_AVERSION
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows).sort_values(["Sharpe", "CAGR"], ascending=False)

    param_summary.to_csv(paths.result_dir / f"{RESULT_PREFIX}_th_set_param_sweep.csv", index=False)
    pd.DataFrame(param_curves).dropna(how="all").to_csv(paths.result_dir / f"{RESULT_PREFIX}_th_set_param_curves.csv")
    comparison.to_csv(paths.result_dir / f"{RESULT_PREFIX}_comparison.csv", index=False)
    pd.DataFrame({base_cash_strategy: base_cash["nav"], side_strategy: side["nav"]}).dropna(how="all").to_csv(
        paths.result_dir / f"{RESULT_PREFIX}_curves.csv"
    )
    best_exposures.reindex(daily_weights.index).ffill().to_csv(paths.result_dir / f"{RESULT_PREFIX}_trigger_exposure.csv")
    side["effective_weights"].to_csv(paths.result_dir / f"{RESULT_PREFIX}_side_switch_effective_weights.csv")
    _sleeve_history(side["effective_weights"]).to_csv(paths.result_dir / f"{RESULT_PREFIX}_side_switch_sleeve_weight_history.csv")
    _latest_frame(side["effective_weights"], side_strategy).to_csv(paths.result_dir / f"{RESULT_PREFIX}_side_switch_latest_weights.csv", index=False)

    print("Best full-period TH SET daily exposure param")
    print(best_param.to_frame("Best").T.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nFull-period comparison")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nLatest full-period side-switch weights")
    latest = _latest_frame(side["effective_weights"], side_strategy)
    print(latest.loc[latest["Effective Weight"] > 1e-8, ["Asset", "Sleeve", "Effective Weight"]].to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
