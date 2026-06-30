from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebook"
NOTEBOOK_FILE = NOTEBOOK_DIR / "pit_reselect_by_step.ipynb"


def md_cell(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code_cell(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md_cell(
            """
            # PIT Reselect By Step

            This notebook reruns the portfolio tests with **true point-in-time stock reselection every rebalance**.

            Core rule for stock selection:

            - `get_sp500_members_as_of(rebalance_date, ...)` for US members
            - `get_set100_members_as_of(rebalance_date, ...)` for Thailand members
            - `select_point_in_time_universe(...)` inside the trailing lookback window
            - select top liquid names at every monthly rebalance
            - then build features, clusters, covariance, and optimizer weights for that rebalance only

            Default behavior is `RUN_BACKTESTS = False`: the notebook loads precomputed files when available and only reruns missing steps. Set it to `True` to force a full rerun.
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import sys
            sys.dont_write_bytecode = True
            import itertools
            import importlib
            import numpy as np
            import pandas as pd
            import plotly.graph_objects as go
            import plotly.io as pio

            ROOT = Path.cwd().resolve()
            while not (ROOT / "src" / "dynamic_factor_copula.py").exists() and ROOT.parent != ROOT:
                ROOT = ROOT.parent
            SRC = ROOT / "src"
            SCRIPTS = ROOT / "scripts"
            for path in [SRC, SCRIPTS]:
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))

            from dynamic_factor_copula import (
                compare_rebalanced_portfolio,
                compute_port_opt_style_metrics,
                curve_from_returns,
                default_paths,
                lag_close_signal_to_next_session,
                load_overlay_compare_prices,
            )
            from run_us_th_joint_model import END_DATE, FEATURE_FLAGS, LOOKBACK_DAYS, N_CLUSTERS, START_DATE
            import us_th_pit_reselect_utils as pit_utils
            pit_utils = importlib.reload(pit_utils)
            import run_no_copula_pit_top30 as no_copula_pit
            no_copula_pit = importlib.reload(no_copula_pit)
            import run_covariance_objective_max_weight_sweep as cov_weight_sweep
            cov_weight_sweep = importlib.reload(cov_weight_sweep)
            import run_mean_covariance_with_overlay_caps as mean_cov_overlay
            mean_cov_overlay = importlib.reload(mean_cov_overlay)
            import run_mean_covariance_penalty_sweep as mean_cov_penalty_sweep
            mean_cov_penalty_sweep = importlib.reload(mean_cov_penalty_sweep)
            import run_mean_covariance_cov_shrink_sweep as mean_cov_cov_shrink
            mean_cov_cov_shrink = importlib.reload(mean_cov_cov_shrink)
            import run_mean_covariance_signal_sweep as mean_cov_signal_sweep
            mean_cov_signal_sweep = importlib.reload(mean_cov_signal_sweep)
            import run_mean_covariance_rp_tilt_sweep as mean_cov_rp_tilt
            mean_cov_rp_tilt = importlib.reload(mean_cov_rp_tilt)
            import run_mean_covariance_stock_cap_sweep as mean_cov_stock_cap
            mean_cov_stock_cap = importlib.reload(mean_cov_stock_cap)
            import run_mean_covariance_us_th_overlay_gold30 as mean_cov_us_th_overlay
            mean_cov_us_th_overlay = importlib.reload(mean_cov_us_th_overlay)
            import run_mean_covariance_us_th_gold30_1y_lookback_latest_year as mean_cov_us_th_1y_latest
            mean_cov_us_th_1y_latest = importlib.reload(mean_cov_us_th_1y_latest)
            import run_mean_covariance_us_th_side_switch_1y as mean_cov_us_th_side_switch
            mean_cov_us_th_side_switch = importlib.reload(mean_cov_us_th_side_switch)
            import run_mean_covariance_us_th_side_switch_1y_full_period as mean_cov_us_th_side_switch_full
            mean_cov_us_th_side_switch_full = importlib.reload(mean_cov_us_th_side_switch_full)
            import run_mean_covariance_gold30_with_th_sleeve as mean_cov_th_sleeve
            mean_cov_th_sleeve = importlib.reload(mean_cov_th_sleeve)
            import run_mean_covariance_th_gated_sleeve as mean_cov_th_gated_sleeve
            mean_cov_th_gated_sleeve = importlib.reload(mean_cov_th_gated_sleeve)
            import run_mean_covariance_th_ma200_regime_gate as mean_cov_th_ma200_regime
            mean_cov_th_ma200_regime = importlib.reload(mean_cov_th_ma200_regime)
            from us_th_pit_reselect_utils import (
                build_asset_caps,
                load_full_us_th_thb_panel,
                run_joint_pit_reselect_model,
                weights_history_to_frame,
            )

            paths = default_paths(ROOT)
            paths.result_dir.mkdir(parents=True, exist_ok=True)
            pio.renderers.default = "notebook"

            RUN_BACKTESTS = False
            RISK_FREE_RATE = 0.03
            INITIAL_VALUE = 10_000.0
            OBJECTIVE_MODES = ["mean_variance", "max_sharpe_mom", "min_vol_mom_tilt"]
            MOMENTUM_MODES = [False, True]
            MAX_WEIGHT_SWEEP = [0.06, 0.08, 0.10, 0.15, 0.20]
            STOCK_MAX_WEIGHT = 0.08
            US_ASSETS = 30
            TH_ASSETS = 30
            MODEL_OVERLAY_ASSETS = ["GC=F", "BTC-USD", "BIL"]
            ALLOCATION_STEP = 0.05
            ALLOCATION_CAPS = {"EQUITY": 1.00, "GOLD": 0.40, "BTC": 0.10, "BIL": 0.50}

            def metric_row_from_curve(curve: pd.Series, strategy: str) -> dict:
                clean = curve.dropna()
                row = compute_port_opt_style_metrics(clean, risk_free_rate=RISK_FREE_RATE).to_dict()
                row["Strategy"] = strategy
                row["Start"] = clean.index.min().date().isoformat()
                row["End"] = clean.index.max().date().isoformat()
                return row

            def parse_bool(value, default: bool = True) -> bool:
                if pd.isna(value):
                    return default
                if isinstance(value, str):
                    return value.strip().lower() in {"true", "yes", "1", "y"}
                return bool(value)

            def write_summary_curves(summary: pd.DataFrame, curves: pd.DataFrame, summary_name: str, curves_name: str) -> None:
                summary.to_csv(paths.result_dir / summary_name, index=False)
                curves.to_csv(paths.result_dir / curves_name)

            def load_summary_curves(summary_name: str, curves_name: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
                summary_path = paths.result_dir / summary_name
                curves_path = paths.result_dir / curves_name
                if summary_path.exists() and curves_path.exists() and not RUN_BACKTESTS:
                    return (
                        pd.read_csv(summary_path),
                        pd.read_csv(curves_path, index_col=0, parse_dates=True),
                    )
                return None

            def capped_weight_grid(caps: dict[str, float], step: float = 0.05) -> pd.DataFrame:
                assets = list(caps)
                total_units = int(round(1.0 / step))
                cap_units = {asset: int(round(cap / step)) for asset, cap in caps.items()}
                rows = []

                def walk(idx: int, remaining: int, current: dict[str, int]) -> None:
                    asset = assets[idx]
                    if idx == len(assets) - 1:
                        if 0 <= remaining <= cap_units[asset]:
                            row = current.copy()
                            row[asset] = remaining
                            rows.append({name: units * step for name, units in row.items()})
                        return
                    for units in range(min(cap_units[asset], remaining) + 1):
                        current[asset] = units
                        walk(idx + 1, remaining - units, current)
                    current.pop(asset, None)

                walk(0, total_units, {})
                return pd.DataFrame(rows).reindex(columns=assets).fillna(0.0)
            """
        ),
        md_cell(
            """
            ## 1 - Stock Only PIT Reselect

            This step tests stocks only:

            - `1.1 US`: S&P 500 PIT membership, reselect top 30 liquid stocks every rebalance
            - `1.2 US + TH`: S&P 500 PIT + SET100 PIT, reselect top 30 US and top 30 TH stocks every rebalance
            - `1.3 Momentum sweep`: without momentum vs with momentum
            - `1.4 Objective sweep`: `mean_variance`, `max_sharpe_mom`, `min_vol_mom_tilt`
            - `1.5 Stock max-weight sweep`: `6%`, `8%`, `10%`, `15%`, `20%`

            `US+TH stock only` is a joint model: US and TH names enter the same optimizer together. It is not a fixed US/TH portion blend.

            Output:

            - `result/pit_reselect_step1_stock_only_momentum_objective_maxweight_summary_thb.csv`
            - `result/pit_reselect_step1_stock_only_momentum_objective_maxweight_curves_thb.csv`
            """
        ),
        code_cell(
            """
            step1_files = (
                "pit_reselect_step1_stock_only_momentum_objective_maxweight_summary_thb.csv",
                "pit_reselect_step1_stock_only_momentum_objective_maxweight_curves_thb.csv",
            )
            loaded = load_summary_curves(*step1_files)
            if loaded is not None:
                stock_summary, stock_curves = loaded
            else:
                prices, volumes, benchmark, vol_proxy, us_all, th_all = load_full_us_th_thb_panel(include_overlay_assets=False)
                rows = []
                curves = {}
                cases = [
                    {"Family": "US stock only", "us_all": us_all, "th_all": [], "us_assets": US_ASSETS, "th_assets": 0},
                    {"Family": "US+TH stock only", "us_all": us_all, "th_all": th_all, "us_assets": US_ASSETS, "th_assets": TH_ASSETS},
                ]
                for case in cases:
                    for momentum_enabled in MOMENTUM_MODES:
                        momentum_label = "with momentum" if momentum_enabled else "no momentum"
                        for stock_max_weight in MAX_WEIGHT_SWEEP:
                            for objective in OBJECTIVE_MODES:
                                print(f"Running stock PIT reselect: {case['Family']} / {momentum_label} / {objective} / max_weight={stock_max_weight:.0%}")
                                results = run_joint_pit_reselect_model(
                                    prices=prices,
                                    volumes=volumes,
                                    benchmark=benchmark,
                                    vol_proxy=vol_proxy,
                                    us_all=case["us_all"],
                                    th_all=case["th_all"],
                                    us_assets=case["us_assets"],
                                    th_assets=case["th_assets"],
                                    objective_mode=objective,
                                    max_weight=stock_max_weight,
                                    include_overlay_assets=False,
                                    include_momentum=momentum_enabled,
                                )
                                for model_name in ["Static Copula", "Dynamic HMM Copula"]:
                                    strategy = f"{case['Family']} {model_name} [{objective}] [{momentum_label}] max{int(stock_max_weight * 100)} PIT reselect"
                                    curve = results["nav"][model_name].loc["2017-12-29":].mul(INITIAL_VALUE)
                                    row = metric_row_from_curve(curve, strategy)
                                    row["Family"] = case["Family"]
                                    row["Model"] = model_name
                                    row["Objective"] = objective
                                    row["Momentum"] = "Yes" if momentum_enabled else "No"
                                    row["Include Momentum"] = momentum_enabled
                                    row["US Assets"] = case["us_assets"]
                                    row["TH Assets"] = case["th_assets"]
                                    row["Stock Max Weight"] = stock_max_weight
                                    row["Selection Rule"] = "Full PIT reselect every rebalance"
                                    rows.append(row)
                                    curves[strategy] = curve
                stock_summary = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
                stock_curves = pd.DataFrame(curves).dropna(how="all")
                write_summary_curves(stock_summary, stock_curves, *step1_files)

            best_stock_row = stock_summary.sort_values("Sharpe", ascending=False).iloc[0]
            STOCK_MAX_WEIGHT = float(best_stock_row["Stock Max Weight"])
            BEST_STOCK_INCLUDE_MOMENTUM = parse_bool(best_stock_row.get("Include Momentum", best_stock_row.get("Momentum", "Yes")), default=True)
            display(stock_summary)
            best_stock_row.to_frame("Best Stock Config")
            """
        ),
        md_cell(
            """
            ## 1.6 - PIT Reselect Copula vs No-Copula Rebalance Sweep

            This section isolates the covariance model for the US PIT top-30 sleeve.

            It compares:

            - `Static Copula`
            - `Dynamic HMM Copula`
            - `Sample Cov Optimizer` with no copula layer

            The rebalance cadence is swept across `1`, `2`, `3`, and `6` months. All variants use the same PIT S&P 500 membership rule, top-30 liquidity selection, `504` trading-day lookback, `max_weight=8%`, and `mom_63` signal.

            Output:

            - `result/pit_reselect_copula_vs_no_copula_rebalance_sweep.csv`
            - `result/no_copula_pit_top30_optimizer_summary.csv`
            - `result/no_copula_pit_top30_optimizer_curves.csv`
            """
        ),
        code_cell(
            """
            no_copula_sweep_file = paths.result_dir / "pit_reselect_copula_vs_no_copula_rebalance_sweep.csv"
            no_copula_summary_file = paths.result_dir / "no_copula_pit_top30_optimizer_summary.csv"
            no_copula_curves_file = paths.result_dir / "no_copula_pit_top30_optimizer_curves.csv"

            if RUN_BACKTESTS or not (no_copula_sweep_file.exists() and no_copula_summary_file.exists() and no_copula_curves_file.exists()):
                no_copula_pit.main()

            copula_vs_no_copula = pd.read_csv(no_copula_sweep_file)
            no_copula_full_summary = pd.read_csv(no_copula_summary_file)
            no_copula_curves = pd.read_csv(no_copula_curves_file, index_col=0, parse_dates=True)

            display_cols = [
                "Strategy",
                "Optimizer Objective",
                "Rebalance Months",
                "CAGR",
                "Sharpe",
                "Max Drawdown",
                "Turnover",
            ]
            display(copula_vs_no_copula.sort_values(["Rebalance Months", "Sharpe"], ascending=[True, False])[display_cols])

            fig = go.Figure()
            for strategy, group in copula_vs_no_copula.groupby("Strategy"):
                ordered = group.sort_values("Rebalance Months")
                fig.add_trace(
                    go.Scatter(
                        x=ordered["Rebalance Months"],
                        y=ordered["Sharpe"],
                        mode="lines+markers",
                        name=strategy,
                    )
                )
            fig.update_layout(
                title="PIT Top-30: Copula vs Sample Covariance by Rebalance Cadence",
                xaxis_title="Rebalance Months",
                yaxis_title="Sharpe",
                template="plotly_white",
                height=420,
            )
            fig.show()

            best_cov_row = copula_vs_no_copula.sort_values("Sharpe", ascending=False).iloc[0]
            best_cov_row.to_frame("Best Copula / No-Copula Config")
            """
        ),
        md_cell(
            """
            ## 1.7 - Sample Covariance Objective + Max-Weight Sweep

            This section keeps the same US PIT top-30 monthly sleeve and compares simple non-copula optimizers:

            - `Mean Covariance`: sample covariance plus `mom_63` expected-return signal through the mean-variance optimizer
            - `Risk Parity`: inverse/risk contribution style optimizer on sample covariance
            - `Min Vol`: pure minimum-variance optimizer on sample covariance

            Max weight is swept across `8%`, `10%`, `15%`, and `20%`.

            Output:

            - `result/covariance_objective_max_weight_sweep.csv`
            - `result/covariance_objective_max_weight_curves.csv`
            - `result/covariance_objective_max_weight_latest_weights.csv`
            """
        ),
        code_cell(
            """
            cov_weight_summary_file = paths.result_dir / "covariance_objective_max_weight_sweep.csv"
            cov_weight_curves_file = paths.result_dir / "covariance_objective_max_weight_curves.csv"
            cov_weight_latest_file = paths.result_dir / "covariance_objective_max_weight_latest_weights.csv"

            if RUN_BACKTESTS or not (cov_weight_summary_file.exists() and cov_weight_curves_file.exists() and cov_weight_latest_file.exists()):
                cov_weight_sweep.main()

            cov_weight_summary = pd.read_csv(cov_weight_summary_file)
            cov_weight_curves = pd.read_csv(cov_weight_curves_file, index_col=0, parse_dates=True)

            display_cols = [
                "Strategy",
                "Max Weight",
                "CAGR",
                "Sharpe",
                "Max Drawdown",
                "Turnover",
            ]
            display(cov_weight_summary.sort_values(["Sharpe", "CAGR"], ascending=False)[display_cols])

            fig = go.Figure()
            for strategy, group in cov_weight_summary.groupby("Strategy"):
                ordered = group.sort_values("Max Weight")
                fig.add_trace(
                    go.Scatter(
                        x=ordered["Max Weight"],
                        y=ordered["Sharpe"],
                        mode="lines+markers",
                        name=strategy,
                    )
                )
            fig.update_layout(
                title="Sample Covariance Optimizers by Max Weight",
                xaxis_title="Max Weight",
                yaxis_title="Sharpe",
                template="plotly_white",
                height=420,
            )
            fig.show()

            best_cov_weight_row = cov_weight_summary.sort_values("Sharpe", ascending=False).iloc[0]
            best_cov_weight_row.to_frame("Best Sample Covariance Objective / Max Weight")
            """
        ),
        md_cell(
            """
            ## 2.1 - Best Stock Sleeve + Fixed Gold/BTC/BIL Allocation

            This step chooses the best Sharpe stock-only config from Step 1 as one `EQUITY` sleeve, then sweeps fixed monthly allocation with:

            - `EQUITY`
            - `GOLD`
            - `BTC`
            - `BIL`

            Caps:

            - Equity: 100%
            - Gold: 40%
            - BTC: 10%
            - BIL: 50%

            Output:

            - `result/pit_reselect_step2_1_from_step1_momentum_equity_gold_btc_bil_allocation_summary_thb.csv`
            - `result/pit_reselect_step2_1_from_step1_momentum_equity_gold_btc_bil_allocation_curves_thb.csv`
            """
        ),
        code_cell(
            """
            step2_1_files = (
                "pit_reselect_step2_1_from_step1_momentum_equity_gold_btc_bil_allocation_summary_thb.csv",
                "pit_reselect_step2_1_from_step1_momentum_equity_gold_btc_bil_allocation_curves_thb.csv",
            )
            loaded = load_summary_curves(*step2_1_files)
            if loaded is not None:
                fixed_alloc_summary, fixed_alloc_curves = loaded
            else:
                best_stock_strategy = str(best_stock_row["Strategy"])
                equity_curve = stock_curves[best_stock_strategy].dropna()
                equity_returns = equity_curve.pct_change(fill_method=None).rename("EQUITY")

                overlay = load_overlay_compare_prices(
                    paths,
                    start_date=START_DATE,
                    end_date=END_DATE,
                    tickers=["GC=F", "BTC-USD", "BIL", "USDTHB=X"],
                ).sort_index().ffill()
                fx = overlay["USDTHB=X"].reindex(equity_curve.index).ffill()
                sleeve_prices = pd.DataFrame(
                    {
                        "GOLD": overlay["GC=F"].reindex(equity_curve.index).ffill().mul(fx),
                        "BTC": overlay["BTC-USD"].reindex(equity_curve.index).ffill().mul(fx),
                        "BIL": overlay["BIL"].reindex(equity_curve.index).ffill().mul(fx),
                    },
                    index=equity_curve.index,
                )
                sleeve_returns = pd.concat([equity_returns, sleeve_prices.pct_change(fill_method=None)], axis=1).dropna()

                rows = []
                curves = {}
                for _, weights in capped_weight_grid(ALLOCATION_CAPS, ALLOCATION_STEP).iterrows():
                    port_returns = compare_rebalanced_portfolio(
                        sleeve_returns,
                        weights=pd.Series(weights, dtype=float),
                        rebalance_months=1,
                    )
                    active = [(asset, float(weights[asset])) for asset in ALLOCATION_CAPS if float(weights[asset]) > 1e-12]
                    label_assets = "/".join(asset for asset, _weight in active)
                    label_weights = "/".join(str(int(round(weight * 100))) for _asset, weight in active)
                    strategy = f"Best stock sleeve + {label_assets} {label_weights}"
                    curve = curve_from_returns(port_returns, initial=INITIAL_VALUE)
                    row = metric_row_from_curve(curve, strategy)
                    for asset in ALLOCATION_CAPS:
                        row[f"{asset} Weight"] = float(weights[asset])
                    row["Source Stock Strategy"] = best_stock_strategy
                    row["Rebalance"] = "Monthly"
                    rows.append(row)
                    curves[strategy] = curve
                fixed_alloc_summary = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
                fixed_alloc_curves = pd.DataFrame(curves).dropna(how="all")
                write_summary_curves(fixed_alloc_summary, fixed_alloc_curves, *step2_1_files)

            best_fixed_alloc_row = fixed_alloc_summary.sort_values("Sharpe", ascending=False).iloc[0]
            display(fixed_alloc_summary.head(20))
            best_fixed_alloc_row.to_frame("Best Fixed Allocation Config")
            """
        ),
        md_cell(
            """
            ## 2.2 - Stocks + Gold + BTC + BIL In One Model

            This step puts stocks, Gold, BTC, and BIL into the same copula/HMM model using true PIT stock reselection every rebalance.

            The objective is inherited from the best stock-only config in Step 1.

            Output:

            - `result/pit_reselect_step2_2_from_step1_momentum_all_assets_with_bil_one_model_summary_thb.csv`
            - `result/pit_reselect_step2_2_from_step1_momentum_all_assets_with_bil_one_model_curves_thb.csv`
            """
        ),
        code_cell(
            """
            step2_2_files = (
                "pit_reselect_step2_2_from_step1_momentum_all_assets_with_bil_one_model_summary_thb.csv",
                "pit_reselect_step2_2_from_step1_momentum_all_assets_with_bil_one_model_curves_thb.csv",
            )
            loaded = load_summary_curves(*step2_2_files)
            best_stock_objective = str(best_stock_row["Objective"])
            best_stock_include_momentum = parse_bool(best_stock_row.get("Include Momentum", best_stock_row.get("Momentum", "Yes")), default=True)
            if loaded is not None:
                all_asset_summary, all_asset_curves = loaded
            else:
                prices, volumes, benchmark, vol_proxy, us_all, th_all = load_full_us_th_thb_panel(
                    include_overlay_assets=True,
                    overlay_asset_tickers=MODEL_OVERLAY_ASSETS,
                )
                all_asset_results = run_joint_pit_reselect_model(
                    prices=prices,
                    volumes=volumes,
                    benchmark=benchmark,
                    vol_proxy=vol_proxy,
                    us_all=us_all,
                    th_all=th_all,
                    us_assets=US_ASSETS,
                    th_assets=TH_ASSETS,
                    objective_mode=best_stock_objective,
                    max_weight=STOCK_MAX_WEIGHT,
                    include_overlay_assets=True,
                    overlay_asset_tickers=MODEL_OVERLAY_ASSETS,
                    include_momentum=best_stock_include_momentum,
                )
                rows = []
                curves = {}
                for model_name in ["Static Copula", "Dynamic HMM Copula"]:
                    strategy = f"Stocks+Gold+BTC+BIL one-model {model_name} [{best_stock_objective}] PIT reselect"
                    curve = all_asset_results["nav"][model_name].loc["2017-12-29":].mul(INITIAL_VALUE)
                    row = metric_row_from_curve(curve, strategy)
                    row["Model"] = model_name
                    row["Objective"] = best_stock_objective
                    row["US Assets"] = US_ASSETS
                    row["TH Assets"] = TH_ASSETS
                    row["Stock Max Weight"] = STOCK_MAX_WEIGHT
                    row["Momentum"] = "Yes" if best_stock_include_momentum else "No"
                    row["Gold/BTC/BIL Cap Mode"] = "same as stock max weight"
                    row["Selection Rule"] = "Full PIT reselect every rebalance"
                    rows.append(row)
                    curves[strategy] = curve
                all_asset_summary = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
                all_asset_curves = pd.DataFrame(curves).dropna(how="all")
                write_summary_curves(all_asset_summary, all_asset_curves, *step2_2_files)

            display(all_asset_summary)
            """
        ),
        md_cell(
            """
            ## 2.3 - Stocks + Gold + BTC + BIL In One Model With Gold/BTC/BIL Caps From 2.1

            This repeats Step 2.2 but sets Gold, BTC, and BIL max weights using the best fixed allocation from Step 2.1.

            Stock caps remain `8%` per stock.

            Output:

            - `result/pit_reselect_step2_3_from_step1_momentum_all_assets_with_bil_capped_from_2_1_summary_thb.csv`
            - `result/pit_reselect_step2_3_from_step1_momentum_all_assets_with_bil_capped_from_2_1_curves_thb.csv`
            - `result/pit_reselect_step2_3_from_step1_momentum_all_assets_with_bil_capped_from_2_1_latest_weights_thb.csv`
            """
        ),
        code_cell(
            """
            step2_3_files = (
                "pit_reselect_step2_3_from_step1_momentum_all_assets_with_bil_capped_from_2_1_summary_thb.csv",
                "pit_reselect_step2_3_from_step1_momentum_all_assets_with_bil_capped_from_2_1_curves_thb.csv",
            )
            loaded = load_summary_curves(*step2_3_files)
            latest_weights_path = paths.result_dir / "pit_reselect_step2_3_from_step1_momentum_all_assets_with_bil_capped_from_2_1_latest_weights_thb.csv"
            gold_cap = float(best_fixed_alloc_row["GOLD Weight"])
            btc_cap = float(best_fixed_alloc_row["BTC Weight"])
            bil_cap = float(best_fixed_alloc_row["BIL Weight"])
            best_stock_include_momentum = parse_bool(best_stock_row.get("Include Momentum", best_stock_row.get("Momentum", "Yes")), default=True)
            if loaded is not None and latest_weights_path.exists():
                capped_all_asset_summary, capped_all_asset_curves = loaded
                capped_latest_weights = pd.read_csv(latest_weights_path)
            else:
                prices, volumes, benchmark, vol_proxy, us_all, th_all = load_full_us_th_thb_panel(
                    include_overlay_assets=True,
                    overlay_asset_tickers=MODEL_OVERLAY_ASSETS,
                )
                asset_caps = build_asset_caps(
                    us_tickers=us_all,
                    th_tickers=th_all,
                    gold_cap=gold_cap,
                    btc_cap=btc_cap,
                    us_cap=STOCK_MAX_WEIGHT,
                    th_cap=STOCK_MAX_WEIGHT,
                    bil_cap=bil_cap,
                )
                capped_results = run_joint_pit_reselect_model(
                    prices=prices,
                    volumes=volumes,
                    benchmark=benchmark,
                    vol_proxy=vol_proxy,
                    us_all=us_all,
                    th_all=th_all,
                    us_assets=US_ASSETS,
                    th_assets=TH_ASSETS,
                    objective_mode=best_stock_objective,
                    max_weight=max(STOCK_MAX_WEIGHT, gold_cap, btc_cap, bil_cap),
                    include_overlay_assets=True,
                    overlay_asset_tickers=MODEL_OVERLAY_ASSETS,
                    asset_caps=asset_caps,
                    include_momentum=best_stock_include_momentum,
                )
                rows = []
                curves = {}
                latest_rows = []
                for model_name in ["Static Copula", "Dynamic HMM Copula"]:
                    strategy = f"Stocks+Gold+BTC+BIL one-model capped {model_name} [{best_stock_objective}] PIT reselect"
                    curve = capped_results["nav"][model_name].loc["2017-12-29":].mul(INITIAL_VALUE)
                    row = metric_row_from_curve(curve, strategy)
                    row["Model"] = model_name
                    row["Objective"] = best_stock_objective
                    row["US Assets"] = US_ASSETS
                    row["TH Assets"] = TH_ASSETS
                    row["US Stock Cap"] = STOCK_MAX_WEIGHT
                    row["TH Stock Cap"] = STOCK_MAX_WEIGHT
                    row["Momentum"] = "Yes" if best_stock_include_momentum else "No"
                    row["Gold Cap From 2.1"] = gold_cap
                    row["BTC Cap From 2.1"] = btc_cap
                    row["BIL Cap From 2.1"] = bil_cap
                    row["Selection Rule"] = "Full PIT reselect every rebalance"
                    rows.append(row)
                    curves[strategy] = curve

                    weight_history = weights_history_to_frame(capped_results["weights_history"][model_name])
                    latest_date = weight_history.index.max()
                    latest = weight_history.loc[latest_date].rename("Portfolio Weight").reset_index()
                    latest.columns = ["Asset", "Portfolio Weight"]
                    latest["Date"] = pd.Timestamp(latest_date).date().isoformat()
                    latest["Model"] = model_name
                    latest["Strategy"] = strategy
                    latest["Sleeve"] = "US Equity"
                    latest.loc[latest["Asset"].str.endswith(".BK"), "Sleeve"] = "TH Equity"
                    latest.loc[latest["Asset"].eq("GC=F"), "Sleeve"] = "Gold"
                    latest.loc[latest["Asset"].eq("BTC-USD"), "Sleeve"] = "BTC"
                    latest.loc[latest["Asset"].eq("BIL"), "Sleeve"] = "BIL"
                    latest_rows.append(latest.sort_values("Portfolio Weight", ascending=False))

                capped_all_asset_summary = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
                capped_all_asset_curves = pd.DataFrame(curves).dropna(how="all")
                capped_latest_weights = pd.concat(latest_rows, ignore_index=True)
                write_summary_curves(capped_all_asset_summary, capped_all_asset_curves, *step2_3_files)
                capped_latest_weights.to_csv(latest_weights_path, index=False)

            display(capped_all_asset_summary)
            display(capped_latest_weights.loc[capped_latest_weights["Portfolio Weight"] > 0].sort_values(["Model", "Portfolio Weight"], ascending=[True, False]).head(80))
            """
        ),
        md_cell(
            """
            ## 2.3b - Mean Covariance + Gold/BTC/BIL In One Model

            This repeats the no-copula `Mean Covariance` test, but adds Gold, BTC, and BIL directly into the optimizer with the same overlay caps used in Step 2.3:

            - US stock cap: `10%`
            - Gold cap: `40%`
            - BTC cap: `5%`
            - BIL cap: `0%`

            Because the Step 2.3-derived BIL cap is `0%`, BIL is present in the model universe but cannot receive capital in this run.

            Output:

            - `result/mean_covariance_gold_btc_bil_capped_summary.csv`
            - `result/mean_covariance_gold_btc_bil_capped_curve.csv`
            - `result/mean_covariance_gold_btc_bil_capped_latest_weights.csv`
            """
        ),
        code_cell(
            """
            mean_cov_overlay_summary_file = paths.result_dir / "mean_covariance_gold_btc_bil_capped_summary.csv"
            mean_cov_overlay_curve_file = paths.result_dir / "mean_covariance_gold_btc_bil_capped_curve.csv"
            mean_cov_overlay_latest_file = paths.result_dir / "mean_covariance_gold_btc_bil_capped_latest_weights.csv"
            mean_cov_overlay_daily_summary_file = paths.result_dir / "mean_covariance_gold_btc_bil_asset_daily_exposure_summary.csv"
            mean_cov_overlay_daily_curves_file = paths.result_dir / "mean_covariance_gold_btc_bil_asset_daily_exposure_curves.csv"
            mean_cov_overlay_daily_exposure_file = paths.result_dir / "mean_covariance_gold_btc_bil_asset_daily_exposure_history.csv"
            mean_cov_overlay_daily_effective_file = paths.result_dir / "mean_covariance_gold_btc_bil_asset_daily_effective_weights.csv"
            mean_cov_gold30_latest_file = paths.result_dir / "mean_covariance_gold30_asset_daily_latest_effective_weights.csv"
            mean_cov_gold30_recheck_file = paths.result_dir / "mean_covariance_gold30_asset_daily_recheck_today_weights.csv"
            mean_cov_gold30_sleeve_history_file = paths.result_dir / "mean_covariance_gold30_asset_daily_sleeve_weight_history.csv"

            if RUN_BACKTESTS or not (
                mean_cov_overlay_summary_file.exists()
                and mean_cov_overlay_curve_file.exists()
                and mean_cov_overlay_latest_file.exists()
                and mean_cov_overlay_daily_summary_file.exists()
                and mean_cov_overlay_daily_curves_file.exists()
                and mean_cov_overlay_daily_exposure_file.exists()
                and mean_cov_overlay_daily_effective_file.exists()
            ):
                mean_cov_overlay.main()

            mean_cov_overlay_summary = pd.read_csv(mean_cov_overlay_summary_file)
            mean_cov_overlay_curve = pd.read_csv(mean_cov_overlay_curve_file, index_col=0, parse_dates=True)
            mean_cov_overlay_latest = pd.read_csv(mean_cov_overlay_latest_file)
            mean_cov_overlay_daily_summary = pd.read_csv(mean_cov_overlay_daily_summary_file)
            mean_cov_overlay_daily_curves = pd.read_csv(mean_cov_overlay_daily_curves_file, index_col=0, parse_dates=True)
            mean_cov_overlay_daily_exposure = pd.read_csv(mean_cov_overlay_daily_exposure_file, index_col=0, parse_dates=True)
            mean_cov_overlay_daily_effective = pd.read_csv(mean_cov_overlay_daily_effective_file, index_col=0, parse_dates=True)
            mean_cov_gold30_effective = mean_cov_overlay_daily_effective.loc[
                mean_cov_overlay_daily_effective["Gold Cap"].round(6).eq(0.30)
            ].drop(columns=["Gold Cap"])
            sleeve_map = {
                asset: (
                    "Gold" if asset == "GC=F" else
                    "BTC" if asset == "BTC-USD" else
                    "BIL" if asset == "BIL" else
                    "Cash / Reduced Exposure" if asset == "Cash / Reduced Exposure" else
                    "US Equity"
                )
                for asset in mean_cov_gold30_effective.columns
            }
            mean_cov_gold30_sleeve_history = mean_cov_gold30_effective.rename(columns=sleeve_map).T.groupby(level=0).sum().T
            mean_cov_gold30_sleeve_history = mean_cov_gold30_sleeve_history.reindex(
                columns=["US Equity", "Gold", "BTC", "BIL", "Cash / Reduced Exposure"]
            ).dropna(axis=1, how="all")
            mean_cov_gold30_sleeve_history.to_csv(mean_cov_gold30_sleeve_history_file)

            if mean_cov_gold30_recheck_file.exists():
                mean_cov_gold30_latest = pd.read_csv(mean_cov_gold30_recheck_file)
                mean_cov_gold30_latest.to_csv(mean_cov_gold30_latest_file, index=False)
            else:
                latest_gold30_date = mean_cov_gold30_effective.index.max()
                mean_cov_gold30_latest = (
                    mean_cov_gold30_effective.loc[latest_gold30_date]
                    .rename("Effective Weight")
                    .reset_index()
                    .rename(columns={"index": "Asset"})
                )
                mean_cov_gold30_latest["Date"] = pd.Timestamp(latest_gold30_date).date().isoformat()
                mean_cov_gold30_latest["Strategy"] = "Mean Covariance Gold30 stock cap 8 mom_63 + asset-level daily exposure"
                mean_cov_gold30_latest["Sleeve"] = "US Equity"
                mean_cov_gold30_latest.loc[mean_cov_gold30_latest["Asset"].eq("GC=F"), "Sleeve"] = "Gold"
                mean_cov_gold30_latest.loc[mean_cov_gold30_latest["Asset"].eq("BTC-USD"), "Sleeve"] = "BTC"
                mean_cov_gold30_latest.loc[mean_cov_gold30_latest["Asset"].eq("BIL"), "Sleeve"] = "BIL"
                mean_cov_gold30_latest.loc[mean_cov_gold30_latest["Asset"].eq("Cash / Reduced Exposure"), "Sleeve"] = "Cash / Reduced Exposure"
                mean_cov_gold30_latest = mean_cov_gold30_latest.loc[mean_cov_gold30_latest["Effective Weight"].abs() > 1e-12]
                mean_cov_gold30_latest = mean_cov_gold30_latest.sort_values("Effective Weight", ascending=False)
                mean_cov_gold30_latest.to_csv(mean_cov_gold30_latest_file, index=False)

            display(mean_cov_overlay_summary)
            display(mean_cov_overlay_daily_summary)
            display(mean_cov_overlay_latest.loc[mean_cov_overlay_latest["Portfolio Weight"] > 0].sort_values("Portfolio Weight", ascending=False))
            display(mean_cov_overlay_daily_exposure.tail(10))
            display(mean_cov_gold30_latest)

            fig = go.Figure()
            for column in mean_cov_gold30_sleeve_history.columns:
                fig.add_trace(
                    go.Scatter(
                        x=mean_cov_gold30_sleeve_history.index,
                        y=mean_cov_gold30_sleeve_history[column],
                        mode="lines",
                        stackgroup="one",
                        name=column,
                    )
                )
            fig.update_layout(
                title="Mean Covariance Gold 30 + Asset-Level Daily Exposure: Sleeve Weight History",
                xaxis_title="Date",
                yaxis_title="Effective Weight",
                yaxis_range=[0, 1],
                width=1150,
                height=520,
                legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0.0),
                margin=dict(l=70, r=30, t=70, b=130),
            )
            fig.show()
            """
        ),
        md_cell(
            """
            ## 2.3b-2 - Mean Covariance Concentration Penalty Sweep

            This section keeps the same no-TH mean-covariance setup as 2.3b, but tests whether the optimizer should be less concentrated.

            Tested parameters:

            - Gold cap: `30%`
            - BTC cap: `5%`
            - BIL cap: `0%`
            - US stock cap: `10%`
            - Rebalance: monthly
            - Objective: `mean_variance + mom_63 + concentration_penalty`
            - `risk_aversion`: `8`, `16`, `24`
            - `concentration_penalty`: `0`, `1`, `3`, `5`, `10`
            - `momentum_strength`: `0.5`, `1.0`
            - Daily exposure: same asset-level lagged close signal as 2.3b

            Concentration diagnostics:

            - `Latest Effective N = 1 / sum(weight^2)`
            - Latest top-5 / top-10 weight
            - Latest Gold/BTC/BIL/US equity weights
            """
        ),
        code_cell(
            """
            penalty_summary_file = paths.result_dir / "mean_covariance_penalty_sweep_summary.csv"
            penalty_daily_summary_file = paths.result_dir / "mean_covariance_penalty_sweep_daily_exposure_summary.csv"
            penalty_curves_file = paths.result_dir / "mean_covariance_penalty_sweep_curves.csv"
            penalty_daily_curves_file = paths.result_dir / "mean_covariance_penalty_sweep_daily_exposure_curves.csv"
            penalty_latest_effective_file = paths.result_dir / "mean_covariance_penalty_sweep_latest_effective_weights.csv"

            if RUN_BACKTESTS or not (
                penalty_summary_file.exists()
                and penalty_daily_summary_file.exists()
                and penalty_curves_file.exists()
                and penalty_daily_curves_file.exists()
                and penalty_latest_effective_file.exists()
            ):
                mean_cov_penalty_sweep.main()

            penalty_summary = pd.read_csv(penalty_summary_file)
            penalty_daily_summary = pd.read_csv(penalty_daily_summary_file)
            penalty_daily_curves = pd.read_csv(penalty_daily_curves_file, index_col=0, parse_dates=True)
            penalty_latest_effective = pd.read_csv(penalty_latest_effective_file)

            penalty_display_cols = [
                "Strategy",
                "Risk Aversion",
                "Concentration Penalty",
                "Momentum Strength",
                "CAGR",
                "Sharpe",
                "Max Drawdown",
                "Average Exposure",
                "Latest Effective N",
                "Latest Top 5 Weight",
                "Latest Top 10 Weight",
                "Latest Gold Weight",
                "Latest BTC Weight",
                "Latest US Equity Weight",
            ]
            display(penalty_daily_summary.sort_values(["Sharpe", "CAGR"], ascending=False)[penalty_display_cols].head(20))

            fig = go.Figure()
            for momentum_strength, group in penalty_daily_summary.groupby("Momentum Strength"):
                plot_group = group.sort_values(["Concentration Penalty", "Risk Aversion"])
                fig.add_trace(
                    go.Scatter(
                        x=plot_group["Concentration Penalty"].astype(str) + " / risk " + plot_group["Risk Aversion"].astype(str),
                        y=plot_group["Sharpe"],
                        mode="markers",
                        name=f"Momentum strength {momentum_strength:g}",
                        text=plot_group["Strategy"],
                    )
                )
            fig.update_layout(
                title="Mean Covariance Gold30 Daily Exposure: Penalty Sweep Sharpe",
                xaxis_title="Concentration Penalty / Risk Aversion",
                yaxis_title="Sharpe",
                template="plotly_white",
                width=1150,
                height=520,
                margin=dict(l=70, r=30, t=70, b=160),
            )
            fig.show()

            best_penalty = penalty_daily_summary.sort_values(["Sharpe", "CAGR"], ascending=False).iloc[0]
            best_penalty_weights = penalty_latest_effective.loc[
                penalty_latest_effective["Strategy"].eq(best_penalty["Strategy"])
                & penalty_latest_effective["Effective Weight"].abs().gt(1e-12)
            ].sort_values("Effective Weight", ascending=False)
            display(best_penalty.to_frame("Best Penalty Config"))
            display(best_penalty_weights)
            """
        ),
        md_cell(
            """
            ## 2.3b-3 - Mean Covariance Optimization Experiments A/B/C

            This section compares three follow-up attempts to reduce concentration without destroying Sharpe:

            - A: covariance shrinkage toward diagonal covariance
            - B: momentum signal transform (`mom_63`, `rank_63`, `zscore_63`, `blend_21_63`)
            - C: risk-parity anchor plus rank-momentum tilt

            Interpretation focus:

            - Higher Sharpe is better.
            - Higher `Latest Effective N` means less concentrated.
            - Lower top-5/top-10 weight means less cap-stacking.
            """
        ),
        code_cell(
            """
            experiment_files = {
                "A Cov shrink": paths.result_dir / "mean_covariance_cov_shrink_sweep_daily_exposure_summary.csv",
                "B Signal transform": paths.result_dir / "mean_covariance_signal_sweep_daily_exposure_summary.csv",
                "C RP anchor tilt": paths.result_dir / "mean_covariance_rp_tilt_sweep_daily_exposure_summary.csv",
            }
            if RUN_BACKTESTS or not all(path.exists() for path in experiment_files.values()):
                if RUN_BACKTESTS or not experiment_files["A Cov shrink"].exists():
                    mean_cov_cov_shrink.main()
                if RUN_BACKTESTS or not experiment_files["B Signal transform"].exists():
                    mean_cov_signal_sweep.main()
                if RUN_BACKTESTS or not experiment_files["C RP anchor tilt"].exists():
                    mean_cov_rp_tilt.main()

            experiment_frames = []
            for experiment, path in experiment_files.items():
                frame = pd.read_csv(path)
                frame["Experiment"] = experiment
                experiment_frames.append(frame)
            mean_cov_abc = pd.concat(experiment_frames, ignore_index=True)
            mean_cov_abc_cols = [
                "Experiment",
                "Strategy",
                "CAGR",
                "Sharpe",
                "Max Drawdown",
                "Latest Effective N",
                "Latest Top 5 Weight",
                "Latest Top 10 Weight",
                "Latest Gold Weight",
                "Latest BTC Weight",
                "Average Exposure",
            ]
            display(mean_cov_abc.sort_values(["Sharpe", "CAGR"], ascending=False)[mean_cov_abc_cols].head(25))

            best_by_experiment = (
                mean_cov_abc.sort_values(["Sharpe", "CAGR"], ascending=False)
                .groupby("Experiment", as_index=False)
                .head(1)
                .sort_values("Experiment")
            )
            display(best_by_experiment[mean_cov_abc_cols])

            fig = go.Figure()
            for experiment, group in mean_cov_abc.groupby("Experiment"):
                fig.add_trace(
                    go.Scatter(
                        x=group["Latest Effective N"],
                        y=group["Sharpe"],
                        mode="markers",
                        name=experiment,
                        text=group["Strategy"],
                    )
                )
            fig.update_layout(
                title="Mean Covariance Gold30 Experiments: Sharpe vs Latest Effective N",
                xaxis_title="Latest Effective N",
                yaxis_title="Sharpe",
                template="plotly_white",
                width=950,
                height=520,
            )
            fig.show()
            """
        ),
        md_cell(
            """
            ## 2.3b-4 - Mean Covariance Stock Max-Weight Sweep

            This test directly addresses the cap-stacking issue by lowering the US stock max weight.

            Settings:

            - Gold cap: `30%`
            - BTC cap: `5%`
            - BIL cap: `0%`
            - US stock cap sweep: `6%`, `8%`, `10%`
            - Signal mode: no momentum vs `mom_63` vs `zscore_63`
            - Risk aversion: `8`
            - Rebalance: monthly
            - Daily exposure: same asset-level lagged close signal as 2.3b
            """
        ),
        code_cell(
            """
            stock_cap_summary_file = paths.result_dir / "mean_covariance_stock_cap_sweep_daily_exposure_summary.csv"
            stock_cap_curves_file = paths.result_dir / "mean_covariance_stock_cap_sweep_daily_exposure_curves.csv"
            stock_cap_latest_file = paths.result_dir / "mean_covariance_stock_cap_sweep_latest_effective_weights.csv"

            if RUN_BACKTESTS or not (
                stock_cap_summary_file.exists()
                and stock_cap_curves_file.exists()
                and stock_cap_latest_file.exists()
            ):
                mean_cov_stock_cap.main()

            stock_cap_summary = pd.read_csv(stock_cap_summary_file)
            stock_cap_curves = pd.read_csv(stock_cap_curves_file, index_col=0, parse_dates=True)
            stock_cap_latest = pd.read_csv(stock_cap_latest_file)
            stock_cap_cols = [
                "Strategy",
                "Stock Cap",
                "Signal Mode",
                "CAGR",
                "Sharpe",
                "Max Drawdown",
                "Average Exposure",
                "Latest Effective N",
                "Latest Top 5 Weight",
                "Latest Top 10 Weight",
                "Latest Gold Weight",
                "Latest BTC Weight",
            ]
            display(stock_cap_summary.sort_values(["Sharpe", "CAGR"], ascending=False)[stock_cap_cols])

            fig = go.Figure()
            for signal_mode, group in stock_cap_summary.groupby("Signal Mode"):
                ordered = group.sort_values("Stock Cap")
                fig.add_trace(
                    go.Scatter(
                        x=ordered["Stock Cap"],
                        y=ordered["Sharpe"],
                        mode="lines+markers",
                        name=signal_mode,
                        text=ordered["Strategy"],
                    )
                )
            fig.update_layout(
                title="Mean Covariance Gold30 Daily Exposure: Stock Cap Sweep",
                xaxis_title="US Stock Max Weight",
                yaxis_title="Sharpe",
                template="plotly_white",
                width=950,
                height=480,
            )
            fig.show()

            best_stock_cap = stock_cap_summary.sort_values(["Sharpe", "CAGR"], ascending=False).iloc[0]
            best_stock_cap_weights = stock_cap_latest.loc[
                stock_cap_latest["Strategy"].eq(best_stock_cap["Strategy"])
                & stock_cap_latest["Effective Weight"].abs().gt(1e-12)
            ].sort_values("Effective Weight", ascending=False)
            display(best_stock_cap.to_frame("Best Stock Cap Config"))
            display(best_stock_cap_weights)
            """
        ),
        md_cell(
            """
            ## 2.3c - Add Thailand Stocks To Mean Covariance Gold 30%

            This variant adds Thailand SET100 PIT top-30 stocks into the same sample-covariance optimizer as US PIT top-30 stocks plus Gold/BTC/BIL.

            Settings:

            - US stocks: top 30, cap `10%`
            - Thailand stocks: top 30, cap `10%`
            - Gold cap: `30%`
            - BTC cap: `5%`
            - BIL cap: `0%`
            - Daily exposure variant: US stocks use SPY trend, Thailand stocks use SET trend, Gold/BTC use their own asset trend

            Output:

            - `result/mean_covariance_us_th_gold_btc_bil_gold30_summary.csv`
            - `result/mean_covariance_us_th_gold_btc_bil_gold30_curves.csv`
            - `result/mean_covariance_us_th_gold_btc_bil_gold30_latest_weights.csv`
            - `result/mean_covariance_us_th_gold_btc_bil_gold30_daily_exposure_history.csv`
            - `result/mean_covariance_us_th_gold_btc_bil_gold30_daily_effective_weights.csv`
            """
        ),
        code_cell(
            """
            mean_cov_us_th_summary_file = paths.result_dir / "mean_covariance_us_th_gold_btc_bil_gold30_summary.csv"
            mean_cov_us_th_curves_file = paths.result_dir / "mean_covariance_us_th_gold_btc_bil_gold30_curves.csv"
            mean_cov_us_th_latest_file = paths.result_dir / "mean_covariance_us_th_gold_btc_bil_gold30_latest_weights.csv"
            mean_cov_us_th_exposure_file = paths.result_dir / "mean_covariance_us_th_gold_btc_bil_gold30_daily_exposure_history.csv"

            if RUN_BACKTESTS or not (
                mean_cov_us_th_summary_file.exists()
                and mean_cov_us_th_curves_file.exists()
                and mean_cov_us_th_latest_file.exists()
                and mean_cov_us_th_exposure_file.exists()
            ):
                mean_cov_us_th_overlay.main()

            mean_cov_us_th_summary = pd.read_csv(mean_cov_us_th_summary_file)
            mean_cov_us_th_curves = pd.read_csv(mean_cov_us_th_curves_file, index_col=0, parse_dates=True)
            mean_cov_us_th_latest = pd.read_csv(mean_cov_us_th_latest_file)
            mean_cov_us_th_exposure = pd.read_csv(mean_cov_us_th_exposure_file, index_col=0, parse_dates=True)

            display(mean_cov_us_th_summary)
            display(mean_cov_us_th_latest.loc[mean_cov_us_th_latest["Portfolio Weight"] > 0].sort_values("Portfolio Weight", ascending=False))
            display(mean_cov_us_th_exposure.tail(10))
            """
        ),
        md_cell(
            """
            ## 2.3c-2 - US+TH Mean Covariance Gold30, 1Y Lookback, Latest-Year Rebalance

            This test keeps the final no-TH setup direction, but lets US and Thailand stocks compete in one optimizer.

            Settings:

            - US stocks: S&P 500 PIT top 30, cap `8%`
            - Thailand stocks: SET100 PIT top 30, cap `8%`
            - Lookback: `252` trading days
            - Backtest: latest available year only
            - Rebalance: monthly during the latest-year test window
            - Gold cap: `30%`
            - BTC cap: `5%`
            - BIL cap: `0%`
            - Objective: sample mean covariance, `mom_63`, risk aversion `8`
            - Daily exposure: US stocks use SPY trend, Thailand stocks use SET trend, Gold/BTC use their own asset trend

            Output:

            - `result/mean_covariance_us_th_gold30_stockcap8_1y_lookback_latest_year_summary.csv`
            - `result/mean_covariance_us_th_gold30_stockcap8_1y_lookback_latest_year_curves.csv`
            - `result/mean_covariance_us_th_gold30_stockcap8_1y_lookback_latest_year_latest_effective_weights.csv`
            - `result/mean_covariance_us_th_gold30_stockcap8_1y_lookback_latest_year_sleeve_weight_history.csv`
            """
        ),
        code_cell(
            """
            us_th_1y_prefix = "mean_covariance_us_th_gold30_stockcap8_1y_lookback_latest_year"
            us_th_1y_summary_file = paths.result_dir / f"{us_th_1y_prefix}_summary.csv"
            us_th_1y_curves_file = paths.result_dir / f"{us_th_1y_prefix}_curves.csv"
            us_th_1y_latest_file = paths.result_dir / f"{us_th_1y_prefix}_latest_effective_weights.csv"
            us_th_1y_sleeve_file = paths.result_dir / f"{us_th_1y_prefix}_sleeve_weight_history.csv"

            if RUN_BACKTESTS or not (
                us_th_1y_summary_file.exists()
                and us_th_1y_curves_file.exists()
                and us_th_1y_latest_file.exists()
                and us_th_1y_sleeve_file.exists()
            ):
                mean_cov_us_th_1y_latest.main()

            us_th_1y_summary = pd.read_csv(us_th_1y_summary_file)
            us_th_1y_curves = pd.read_csv(us_th_1y_curves_file, index_col=0, parse_dates=True)
            us_th_1y_latest = pd.read_csv(us_th_1y_latest_file)
            us_th_1y_sleeve = pd.read_csv(us_th_1y_sleeve_file, index_col=0, parse_dates=True)

            display(us_th_1y_summary)
            display(us_th_1y_latest.loc[us_th_1y_latest["Effective Weight"] > 0].sort_values("Effective Weight", ascending=False))

            fig = go.Figure()
            for column in us_th_1y_curves.columns:
                fig.add_trace(go.Scatter(x=us_th_1y_curves.index, y=us_th_1y_curves[column], mode="lines", name=column))
            fig.update_layout(
                title="US+TH Mean Covariance Gold30 Stock Cap 8: 1Y Lookback Latest-Year Backtest",
                yaxis_title="Portfolio Value",
                template="plotly_white",
                width=950,
                height=480,
            )
            fig.show()

            sleeve_cols = [column for column in ["US Equity", "TH Equity", "Gold", "BTC", "BIL", "Cash"] if column in us_th_1y_sleeve.columns]
            fig = go.Figure()
            for column in sleeve_cols:
                fig.add_trace(go.Scatter(x=us_th_1y_sleeve.index, y=us_th_1y_sleeve[column], stackgroup="one", name=column))
            fig.update_layout(
                title="US+TH Mean Covariance Gold30 Stock Cap 8: Daily Effective Sleeve Weights",
                yaxis_title="Effective Weight",
                template="plotly_white",
                width=950,
                height=480,
            )
            fig.show()
            """
        ),
        md_cell(
            """
            ## 2.3c-3 - US+TH Side-Switch Daily Exposure

            This test first searches Thailand stock daily-exposure parameters using `^SET.BK`, then applies a side-switch rule to the same US+TH mean-covariance PIT portfolio.

            Rule:

            - If US stock exposure is below full exposure and TH stock exposure is full, move US stock weight into all active TH stocks pro-rata.
            - If TH stock exposure is below full exposure and US stock exposure is full, move TH stock weight into all active US stocks pro-rata.
            - If both US and TH exposure are full, use the model weights unchanged.
            - If both are below full exposure, keep the stock-side cash drag.

            Settings remain aligned with 2.3c-2:

            - US stocks: S&P 500 PIT top 30, cap `8%`
            - Thailand stocks: SET100 PIT top 30, cap `8%`
            - Lookback: `252` trading days
            - Backtest: latest available year only
            - Objective: sample mean covariance, `mom_63`, risk aversion `8`

            Output:

            - `result/mean_covariance_us_th_gold30_stockcap8_1y_side_switch_th_set_param_sweep.csv`
            - `result/mean_covariance_us_th_gold30_stockcap8_1y_side_switch_comparison.csv`
            - `result/mean_covariance_us_th_gold30_stockcap8_1y_side_switch_side_switch_latest_weights.csv`
            - `result/mean_covariance_us_th_gold30_stockcap8_1y_side_switch_side_switch_sleeve_weight_history.csv`
            """
        ),
        code_cell(
            """
            side_switch_prefix = "mean_covariance_us_th_gold30_stockcap8_1y_side_switch"
            side_switch_param_file = paths.result_dir / f"{side_switch_prefix}_th_set_param_sweep.csv"
            side_switch_comparison_file = paths.result_dir / f"{side_switch_prefix}_comparison.csv"
            side_switch_curves_file = paths.result_dir / f"{side_switch_prefix}_curves.csv"
            side_switch_latest_file = paths.result_dir / f"{side_switch_prefix}_side_switch_latest_weights.csv"
            side_switch_sleeve_file = paths.result_dir / f"{side_switch_prefix}_side_switch_sleeve_weight_history.csv"

            if RUN_BACKTESTS or not (
                side_switch_param_file.exists()
                and side_switch_comparison_file.exists()
                and side_switch_curves_file.exists()
                and side_switch_latest_file.exists()
                and side_switch_sleeve_file.exists()
            ):
                mean_cov_us_th_side_switch.main()

            side_switch_param = pd.read_csv(side_switch_param_file)
            side_switch_comparison = pd.read_csv(side_switch_comparison_file)
            side_switch_curves = pd.read_csv(side_switch_curves_file, index_col=0, parse_dates=True)
            side_switch_latest = pd.read_csv(side_switch_latest_file)
            side_switch_sleeve = pd.read_csv(side_switch_sleeve_file, index_col=0, parse_dates=True)

            display(side_switch_param.sort_values(["Sharpe", "CAGR"], ascending=False).head(10))
            display(side_switch_comparison)
            display(side_switch_latest.loc[side_switch_latest["Effective Weight"] > 0].sort_values("Effective Weight", ascending=False))

            fig = go.Figure()
            for below_exposure, group in side_switch_param.groupby("TH Below Exposure"):
                ordered = group.sort_values("TH MA Period")
                fig.add_trace(
                    go.Scatter(
                        x=ordered["TH MA Period"],
                        y=ordered["Sharpe"],
                        mode="lines+markers",
                        name=f"below {below_exposure:.0%}",
                    )
                )
            fig.update_layout(
                title="TH SET Daily Exposure Param Sweep",
                xaxis_title="SET MA Period",
                yaxis_title="Sharpe",
                template="plotly_white",
                width=950,
                height=480,
            )
            fig.show()

            fig = go.Figure()
            for column in side_switch_curves.columns:
                fig.add_trace(go.Scatter(x=side_switch_curves.index, y=side_switch_curves[column], mode="lines", name=column))
            fig.update_layout(
                title="US+TH Mean Covariance Gold30: Cash Drag vs Side Switch",
                yaxis_title="Portfolio Value",
                template="plotly_white",
                width=950,
                height=480,
            )
            fig.show()

            sleeve_cols = [column for column in ["US Equity", "TH Equity", "Gold", "BTC", "BIL", "Cash"] if column in side_switch_sleeve.columns]
            fig = go.Figure()
            for column in sleeve_cols:
                fig.add_trace(go.Scatter(x=side_switch_sleeve.index, y=side_switch_sleeve[column], stackgroup="one", name=column))
            fig.update_layout(
                title="US+TH Side-Switch Daily Effective Sleeve Weights",
                yaxis_title="Effective Weight",
                template="plotly_white",
                width=950,
                height=480,
            )
            fig.show()
            """
        ),
        md_cell(
            """
            ## 2.3c-4 - US+TH Side-Switch, Same Timing As Final Strategy

            This reruns the side-switch test on the same comparison window as the current final strategy:

            - Evaluation period: `2018-01-02` to `2026-04-29`
            - Lookback: still `252` trading days
            - US/TH PIT reselection remains active at each monthly rebalance
            - TH daily exposure parameter is searched again on this longer period

            This is the timing-aligned check for whether the latest-year side-switch result generalizes.

            Output:

            - `result/mean_covariance_us_th_gold30_stockcap8_1y_side_switch_full_period_th_set_param_sweep.csv`
            - `result/mean_covariance_us_th_gold30_stockcap8_1y_side_switch_full_period_comparison.csv`
            - `result/mean_covariance_us_th_gold30_stockcap8_1y_side_switch_full_period_side_switch_latest_weights.csv`
            - `result/mean_covariance_us_th_gold30_stockcap8_1y_side_switch_full_period_side_switch_sleeve_weight_history.csv`
            """
        ),
        code_cell(
            """
            side_switch_full_prefix = "mean_covariance_us_th_gold30_stockcap8_1y_side_switch_full_period"
            side_switch_full_param_file = paths.result_dir / f"{side_switch_full_prefix}_th_set_param_sweep.csv"
            side_switch_full_comparison_file = paths.result_dir / f"{side_switch_full_prefix}_comparison.csv"
            side_switch_full_curves_file = paths.result_dir / f"{side_switch_full_prefix}_curves.csv"
            side_switch_full_latest_file = paths.result_dir / f"{side_switch_full_prefix}_side_switch_latest_weights.csv"
            side_switch_full_sleeve_file = paths.result_dir / f"{side_switch_full_prefix}_side_switch_sleeve_weight_history.csv"

            if RUN_BACKTESTS or not (
                side_switch_full_param_file.exists()
                and side_switch_full_comparison_file.exists()
                and side_switch_full_curves_file.exists()
                and side_switch_full_latest_file.exists()
                and side_switch_full_sleeve_file.exists()
            ):
                mean_cov_us_th_side_switch_full.main()

            side_switch_full_param = pd.read_csv(side_switch_full_param_file)
            side_switch_full_comparison = pd.read_csv(side_switch_full_comparison_file)
            side_switch_full_curves = pd.read_csv(side_switch_full_curves_file, index_col=0, parse_dates=True)
            side_switch_full_latest = pd.read_csv(side_switch_full_latest_file)
            side_switch_full_sleeve = pd.read_csv(side_switch_full_sleeve_file, index_col=0, parse_dates=True)

            display(side_switch_full_param.sort_values(["Sharpe", "CAGR"], ascending=False).head(10))
            display(side_switch_full_comparison)
            display(side_switch_full_latest.loc[side_switch_full_latest["Effective Weight"] > 0].sort_values("Effective Weight", ascending=False))

            fig = go.Figure()
            for below_exposure, group in side_switch_full_param.groupby("TH Below Exposure"):
                ordered = group.sort_values("TH MA Period")
                fig.add_trace(
                    go.Scatter(
                        x=ordered["TH MA Period"],
                        y=ordered["Sharpe"],
                        mode="lines+markers",
                        name=f"below {below_exposure:.0%}",
                    )
                )
            fig.update_layout(
                title="Full-Period TH SET Daily Exposure Param Sweep",
                xaxis_title="SET MA Period",
                yaxis_title="Sharpe",
                template="plotly_white",
                width=950,
                height=480,
            )
            fig.show()

            fig = go.Figure()
            for column in side_switch_full_curves.columns:
                fig.add_trace(go.Scatter(x=side_switch_full_curves.index, y=side_switch_full_curves[column], mode="lines", name=column))
            fig.update_layout(
                title="Full-Period US+TH Mean Covariance Gold30: Cash Drag vs Side Switch",
                yaxis_title="Portfolio Value",
                template="plotly_white",
                width=950,
                height=480,
            )
            fig.show()
            """
        ),
        md_cell(
            """
            ## 2.3d - Add Thailand As A Small Fixed Sleeve

            Instead of letting Thailand stocks compete freely inside the one-model optimizer, this variant adds Thailand as a small fixed sleeve.

            It uses:

            - Core: Mean Covariance Gold 30% portfolio
            - Core variants: raw and asset-level daily exposure
            - Thailand sleeve: SET100 PIT top-30 Static Copula sleeve
            - TH sleeve weights: `0%`, `5%`, `10%`, `15%`, `20%`
            - Currency: THB, with USD core returns translated through USDTHB before mixing

            Output:

            - `result/mean_covariance_gold30_with_th_sleeve_summary_thb.csv`
            - `result/mean_covariance_gold30_with_th_sleeve_curves_thb.csv`
            - `result/mean_covariance_gold30_with_th_sleeve_latest_th_weights.csv`
            """
        ),
        code_cell(
            """
            mean_cov_th_sleeve_summary_file = paths.result_dir / "mean_covariance_gold30_with_th_sleeve_summary_thb.csv"
            mean_cov_th_sleeve_curves_file = paths.result_dir / "mean_covariance_gold30_with_th_sleeve_curves_thb.csv"
            mean_cov_th_sleeve_latest_file = paths.result_dir / "mean_covariance_gold30_with_th_sleeve_latest_th_weights.csv"

            if RUN_BACKTESTS or not (
                mean_cov_th_sleeve_summary_file.exists()
                and mean_cov_th_sleeve_curves_file.exists()
                and mean_cov_th_sleeve_latest_file.exists()
            ):
                mean_cov_th_sleeve.main()

            mean_cov_th_sleeve_summary = pd.read_csv(mean_cov_th_sleeve_summary_file)
            mean_cov_th_sleeve_curves = pd.read_csv(mean_cov_th_sleeve_curves_file, index_col=0, parse_dates=True)
            mean_cov_th_sleeve_latest = pd.read_csv(mean_cov_th_sleeve_latest_file)

            display(mean_cov_th_sleeve_summary)
            display(mean_cov_th_sleeve_latest.loc[mean_cov_th_sleeve_latest["TH Sleeve Weight"] > 0].sort_values("TH Sleeve Weight", ascending=False))
            """
        ),
        md_cell(
            """
            ## 2.3e - Gated Thailand Sleeve

            This test keeps the final no-TH portfolio as the core and only turns on a Thailand sleeve when the Thailand market passes a gate.

            Gate candidates:

            - `absolute`: SET index above its moving average
            - `absolute_relative`: SET above MA and SET/SPY_THB ratio above MA
            - `relative_momentum`: SET/SPY_THB ratio above MA and SET momentum beats SPY_THB momentum
            - `all`: all three conditions pass

            Settings:

            - Core: final no-TH `Mean Covariance Gold30 stockcap8 mom_63 + asset-level daily exposure`
            - TH sleeve: SET100 PIT top-30 Static Copula sleeve with momentum
            - Currency: THB
            - Evaluation period: same final comparison window
            - TH sleeve weight when gate is on: `5%`, `10%`, `15%`, `20%`

            Output:

            - `result/mean_covariance_th_gated_sleeve_summary_thb.csv`
            - `result/mean_covariance_th_gated_sleeve_curves_thb.csv`
            - `result/mean_covariance_th_gated_sleeve_annual_returns_thb.csv`
            - `result/mean_covariance_th_gated_sleeve_best_weight_history_thb.csv`
            - `result/mean_covariance_th_gated_sleeve_best_gate_history_thb.csv`
            """
        ),
        code_cell(
            """
            th_gated_summary_file = paths.result_dir / "mean_covariance_th_gated_sleeve_summary_thb.csv"
            th_gated_curves_file = paths.result_dir / "mean_covariance_th_gated_sleeve_curves_thb.csv"
            th_gated_annual_file = paths.result_dir / "mean_covariance_th_gated_sleeve_annual_returns_thb.csv"
            th_gated_weights_file = paths.result_dir / "mean_covariance_th_gated_sleeve_best_weight_history_thb.csv"

            if RUN_BACKTESTS or not (
                th_gated_summary_file.exists()
                and th_gated_curves_file.exists()
                and th_gated_annual_file.exists()
                and th_gated_weights_file.exists()
            ):
                mean_cov_th_gated_sleeve.main()

            th_gated_summary = pd.read_csv(th_gated_summary_file)
            th_gated_curves = pd.read_csv(th_gated_curves_file, index_col=0, parse_dates=True)
            th_gated_annual = pd.read_csv(th_gated_annual_file)
            th_gated_weights = pd.read_csv(th_gated_weights_file, index_col=0, parse_dates=True)

            display(th_gated_summary.head(12))
            display(th_gated_annual.pivot(index="Year", columns="Strategy", values="Annual Return"))
            display(th_gated_weights.tail(10))

            best_gated = th_gated_summary.iloc[0]["Strategy"]
            fig = go.Figure()
            for column in ["Core no-TH final", best_gated]:
                fig.add_trace(go.Scatter(x=th_gated_curves.index, y=th_gated_curves[column], mode="lines", name=column))
            fig.update_layout(
                title="Gated Thailand Sleeve vs Core No-TH Final",
                yaxis_title="Portfolio Value (THB)",
                template="plotly_white",
                width=950,
                height=480,
            )
            fig.show()
            """
        ),
        md_cell(
            """
            ## 2.3f - Thailand MA200 Regime Gate

            This section focuses on visual regime detection from SET and MA200.

            It tests:

            - SET above MA200
            - MA200 slope is positive
            - optional entry buffer above MA200 to avoid weak rebounds
            - entry/exit confirmation windows to merge short whipsaws

            The intent is to catch the two visually obvious Thailand-on regimes rather than maximize the broad parameter sweep.

            Output:

            - `result/mean_covariance_th_ma200_regime_gate_summary_thb.csv`
            - `result/mean_covariance_th_ma200_regime_gate_curves_thb.csv`
            - `result/mean_covariance_th_ma200_regime_gate_annual_returns_thb.csv`
            - `result/mean_covariance_th_ma200_regime_gate_best_gate_history_thb.csv`
            """
        ),
        code_cell(
            """
            th_ma200_summary_file = paths.result_dir / "mean_covariance_th_ma200_regime_gate_summary_thb.csv"
            th_ma200_curves_file = paths.result_dir / "mean_covariance_th_ma200_regime_gate_curves_thb.csv"
            th_ma200_annual_file = paths.result_dir / "mean_covariance_th_ma200_regime_gate_annual_returns_thb.csv"

            if RUN_BACKTESTS or not (
                th_ma200_summary_file.exists()
                and th_ma200_curves_file.exists()
                and th_ma200_annual_file.exists()
            ):
                mean_cov_th_ma200_regime.main()

            th_ma200_summary = pd.read_csv(th_ma200_summary_file)
            th_ma200_curves = pd.read_csv(th_ma200_curves_file, index_col=0, parse_dates=True)
            th_ma200_annual = pd.read_csv(th_ma200_annual_file)

            display(th_ma200_summary.head(12))
            visual_ma200 = th_ma200_summary.loc[
                th_ma200_summary["Strategy"].eq("TH MA200 regime TH5 slope63 buffer8% entry1 exit20")
            ]
            display(visual_ma200)
            display(th_ma200_annual.pivot(index="Year", columns="Strategy", values="Annual Return"))

            fig = go.Figure()
            best_ma200 = th_ma200_summary.iloc[0]["Strategy"]
            for column in ["Core no-TH final", best_ma200]:
                fig.add_trace(go.Scatter(x=th_ma200_curves.index, y=th_ma200_curves[column], mode="lines", name=column))
            fig.update_layout(
                title="Thailand MA200 Regime Gate",
                yaxis_title="Portfolio Value (THB)",
                template="plotly_white",
                width=950,
                height=480,
            )
            fig.show()
            """
        ),
        md_cell(
            """
            ## 2.4 - Best Stock Model Assets + Gold/BTC/BIL/IEF Re-Optimized

            This section uses the best Sharpe stock-only configuration from Step 1 as the source stock model.

            At each rebalance it:

            - reselects the stock assets using the same PIT rule as the best stock model
            - adds `Gold`, `BTC`, `BIL`, and `IEF`
            - runs the optimizer again on those assets
            - rebalances monthly

            This is intentionally different from the fixed allocation in Step 2.1. Here, Gold, BTC, BIL, and IEF compete with the selected stocks inside the optimizer.

            Caps:

            - stocks: best stock max weight from Step 1
            - Gold: 30%
            - BTC: 10%
            - BIL: 50%
            - IEF: 30%

            Output:

            - `result/pit_reselect_step2_4_from_step1_momentum_best_stock_assets_gold_btc_bil_ief_reoptimized_summary_thb.csv`
            - `result/pit_reselect_step2_4_from_step1_momentum_best_stock_assets_gold_btc_bil_ief_reoptimized_curves_thb.csv`
            - `result/pit_reselect_step2_4_from_step1_momentum_best_stock_assets_gold_btc_bil_ief_reoptimized_latest_weights_thb.csv`
            """
        ),
        code_cell(
            """
            step2_4_files = (
                "pit_reselect_step2_4_from_step1_momentum_best_stock_assets_gold_btc_bil_ief_reoptimized_summary_thb.csv",
                "pit_reselect_step2_4_from_step1_momentum_best_stock_assets_gold_btc_bil_ief_reoptimized_curves_thb.csv",
            )
            step2_4_latest_weights_path = paths.result_dir / "pit_reselect_step2_4_from_step1_momentum_best_stock_assets_gold_btc_bil_ief_reoptimized_latest_weights_thb.csv"
            step2_4_weight_history_path = paths.result_dir / "pit_reselect_step2_4_from_step1_momentum_best_stock_assets_gold_btc_bil_ief_reoptimized_weight_history_thb.csv"
            loaded = load_summary_curves(*step2_4_files)
            best_stock_family = str(best_stock_row["Family"])
            best_stock_objective = str(best_stock_row["Objective"])
            best_stock_max_weight = float(best_stock_row["Stock Max Weight"])
            best_stock_include_momentum = parse_bool(best_stock_row.get("Include Momentum", best_stock_row.get("Momentum", "Yes")), default=True)
            reopt_overlay_assets = ["GC=F", "BTC-USD", "BIL", "IEF"]
            reopt_overlay_caps = {"GC=F": 0.30, "BTC-USD": 0.10, "BIL": 0.50, "IEF": 0.30}
            precomputed_step2_4_ready = (
                loaded is not None
                and step2_4_latest_weights_path.exists()
                and step2_4_weight_history_path.exists()
            )
            if precomputed_step2_4_ready:
                reoptimized_summary, reoptimized_curves = loaded
                reoptimized_latest_weights = pd.read_csv(step2_4_latest_weights_path)
                reoptimized_weight_history = pd.read_csv(step2_4_weight_history_path, index_col=0, parse_dates=True)
            else:
                if loaded is not None and not step2_4_weight_history_path.exists():
                    print("Step 2.4 summary/curves exist, but weight history is missing. Rerunning Step 2.4 to build the stacked-bar chart data.")
                prices, volumes, benchmark, vol_proxy, us_all, th_all = load_full_us_th_thb_panel(
                    include_overlay_assets=True,
                    overlay_asset_tickers=reopt_overlay_assets,
                )
                use_th = best_stock_family == "US+TH stock only"
                source_th_all = th_all if use_th else []
                source_th_assets = TH_ASSETS if use_th else 0
                reopt_asset_caps = build_asset_caps(
                    us_tickers=us_all,
                    th_tickers=source_th_all,
                    gold_cap=reopt_overlay_caps["GC=F"],
                    btc_cap=reopt_overlay_caps["BTC-USD"],
                    us_cap=best_stock_max_weight,
                    th_cap=best_stock_max_weight,
                    bil_cap=reopt_overlay_caps["BIL"],
                )
                reopt_asset_caps["IEF"] = reopt_overlay_caps["IEF"]

                reoptimized_results = run_joint_pit_reselect_model(
                    prices=prices,
                    volumes=volumes,
                    benchmark=benchmark,
                    vol_proxy=vol_proxy,
                    us_all=us_all,
                    th_all=source_th_all,
                    us_assets=US_ASSETS,
                    th_assets=source_th_assets,
                    objective_mode=best_stock_objective,
                    max_weight=max(best_stock_max_weight, *reopt_overlay_caps.values()),
                    include_overlay_assets=True,
                    overlay_asset_tickers=reopt_overlay_assets,
                    asset_caps=reopt_asset_caps,
                    include_momentum=best_stock_include_momentum,
                )

                rows = []
                curves = {}
                latest_rows = []
                weight_history_frames = {}
                for model_name in ["Static Copula", "Dynamic HMM Copula"]:
                    strategy = (
                        f"Best stock assets + Gold/BTC/BIL/IEF reoptimized {model_name} "
                        f"[{best_stock_family}] [{best_stock_objective}] max{int(best_stock_max_weight * 100)} PIT reselect"
                    )
                    curve = reoptimized_results["nav"][model_name].loc["2017-12-29":].mul(INITIAL_VALUE)
                    row = metric_row_from_curve(curve, strategy)
                    row["Source Stock Family"] = best_stock_family
                    row["Model"] = model_name
                    row["Objective"] = best_stock_objective
                    row["Momentum"] = "Yes" if best_stock_include_momentum else "No"
                    row["US Assets"] = US_ASSETS
                    row["TH Assets"] = source_th_assets
                    row["Stock Max Weight"] = best_stock_max_weight
                    row["Gold Max Weight"] = reopt_overlay_caps["GC=F"]
                    row["BTC Max Weight"] = reopt_overlay_caps["BTC-USD"]
                    row["BIL Max Weight"] = reopt_overlay_caps["BIL"]
                    row["IEF Max Weight"] = reopt_overlay_caps["IEF"]
                    row["Overlay Assets"] = ",".join(reopt_overlay_assets)
                    row["Selection Rule"] = "Full PIT stock reselect every rebalance, then add Gold/BTC/BIL/IEF to optimizer"
                    rows.append(row)
                    curves[strategy] = curve

                    weight_history = weights_history_to_frame(reoptimized_results["weights_history"][model_name])
                    weight_history_frames[model_name] = weight_history
                    latest_date = weight_history.index.max()
                    latest = weight_history.loc[latest_date].rename("Portfolio Weight").reset_index()
                    latest.columns = ["Asset", "Portfolio Weight"]
                    latest["Date"] = pd.Timestamp(latest_date).date().isoformat()
                    latest["Model"] = model_name
                    latest["Strategy"] = strategy
                    latest["Sleeve"] = "US Equity"
                    latest.loc[latest["Asset"].str.endswith(".BK"), "Sleeve"] = "TH Equity"
                    latest.loc[latest["Asset"].eq("GC=F"), "Sleeve"] = "Gold"
                    latest.loc[latest["Asset"].eq("BTC-USD"), "Sleeve"] = "BTC"
                    latest.loc[latest["Asset"].eq("BIL"), "Sleeve"] = "BIL"
                    latest.loc[latest["Asset"].eq("IEF"), "Sleeve"] = "IEF"
                    latest_rows.append(latest.sort_values("Portfolio Weight", ascending=False))

                reoptimized_summary = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
                reoptimized_curves = pd.DataFrame(curves).dropna(how="all")
                reoptimized_latest_weights = pd.concat(latest_rows, ignore_index=True)
                best_reoptimized_model = str(reoptimized_summary.iloc[0]["Model"])
                reoptimized_weight_history = weight_history_frames[best_reoptimized_model]
                write_summary_curves(reoptimized_summary, reoptimized_curves, *step2_4_files)
                reoptimized_latest_weights.to_csv(step2_4_latest_weights_path, index=False)
                reoptimized_weight_history.to_csv(step2_4_weight_history_path)

            if reoptimized_weight_history.empty:
                dynamic_weights_df = pd.DataFrame()
                chart_weights = pd.DataFrame()
            else:
                dynamic_weights_df = reoptimized_weight_history.copy()
                dynamic_weights_df = dynamic_weights_df.loc[:, dynamic_weights_df.max(axis=0) > 1e-12]
                fig = go.Figure()
                for column in dynamic_weights_df.columns:
                    fig.add_trace(go.Bar(x=dynamic_weights_df.index, y=dynamic_weights_df[column], name=column))
                fig.update_layout(
                    title="2.4 Best stock assets + Gold/BTC/BIL/IEF reoptimized Weight History",
                    barmode="stack",
                    xaxis_title="Rebalance Date",
                    yaxis_title="Portfolio Weight",
                    yaxis_range=[0, 1],
                    height=600,
                )
                fig.show()

                chart_weights = dynamic_weights_df.copy()
                sleeve_map = {
                    asset: (
                        "TH Equity" if str(asset).endswith(".BK") else
                        "Gold" if asset == "GC=F" else
                        "BTC" if asset == "BTC-USD" else
                        "BIL" if asset == "BIL" else
                        "IEF" if asset == "IEF" else
                        "US Equity"
                    )
                    for asset in chart_weights.columns
                }
                chart_weights = chart_weights.rename(columns=sleeve_map).T.groupby(level=0).sum().T
                chart_weights = chart_weights.reindex(columns=["US Equity", "TH Equity", "Gold", "BTC", "BIL", "IEF"]).dropna(axis=1, how="all")
                chart_weights.to_csv(paths.result_dir / "pit_reselect_step2_4_from_step1_momentum_best_stock_assets_gold_btc_bil_ief_reoptimized_sleeve_weight_history_thb.csv")
            display(reoptimized_summary)
            display(dynamic_weights_df.tail(12))
            display(chart_weights.tail(12))
            display(reoptimized_latest_weights.loc[reoptimized_latest_weights["Portfolio Weight"] > 0].sort_values(["Model", "Portfolio Weight"], ascending=[True, False]).head(80))
            """
        ),
        md_cell(
            """
            ## Step 2.5 - Daily Exposure On Best 2.4 Config

            This section tests daily exposure without lookahead.

            Signal timing rule:

            - compute the close-based signal after today's close
            - shift it by one trading session with `lag_close_signal_to_next_session(...)`
            - apply the exposure to the next available daily return

            Parameter source:

            - start from the best daily exposure parameters from `notebook/best_param_by_step.ipynb`
            - for the stock group, use the S&P 500 trend signal and test whether an additional VIX cap helps
            - test missing overlay groups in this notebook (`BIL`, `IEF`) with the same moving-average grid
            - apply the best stock/VIX daily exposure variants to the best section 2.4 portfolio curve
            """
        ),
        code_cell(
            """
            step2_5_files = (
                "pit_reselect_step2_5_daily_exposure_on_step2_4_summary_thb.csv",
                "pit_reselect_step2_5_daily_exposure_on_step2_4_curves_thb.csv",
            )
            step2_5_asset_sweep_path = paths.result_dir / "pit_reselect_step2_5_daily_exposure_asset_sweep_thb.csv"
            step2_5_best_asset_path = paths.result_dir / "pit_reselect_step2_5_daily_exposure_best_by_asset_thb.csv"
            step2_5_asset_exposure_history_path = paths.result_dir / "pit_reselect_step2_5_daily_exposure_best_asset_history_thb.csv"
            step2_5_exposure_history_path = paths.result_dir / "pit_reselect_step2_5_daily_exposure_history_thb.csv"
            step2_5_effective_weight_history_path = paths.result_dir / "pit_reselect_step2_5_effective_sleeve_weight_history_thb.csv"
            loaded = load_summary_curves(*step2_5_files)

            DAILY_EXPOSURE_MA_PERIODS = [50, 75, 100, 150, 200, 250, 300]
            DAILY_EXPOSURE_BELOW = [0.00, 0.25, 0.50, 0.65, 0.80, 1.00]
            VIX_CAP_RULES = [
                ("No VIX", np.inf, np.inf, 1.00, 1.00),
                ("VIX warn28/crash35 cap50/25", 28.0, 35.0, 0.50, 0.25),
                ("VIX warn25/crash35 cap65/35", 25.0, 35.0, 0.65, 0.35),
                ("VIX warn20/crash30 cap80/50", 20.0, 30.0, 0.80, 0.50),
            ]

            def returns_metrics_row(returns: pd.Series, strategy: str) -> dict:
                curve = curve_from_returns(returns.fillna(0.0), initial=INITIAL_VALUE)
                row = compute_port_opt_style_metrics(curve, risk_free_rate=RISK_FREE_RATE).to_dict()
                row["Strategy"] = strategy
                row["Start"] = curve.dropna().index.min().date().isoformat()
                row["End"] = curve.dropna().index.max().date().isoformat()
                return row

            def close_trend_exposure(price: pd.Series, ma_period: int, below_exposure: float) -> pd.Series:
                price = price.astype(float).sort_index().ffill()
                min_periods = max(20, int(ma_period * 0.20))
                ma = price.rolling(ma_period, min_periods=min_periods).mean()
                signal = pd.Series(1.0, index=price.index, dtype=float)
                signal.loc[price < ma] = below_exposure
                signal.loc[ma.isna()] = 1.0
                return lag_close_signal_to_next_session(signal, initial=1.0)

            def close_vix_cap(
                vix: pd.Series,
                warn_level: float,
                crash_level: float,
                warn_cap: float,
                crash_cap: float,
            ) -> pd.Series:
                vix = vix.astype(float).sort_index().ffill()
                cap = pd.Series(1.0, index=vix.index, dtype=float)
                cap.loc[vix >= warn_level] = warn_cap
                cap.loc[vix >= crash_level] = crash_cap
                return lag_close_signal_to_next_session(cap, initial=1.0)

            def apply_daily_exposure_to_returns(returns: pd.Series, exposure: pd.Series) -> pd.Series:
                aligned = pd.concat(
                    [returns.rename("returns"), exposure.rename("exposure")],
                    axis=1,
                ).dropna(subset=["returns"])
                aligned["exposure"] = aligned["exposure"].ffill().fillna(1.0)
                return aligned["returns"].mul(aligned["exposure"]).rename("Daily Exposure Return")

            def daily_exposure_variant_from_strategy(strategy: str) -> str:
                text = str(strategy)
                if text.endswith(" raw") or text == "Raw":
                    return "Raw"
                if "VIX warn28/crash35 cap50/25" in text or "VIX 28/35 cap 50/25" in text:
                    return "S&P trend + VIX 28/35 cap 50/25"
                if "VIX warn25/crash35 cap65/35" in text or "VIX 25/35 cap 65/35" in text:
                    return "S&P trend + VIX 25/35 cap 65/35"
                if "VIX warn20/crash30 cap80/50" in text or "VIX 20/30 cap 80/50" in text:
                    return "S&P trend + VIX 20/30 cap 80/50"
                if "daily exposure: S&P MA" in text or "S&P trend" in text:
                    return "S&P trend"
                return text

            def add_daily_exposure_display_columns(summary: pd.DataFrame) -> pd.DataFrame:
                frame = summary.copy()
                if "Daily Exposure Variant" not in frame.columns:
                    frame["Daily Exposure Variant"] = frame["Strategy"].map(daily_exposure_variant_from_strategy)
                if "Strategy Full" not in frame.columns:
                    frame["Strategy Full"] = frame["Strategy"]
                frame["Strategy"] = frame["Daily Exposure Variant"]
                return frame

            def exposure_column_for_variant(variant: str, exposure_history: pd.DataFrame) -> str:
                variant = str(variant)
                if variant == "Raw":
                    return "Step 2.4 Raw"
                if variant in exposure_history.columns:
                    return variant
                for column in exposure_history.columns:
                    if daily_exposure_variant_from_strategy(column) == variant:
                        return column
                return exposure_history.columns[0]

            best_param_signal_path = paths.result_dir / "best_param_step3b_best_signal_config_used.csv"
            if best_param_signal_path.exists():
                best_param_signal_config = pd.read_csv(best_param_signal_path, index_col=0)
            else:
                best_param_signal_config = pd.DataFrame(
                    {
                        "Asset": {"SPY": "S&P 500", "GOLD": "Gold", "BTC": "BTC"},
                        "MA Period": {"SPY": 300, "GOLD": 50, "BTC": 50},
                        "Below Exposure": {"SPY": 0.50, "GOLD": 1.00, "BTC": 0.00},
                    }
                )

            if (
                loaded is not None
                and step2_5_asset_sweep_path.exists()
                and step2_5_best_asset_path.exists()
                and step2_5_asset_exposure_history_path.exists()
                and step2_5_exposure_history_path.exists()
            ):
                daily_exposure_24_summary, daily_exposure_24_curves = loaded
                daily_exposure_asset_sweep = pd.read_csv(step2_5_asset_sweep_path)
                daily_exposure_best_by_asset = pd.read_csv(step2_5_best_asset_path)
                daily_exposure_best_asset_history = pd.read_csv(step2_5_asset_exposure_history_path, index_col=0, parse_dates=True)
                daily_exposure_history = pd.read_csv(step2_5_exposure_history_path, index_col=0, parse_dates=True)
                daily_exposure_24_summary = add_daily_exposure_display_columns(daily_exposure_24_summary)
                if step2_5_effective_weight_history_path.exists():
                    daily_exposure_effective_weight_history = pd.read_csv(
                        step2_5_effective_weight_history_path,
                        index_col=0,
                        parse_dates=True,
                    )
                else:
                    daily_exposure_effective_weight_history = pd.DataFrame()
            else:
                overlay_prices = load_overlay_compare_prices(
                    paths,
                    start_date=START_DATE,
                    end_date=END_DATE,
                    tickers=["SPY", "GC=F", "BTC-USD", "BIL", "IEF", "^VIX", "USDTHB=X"],
                ).sort_index().ffill()
                fx = overlay_prices["USDTHB=X"].ffill()
                thb_price = pd.DataFrame(
                    {
                        "S&P 500": overlay_prices["SPY"].mul(fx),
                        "Gold": overlay_prices["GC=F"].mul(fx),
                        "BTC": overlay_prices["BTC-USD"].mul(fx),
                        "BIL": overlay_prices["BIL"].mul(fx),
                        "IEF": overlay_prices["IEF"].mul(fx),
                    }
                ).dropna(how="all").ffill()
                signal_price = {
                    "S&P 500": overlay_prices["SPY"],
                    "Gold": overlay_prices["GC=F"],
                    "BTC": overlay_prices["BTC-USD"],
                    "BIL": overlay_prices["BIL"],
                    "IEF": overlay_prices["IEF"],
                }
                vix = overlay_prices["^VIX"]

                asset_rows = []
                asset_exposure_columns = {}
                for asset in ["S&P 500", "Gold", "BTC", "BIL", "IEF"]:
                    returns = thb_price[asset].pct_change(fill_method=None).fillna(0.0)
                    raw_row = returns_metrics_row(returns, f"{asset} raw buy hold THB")
                    raw_row["Asset"] = asset
                    raw_row["MA Period"] = 0
                    raw_row["Below Exposure"] = 1.0
                    raw_row["VIX Rule"] = "No VIX"
                    raw_row["Rule"] = "Raw"
                    raw_row["Parameter Source"] = "Raw"
                    asset_rows.append(raw_row)

                    for ma_period in DAILY_EXPOSURE_MA_PERIODS:
                        for below_exposure in DAILY_EXPOSURE_BELOW:
                            trend = close_trend_exposure(signal_price[asset], ma_period, below_exposure)
                            vix_rules = VIX_CAP_RULES if asset == "S&P 500" else [VIX_CAP_RULES[0]]
                            for vix_rule, warn_level, crash_level, warn_cap, crash_cap in vix_rules:
                                if vix_rule == "No VIX":
                                    exposure = trend
                                else:
                                    exposure = pd.concat(
                                        [
                                            trend.rename("trend"),
                                            close_vix_cap(vix, warn_level, crash_level, warn_cap, crash_cap).rename("vix"),
                                        ],
                                        axis=1,
                                    ).min(axis=1)
                                exposed_returns = apply_daily_exposure_to_returns(returns, exposure)
                                label = f"{asset} MA{ma_period} below{below_exposure:.2f} {vix_rule}"
                                row = returns_metrics_row(exposed_returns, label)
                                row["Asset"] = asset
                                row["MA Period"] = ma_period
                                row["Below Exposure"] = below_exposure
                                row["VIX Rule"] = vix_rule
                                row["Rule"] = "Trend exposure, lag 1 session"
                                row["Parameter Source"] = (
                                    "best_param_by_step candidate grid"
                                    if asset in {"S&P 500", "Gold", "BTC"}
                                    else "new PIT notebook test"
                                )
                                asset_rows.append(row)
                                asset_exposure_columns[label] = exposure

                daily_exposure_asset_sweep = pd.DataFrame(asset_rows).sort_values(["Asset", "Sharpe"], ascending=[True, False])
                daily_exposure_best_by_asset = (
                    daily_exposure_asset_sweep.loc[daily_exposure_asset_sweep["Rule"] != "Raw"]
                    .sort_values(["Asset", "Sharpe"], ascending=[True, False])
                    .groupby("Asset", as_index=False)
                    .head(1)
                    .sort_values("Asset")
                )
                best_asset_exposure_columns = {}
                for _, row in daily_exposure_best_by_asset.iterrows():
                    strategy = str(row["Strategy"])
                    asset = str(row["Asset"])
                    if strategy in asset_exposure_columns:
                        best_asset_exposure_columns[asset] = asset_exposure_columns[strategy]
                    else:
                        best_asset_exposure_columns[asset] = pd.Series(1.0, index=thb_price.index)
                daily_exposure_best_asset_history = pd.DataFrame(best_asset_exposure_columns).dropna(how="all")

                best_24_row = reoptimized_summary.sort_values("Sharpe", ascending=False).iloc[0]
                best_24_strategy = str(best_24_row["Strategy"])
                best_24_curve = reoptimized_curves[best_24_strategy].dropna()
                best_24_returns = best_24_curve.pct_change(fill_method=None).fillna(0.0)

                spx_cfg = best_param_signal_config.loc["SPY"] if "SPY" in best_param_signal_config.index else pd.Series({"MA Period": 300, "Below Exposure": 0.50})
                spx_trend = close_trend_exposure(
                    overlay_prices["SPY"],
                    ma_period=int(spx_cfg["MA Period"]),
                    below_exposure=float(spx_cfg["Below Exposure"]),
                )

                portfolio_rows = []
                portfolio_curves = {}
                portfolio_exposures = {"Step 2.4 Raw": pd.Series(1.0, index=best_24_returns.index)}
                raw_label = "Raw"
                portfolio_rows.append(returns_metrics_row(best_24_returns, raw_label))
                portfolio_curves[raw_label] = best_24_curve

                trend_label = f"S&P trend MA{int(spx_cfg['MA Period'])} below{float(spx_cfg['Below Exposure']):.2f}"
                trend_returns = apply_daily_exposure_to_returns(best_24_returns, spx_trend)
                portfolio_rows.append(returns_metrics_row(trend_returns, trend_label))
                portfolio_curves[trend_label] = curve_from_returns(trend_returns, initial=INITIAL_VALUE)
                portfolio_exposures["S&P trend"] = spx_trend.reindex(best_24_returns.index).ffill()

                for vix_rule, warn_level, crash_level, warn_cap, crash_cap in VIX_CAP_RULES[1:]:
                    vix_cap = close_vix_cap(vix, warn_level, crash_level, warn_cap, crash_cap)
                    combined_exposure = pd.concat([spx_trend.rename("trend"), vix_cap.rename("vix")], axis=1).min(axis=1)
                    label = daily_exposure_variant_from_strategy(f"S&P trend + {vix_rule}")
                    variant_returns = apply_daily_exposure_to_returns(best_24_returns, combined_exposure)
                    portfolio_rows.append(returns_metrics_row(variant_returns, label))
                    portfolio_curves[label] = curve_from_returns(variant_returns, initial=INITIAL_VALUE)
                    portfolio_exposures[vix_rule] = combined_exposure.reindex(best_24_returns.index).ffill()

                daily_exposure_24_summary = add_daily_exposure_display_columns(pd.DataFrame(portfolio_rows).sort_values("Sharpe", ascending=False))
                daily_exposure_24_summary["Base Section"] = "2.4"
                daily_exposure_24_summary["Base Strategy"] = best_24_strategy
                daily_exposure_24_summary["Signal Asset"] = "S&P 500"
                daily_exposure_24_summary["Lookahead"] = "No, close signal shifted one session"
                daily_exposure_24_curves = pd.DataFrame(portfolio_curves).dropna(how="all")
                daily_exposure_history = pd.DataFrame(portfolio_exposures).dropna(how="all")

                daily_exposure_asset_sweep.to_csv(step2_5_asset_sweep_path, index=False)
                daily_exposure_best_by_asset.to_csv(step2_5_best_asset_path, index=False)
                daily_exposure_best_asset_history.to_csv(step2_5_asset_exposure_history_path)
                daily_exposure_history.to_csv(step2_5_exposure_history_path)
                write_summary_curves(daily_exposure_24_summary, daily_exposure_24_curves, *step2_5_files)

                daily_exposure_effective_weight_history = pd.DataFrame()

            daily_exposure_24_summary = add_daily_exposure_display_columns(daily_exposure_24_summary)
            daily_exposure_24_summary["Exposure Target"] = "Whole section 2.4 portfolio"
            daily_exposure_24_summary["Signal Scope"] = "S&P 500 only"
            daily_exposure_24_summary["Asset-Level Gold/BTC Signal Applied"] = "No"
            daily_exposure_24_summary["Portfolio Exposure Note"] = (
                "Portfolio-level test reduces the full 2.4 portfolio using the lagged S&P signal; "
                "asset-level Gold/BTC/BIL/IEF signal tests are shown separately above."
            )
            daily_exposure_24_summary.to_csv(paths.result_dir / step2_5_files[0], index=False)

            display(best_param_signal_config)
            display(daily_exposure_best_by_asset)

            if daily_exposure_best_asset_history.empty:
                print("No asset-level exposure history available. Rerun this section to rebuild it.")
            else:
                display(daily_exposure_best_asset_history.tail(10))
                fig = go.Figure()
                for column in daily_exposure_best_asset_history.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=daily_exposure_best_asset_history.index,
                            y=daily_exposure_best_asset_history[column],
                            mode="lines",
                            line_shape="hv",
                            name=column,
                        )
                    )
                fig.update_layout(
                    title="Step 2.5 Best Daily Exposure By Asset",
                    xaxis_title="Date",
                    yaxis_title="Daily Exposure",
                    yaxis_range=[-0.02, 1.05],
                    width=1150,
                    height=420,
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.25,
                        xanchor="left",
                        x=0.0,
                    ),
                    margin=dict(l=70, r=30, t=70, b=120),
                )
                display(fig)

            daily_exposure_24_display_cols = [
                "Daily Exposure Variant",
                "Exposure Target",
                "Signal Scope",
                "Asset-Level Gold/BTC Signal Applied",
                "Total Return",
                "CAGR",
                "Annual Vol",
                "Sharpe",
                "Sortino",
                "Max Drawdown",
                "Hit Rate",
                "Start",
                "End",
                "Lookahead",
            ]
            display(daily_exposure_24_summary[daily_exposure_24_display_cols])

            if daily_exposure_history.empty:
                print("No portfolio-level exposure history available. Rerun this section to rebuild it.")
            else:
                display(daily_exposure_history.tail(10))
                fig = go.Figure()
                for column in daily_exposure_history.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=daily_exposure_history.index,
                            y=daily_exposure_history[column],
                            mode="lines",
                            line_shape="hv",
                            name=daily_exposure_variant_from_strategy(column),
                        )
                    )
                fig.update_layout(
                    title="Step 2.5 Portfolio Daily Exposure Applied To Best Step 2.4",
                    xaxis_title="Date",
                    yaxis_title="Daily Exposure",
                    yaxis_range=[-0.02, 1.05],
                    width=1150,
                    height=420,
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.25,
                        xanchor="left",
                        x=0.0,
                    ),
                    margin=dict(l=70, r=30, t=70, b=120),
                )
                display(fig)

            step24_sleeve_weight_history_path = paths.result_dir / "pit_reselect_step2_4_from_step1_momentum_best_stock_assets_gold_btc_bil_ief_reoptimized_sleeve_weight_history_thb.csv"
            if step24_sleeve_weight_history_path.exists() and not daily_exposure_history.empty:
                step24_sleeve_weight_history = pd.read_csv(
                    step24_sleeve_weight_history_path,
                    index_col=0,
                    parse_dates=True,
                ).sort_index()
                best_exposure_variant = str(daily_exposure_24_summary.sort_values("Sharpe", ascending=False).iloc[0]["Daily Exposure Variant"])
                best_exposure_column = exposure_column_for_variant(best_exposure_variant, daily_exposure_history)
                best_exposure = daily_exposure_history[best_exposure_column].rename("Daily Exposure").sort_index().ffill().fillna(1.0)
                sleeve_daily = step24_sleeve_weight_history.reindex(best_exposure.index).ffill().fillna(0.0)
                sleeve_cols = [column for column in ["US Equity", "TH Equity", "Gold", "BTC", "BIL", "IEF"] if column in sleeve_daily.columns]
                daily_exposure_effective_weight_history = sleeve_daily[sleeve_cols].mul(best_exposure, axis=0)
                daily_exposure_effective_weight_history["Cash / Reduced Exposure"] = (1.0 - best_exposure).clip(lower=0.0)
                daily_exposure_effective_weight_history.to_csv(step2_5_effective_weight_history_path)

                display(
                    pd.DataFrame(
                        [
                            {
                                "Best Daily Exposure Variant": best_exposure_variant,
                                "Exposure Source Column": best_exposure_column,
                                "Weight Source": "Step 2.4 sleeve weight history",
                                "Gold/BTC Handling": "Scaled by the same portfolio-level exposure, not separate Gold/BTC signals",
                            }
                        ]
                    )
                )
                display(daily_exposure_effective_weight_history.tail(10))

                fig = go.Figure()
                for column in daily_exposure_effective_weight_history.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=daily_exposure_effective_weight_history.index,
                            y=daily_exposure_effective_weight_history[column],
                            mode="lines",
                            stackgroup="one",
                            name=column,
                        )
                    )
                fig.update_layout(
                    title="Step 2.5 Effective Sleeve Weights After Daily Exposure From Step 2.4",
                    xaxis_title="Date",
                    yaxis_title="Effective Portfolio Weight",
                    yaxis_range=[0, 1],
                    width=1150,
                    height=520,
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.22,
                        xanchor="left",
                        x=0.0,
                    ),
                    margin=dict(l=70, r=30, t=70, b=130),
                )
                display(fig)
            else:
                print("Step 2.4 sleeve weight history is missing, so Step 2.5 effective sleeve weights cannot be plotted.")

            fig = go.Figure()
            curve_name_map = {column: daily_exposure_variant_from_strategy(column) for column in daily_exposure_24_curves.columns}
            for column in daily_exposure_24_curves.columns:
                fig.add_trace(go.Scatter(x=daily_exposure_24_curves.index, y=daily_exposure_24_curves[column], mode="lines", name=curve_name_map[column]))
            fig.update_layout(
                title="Step 2.5 Daily Exposure On Best Step 2.4 Portfolio",
                xaxis_title="Date",
                yaxis_title="Portfolio Value (THB base)",
                width=1150,
                height=650,
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.22,
                    xanchor="left",
                    x=0.0,
                ),
                margin=dict(l=70, r=30, t=70, b=150),
            )
            display(fig)
            """
        ),
        md_cell(
            """
            ## Final Best Sharpe Review

            Compact comparison of the best Sharpe result from each stage.
            """
        ),
        code_cell(
            """
            if "daily_exposure_variant_from_strategy" not in globals():
                def daily_exposure_variant_from_strategy(strategy: str) -> str:
                    text = str(strategy)
                    if text.endswith(" raw") or text == "Raw":
                        return "Raw"
                    if "VIX warn28/crash35 cap50/25" in text or "VIX 28/35 cap 50/25" in text:
                        return "S&P trend + VIX 28/35 cap 50/25"
                    if "VIX warn25/crash35 cap65/35" in text or "VIX 25/35 cap 65/35" in text:
                        return "S&P trend + VIX 25/35 cap 65/35"
                    if "VIX warn20/crash30 cap80/50" in text or "VIX 20/30 cap 80/50" in text:
                        return "S&P trend + VIX 20/30 cap 80/50"
                    if "daily exposure: S&P MA" in text or "S&P trend" in text:
                        return "S&P trend"
                    return text

            if "add_daily_exposure_display_columns" not in globals():
                def add_daily_exposure_display_columns(summary: pd.DataFrame) -> pd.DataFrame:
                    frame = summary.copy()
                    if "Daily Exposure Variant" not in frame.columns:
                        frame["Daily Exposure Variant"] = frame["Strategy"].map(daily_exposure_variant_from_strategy)
                    if "Strategy Full" not in frame.columns:
                        frame["Strategy Full"] = frame["Strategy"]
                    frame["Strategy"] = frame["Daily Exposure Variant"]
                    return frame

            if "exposure_column_for_variant" not in globals():
                def exposure_column_for_variant(variant: str, exposure_history: pd.DataFrame) -> str:
                    variant = str(variant)
                    if variant == "Raw":
                        return "Step 2.4 Raw"
                    if variant in exposure_history.columns:
                        return variant
                    for column in exposure_history.columns:
                        if daily_exposure_variant_from_strategy(column) == variant:
                            return column
                    return exposure_history.columns[0]

            def final_summary_frame(var_name: str, summary_file: str) -> pd.DataFrame:
                frame = globals().get(var_name)
                if frame is None:
                    frame = pd.read_csv(paths.result_dir / summary_file)
                frame = frame.copy()
                if var_name == "daily_exposure_24_summary":
                    frame = add_daily_exposure_display_columns(frame)
                return frame

            def preferred_stock_cap_mom63_frame() -> pd.DataFrame:
                frame = pd.read_csv(paths.result_dir / "mean_covariance_stock_cap_sweep_daily_exposure_summary.csv")
                return frame.loc[
                    frame["Stock Cap"].round(6).eq(0.08)
                    & frame["Signal Mode"].astype(str).eq("mom_63")
                ].copy()

            def pct_text(value, decimals: int = 0) -> str:
                if pd.isna(value):
                    return "n/a"
                return f"{float(value) * 100:.{decimals}f}%"

            def active_weight_text(row: pd.Series, columns: list[str]) -> str:
                parts = []
                for column in columns:
                    value = float(row.get(column, 0.0) or 0.0)
                    if abs(value) > 1e-12:
                        parts.append(f"{column.replace(' Weight', '')}={value:.0%}")
                return ", ".join(parts) if parts else "n/a"

            def infer_curve_period(source_file: str, strategy: str) -> tuple[str, str]:
                curve_path = paths.result_dir / source_file
                if not curve_path.exists():
                    return "", ""
                curves = pd.read_csv(curve_path, index_col=0, parse_dates=True)
                if strategy in curves.columns:
                    series = curves[strategy].dropna()
                else:
                    series = curves.dropna(how="all").iloc[:, 0].dropna() if not curves.empty else pd.Series(dtype=float)
                if series.empty:
                    return "", ""
                return series.index.min().date().isoformat(), series.index.max().date().isoformat()

            def period_text(row: pd.Series) -> str:
                return f"{row.get('Start', '')} to {row.get('End', '')}"

            def final_config_text(step: str, row: pd.Series) -> str:
                if step.startswith("1."):
                    return (
                        f"Stock-only PIT reselect; family={row.get('Family')}; model={row.get('Model')}; "
                        f"objective={row.get('Objective')}; momentum={row.get('Momentum')}; "
                        f"US assets={row.get('US Assets')}; TH assets={row.get('TH Assets')}; "
                        f"stock max weight={pct_text(row.get('Stock Max Weight'))}; "
                        "selection=get members as of rebalance date, select top liquid names in trailing window, then build features/clusters/optimizer."
                    )
                if step.startswith("2.1"):
                    return (
                        "Monthly fixed allocation using the best Step 1 stock sleeve plus overlay assets; "
                        f"active weights: {active_weight_text(row, ['EQUITY Weight', 'GOLD Weight', 'BTC Weight', 'BIL Weight'])}; "
                        f"source stock strategy={row.get('Source Stock Strategy')}; rebalance={row.get('Rebalance')}."
                    )
                if step.startswith("2.2"):
                    return (
                        "One optimizer model containing stocks, Gold, BTC, and BIL together; "
                        f"model={row.get('Model')}; objective={row.get('Objective')}; momentum={row.get('Momentum')}; "
                        f"US assets={row.get('US Assets')}; TH assets={row.get('TH Assets')}; "
                        f"stock max weight={pct_text(row.get('Stock Max Weight'))}; "
                        f"overlay cap mode={row.get('Gold/BTC/BIL Cap Mode')}; full PIT stock reselect every rebalance."
                    )
                if step.startswith("2.3b"):
                    return (
                        "No-TH sample-covariance one-model optimizer with asset-level daily exposure; "
                        f"base strategy={row.get('Base Strategy')}; objective={row.get('Objective')}; "
                        f"US assets={row.get('US Assets', 30)}; US stock cap={pct_text(row.get('US Stock Cap', row.get('Stock Cap')))}; "
                        f"signal mode={row.get('Signal Mode', 'mom_63')}; "
                        f"Gold cap={pct_text(row.get('Gold Cap'))}; BTC cap={pct_text(row.get('BTC Cap'))}; "
                        f"BIL cap={pct_text(row.get('BIL Cap'))}; "
                        f"stock exposure={row.get('Stock Exposure Signal')}; "
                        f"Gold exposure={row.get('Gold Exposure Signal')}; BTC exposure={row.get('BTC Exposure Signal')}; "
                        "all close-based signals are lagged one session."
                    )
                if step.startswith("2.3"):
                    return (
                        "One optimizer model with overlay caps copied from Step 2.1 allocation winner; "
                        f"model={row.get('Model')}; objective={row.get('Objective')}; momentum={row.get('Momentum')}; "
                        f"US cap={pct_text(row.get('US Stock Cap'))}; TH cap={pct_text(row.get('TH Stock Cap'))}; "
                        f"Gold cap={pct_text(row.get('Gold Cap From 2.1'))}; BTC cap={pct_text(row.get('BTC Cap From 2.1'))}; "
                        f"BIL cap={pct_text(row.get('BIL Cap From 2.1'))}; full PIT stock reselect every rebalance."
                    )
                if step.startswith("2.4"):
                    return (
                        "Reoptimize the best Step 1 stock assets together with Gold/BTC/BIL/IEF; "
                        f"source stock family={row.get('Source Stock Family')}; model={row.get('Model')}; "
                        f"objective={row.get('Objective')}; momentum={row.get('Momentum')}; "
                        f"US assets={row.get('US Assets')}; TH assets={row.get('TH Assets')}; "
                        f"stock cap={pct_text(row.get('Stock Max Weight'))}; Gold cap={pct_text(row.get('Gold Max Weight'))}; "
                        f"BTC cap={pct_text(row.get('BTC Max Weight'))}; BIL cap={pct_text(row.get('BIL Max Weight'))}; "
                        f"IEF cap={pct_text(row.get('IEF Max Weight'))}; overlay assets={row.get('Overlay Assets')}."
                    )
                if step.startswith("2.5"):
                    return (
                        "Portfolio-level daily exposure test on the best Step 2.4 portfolio; "
                        f"variant={row.get('Daily Exposure Variant', row.get('Strategy'))}; "
                        f"exposure target={row.get('Exposure Target')}; signal scope={row.get('Signal Scope')}; "
                        f"signal asset={row.get('Signal Asset')}; lookahead rule={row.get('Lookahead')}; "
                        f"asset-level Gold/BTC signal applied={row.get('Asset-Level Gold/BTC Signal Applied')}. "
                        f"Base strategy={row.get('Base Strategy')}."
                    )
                return str(row.get("Strategy", ""))

            final_sources = [
                (
                    1.0,
                    "1. Stock only",
                    final_summary_frame("stock_summary", "pit_reselect_step1_stock_only_momentum_objective_maxweight_summary_thb.csv"),
                    "pit_reselect_step1_stock_only_momentum_objective_maxweight_curves_thb.csv",
                ),
                (
                    2.1,
                    "2.1 Equity + Gold/BTC/BIL allocation",
                    final_summary_frame("fixed_alloc_summary", "pit_reselect_step2_1_from_step1_momentum_equity_gold_btc_bil_allocation_summary_thb.csv"),
                    "pit_reselect_step2_1_from_step1_momentum_equity_gold_btc_bil_allocation_curves_thb.csv",
                ),
                (
                    2.2,
                    "2.2 Stocks + Gold/BTC/BIL one model",
                    final_summary_frame("all_asset_summary", "pit_reselect_step2_2_from_step1_momentum_all_assets_with_bil_one_model_summary_thb.csv"),
                    "pit_reselect_step2_2_from_step1_momentum_all_assets_with_bil_one_model_curves_thb.csv",
                ),
                (
                    2.3,
                    "2.3 Capped one model from 2.1",
                    final_summary_frame("capped_all_asset_summary", "pit_reselect_step2_3_from_step1_momentum_all_assets_with_bil_capped_from_2_1_summary_thb.csv"),
                    "pit_reselect_step2_3_from_step1_momentum_all_assets_with_bil_capped_from_2_1_curves_thb.csv",
                ),
                (
                    2.35,
                    "2.3b No-TH mean covariance + asset daily exposure",
                    final_summary_frame("mean_cov_overlay_daily_summary", "mean_covariance_gold_btc_bil_asset_daily_exposure_summary.csv"),
                    "mean_covariance_gold_btc_bil_asset_daily_exposure_curves.csv",
                ),
                (
                    2.36,
                    "2.3b-4 Final Mean Covariance Gold30 stock cap 8 mom_63",
                    preferred_stock_cap_mom63_frame(),
                    "mean_covariance_stock_cap_sweep_daily_exposure_curves.csv",
                ),
                (
                    2.4,
                    "2.4 Best stock assets + Gold/BTC/BIL/IEF reoptimized",
                    final_summary_frame("reoptimized_summary", "pit_reselect_step2_4_from_step1_momentum_best_stock_assets_gold_btc_bil_ief_reoptimized_summary_thb.csv"),
                    "pit_reselect_step2_4_from_step1_momentum_best_stock_assets_gold_btc_bil_ief_reoptimized_curves_thb.csv",
                ),
                (
                    2.5,
                    "2.5 Daily exposure on best 2.4",
                    final_summary_frame("daily_exposure_24_summary", "pit_reselect_step2_5_daily_exposure_on_step2_4_summary_thb.csv"),
                    "pit_reselect_step2_5_daily_exposure_on_step2_4_curves_thb.csv",
                ),
            ]

            final_rows = []
            for step_order, step, frame, source_file in final_sources:
                row = frame.sort_values("Sharpe", ascending=False).iloc[0].copy()
                if pd.isna(row.get("Start", np.nan)) or pd.isna(row.get("End", np.nan)) or not str(row.get("Start", "")).strip():
                    inferred_start, inferred_end = infer_curve_period(source_file, str(row.get("Strategy", "")))
                    row["Start"] = inferred_start
                    row["End"] = inferred_end
                row["Step Order"] = step_order
                row["Step"] = step
                row["Config"] = final_config_text(step, row)
                row["Precompute Port Growth File"] = str((paths.result_dir / source_file).relative_to(ROOT))
                row["Precompute Period"] = period_text(row)
                final_rows.append(row)
            final_best = pd.DataFrame(final_rows).sort_values("Step Order")
            best_overall = final_best.loc[
                final_best["Step"].eq("2.3b-4 Final Mean Covariance Gold30 stock cap 8 mom_63")
            ].iloc[0]
            reference_start = str(best_overall.get("Start", ""))
            reference_end = str(best_overall.get("End", ""))
            final_best["Same Period As Overall Best"] = (
                final_best["Start"].astype(str).eq(reference_start)
                & final_best["End"].astype(str).eq(reference_end)
            )
            final_best["Timing Note"] = np.where(
                final_best["Same Period As Overall Best"],
                "Comparable on exact same start/end dates as overall best",
                "Not exact same timing; compare Sharpe directionally or rerun on common overlap",
            )
            final_best.to_csv(paths.result_dir / "pit_reselect_by_step_best_sharpe_summary_thb.csv", index=False)
            final_best[["Step", "Strategy", "Start", "End", "Sharpe", "Same Period As Overall Best", "Timing Note"]].to_csv(
                paths.result_dir / "pit_reselect_by_step_timing_audit.csv",
                index=False,
            )
            recommended_strategy = "Mean Covariance Gold30 stock cap 8 mom_63 + asset-level daily exposure"
            gold30_latest_weights_path = paths.result_dir / "mean_covariance_gold30_asset_daily_latest_effective_weights.csv"
            gold30_sleeve_history_path = paths.result_dir / "mean_covariance_gold30_asset_daily_sleeve_weight_history.csv"
            latest_recommended_weights = pd.read_csv(gold30_latest_weights_path)
            recommended_sleeve_history = pd.read_csv(
                gold30_sleeve_history_path,
                index_col=0,
                parse_dates=True,
            )
            recommended_last_date = latest_recommended_weights["Date"].iloc[0]
            recommended_sleeve_latest = (
                recommended_sleeve_history.loc[recommended_sleeve_history.index.max()]
                .rename("Effective Weight")
                .reset_index()
                .rename(columns={"index": "Sleeve"})
            )
            recommended_sleeve_latest["Date"] = pd.Timestamp(recommended_sleeve_history.index.max()).date().isoformat()
            recommended_sleeve_latest["Effective Weight %"] = recommended_sleeve_latest["Effective Weight"].mul(100.0)
            recommended_sleeve_latest = recommended_sleeve_latest.loc[
                recommended_sleeve_latest["Effective Weight"].abs() > 1e-12
            ].sort_values("Effective Weight", ascending=False)

            md_lines = [
                "# PIT Reselect By Step - Port Opt Advance Handoff",
                "",
                "This file lists the best-Sharpe result from each PIT-reselect notebook step. The table is ordered by notebook step, not by Sharpe.",
                "",
                "Important daily exposure note:",
                "",
                "- The recommended latest-weight view now uses the Step 2.3b-4 mean-covariance Gold 30%, stock cap 8%, mom_63, asset-level daily exposure strategy.",
                "- Step 2.4 and Step 2.5 remain in the comparison table, but they are no longer used as the latest recommended weight source.",
                "",
                "## Overall Best Sharpe",
                "",
                f"- Step: `{best_overall['Step']}`",
                f"- Strategy: `{best_overall['Strategy']}`",
                f"- Sharpe: `{float(best_overall['Sharpe']):.4f}`",
                f"- CAGR: `{float(best_overall['CAGR']):.4f}`",
                f"- Max Drawdown: `{float(best_overall['Max Drawdown']):.4f}`",
                f"- Precompute port growth file path: `{best_overall['Precompute Port Growth File']}`",
                f"- Precompute period: `{best_overall['Precompute Period']}`",
                "",
                "## Latest Recommended Effective Sleeve Weights",
                "",
                f"- Strategy: `{recommended_strategy}`",
                f"- Date: `{recommended_last_date}`",
                f"- Weight source: `result\\{gold30_sleeve_history_path.name}`",
                "- Rule: asset-level daily exposure is applied per sleeve/asset after the Gold 30%, stock cap 8%, mom_63 mean-covariance optimizer.",
                "",
                "| Sleeve | Effective Weight |",
                "|---|---:|",
                *[
                    f"| {row['Sleeve']} | {float(row['Effective Weight']):.4f} |"
                    for _, row in recommended_sleeve_latest.iterrows()
                ],
                "",
                "## Latest Recommended Effective Security Weights",
                "",
                f"- Strategy: `{recommended_strategy}`",
                f"- Output file: `result\\{gold30_latest_weights_path.name}`",
                "",
                "## Best Sharpe By Step",
                "",
            ]
            for _, row in final_best.iterrows():
                md_lines.extend(
                    [
                        f"### {row['Step']}",
                        "",
                        f"- Strategy name: `{row['Strategy']}`",
                        f"- Config: {row['Config']}",
                        "- Metrics:",
                        f"  - Total Return: `{float(row['Total Return']):.4f}`",
                        f"  - CAGR: `{float(row['CAGR']):.4f}`",
                        f"  - Annual Vol: `{float(row['Annual Vol']):.4f}`",
                        f"  - Sharpe: `{float(row['Sharpe']):.4f}`",
                        f"  - Sortino: `{float(row['Sortino']):.4f}`",
                        f"  - Max Drawdown: `{float(row['Max Drawdown']):.4f}`",
                        f"  - Hit Rate: `{float(row['Hit Rate']):.4f}`",
                        f"- Precompute port growth file path: `{row['Precompute Port Growth File']}`",
                        f"- Precompute period: `{row['Precompute Period']}`",
                        "",
                    ]
                )
            doc_dir = ROOT / "doc" / "port_opt_advance"
            doc_dir.mkdir(parents=True, exist_ok=True)
            (doc_dir / "PIT_RESELECT_BY_STEP_HANDOFF.md").write_text("\\n".join(md_lines), encoding="utf-8")

            display(final_best)

            if not recommended_sleeve_history.empty:
                fig = go.Figure()
                for column in recommended_sleeve_history.columns:
                    fig.add_trace(
                        go.Bar(
                            x=recommended_sleeve_history.index,
                            y=recommended_sleeve_history[column],
                            name=column,
                        )
                    )
                fig.update_layout(
                    title="Mean Covariance Gold 30 + Asset-Level Daily Exposure: Sleeve Weight History",
                    barmode="stack",
                    xaxis_title="Date",
                    yaxis_title="Effective Weight",
                    yaxis_range=[0, 1],
                    width=1150,
                    height=520,
                    legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0.0),
                    margin=dict(l=70, r=30, t=70, b=130),
                )
                display(fig)

            display_cols = ["Step", "Strategy", "Sharpe", "CAGR", "Max Drawdown", "Precompute Port Growth File"]
            display(final_best.loc[final_best["Step"].astype(str).str.startswith("2.3b"), display_cols])

            display(recommended_sleeve_latest)
            display(latest_recommended_weights)
            """
        ),
        md_cell(
            """
            ## Final - Latest Recheck Weights Today

            This is the easy-to-find final section for the latest recheck weights of:

            `Mean Covariance Gold30 stock cap 8 mom_63 + asset-level daily exposure`

            It uses the latest common close available in the local cache, with lagged asset-level daily exposure signals.
            """
        ),
        code_cell(
            """
            latest_recheck_weights_file = paths.result_dir / "mean_covariance_gold30_asset_daily_recheck_today_weights.csv"
            latest_recheck_meta_file = paths.result_dir / "mean_covariance_gold30_asset_daily_recheck_today_meta.csv"
            latest_recheck_sleeve_file = paths.result_dir / "mean_covariance_gold30_asset_daily_recheck_today_sleeve_weights.csv"
            latest_recheck_history_file = paths.result_dir / "mean_covariance_gold30_asset_daily_sleeve_weight_history.csv"

            if not latest_recheck_weights_file.exists():
                raise FileNotFoundError(
                    "Run scripts/recheck_mean_covariance_gold30_today_weights.py before opening this final section."
                )

            latest_recheck_weights = pd.read_csv(latest_recheck_weights_file)
            latest_recheck_meta = pd.read_csv(latest_recheck_meta_file) if latest_recheck_meta_file.exists() else pd.DataFrame()
            latest_recheck_sleeve = pd.read_csv(latest_recheck_sleeve_file) if latest_recheck_sleeve_file.exists() else pd.DataFrame()
            latest_recheck_history = pd.read_csv(latest_recheck_history_file, index_col=0, parse_dates=True)

            display(latest_recheck_meta)
            display(latest_recheck_sleeve)
            display(latest_recheck_weights)

            fig = go.Figure()
            for column in latest_recheck_history.columns:
                fig.add_trace(
                    go.Bar(
                        x=latest_recheck_history.index,
                        y=latest_recheck_history[column],
                        name=column,
                    )
                )
            fig.update_layout(
                title="Final Latest Recheck: Mean Covariance Gold30 Stock Cap 8 mom_63 Asset-Level Daily Exposure Weight History",
                barmode="stack",
                xaxis_title="Date",
                yaxis_title="Effective Weight",
                yaxis_range=[0, 1],
                width=1150,
                height=520,
                legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0.0),
                margin=dict(l=70, r=30, t=70, b=130),
            )
            display(fig)
            """
        ),
    ]
    return nb


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    with NOTEBOOK_FILE.open("w", encoding="utf-8") as handle:
        nbf.write(notebook, handle)
    print(NOTEBOOK_FILE.name)


if __name__ == "__main__":
    main()
