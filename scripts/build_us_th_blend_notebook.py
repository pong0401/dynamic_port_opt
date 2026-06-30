from __future__ import annotations

import textwrap
from pathlib import Path
import sys

import nbformat as nbf
import pandas as pd
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynamic_factor_copula import default_paths  # noqa: E402


NOTEBOOK_DIR = ROOT / "notebook"
NOTEBOOK_FILE = NOTEBOOK_DIR / "us_th_blend_poc.ipynb"


def md_cell(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code_cell(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def html_output(html: str, text: str = ""):
    return nbf.v4.new_output("display_data", data={"text/html": html, "text/plain": text})


def plotly_output(fig: go.Figure, text: str = "plotly chart"):
    return html_output(fig.to_html(include_plotlyjs="cdn", full_html=False), text)


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md_cell(
            """
            # US + Thailand Blend PoC

            Notebook เฉพาะสำหรับทดลอง US Static HMM sleeve, Thailand SET100 PIT sleeve, Gold, BTC และ joint-model variants.

            จุดประสงค์:

            - ตรวจผล blend แบบแยก US/TH sleeves แล้วค่อย rebalance รวม
            - ตรวจผล joint model ที่แปลง US + TH เป็น THB-base ก่อนเข้า copula model พร้อมกัน
            - เทียบผลกับ all-asset model ที่ใส่ US stocks, Thai stocks, Gold, BTC เข้า model เดียวกัน
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import sys

            ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebook" else Path.cwd().resolve()
            SRC = ROOT / "src"
            if str(SRC) not in sys.path:
                sys.path.insert(0, str(SRC))

            import pandas as pd
            import plotly.graph_objects as go

            from dynamic_factor_copula import default_paths

            pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
            paths = default_paths(ROOT)
            paths.result_dir
            """
        ),
        md_cell(
            """
            ## Parameters Used

            These are the validated sleeve parameters used by the current US + Thailand experiments.
            """
        ),
        code_cell(
            """
            params = pd.DataFrame(
                [
                    {"Parameter": "US universe", "Value": "sp500_pit"},
                    {"Parameter": "Thailand universe", "Value": "set100_pit"},
                    {"Parameter": "n_assets per sleeve", "Value": 30},
                    {"Parameter": "n_clusters", "Value": 4},
                    {"Parameter": "lookback_days", "Value": 504},
                    {"Parameter": "rebalance_freq", "Value": "ME"},
                    {"Parameter": "max_weight", "Value": 0.08},
                    {"Parameter": "momentum features", "Value": True},
                    {"Parameter": "momentum optimizer signal", "Value": True},
                    {"Parameter": "momentum_signal_mode", "Value": "mom_63"},
                    {"Parameter": "dropped features", "Value": "resid_vol, drawdown, downside_beta"},
                    {"Parameter": "strategic rebalance months", "Value": 1},
                    {"Parameter": "report currency", "Value": "THB"},
                ]
            )
            params
            """
        ),
        md_cell("## Thailand Sleeve Check"),
        code_cell(
            """
            thai_metrics = pd.read_csv(paths.result_dir / "thai_set100_pit_metrics.csv", index_col=0)
            thai_members = pd.read_csv(paths.result_dir / "latest_th_hmm_members.csv")
            display(thai_metrics)
            display(thai_members.head(30))
            """
        ),
        md_cell(
            """
            ## Separate Sleeve Blend

            This is the original blend experiment: run US Static HMM and Thailand Static HMM separately, apply daily exposure overlays, then rebalance US/TH/Gold/BTC sleeves together.
            """
        ),
        code_cell(
            """
            blend_summary_thb = pd.read_csv(paths.result_dir / "us_th_gold_btc_blended_summary_thb.csv")
            blend_curves_thb = pd.read_csv(paths.result_dir / "us_th_gold_btc_blended_curves_thb.csv", index_col=0, parse_dates=True)

            fig = go.Figure()
            for column in blend_curves_thb.columns:
                fig.add_trace(go.Scatter(x=blend_curves_thb.index, y=blend_curves_thb[column], mode="lines", name=column))
            fig.update_layout(
                title="Separate Sleeve Blend: US HMM, Thailand HMM, Gold, BTC (THB)",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                height=600,
            )
            fig.show()

            blend_summary_thb
            """
        ),
        md_cell(
            """
            ## Joint Model Variants

            This experiment converts US and Thailand assets into THB-base prices before modeling. It includes:

            - joint US+TH equity model, then 60/30/10 with Gold/BTC
            - all-asset model where US stocks, Thai stocks, Gold, and BTC enter the copula model together
            """
        ),
        code_cell(
            """
            joint_summary_thb = pd.read_csv(paths.result_dir / "us_th_joint_model_summary_thb.csv")
            joint_curves_thb = pd.read_csv(paths.result_dir / "us_th_joint_model_curves_thb.csv", index_col=0, parse_dates=True)

            fig = go.Figure()
            for column in joint_curves_thb.columns:
                fig.add_trace(go.Scatter(x=joint_curves_thb.index, y=joint_curves_thb[column], mode="lines", name=column))
            fig.update_layout(
                title="Joint Model Variants (THB)",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                height=600,
            )
            fig.show()

            joint_summary_thb
            """
        ),
        md_cell("## Combined Ranking"),
        code_cell(
            """
            combined = pd.concat(
                [
                    blend_summary_thb.assign(Experiment="Separate sleeve blend"),
                    joint_summary_thb.assign(Experiment="Joint model"),
                ],
                ignore_index=True,
            )
            combined = combined.sort_values("Sharpe", ascending=False)
            combined[
                ["Experiment", "Strategy", "CAGR", "Annual Vol", "Sharpe", "Sortino", "Max Drawdown", "Hit Rate", "Start", "End"]
            ]
            """
        ),
        md_cell(
            """
            ## Focus Comparison

            This isolates the two requested separate-sleeve references and compares them with the new joint-model variants.
            """
        ),
        code_cell(
            """
            focus_names = [
                "US/TH/Gold/BTC 40/20/30/10",
                "US HMM/Gold/BTC 60/30/10",
                "Joint US+TH Dynamic HMM Copula/Gold/BTC 60/30/10",
                "Joint US+TH Static Copula/Gold/BTC 60/30/10",
                "All assets in one Dynamic HMM Copula model",
            ]
            focus = combined.loc[combined["Strategy"].isin(focus_names)].copy()
            focus["CAGR - US/TH 40/20"] = focus["CAGR"] - float(
                focus.loc[focus["Strategy"] == "US/TH/Gold/BTC 40/20/30/10", "CAGR"].iloc[0]
            )
            focus["Sharpe - US/TH 40/20"] = focus["Sharpe"] - float(
                focus.loc[focus["Strategy"] == "US/TH/Gold/BTC 40/20/30/10", "Sharpe"].iloc[0]
            )
            focus["Max DD delta vs US/TH 40/20"] = focus["Max Drawdown"] - float(
                focus.loc[focus["Strategy"] == "US/TH/Gold/BTC 40/20/30/10", "Max Drawdown"].iloc[0]
            )
            focus = focus.sort_values("Sharpe", ascending=False)
            focus[
                [
                    "Experiment",
                    "Strategy",
                    "CAGR",
                    "Annual Vol",
                    "Sharpe",
                    "Sortino",
                    "Max Drawdown",
                    "Hit Rate",
                    "CAGR - US/TH 40/20",
                    "Sharpe - US/TH 40/20",
                    "Max DD delta vs US/TH 40/20",
                    "Start",
                    "End",
                ]
            ]
            """
        ),
        md_cell(
            """
            ## Objective Sweep

            This tests alternate optimizer objectives for the joint-model setup.

            Objective modes:

            - `mean_variance`: current baseline objective
            - `max_sharpe_mom`: maximize momentum per unit variance
            - `min_vol_mom_tilt`: lower-volatility objective with a smaller momentum tilt
            - `risk_parity_mom_tilt`: risk parity base with momentum tilt
            """
        ),
        code_cell(
            """
            objective_sweep = pd.read_csv(paths.result_dir / "us_th_joint_model_objective_sweep_thb.csv")
            objective_curves = pd.read_csv(
                paths.result_dir / "us_th_joint_model_objective_sweep_curves_thb.csv",
                index_col=0,
                parse_dates=True,
            )

            fig = go.Figure()
            for column in objective_curves.columns:
                if "Joint US+TH Dynamic HMM" in column:
                    fig.add_trace(go.Scatter(x=objective_curves.index, y=objective_curves[column], mode="lines", name=column))
            fig.update_layout(
                title="Objective Sweep: Joint US+TH Dynamic HMM/Gold/BTC 60/30/10",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                height=600,
            )
            fig.show()

            objective_sweep[
                [
                    "Model_Group",
                    "Objective",
                    "Strategy",
                    "CAGR",
                    "Annual Vol",
                    "Sharpe",
                    "Sortino",
                    "Max Drawdown",
                    "Hit Rate",
                    "Start",
                    "End",
                ]
            ].sort_values("Sharpe", ascending=False)
            """
        ),
        md_cell(
            """
            ## Interpretation

            Current read:

            - The best separate-sleeve Sharpe is `US/TH/Gold/BTC 45/15/30/10`.
            - The joint US+TH equity model improves Sharpe versus the separate-sleeve blend, but its comparison period starts later because the THB-base joint model needs a 504-day lookback from the available overlay/FX cache.
            - The all-asset one-model variant has much higher volatility and drawdown, so it is less attractive despite high CAGR.
            """
        ),
        md_cell(
            """
            ## Backtest Runner / Cached Results

            Default mode loads saved CSV results. Set `RUN_BACKTEST = True` to rerun the heavy scripts and overwrite
            the result files, then rerun the display cells.
            """
        ),
        code_cell(
            """
            RUN_BACKTEST = False

            if RUN_BACKTEST:
                import subprocess
                subprocess.run([sys.executable, str(ROOT / "scripts" / "run_us_th_blended.py")], check=True)
                subprocess.run([sys.executable, str(ROOT / "scripts" / "run_us_th_joint_model.py")], check=True)

            cached_files = pd.DataFrame(
                [
                    {"Result": "Separate sleeve blend", "File": "us_th_gold_btc_blended_summary_thb.csv"},
                    {"Result": "Joint model variants", "File": "us_th_joint_model_summary_thb.csv"},
                    {"Result": "Objective sweep", "File": "us_th_joint_model_objective_sweep_thb.csv"},
                    {"Result": "Best config fee/realloc extension", "File": "us_th_best_config_extension_summary_thb.csv"},
                    {"Result": "Asset count + max weight sweep", "File": "us_th_asset_count_max_weight_sweep_thb.csv"},
                    {"Result": "Best asset sweep fee/realloc extension", "File": "us_th_best_asset_sweep_fee_realloc_summary_thb.csv"},
                    {"Result": "All-asset static capped rebalance", "File": "us_th_all_asset_static_caps_summary_thb.csv"},
                ]
            )
            cached_files["Exists"] = cached_files["File"].map(lambda name: (paths.result_dir / name).exists())
            cached_files["Last Modified"] = cached_files["File"].map(
                lambda name: pd.Timestamp((paths.result_dir / name).stat().st_mtime, unit="s") if (paths.result_dir / name).exists() else pd.NaT
            )
            cached_files
            """
        ),
        md_cell(
            """
            ## Fee, Slippage, Reallocation Test

            Best config continuation uses the best objective found above: `min_vol_mom_tilt`.

            Cost assumption:

            - Dime-style commission: `0.15%` per traded notional, based on Dime's published stock commission examples.
            - Slippage assumption: `0.02%` per traded notional.
            - Total modeled cost: `0.17%`, or `17 bps`, per effective weight change.

            Implementation note: small per-share US regulatory charges such as CAT/SEC/TAF are not modeled because this
            notebook works at portfolio-weight level rather than share-count/order level.

            Reallocation test: when a sleeve's daily exposure falls below target, the idle exposure can either stay in cash
            or be reallocated to sleeves with full exposure that day.

            Fee references:

            - Dime US stocks: https://dime.co.th/en/invest/stock-us
            - Dime Thai stocks: https://dime.co.th/en/invest/stock-th
            """
        ),
        code_cell(
            """
            extension_summary = pd.read_csv(paths.result_dir / "us_th_best_config_extension_summary_thb.csv")
            extension_curves = pd.read_csv(paths.result_dir / "us_th_best_config_extension_curves_thb.csv", index_col=0, parse_dates=True)

            fig = go.Figure()
            for column in extension_curves.columns:
                fig.add_trace(go.Scatter(x=extension_curves.index, y=extension_curves[column], mode="lines", name=column))
            fig.update_layout(
                title="Best Config Extension: Fee/Slippage and Idle Exposure Reallocation",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                height=600,
            )
            fig.show()

            extension_summary
            """
        ),
        md_cell(
            """
            ## Asset Count + Max Weight Sweep

            This tests whether adding more Thai stocks and changing the optimizer `max_weight` improves the joint US+TH model.

            The sweep uses the best objective from the objective sweep: `min_vol_mom_tilt`.
            """
        ),
        code_cell(
            """
            asset_weight_sweep = pd.read_csv(paths.result_dir / "us_th_asset_count_max_weight_sweep_thb.csv")
            asset_weight_curves = pd.read_csv(
                paths.result_dir / "us_th_asset_count_max_weight_sweep_curves_thb.csv",
                index_col=0,
                parse_dates=True,
            )

            best_by_case = asset_weight_sweep.sort_values("Sharpe", ascending=False)
            fig = go.Figure()
            for column in asset_weight_curves.columns:
                if "Dynamic HMM" in column:
                    fig.add_trace(go.Scatter(x=asset_weight_curves.index, y=asset_weight_curves[column], mode="lines", name=column))
            fig.update_layout(
                title="Asset Count + Max Weight Sweep: Dynamic HMM Variants",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                height=600,
            )
            fig.show()

            best_by_case[
                [
                    "Case",
                    "US Assets",
                    "TH Assets",
                    "Max Weight",
                    "CAGR",
                    "Annual Vol",
                    "Sharpe",
                    "Sortino",
                    "Max Drawdown",
                    "Hit Rate",
                    "Start",
                    "End",
                ]
            ]
            """
        ),
        md_cell(
            """
            ## Best Asset Sweep Fee + Reallocation Test

            This takes the best asset/max-weight sweep result, `US30/TH30/max6 Dynamic`, then applies fee/slippage and idle exposure reallocation tests.
            """
        ),
        code_cell(
            """
            best_asset_extension = pd.read_csv(paths.result_dir / "us_th_best_asset_sweep_fee_realloc_summary_thb.csv")
            best_asset_curves = pd.read_csv(
                paths.result_dir / "us_th_best_asset_sweep_fee_realloc_curves_thb.csv",
                index_col=0,
                parse_dates=True,
            )

            fig = go.Figure()
            for column in best_asset_curves.columns:
                fig.add_trace(go.Scatter(x=best_asset_curves.index, y=best_asset_curves[column], mode="lines", name=column))
            fig.update_layout(
                title="Best Asset Sweep Extension: Fee/Slippage and Idle Exposure Reallocation",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                height=600,
            )
            fig.show()

            best_asset_extension
            """
        ),
        md_cell(
            """
            ## Asset Weight History

            Weight history for the best asset/max-weight sweep case: `US30/TH30/max6 Dynamic`.

            The equity sleeve weights sum to 100% inside the joint US+TH equity sleeve. The full portfolio weights scale those stock weights by the 60% equity sleeve target and add Gold 30% plus BTC 10%.
            """
        ),
        code_cell(
            """
            sleeve_weight_history = pd.read_csv(
                paths.result_dir / "us_th_best_asset_sweep_dynamic_weight_history_thb.csv",
                index_col=0,
                parse_dates=True,
            )
            full_asset_weight_history = pd.read_csv(
                paths.result_dir / "us_th_best_asset_sweep_full_asset_weight_history_thb.csv",
                index_col=0,
                parse_dates=True,
            )

            equity_weight_history = full_asset_weight_history.drop(columns=["GOLD", "BTC"], errors="ignore")
            top_assets = equity_weight_history.max().sort_values(ascending=False).head(20).index
            chart_weights = full_asset_weight_history[["GOLD", "BTC"]].copy()
            chart_weights = pd.concat([chart_weights, full_asset_weight_history[top_assets]], axis=1)
            chart_weights["Other Equity"] = equity_weight_history.drop(columns=top_assets, errors="ignore").sum(axis=1)
            chart_weights = chart_weights[["GOLD", "BTC", *top_assets, "Other Equity"]]

            fig = go.Figure()
            for column in chart_weights.columns:
                fig.add_trace(
                    go.Bar(
                        x=chart_weights.index,
                        y=chart_weights[column],
                        name=column,
                    )
                )
            fig.update_layout(
                title="US30/TH30/max6 Dynamic: Full Portfolio Asset Weight History (Top Assets + Other Equity)",
                xaxis_title="Rebalance Date",
                yaxis_title="Portfolio Weight",
                barmode="stack",
                height=650,
            )
            fig.show()

            latest_asset_weights = pd.read_csv(paths.result_dir / "us_th_best_asset_sweep_latest_asset_weights_thb.csv")
            latest_asset_weights = latest_asset_weights.loc[latest_asset_weights["Portfolio Weight"] > 1e-10]
            latest_asset_weights
            """
        ),
        md_cell(
            """
            ## Latest Asset Weight Standalone

            This cell is standalone: run it by itself to read the latest saved asset weights. If the saved files do not exist, it reruns only the best asset/max-weight extension job first.
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import subprocess
            import sys

            import pandas as pd

            ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebook" else Path.cwd().resolve()
            result_dir = ROOT / "result"
            latest_weight_file = result_dir / "us_th_best_asset_sweep_latest_asset_weights_thb.csv"

            if not latest_weight_file.exists():
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "run_us_th_joint_model.py"),
                        "--best-asset-extension-only",
                    ],
                    check=True,
                )

            latest_asset_weights = pd.read_csv(latest_weight_file)
            latest_asset_weights["Portfolio Weight %"] = latest_asset_weights["Portfolio Weight"] * 100.0
            latest_asset_weights["Equity Sleeve Weight %"] = latest_asset_weights["Equity Sleeve Weight"] * 100.0
            latest_asset_weights = latest_asset_weights.loc[latest_asset_weights["Portfolio Weight"] > 1e-10]
            latest_asset_weights = latest_asset_weights[
                ["Date", "Sleeve", "Asset", "Portfolio Weight %", "Equity Sleeve Weight %"]
            ].sort_values("Portfolio Weight %", ascending=False)
            pd.set_option("display.max_rows", len(latest_asset_weights))
            latest_asset_weights
            """
        ),
        md_cell(
            """
            ## Daily Asset Exposure

            Daily effective asset exposure for `US30/TH30/max6 Dynamic`.

            - `cash drag`: reduced exposure stays in cash.
            - `realloc idle`: reduced exposure is reallocated to sleeves with full exposure when available.
            """
        ),
        code_cell(
            """
            daily_cash_drag = pd.read_csv(
                paths.result_dir / "us_th_best_asset_sweep_daily_asset_exposure_cash_drag_thb.csv",
                index_col=0,
                parse_dates=True,
            )
            daily_realloc_idle = pd.read_csv(
                paths.result_dir / "us_th_best_asset_sweep_daily_asset_exposure_realloc_idle_thb.csv",
                index_col=0,
                parse_dates=True,
            )

            exposure_to_plot = daily_cash_drag.copy()
            equity_cols = exposure_to_plot.drop(columns=["GOLD", "BTC", "CASH"], errors="ignore")
            top_assets = equity_cols.max().sort_values(ascending=False).head(15).index
            chart_exposure = exposure_to_plot[["GOLD", "BTC", "CASH"]].copy()
            chart_exposure = pd.concat([chart_exposure, exposure_to_plot[top_assets]], axis=1)
            chart_exposure["Other Equity"] = equity_cols.drop(columns=top_assets, errors="ignore").sum(axis=1)
            chart_exposure = chart_exposure[["GOLD", "BTC", "CASH", *top_assets, "Other Equity"]]

            fig = go.Figure()
            for column in chart_exposure.columns:
                fig.add_trace(
                    go.Scatter(
                        x=chart_exposure.index,
                        y=chart_exposure[column],
                        mode="lines",
                        stackgroup="one",
                        name=column,
                    )
                )
            fig.update_layout(
                title="Daily Asset Exposure: Cash Drag (Top Assets + Other Equity + Cash)",
                xaxis_title="Date",
                yaxis_title="Portfolio Exposure",
                height=650,
            )
            fig.show()

            latest_daily_exposure = daily_cash_drag.iloc[-1].rename("Portfolio Exposure").reset_index()
            latest_daily_exposure.columns = ["Asset", "Portfolio Exposure"]
            latest_daily_exposure = latest_daily_exposure.loc[latest_daily_exposure["Portfolio Exposure"] > 1e-10]
            latest_daily_exposure["Portfolio Exposure %"] = latest_daily_exposure["Portfolio Exposure"] * 100.0
            latest_daily_exposure.insert(0, "Date", daily_cash_drag.index[-1].date().isoformat())
            latest_daily_exposure.sort_values("Portfolio Exposure %", ascending=False)
            """
        ),
        md_cell(
            """
            ## US/TH Side Trigger Reallocation Test

            This tests whether the stock sleeve should stay partly in cash or reallocate to the other equity side when only one market trigger is active.

            Trigger setup:

            - US stocks use `SPY` plus `^VIX`.
            - Thai stocks use `^SET.BK`.
            - Gold and BTC keep their own trend triggers.
            """
        ),
        code_cell(
            """
            side_trigger_summary = pd.read_csv(paths.result_dir / "us_th_side_trigger_reallocation_summary_thb.csv")
            side_trigger_curves = pd.read_csv(
                paths.result_dir / "us_th_side_trigger_reallocation_curves_thb.csv",
                index_col=0,
                parse_dates=True,
            )

            fig = go.Figure()
            for column in side_trigger_curves.columns:
                fig.add_trace(go.Scatter(x=side_trigger_curves.index, y=side_trigger_curves[column], mode="lines", name=column))
            fig.update_layout(
                title="US/TH Side Trigger: Cash Drag vs Reallocate to Active Stock Side",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                height=600,
            )
            fig.show()

            side_trigger_summary
            """
        ),
        md_cell(
            """
            ## Latest Trigger-by-Side Asset Weight

            Latest effective asset exposure for the selected realistic config: side-specific triggers with fee/slippage and stock-sleeve reallocation.
            """
        ),
        code_cell(
            """
            latest_side_trigger_weights = pd.read_csv(paths.result_dir / "us_th_side_trigger_latest_asset_weights_thb.csv")
            latest_side_trigger_weights
            """
        ),
        md_cell(
            """
            ## Best Config Saved

            Best config is selected from fee/slippage-adjusted results only.
            """
        ),
        code_cell(
            """
            best_config = pd.read_csv(paths.result_dir / "us_th_best_config_side_trigger_fee_slippage.csv")
            best_config
            """
        ),
        md_cell(
            """
            ## All-Asset Static Model: Gold/BTC Asset Caps

            This final rebalance backtest uses the Strategy B style config: US/TH stocks plus Gold/BTC inside one Static model.

            Optimizer constraints:

            - stock default max weight: `8%`
            - Gold (`GC=F`) cap: `30%`
            - BTC (`BTC-USD`) cap: `10%`
            """
        ),
        code_cell(
            """
            RUN_ALL_ASSET_STATIC_CAPS_BACKTEST = False

            if RUN_ALL_ASSET_STATIC_CAPS_BACKTEST:
                import subprocess
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "run_us_th_joint_model.py"),
                        "--all-asset-static-caps-only",
                    ],
                    check=True,
                )

            cap_config = pd.DataFrame(
                [
                    {"Asset": "Stocks", "Constraint": "default_max_weight", "Value": 0.08},
                    {"Asset": "GC=F", "Constraint": "asset_cap", "Value": 0.30},
                    {"Asset": "BTC-USD", "Constraint": "asset_cap", "Value": 0.10},
                ]
            )
            all_asset_static_summary = pd.read_csv(paths.result_dir / "us_th_all_asset_static_caps_summary_thb.csv")
            all_asset_static_curves = pd.read_csv(
                paths.result_dir / "us_th_all_asset_static_caps_curves_thb.csv",
                index_col=0,
                parse_dates=True,
            )
            all_asset_static_weights = pd.read_csv(
                paths.result_dir / "us_th_all_asset_static_caps_weight_history_thb.csv",
                index_col=0,
                parse_dates=True,
            )
            all_asset_static_latest = pd.read_csv(paths.result_dir / "us_th_all_asset_static_caps_latest_weights_thb.csv")

            fig = go.Figure()
            for column in all_asset_static_curves.columns:
                fig.add_trace(go.Scatter(x=all_asset_static_curves.index, y=all_asset_static_curves[column], mode="lines", name=column))
            fig.update_layout(
                title="US/TH Stocks + Gold/BTC: All Assets Static Model Capped Rebalance",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                height=600,
            )
            fig.show()

            top_assets = all_asset_static_weights.max().sort_values(ascending=False).head(20).index
            chart_weights = all_asset_static_weights[top_assets].copy()
            chart_weights["Other"] = all_asset_static_weights.drop(columns=top_assets, errors="ignore").sum(axis=1)
            fig = go.Figure()
            for column in chart_weights.columns:
                fig.add_trace(go.Bar(x=chart_weights.index, y=chart_weights[column], name=column))
            fig.update_layout(
                title="All-Asset Static Model: Rebalance Weight History",
                xaxis_title="Rebalance Date",
                yaxis_title="Portfolio Weight",
                barmode="stack",
                height=650,
            )
            fig.show()

            display(cap_config)
            display(all_asset_static_summary)
            all_asset_static_latest.loc[all_asset_static_latest["Portfolio Weight"] > 1e-10]
            """
        ),
    ]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
    return nb


def hydrate_outputs(nb: nbf.NotebookNode) -> None:
    paths = default_paths(ROOT)

    params = pd.DataFrame(
        [
            {"Parameter": "US universe", "Value": "sp500_pit"},
            {"Parameter": "Thailand universe", "Value": "set100_pit"},
            {"Parameter": "n_assets per sleeve", "Value": 30},
            {"Parameter": "n_clusters", "Value": 4},
            {"Parameter": "lookback_days", "Value": 504},
            {"Parameter": "rebalance_freq", "Value": "ME"},
            {"Parameter": "max_weight", "Value": 0.08},
            {"Parameter": "momentum features", "Value": True},
            {"Parameter": "momentum optimizer signal", "Value": True},
            {"Parameter": "momentum_signal_mode", "Value": "mom_63"},
            {"Parameter": "dropped features", "Value": "resid_vol, drawdown, downside_beta"},
            {"Parameter": "strategic rebalance months", "Value": 1},
            {"Parameter": "report currency", "Value": "THB"},
        ]
    )
    nb.cells[3]["outputs"] = [html_output(params.to_html(index=False), params.to_string(index=False))]
    nb.cells[3]["execution_count"] = 1

    thai_metrics = pd.read_csv(paths.result_dir / "thai_set100_pit_metrics.csv", index_col=0)
    thai_members = pd.read_csv(paths.result_dir / "latest_th_hmm_members.csv")
    thai_html = thai_metrics.to_html(float_format=lambda x: f"{x:,.4f}") + thai_members.head(30).to_html(index=False)
    thai_text = thai_metrics.to_string() + "\n\n" + thai_members.head(30).to_string(index=False)
    nb.cells[5]["outputs"] = [html_output(thai_html, thai_text)]
    nb.cells[5]["execution_count"] = 2

    blend_summary_thb = pd.read_csv(paths.result_dir / "us_th_gold_btc_blended_summary_thb.csv")
    blend_curves_thb = pd.read_csv(paths.result_dir / "us_th_gold_btc_blended_curves_thb.csv", index_col=0, parse_dates=True)
    fig = go.Figure()
    for column in blend_curves_thb.columns:
        fig.add_trace(go.Scatter(x=blend_curves_thb.index, y=blend_curves_thb[column], mode="lines", name=column))
    fig.update_layout(
        title="Separate Sleeve Blend: US HMM, Thailand HMM, Gold, BTC (THB)",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        height=600,
    )
    nb.cells[7]["outputs"] = [
        plotly_output(fig),
        html_output(
            blend_summary_thb.to_html(index=False, float_format=lambda x: f"{x:,.4f}"),
            blend_summary_thb.to_string(index=False),
        ),
    ]
    nb.cells[7]["execution_count"] = 3

    joint_summary_thb = pd.read_csv(paths.result_dir / "us_th_joint_model_summary_thb.csv")
    joint_curves_thb = pd.read_csv(paths.result_dir / "us_th_joint_model_curves_thb.csv", index_col=0, parse_dates=True)
    fig = go.Figure()
    for column in joint_curves_thb.columns:
        fig.add_trace(go.Scatter(x=joint_curves_thb.index, y=joint_curves_thb[column], mode="lines", name=column))
    fig.update_layout(
        title="Joint Model Variants (THB)",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        height=600,
    )
    nb.cells[9]["outputs"] = [
        plotly_output(fig),
        html_output(
            joint_summary_thb.to_html(index=False, float_format=lambda x: f"{x:,.4f}"),
            joint_summary_thb.to_string(index=False),
        ),
    ]
    nb.cells[9]["execution_count"] = 4

    combined = pd.concat(
        [
            blend_summary_thb.assign(Experiment="Separate sleeve blend"),
            joint_summary_thb.assign(Experiment="Joint model"),
        ],
        ignore_index=True,
    ).sort_values("Sharpe", ascending=False)
    combined = combined[
        ["Experiment", "Strategy", "CAGR", "Annual Vol", "Sharpe", "Sortino", "Max Drawdown", "Hit Rate", "Start", "End"]
    ]
    nb.cells[11]["outputs"] = [
        html_output(combined.to_html(index=False, float_format=lambda x: f"{x:,.4f}"), combined.to_string(index=False))
    ]
    nb.cells[11]["execution_count"] = 5

    focus_names = [
        "US/TH/Gold/BTC 40/20/30/10",
        "US HMM/Gold/BTC 60/30/10",
        "Joint US+TH Dynamic HMM Copula/Gold/BTC 60/30/10",
        "Joint US+TH Static Copula/Gold/BTC 60/30/10",
        "All assets in one Dynamic HMM Copula model",
    ]
    focus = combined.loc[combined["Strategy"].isin(focus_names)].copy()
    base_cagr = float(focus.loc[focus["Strategy"] == "US/TH/Gold/BTC 40/20/30/10", "CAGR"].iloc[0])
    base_sharpe = float(focus.loc[focus["Strategy"] == "US/TH/Gold/BTC 40/20/30/10", "Sharpe"].iloc[0])
    base_drawdown = float(focus.loc[focus["Strategy"] == "US/TH/Gold/BTC 40/20/30/10", "Max Drawdown"].iloc[0])
    focus["CAGR - US/TH 40/20"] = focus["CAGR"] - base_cagr
    focus["Sharpe - US/TH 40/20"] = focus["Sharpe"] - base_sharpe
    focus["Max DD delta vs US/TH 40/20"] = focus["Max Drawdown"] - base_drawdown
    focus = focus.sort_values("Sharpe", ascending=False)
    focus = focus[
        [
            "Experiment",
            "Strategy",
            "CAGR",
            "Annual Vol",
            "Sharpe",
            "Sortino",
            "Max Drawdown",
            "Hit Rate",
            "CAGR - US/TH 40/20",
            "Sharpe - US/TH 40/20",
            "Max DD delta vs US/TH 40/20",
            "Start",
            "End",
        ]
    ]
    nb.cells[13]["outputs"] = [
        html_output(focus.to_html(index=False, float_format=lambda x: f"{x:,.4f}"), focus.to_string(index=False))
    ]
    nb.cells[13]["execution_count"] = 6

    objective_sweep = pd.read_csv(paths.result_dir / "us_th_joint_model_objective_sweep_thb.csv")
    objective_curves = pd.read_csv(
        paths.result_dir / "us_th_joint_model_objective_sweep_curves_thb.csv",
        index_col=0,
        parse_dates=True,
    )
    fig = go.Figure()
    for column in objective_curves.columns:
        if "Joint US+TH Dynamic HMM" in column:
            fig.add_trace(go.Scatter(x=objective_curves.index, y=objective_curves[column], mode="lines", name=column))
    fig.update_layout(
        title="Objective Sweep: Joint US+TH Dynamic HMM/Gold/BTC 60/30/10",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        height=600,
    )
    objective_sweep_display = objective_sweep[
        [
            "Model_Group",
            "Objective",
            "Strategy",
            "CAGR",
            "Annual Vol",
            "Sharpe",
            "Sortino",
            "Max Drawdown",
            "Hit Rate",
            "Start",
            "End",
        ]
    ].sort_values("Sharpe", ascending=False)
    nb.cells[15]["outputs"] = [
        plotly_output(fig),
        html_output(
            objective_sweep_display.to_html(index=False, float_format=lambda x: f"{x:,.4f}"),
            objective_sweep_display.to_string(index=False),
        ),
    ]
    nb.cells[15]["execution_count"] = 7

    cached_files = pd.DataFrame(
        [
            {"Result": "Separate sleeve blend", "File": "us_th_gold_btc_blended_summary_thb.csv"},
            {"Result": "Joint model variants", "File": "us_th_joint_model_summary_thb.csv"},
            {"Result": "Objective sweep", "File": "us_th_joint_model_objective_sweep_thb.csv"},
            {"Result": "Best config fee/realloc extension", "File": "us_th_best_config_extension_summary_thb.csv"},
            {"Result": "Asset count + max weight sweep", "File": "us_th_asset_count_max_weight_sweep_thb.csv"},
            {"Result": "Best asset sweep fee/realloc extension", "File": "us_th_best_asset_sweep_fee_realloc_summary_thb.csv"},
            {"Result": "All-asset static capped rebalance", "File": "us_th_all_asset_static_caps_summary_thb.csv"},
        ]
    )
    cached_files["Exists"] = cached_files["File"].map(lambda name: (paths.result_dir / name).exists())
    cached_files["Last Modified"] = cached_files["File"].map(
        lambda name: pd.Timestamp((paths.result_dir / name).stat().st_mtime, unit="s") if (paths.result_dir / name).exists() else pd.NaT
    )
    nb.cells[18]["outputs"] = [
        html_output(cached_files.to_html(index=False), cached_files.to_string(index=False))
    ]
    nb.cells[18]["execution_count"] = 8

    extension_summary = pd.read_csv(paths.result_dir / "us_th_best_config_extension_summary_thb.csv")
    extension_curves = pd.read_csv(paths.result_dir / "us_th_best_config_extension_curves_thb.csv", index_col=0, parse_dates=True)
    fig = go.Figure()
    for column in extension_curves.columns:
        fig.add_trace(go.Scatter(x=extension_curves.index, y=extension_curves[column], mode="lines", name=column))
    fig.update_layout(
        title="Best Config Extension: Fee/Slippage and Idle Exposure Reallocation",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        height=600,
    )
    nb.cells[20]["outputs"] = [
        plotly_output(fig),
        html_output(
            extension_summary.to_html(index=False, float_format=lambda x: f"{x:,.4f}"),
            extension_summary.to_string(index=False),
        ),
    ]
    nb.cells[20]["execution_count"] = 9

    asset_weight_sweep = pd.read_csv(paths.result_dir / "us_th_asset_count_max_weight_sweep_thb.csv")
    asset_weight_curves = pd.read_csv(
        paths.result_dir / "us_th_asset_count_max_weight_sweep_curves_thb.csv",
        index_col=0,
        parse_dates=True,
    )
    best_by_case = asset_weight_sweep.sort_values("Sharpe", ascending=False)
    fig = go.Figure()
    for column in asset_weight_curves.columns:
        if "Dynamic HMM" in column:
            fig.add_trace(go.Scatter(x=asset_weight_curves.index, y=asset_weight_curves[column], mode="lines", name=column))
    fig.update_layout(
        title="Asset Count + Max Weight Sweep: Dynamic HMM Variants",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        height=600,
    )
    asset_weight_display = best_by_case[
        [
            "Case",
            "US Assets",
            "TH Assets",
            "Max Weight",
            "CAGR",
            "Annual Vol",
            "Sharpe",
            "Sortino",
            "Max Drawdown",
            "Hit Rate",
            "Start",
            "End",
        ]
    ]
    nb.cells[22]["outputs"] = [
        plotly_output(fig),
        html_output(
            asset_weight_display.to_html(index=False, float_format=lambda x: f"{x:,.4f}"),
            asset_weight_display.to_string(index=False),
        ),
    ]
    nb.cells[22]["execution_count"] = 10

    best_asset_extension = pd.read_csv(paths.result_dir / "us_th_best_asset_sweep_fee_realloc_summary_thb.csv")
    best_asset_curves = pd.read_csv(
        paths.result_dir / "us_th_best_asset_sweep_fee_realloc_curves_thb.csv",
        index_col=0,
        parse_dates=True,
    )
    fig = go.Figure()
    for column in best_asset_curves.columns:
        fig.add_trace(go.Scatter(x=best_asset_curves.index, y=best_asset_curves[column], mode="lines", name=column))
    fig.update_layout(
        title="Best Asset Sweep Extension: Fee/Slippage and Idle Exposure Reallocation",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        height=600,
    )
    nb.cells[24]["outputs"] = [
        plotly_output(fig),
        html_output(
            best_asset_extension.to_html(index=False, float_format=lambda x: f"{x:,.4f}"),
            best_asset_extension.to_string(index=False),
        ),
    ]
    nb.cells[24]["execution_count"] = 11

    sleeve_weight_file = paths.result_dir / "us_th_best_asset_sweep_dynamic_weight_history_thb.csv"
    full_weight_file = paths.result_dir / "us_th_best_asset_sweep_full_asset_weight_history_thb.csv"
    latest_weight_file = paths.result_dir / "us_th_best_asset_sweep_latest_asset_weights_thb.csv"
    if sleeve_weight_file.exists() and full_weight_file.exists() and latest_weight_file.exists():
        sleeve_weight_history = pd.read_csv(sleeve_weight_file, index_col=0, parse_dates=True)
        full_asset_weight_history = pd.read_csv(full_weight_file, index_col=0, parse_dates=True)
        equity_weight_history = full_asset_weight_history.drop(columns=["GOLD", "BTC"], errors="ignore")
        top_assets = equity_weight_history.max().sort_values(ascending=False).head(20).index
        chart_weights = full_asset_weight_history[["GOLD", "BTC"]].copy()
        chart_weights = pd.concat([chart_weights, full_asset_weight_history[top_assets]], axis=1)
        chart_weights["Other Equity"] = equity_weight_history.drop(columns=top_assets, errors="ignore").sum(axis=1)
        chart_weights = chart_weights[["GOLD", "BTC", *top_assets, "Other Equity"]]
        fig = go.Figure()
        for column in chart_weights.columns:
            fig.add_trace(
                go.Bar(
                    x=chart_weights.index,
                    y=chart_weights[column],
                    name=column,
                )
            )
        fig.update_layout(
            title="US30/TH30/max6 Dynamic: Full Portfolio Asset Weight History (Top Assets + Other Equity)",
            xaxis_title="Rebalance Date",
            yaxis_title="Portfolio Weight",
            barmode="stack",
            height=650,
        )
        latest_asset_weights = pd.read_csv(latest_weight_file)
        latest_asset_weights = latest_asset_weights.loc[latest_asset_weights["Portfolio Weight"] > 1e-10]
        nb.cells[26]["outputs"] = [
            plotly_output(fig),
            html_output(
                latest_asset_weights.to_html(index=False, float_format=lambda x: f"{x:,.4f}"),
                latest_asset_weights.to_string(index=False),
            ),
        ]
        nb.cells[26]["execution_count"] = 12

        standalone_display = latest_asset_weights.copy()
        standalone_display["Portfolio Weight %"] = standalone_display["Portfolio Weight"] * 100.0
        standalone_display["Equity Sleeve Weight %"] = standalone_display["Equity Sleeve Weight"] * 100.0
        standalone_display = standalone_display.loc[standalone_display["Portfolio Weight"] > 1e-10]
        standalone_display = standalone_display[
            ["Date", "Sleeve", "Asset", "Portfolio Weight %", "Equity Sleeve Weight %"]
        ].sort_values("Portfolio Weight %", ascending=False)
        nb.cells[28]["outputs"] = [
            html_output(
                standalone_display.to_html(index=False, float_format=lambda x: f"{x:,.4f}"),
                standalone_display.to_string(index=False),
            )
        ]
        nb.cells[28]["execution_count"] = 13

    daily_cash_drag_file = paths.result_dir / "us_th_best_asset_sweep_daily_asset_exposure_cash_drag_thb.csv"
    daily_realloc_file = paths.result_dir / "us_th_best_asset_sweep_daily_asset_exposure_realloc_idle_thb.csv"
    if daily_cash_drag_file.exists() and daily_realloc_file.exists():
        daily_cash_drag = pd.read_csv(daily_cash_drag_file, index_col=0, parse_dates=True)
        daily_realloc_idle = pd.read_csv(daily_realloc_file, index_col=0, parse_dates=True)
        exposure_to_plot = daily_cash_drag.copy()
        equity_cols = exposure_to_plot.drop(columns=["GOLD", "BTC", "CASH"], errors="ignore")
        top_assets = equity_cols.max().sort_values(ascending=False).head(15).index
        chart_exposure = exposure_to_plot[["GOLD", "BTC", "CASH"]].copy()
        chart_exposure = pd.concat([chart_exposure, exposure_to_plot[top_assets]], axis=1)
        chart_exposure["Other Equity"] = equity_cols.drop(columns=top_assets, errors="ignore").sum(axis=1)
        chart_exposure = chart_exposure[["GOLD", "BTC", "CASH", *top_assets, "Other Equity"]]

        fig = go.Figure()
        for column in chart_exposure.columns:
            fig.add_trace(
                go.Scatter(
                    x=chart_exposure.index,
                    y=chart_exposure[column],
                    mode="lines",
                    stackgroup="one",
                    name=column,
                )
            )
        fig.update_layout(
            title="Daily Asset Exposure: Cash Drag (Top Assets + Other Equity + Cash)",
            xaxis_title="Date",
            yaxis_title="Portfolio Exposure",
            height=650,
        )
        latest_daily_exposure = daily_cash_drag.iloc[-1].rename("Portfolio Exposure").reset_index()
        latest_daily_exposure.columns = ["Asset", "Portfolio Exposure"]
        latest_daily_exposure = latest_daily_exposure.loc[latest_daily_exposure["Portfolio Exposure"] > 1e-10]
        latest_daily_exposure["Portfolio Exposure %"] = latest_daily_exposure["Portfolio Exposure"] * 100.0
        latest_daily_exposure.insert(0, "Date", daily_cash_drag.index[-1].date().isoformat())
        latest_daily_exposure = latest_daily_exposure.sort_values("Portfolio Exposure %", ascending=False)
        nb.cells[30]["outputs"] = [
            plotly_output(fig),
            html_output(
                latest_daily_exposure.to_html(index=False, float_format=lambda x: f"{x:,.4f}"),
                latest_daily_exposure.to_string(index=False),
            ),
        ]
        nb.cells[30]["execution_count"] = 14

    side_trigger_summary_file = paths.result_dir / "us_th_side_trigger_reallocation_summary_thb.csv"
    side_trigger_curves_file = paths.result_dir / "us_th_side_trigger_reallocation_curves_thb.csv"
    if side_trigger_summary_file.exists() and side_trigger_curves_file.exists():
        side_trigger_summary = pd.read_csv(side_trigger_summary_file)
        side_trigger_curves = pd.read_csv(side_trigger_curves_file, index_col=0, parse_dates=True)
        fig = go.Figure()
        for column in side_trigger_curves.columns:
            fig.add_trace(go.Scatter(x=side_trigger_curves.index, y=side_trigger_curves[column], mode="lines", name=column))
        fig.update_layout(
            title="US/TH Side Trigger: Cash Drag vs Reallocate to Active Stock Side",
            xaxis_title="Date",
            yaxis_title="Portfolio Value",
            height=600,
        )
        nb.cells[32]["outputs"] = [
            plotly_output(fig),
            html_output(
                side_trigger_summary.to_html(index=False, float_format=lambda x: f"{x:,.4f}"),
                side_trigger_summary.to_string(index=False),
            ),
        ]
        nb.cells[32]["execution_count"] = 15

    latest_side_trigger_weights_file = paths.result_dir / "us_th_side_trigger_latest_asset_weights_thb.csv"
    if latest_side_trigger_weights_file.exists():
        latest_side_trigger_weights = pd.read_csv(latest_side_trigger_weights_file)
        nb.cells[34]["outputs"] = [
            html_output(
                latest_side_trigger_weights.to_html(index=False, float_format=lambda x: f"{x:,.4f}"),
                latest_side_trigger_weights.to_string(index=False),
            )
        ]
        nb.cells[34]["execution_count"] = 16

    best_config_file = paths.result_dir / "us_th_best_config_side_trigger_fee_slippage.csv"
    if best_config_file.exists():
        best_config = pd.read_csv(best_config_file)
        nb.cells[36]["outputs"] = [
            html_output(
                best_config.to_html(index=False, float_format=lambda x: f"{x:,.4f}"),
                best_config.to_string(index=False),
            )
        ]
        nb.cells[36]["execution_count"] = 17

    all_asset_static_summary_file = paths.result_dir / "us_th_all_asset_static_caps_summary_thb.csv"
    all_asset_static_curves_file = paths.result_dir / "us_th_all_asset_static_caps_curves_thb.csv"
    all_asset_static_weights_file = paths.result_dir / "us_th_all_asset_static_caps_weight_history_thb.csv"
    all_asset_static_latest_file = paths.result_dir / "us_th_all_asset_static_caps_latest_weights_thb.csv"
    if (
        all_asset_static_summary_file.exists()
        and all_asset_static_curves_file.exists()
        and all_asset_static_weights_file.exists()
        and all_asset_static_latest_file.exists()
    ):
        cap_config = pd.DataFrame(
            [
                {"Asset": "Stocks", "Constraint": "default_max_weight", "Value": 0.08},
                {"Asset": "GC=F", "Constraint": "asset_cap", "Value": 0.30},
                {"Asset": "BTC-USD", "Constraint": "asset_cap", "Value": 0.10},
            ]
        )
        all_asset_static_summary = pd.read_csv(all_asset_static_summary_file)
        all_asset_static_curves = pd.read_csv(all_asset_static_curves_file, index_col=0, parse_dates=True)
        all_asset_static_weights = pd.read_csv(all_asset_static_weights_file, index_col=0, parse_dates=True)
        all_asset_static_latest = pd.read_csv(all_asset_static_latest_file)
        fig = go.Figure()
        for column in all_asset_static_curves.columns:
            fig.add_trace(go.Scatter(x=all_asset_static_curves.index, y=all_asset_static_curves[column], mode="lines", name=column))
        fig.update_layout(
            title="US/TH Stocks + Gold/BTC: All Assets Static Model Capped Rebalance",
            xaxis_title="Date",
            yaxis_title="Portfolio Value",
            height=600,
        )
        top_assets = all_asset_static_weights.max().sort_values(ascending=False).head(20).index
        chart_weights = all_asset_static_weights[top_assets].copy()
        chart_weights["Other"] = all_asset_static_weights.drop(columns=top_assets, errors="ignore").sum(axis=1)
        weight_fig = go.Figure()
        for column in chart_weights.columns:
            weight_fig.add_trace(go.Bar(x=chart_weights.index, y=chart_weights[column], name=column))
        weight_fig.update_layout(
            title="All-Asset Static Model: Rebalance Weight History",
            xaxis_title="Rebalance Date",
            yaxis_title="Portfolio Weight",
            barmode="stack",
            height=650,
        )
        latest_display = all_asset_static_latest.loc[all_asset_static_latest["Portfolio Weight"] > 1e-10]
        html = (
            cap_config.to_html(index=False, float_format=lambda x: f"{x:,.4f}")
            + all_asset_static_summary.to_html(index=False, float_format=lambda x: f"{x:,.4f}")
            + latest_display.to_html(index=False, float_format=lambda x: f"{x:,.4f}")
        )
        text = (
            cap_config.to_string(index=False)
            + "\n\n"
            + all_asset_static_summary.to_string(index=False)
            + "\n\n"
            + latest_display.to_string(index=False)
        )
        nb.cells[38]["outputs"] = [
            plotly_output(fig),
            plotly_output(weight_fig),
            html_output(html, text),
        ]
        nb.cells[38]["execution_count"] = 18


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    hydrate_outputs(notebook)
    with NOTEBOOK_FILE.open("w", encoding="utf-8") as handle:
        nbf.write(notebook, handle)
    print(NOTEBOOK_FILE.name)


if __name__ == "__main__":
    main()
