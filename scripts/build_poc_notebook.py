from __future__ import annotations

import base64
import gc
from io import BytesIO
from pathlib import Path
import sys
import textwrap

import matplotlib.pyplot as plt
import nbformat as nbf
import pandas as pd
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynamic_factor_copula import (
    backtest_dynamic_factor_copula,
    build_overlay_comparison,
    compare_apply_returns,
    compare_rebalanced_portfolio,
    compare_sp_exposure,
    compare_trend_exposure,
    compute_port_opt_style_metrics,
    curve_from_returns,
    default_paths,
)


NOTEBOOK_DIR = ROOT / "notebook"
NOTEBOOK_FILE = NOTEBOOK_DIR / "multi_factor_copula_poc.ipynb"


def md_cell(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code_cell(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def html_output(html: str, text: str = ""):
    return nbf.v4.new_output("display_data", data={"text/html": html, "text/plain": text})


def png_output(fig) -> nbf.NotebookNode:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return nbf.v4.new_output("display_data", data={"image/png": encoded, "text/plain": "plot"})


def plotly_output(fig: go.Figure, text: str = "plotly chart"):
    return html_output(fig.to_html(include_plotlyjs="cdn", full_html=False), text)


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md_cell(
            """
            # Dynamic Multi-Factor Copula PoC

            Notebook นี้ทำ backtest เพื่อ **prove concept** ของแนวคิดจาก thesis:

            - เริ่มจาก **initial clustering**
            - อัปเดต cluster assignment แบบ **dynamic** ด้วย HMM-style forward filtering
            - ใช้ **Gaussian multi-factor copula approximation** เพื่อ forecast covariance
            - เปรียบเทียบกับ baseline แบบ **Equal Weight**, **Risk Parity**, และ **Static Copula**
            - ใช้ **full point-in-time S&P 500 membership** แล้วคัด top liquid assets ใหม่ทุก rebalance

            ข้อสำคัญ:

            - นี่เป็น **proof-of-concept implementation** ไม่ใช่ full thesis replication
            - ส่วน GAS ถูกทำในรูป **GAS-inspired centroid update** เพื่อรักษาแกนของแนวคิด time-varying loadings
            - data หลักอ่านจาก cache ใน `../port_opt_advance` และจะใช้ local cache เพิ่มเฉพาะกรณี asset ขาดจริง
            - universe ใช้ `sp500_ticker_start_end.csv` เพื่อทำ PIT membership check จริงในแต่ละรอบ
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

            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import plotly.graph_objects as go

            from dynamic_factor_copula import (
                backtest_dynamic_factor_copula,
                build_overlay_comparison,
                compare_apply_returns,
                compare_rebalanced_portfolio,
                compare_sp_exposure,
                compare_trend_exposure,
                compute_port_opt_style_metrics,
                curve_from_returns,
                default_paths,
            )

            plt.style.use("seaborn-v0_8-whitegrid")
            pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
            paths = default_paths(ROOT)
            BASE_FEATURE_FLAGS = {"resid_vol": False, "drawdown": False, "downside_beta": False}
            paths
            """
        ),
        md_cell(
            """
            ## Run Backtest

            config ด้านล่างตั้งใจให้ runtime พอเหมาะและยังเห็นพฤติกรรม dynamic clustering ชัดเจน:

            - universe = top 30 liquid names จากสมาชิก S&P 500 จริง ณ วัน rebalance
            - lookback 3 ปี
            - rebalance รายเดือน
            - 4 clusters
            """
        ),
        code_cell(
            """
            results = backtest_dynamic_factor_copula(
                start_date="2012-01-01",
                end_date="2026-04-30",
                n_assets=30,
                n_clusters=4,
                lookback_days=756,
                rebalance_freq="ME",
                max_weight=0.08,
                point_in_time_liquid=True,
                universe_mode="sp500_pit",
                feature_flags=BASE_FEATURE_FLAGS,
                paths=paths,
            )

            metrics = results["metrics"].sort_values("Sharpe", ascending=False)
            metrics
            """
        ),
        md_cell("## Equity Curves"),
        code_cell(
            """
            nav = results["nav"]
            fig, ax = plt.subplots(figsize=(12, 6))
            for name in ["Benchmark", "Equal Weight", "Risk Parity", "Static Copula", "Dynamic HMM Copula"]:
                nav[name].plot(ax=ax, label=name, linewidth=2)
            ax.set_title("Backtest NAV Comparison")
            ax.set_ylabel("Growth of $1")
            ax.legend()
            plt.show()
            """
        ),
        md_cell("## Performance Table"),
        code_cell(
            """
            display_cols = ["Total Return", "CAGR", "Annual Vol", "Sharpe", "Sortino", "Max Drawdown", "Benchmark Relative Return", "Turnover"]
            metrics[display_cols].style.format({
                "Total Return": "{:.2%}",
                "CAGR": "{:.2%}",
                "Annual Vol": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Sortino": "{:.2f}",
                "Max Drawdown": "{:.2%}",
                "Benchmark Relative Return": "{:.2%}",
                "Turnover": "{:.2f}",
            })
            """
        ),
        md_cell("## Dynamic Cluster Migration"),
        code_cell(
            """
            assignments = pd.DataFrame(results["dynamic_state"]["assignment_history"]).T.sort_index()
            assignments.index.name = "rebalance_date"
            static_assignments = results["initial_clusters"]["labels"]

            changed_share = (assignments.ne(static_assignments, axis=1)).mean(axis=1)
            fig, ax = plt.subplots(figsize=(12, 4))
            changed_share.plot(ax=ax, color="#c2410c", linewidth=2)
            ax.set_title("Share of Assets Reassigned vs Initial Clusters")
            ax.set_ylabel("Share")
            ax.set_ylim(0, 1)
            plt.show()

            changed_share.describe()
            """
        ),
        md_cell("## PIT Universe Summary"),
        code_cell(
            """
            universe_history = results["universe_history"]
            summary_rows = []
            for date, members in universe_history.items():
                summary_rows.append({
                    "rebalance_date": date,
                    "selected_assets": len(members),
                    "sample_members": ", ".join(members[:10]),
                })
            universe_summary = pd.DataFrame(summary_rows).sort_values("rebalance_date")
            universe_summary.head()
            """
        ),
        md_cell("## Universe Turnover"),
        code_cell(
            """
            universe_history = results["universe_history"]
            ordered_dates = sorted(universe_history)
            turnover_rows = []
            for prev_date, curr_date in zip(ordered_dates[:-1], ordered_dates[1:]):
                prev_set = set(universe_history[prev_date])
                curr_set = set(universe_history[curr_date])
                entered = curr_set - prev_set
                exited = prev_set - curr_set
                turnover_rows.append({
                    "rebalance_date": curr_date,
                    "entered_count": len(entered),
                    "exited_count": len(exited),
                    "membership_turnover": len(entered | exited) / max(len(curr_set), 1),
                })
            turnover_df = pd.DataFrame(turnover_rows)
            turnover_df.head()
            """
        ),
        md_cell("## Universe Turnover Plot"),
        code_cell(
            """
            fig, ax = plt.subplots(figsize=(12, 4))
            turnover_df.set_index("rebalance_date")["membership_turnover"].plot(ax=ax, color="#0f766e", linewidth=2)
            ax.set_title("Point-in-Time Universe Membership Turnover")
            ax.set_ylabel("Turnover")
            ax.set_ylim(0, 1)
            plt.show()
            """
        ),
        md_cell("## Cluster Posterior Snapshot"),
        code_cell(
            """
            latest_date = max(results["dynamic_state"]["posterior_history"])
            latest_posterior = results["dynamic_state"]["posterior_history"][latest_date].copy()
            latest_posterior["assigned_cluster"] = latest_posterior.idxmax(axis=1)
            latest_posterior.sort_values("assigned_cluster")
            """
        ),
        md_cell("## Recent Portfolio Weights"),
        code_cell(
            """
            latest_weight_date = max(results["weights_history"]["Dynamic HMM Copula"])
            recent_weights = pd.DataFrame({
                "dynamic": results["weights_history"]["Dynamic HMM Copula"][latest_weight_date],
                "static": results["weights_history"]["Static Copula"][latest_weight_date],
                "equal_weight": results["weights_history"]["Equal Weight"][latest_weight_date],
            }).fillna(0.0).sort_values("dynamic", ascending=False)
            recent_weights.head(16)
            """
        ),
        md_cell("## Most Frequent PIT Members"),
        code_cell(
            """
            from collections import Counter

            counts = Counter()
            for members in universe_history.values():
                counts.update(members)
            frequent_members = pd.DataFrame(
                [{"ticker": ticker, "selected_rebalances": count} for ticker, count in counts.most_common(30)]
            )
            frequent_members
            """
        ),
        md_cell("## Static Weight History"),
        code_cell(
            """
            static_weights = []
            for rebalance_date, weights in sorted(results["weights_history"]["Static Copula"].items()):
                row = weights.rename(rebalance_date)
                static_weights.append(row)
            static_weights_df = pd.DataFrame(static_weights).fillna(0.0).sort_index(axis=1)
            fig = go.Figure()
            for column in static_weights_df.columns:
                fig.add_trace(go.Bar(x=static_weights_df.index, y=static_weights_df[column], name=column))
            fig.update_layout(
                title="Static Copula Weight History",
                barmode="stack",
                xaxis_title="Rebalance Date",
                yaxis_title="Portfolio Weight",
                yaxis_range=[0, 1],
                height=600,
            )
            fig.show()
            """
        ),
        md_cell("## Dynamic Weight History"),
        code_cell(
            """
            dynamic_weights = []
            for rebalance_date, weights in sorted(results["weights_history"]["Dynamic HMM Copula"].items()):
                row = weights.rename(rebalance_date)
                dynamic_weights.append(row)
            dynamic_weights_df = pd.DataFrame(dynamic_weights).fillna(0.0).sort_index(axis=1)
            fig = go.Figure()
            for column in dynamic_weights_df.columns:
                fig.add_trace(go.Bar(x=dynamic_weights_df.index, y=dynamic_weights_df[column], name=column))
            fig.update_layout(
                title="Dynamic HMM Copula Weight History",
                barmode="stack",
                xaxis_title="Rebalance Date",
                yaxis_title="Portfolio Weight",
                yaxis_range=[0, 1],
                height=600,
            )
            fig.show()
            """
        ),
        md_cell("## Risk Parity Weight History"),
        code_cell(
            """
            risk_parity_weights = []
            for rebalance_date, weights in sorted(results["weights_history"]["Risk Parity"].items()):
                row = weights.rename(rebalance_date)
                risk_parity_weights.append(row)
            risk_parity_weights_df = pd.DataFrame(risk_parity_weights).fillna(0.0).sort_index(axis=1)
            fig = go.Figure()
            for column in risk_parity_weights_df.columns:
                fig.add_trace(go.Bar(x=risk_parity_weights_df.index, y=risk_parity_weights_df[column], name=column))
            fig.update_layout(
                title="Risk Parity Weight History",
                barmode="stack",
                xaxis_title="Rebalance Date",
                yaxis_title="Portfolio Weight",
                yaxis_range=[0, 1],
                height=600,
            )
            fig.show()
            """
        ),
        md_cell("## Latest Weight Comparison"),
        code_cell(
            """
            latest_weight_date = max(results["weights_history"]["Dynamic HMM Copula"])
            latest_weight_compare = pd.DataFrame({
                "Equal Weight": results["weights_history"]["Equal Weight"][latest_weight_date],
                "Risk Parity": results["weights_history"]["Risk Parity"][latest_weight_date],
                "Static Copula": results["weights_history"]["Static Copula"][latest_weight_date],
                "Dynamic HMM Copula": results["weights_history"]["Dynamic HMM Copula"][latest_weight_date],
            }).fillna(0.0)
            latest_weight_compare["max_weight"] = latest_weight_compare.max(axis=1)
            latest_weight_compare = latest_weight_compare.sort_values("max_weight", ascending=False).drop(columns="max_weight")
            latest_weight_compare.head(20)
            """
        ),
        md_cell(
            """
            ## Overlay Summary

            Daily exposure is a close-to-next-session risk-control rule. The signal is computed from information known at the close, then applied to the next trading session's return so the backtest does not use same-day close information to trade the same day.

            Rules used in this section:

            - S&P 500 and Static HMM equity sleeve: start at 100% exposure, then cap exposure at 65% when SPY is below its 200-day moving average, 50% when SPY drawdown is at least 8% or VIX is at least 28, and 25% when SPY drawdown is at least 15% or VIX is at least 35.
            - Gold: start at 100% exposure, then cap exposure at 50% when gold is below its 200-day moving average.
            - BTC: start at 100% exposure, then cut to 0% when BTC is below its 200-day moving average.
            - All daily exposure signals are lagged one session: a signal observed at today's close affects tomorrow's exposure.

            The summary compares raw exposure, daily-exposure variants, and 60/30/10 overlays. The Daily Exposure Impact table below isolates whether the daily exposure rule helped by comparing each sleeve against its own no-overlay baseline over the same date range.
            """
        ),
        code_cell(
            """
            results_no_momentum = backtest_dynamic_factor_copula(
                start_date="2012-01-01",
                end_date="2026-04-30",
                n_assets=30,
                n_clusters=4,
                lookback_days=756,
                rebalance_freq="ME",
                max_weight=0.08,
                point_in_time_liquid=True,
                universe_mode="sp500_pit",
                include_momentum=False,
                feature_flags=BASE_FEATURE_FLAGS,
                paths=paths,
            )
            overlay_results = build_overlay_comparison(results, paths=paths, mix_weights=(0.60, 0.30, 0.10))
            overlay_results_no_momentum = build_overlay_comparison(results_no_momentum, paths=paths, mix_weights=(0.60, 0.30, 0.10))

            overlay_summary = pd.DataFrame(
                [
                    overlay_results["summary"].loc["S&P 500 daily exposure"],
                    overlay_results["summary"].loc["S&P/Gold/BTC 60/30/10 daily exposure"],
                    overlay_results["summary"].loc["Static HMM daily exposure"],
                    overlay_results_no_momentum["summary"].loc["Static HMM/Gold/BTC 60/30/10 daily exposure"],
                    overlay_results["summary"].loc["Static HMM/Gold/BTC 60/30/10 daily exposure"],
                ],
                index=[
                    "S&P 500 daily exposure",
                    "S&P/Gold/BTC 60/30/10 daily exposure",
                    "Static HMM daily exposure",
                    "Static HMM/Gold/BTC 60/30/10 daily exposure",
                    "Static HMM with momentum/Gold/BTC 60/30/10 daily exposure",
                ],
            )
            sp500_curve = pd.Series(overlay_results["curves"]["S&P 500"]).dropna()
            sp500_daily_curve = pd.Series(overlay_results["curves"]["S&P 500 daily exposure"]).dropna()
            static_daily_curve = pd.Series(overlay_results["curves"]["Static HMM daily exposure"]).dropna()
            static_raw_curve = results["nav"]["Static Copula"].reindex(static_daily_curve.index).ffill().dropna()
            static_raw_curve = static_raw_curve / static_raw_curve.iloc[0] * 10_000.0

            daily_exposure_impact = pd.DataFrame(
                [
                    compute_port_opt_style_metrics(sp500_daily_curve, risk_free_rate=0.03)
                    - compute_port_opt_style_metrics(sp500_curve, risk_free_rate=0.03),
                    compute_port_opt_style_metrics(static_daily_curve, risk_free_rate=0.03)
                    - compute_port_opt_style_metrics(static_raw_curve, risk_free_rate=0.03),
                ],
                index=[
                    "S&P 500 daily exposure minus raw S&P 500",
                    "Static HMM daily exposure minus raw Static HMM",
                ],
            )
            daily_exposure_impact["Sharpe Helped"] = daily_exposure_impact["Sharpe"].map(lambda x: "Yes" if x > 0 else "No")
            daily_exposure_impact["Drawdown Helped"] = daily_exposure_impact["Max Drawdown"].map(lambda x: "Yes" if x > 0 else "No")

            display(overlay_summary)
            daily_exposure_impact
            """
        ),
        md_cell("## Overlay Equity Curve Comparison"),
        code_cell(
            """
            overlay_curves = pd.DataFrame(
                {
                    "S&P 500": pd.Series(overlay_results["curves"]["S&P 500"]),
                    "S&P 500 daily exposure": pd.Series(overlay_results["curves"]["S&P 500 daily exposure"]),
                    "S&P/Gold/BTC 60/30/10 daily exposure": pd.Series(overlay_results["curves"]["S&P/Gold/BTC 60/30/10 daily exposure"]),
                    "Static HMM daily exposure": pd.Series(overlay_results["curves"]["Static HMM daily exposure"]),
                    "Static HMM/Gold/BTC 60/30/10 daily exposure": pd.Series(overlay_results_no_momentum["curves"]["Static HMM/Gold/BTC 60/30/10 daily exposure"]),
                    "Static HMM with momentum/Gold/BTC 60/30/10 daily exposure": pd.Series(overlay_results["curves"]["Static HMM/Gold/BTC 60/30/10 daily exposure"]),
                }
            ).ffill()
            overlay_curves = overlay_curves[
                [
                    "S&P 500",
                    "S&P 500 daily exposure",
                    "S&P/Gold/BTC 60/30/10 daily exposure",
                    "Static HMM daily exposure",
                    "Static HMM/Gold/BTC 60/30/10 daily exposure",
                    "Static HMM with momentum/Gold/BTC 60/30/10 daily exposure",
                ]
            ]
            overlay_curves.plot(figsize=(13, 7), linewidth=2, title="S&P500 vs Overlay Variants incl. Static HMM + Gold/BTC")
            plt.ylabel("Portfolio Value")
            plt.show()
            """
        ),
        md_cell("## Overlay Daily Exposure History"),
        code_cell(
            """
            exposure_compare = overlay_results["exposure_compare"]
            fig = go.Figure()
            for column in exposure_compare.columns:
                fig.add_trace(go.Scatter(x=exposure_compare.index, y=exposure_compare[column], mode="lines", name=column, line=dict(width=2.4)))
            fig.update_layout(
                title="Daily Exposure Comparison",
                xaxis_title="Date",
                yaxis_title="Exposure",
                yaxis_range=[0, 1.05],
                height=550,
                template="plotly_white",
            )
            fig.show()
            """
        ),
        md_cell("## Static HMM Mix Sweep"),
        code_cell(
            """
            overlay_prices = pd.read_parquet(paths.local_cache_root / "overlay_compare_prices.parquet").sort_index()
            overlay_prices.index = pd.to_datetime(overlay_prices.index)
            overlay_prices = overlay_prices.dropna(subset=["SPY", "GC=F", "BTC-USD", "^VIX", "USDTHB=X"]).loc[: "2026-04-29"]
            fx_returns = overlay_prices["USDTHB=X"].pct_change(fill_method=None).fillna(0.0)
            gold = overlay_prices["GC=F"]
            btc = overlay_prices["BTC-USD"]
            gold_overlay = compare_apply_returns(
                gold.pct_change(fill_method=None).fillna(0.0),
                compare_trend_exposure(gold, 0.50),
                "USD_STATIC",
                fx_returns,
            )
            btc_overlay = compare_apply_returns(
                btc.pct_change(fill_method=None).fillna(0.0),
                compare_trend_exposure(btc, 0.00),
                "USD_STATIC",
                fx_returns,
            )
            static_hmm_overlay = results["nav"]["Static Copula"].pct_change(fill_method=None).fillna(0.0).reindex(overlay_prices.index).fillna(0.0)

            mix_rows = []
            mix_curves = {}
            for mix in [(1.00, 0.00, 0.00), (0.70, 0.20, 0.10), (0.65, 0.25, 0.10), (0.60, 0.30, 0.10)]:
                mix_name = f"{int(mix[0]*100)}/{int(mix[1]*100)}/{int(mix[2]*100)}"
                sleeve = pd.concat(
                    {
                        "STATIC_HMM_OVERLAY": static_hmm_overlay,
                        "GOLD": gold_overlay,
                        "BTC": btc_overlay,
                    },
                    axis=1,
                ).dropna()
                port_ret = compare_rebalanced_portfolio(
                    sleeve,
                    weights=pd.Series({"STATIC_HMM_OVERLAY": mix[0], "GOLD": mix[1], "BTC": mix[2]}, dtype=float),
                )
                curve = curve_from_returns(port_ret)
                metrics = compute_port_opt_style_metrics(curve, risk_free_rate=0.03)
                mix_rows.append({"Mix": mix_name, **metrics.to_dict()})
                mix_curves[mix_name] = curve
            mix_sweep = pd.DataFrame(mix_rows)

            fig = go.Figure()
            for mix_name, curve in mix_curves.items():
                fig.add_trace(go.Scatter(x=curve.index, y=curve.values, mode="lines", name=mix_name))
            fig.update_layout(
                title="Static HMM with Momentum + Gold/BTC Mix Sweep",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                height=550,
            )
            fig.show()
            mix_sweep
            """
        ),
        md_cell("## Static HMM with Momentum 60/30/10 Asset Weight History"),
        code_cell(
            """
            static_hmm_603010_rows = []
            for rebalance_date, weights in sorted(results["weights_history"]["Static Copula"].items()):
                row = (0.60 * weights).copy()
                row.loc["GOLD"] = 0.30
                row.loc["BTC"] = 0.10
                static_hmm_603010_rows.append(row.rename(rebalance_date))
            static_hmm_603010_df = pd.DataFrame(static_hmm_603010_rows).fillna(0.0).sort_index(axis=1)

            fig = go.Figure()
            for column in static_hmm_603010_df.columns:
                fig.add_trace(go.Bar(x=static_hmm_603010_df.index, y=static_hmm_603010_df[column], name=column))
            fig.update_layout(
                title="Static HMM with Momentum/Gold/BTC 60/30/10 Asset Weight History",
                barmode="stack",
                xaxis_title="Rebalance Date",
                yaxis_title="Portfolio Weight",
                yaxis_range=[0, 1],
                height=650,
            )
            fig.show()
            """
        ),
        md_cell("## Momentum Attribution"),
        code_cell(
            """
            results_features_only = backtest_dynamic_factor_copula(
                start_date="2012-01-01",
                end_date="2026-04-30",
                n_assets=30,
                n_clusters=4,
                lookback_days=756,
                rebalance_freq="ME",
                max_weight=0.08,
                point_in_time_liquid=True,
                universe_mode="sp500_pit",
                include_momentum_features=True,
                include_momentum_signal=False,
                feature_flags=BASE_FEATURE_FLAGS,
                paths=paths,
            )
            results_signal_only = backtest_dynamic_factor_copula(
                start_date="2012-01-01",
                end_date="2026-04-30",
                n_assets=30,
                n_clusters=4,
                lookback_days=756,
                rebalance_freq="ME",
                max_weight=0.08,
                point_in_time_liquid=True,
                universe_mode="sp500_pit",
                include_momentum_features=False,
                include_momentum_signal=True,
                feature_flags=BASE_FEATURE_FLAGS,
                paths=paths,
            )

            attribution_rows = []
            attribution_cases = [
                ("No momentum", results_no_momentum, False, False),
                ("Features only", results_features_only, True, False),
                ("Signal only", results_signal_only, False, True),
                ("Features + Signal", results, True, True),
            ]
            for case_name, case_results, use_feat, use_sig in attribution_cases:
                case_overlay = build_overlay_comparison(case_results, paths=paths, mix_weights=(0.60, 0.30, 0.10))
                metrics = case_overlay["summary"].loc["Static HMM/Gold/BTC 60/30/10 daily exposure"]
                attribution_rows.append(
                    {
                        "Case": case_name,
                        "Momentum Features": use_feat,
                        "Momentum Signal": use_sig,
                        **metrics.to_dict(),
                    }
                )
            attribution_summary = pd.DataFrame(attribution_rows)

            fig = go.Figure()
            fig.add_trace(go.Bar(x=attribution_summary["Case"], y=attribution_summary["CAGR"], name="CAGR"))
            fig.add_trace(go.Bar(x=attribution_summary["Case"], y=attribution_summary["Sharpe"], name="Sharpe", yaxis="y2"))
            fig.update_layout(
                title="Momentum Attribution: Features vs Optimizer Signal",
                barmode="group",
                xaxis_title="Case",
                yaxis=dict(title="CAGR"),
                yaxis2=dict(title="Sharpe", overlaying="y", side="right"),
                height=550,
            )
            fig.show()
            attribution_summary
            """
        ),
        md_cell("## Lookback Sweep (60/30/10)"),
        code_cell(
            """
            lookback_rows = []
            lookback_curves = {}
            for lookback_days in [252, 504, 756]:
                lookback_results = backtest_dynamic_factor_copula(
                    start_date="2012-01-01",
                    end_date="2026-04-30",
                    n_assets=30,
                    n_clusters=4,
                    lookback_days=lookback_days,
                    rebalance_freq="ME",
                    max_weight=0.08,
                    point_in_time_liquid=True,
                    universe_mode="sp500_pit",
                    include_momentum=True,
                    feature_flags=BASE_FEATURE_FLAGS,
                    paths=paths,
                )
                lookback_overlay = build_overlay_comparison(lookback_results, paths=paths, mix_weights=(0.60, 0.30, 0.10))
                curve_name = "Static HMM/Gold/BTC 60/30/10 daily exposure"
                metrics = lookback_overlay["summary"].loc[curve_name]
                lookback_rows.append({"Lookback Days": lookback_days, **metrics.to_dict()})
                lookback_curves[f"{lookback_days} days"] = lookback_overlay["curves"][curve_name]
            lookback_sweep = pd.DataFrame(lookback_rows)

            fig = go.Figure()
            for label, curve in lookback_curves.items():
                fig.add_trace(go.Scatter(x=curve.index, y=curve.values, mode="lines", name=label))
            fig.update_layout(
                title="Lookback Sweep: Static HMM with Momentum/Gold/BTC 60/30/10",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                height=550,
            )
            fig.show()
            lookback_sweep
            """
        ),
        md_cell("## Strategic Rebalance Sweep (60/30/10)"),
        code_cell(
            """
            rebalance_rows = []
            rebalance_curves = {}
            for months in [1, 3, 6]:
                rebalance_overlay = build_overlay_comparison(
                    results,
                    paths=paths,
                    mix_weights=(0.60, 0.30, 0.10),
                    strategic_rebalance_months=months,
                )
                curve_name = "Static HMM/Gold/BTC 60/30/10 daily exposure"
                metrics = rebalance_overlay["summary"].loc[curve_name]
                rebalance_rows.append({"Strategic Rebalance (Months)": months, **metrics.to_dict()})
                rebalance_curves[f"{months} month"] = rebalance_overlay["curves"][curve_name]
            rebalance_sweep = pd.DataFrame(rebalance_rows)

            fig = go.Figure()
            for label, curve in rebalance_curves.items():
                fig.add_trace(go.Scatter(x=curve.index, y=curve.values, mode="lines", name=label))
            fig.update_layout(
                title="Strategic Rebalance Sweep: Static HMM with Momentum/Gold/BTC 60/30/10",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                height=550,
            )
            fig.show()
            rebalance_sweep
            """
        ),
        md_cell("## Factor Ablation (60/30/10)"),
        code_cell(
            """
            factor_cases = [
                ("All features", {}),
                ("No resid_vol", {"resid_vol": False}),
                ("No drawdown", {"drawdown": False}),
                ("No downside_beta", {"downside_beta": False}),
            ]
            factor_rows = []
            factor_curves = {}
            for case_name, feature_flags in factor_cases:
                factor_results = backtest_dynamic_factor_copula(
                    start_date="2012-01-01",
                    end_date="2026-04-30",
                    n_assets=30,
                    n_clusters=4,
                    lookback_days=756,
                    rebalance_freq="ME",
                    max_weight=0.08,
                    point_in_time_liquid=True,
                    universe_mode="sp500_pit",
                    include_momentum=True,
                    feature_flags=feature_flags,
                    paths=paths,
                )
                factor_overlay = build_overlay_comparison(factor_results, paths=paths, mix_weights=(0.60, 0.30, 0.10))
                curve_name = "Static HMM/Gold/BTC 60/30/10 daily exposure"
                metrics = factor_overlay["summary"].loc[curve_name]
                factor_rows.append({"Case": case_name, **metrics.to_dict()})
                factor_curves[case_name] = factor_overlay["curves"][curve_name]
            factor_ablation = pd.DataFrame(factor_rows)

            fig = go.Figure()
            for label, curve in factor_curves.items():
                fig.add_trace(go.Scatter(x=curve.index, y=curve.values, mode="lines", name=label))
            fig.update_layout(
                title="Factor Ablation: Static HMM with Momentum/Gold/BTC 60/30/10",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                height=550,
            )
            fig.show()
            factor_ablation
            """
        ),
        md_cell("## Momentum Signal Variants (60/30/10)"),
        code_cell(
            """
            signal_rows = []
            signal_curves = {}
            signal_modes = [
                "mom_21",
                "mom_63",
                "blend_21_63",
                "zscore_63",
                "rank_63",
            ]
            for signal_mode in signal_modes:
                signal_results = backtest_dynamic_factor_copula(
                    start_date="2012-01-01",
                    end_date="2026-04-30",
                    n_assets=30,
                    n_clusters=4,
                    lookback_days=756,
                    rebalance_freq="ME",
                    max_weight=0.08,
                    point_in_time_liquid=True,
                    universe_mode="sp500_pit",
                    include_momentum_features=True,
                    include_momentum_signal=True,
                    momentum_signal_mode=signal_mode,
                    feature_flags=BASE_FEATURE_FLAGS,
                    paths=paths,
                )
                signal_overlay = build_overlay_comparison(signal_results, paths=paths, mix_weights=(0.60, 0.30, 0.10))
                curve_name = "Static HMM/Gold/BTC 60/30/10 daily exposure"
                metrics = signal_overlay["summary"].loc[curve_name]
                signal_rows.append({"Signal Mode": signal_mode, **metrics.to_dict()})
                signal_curves[signal_mode] = signal_overlay["curves"][curve_name]
            signal_variant_summary = pd.DataFrame(signal_rows)

            fig = go.Figure()
            for label, curve in signal_curves.items():
                fig.add_trace(go.Scatter(x=curve.index, y=curve.values, mode="lines", name=label))
            fig.update_layout(
                title="Momentum Signal Variants: Static HMM with Momentum/Gold/BTC 60/30/10",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                height=550,
            )
            fig.show()
            signal_variant_summary
            """
        ),
        md_cell("## THB Overlay Summary"),
        code_cell(
            """
            overlay_results_thb = build_overlay_comparison(
                results,
                paths=paths,
                mix_weights=(0.60, 0.30, 0.10),
                report_currency="THB",
            )
            overlay_results_no_momentum_thb = build_overlay_comparison(
                results_no_momentum,
                paths=paths,
                mix_weights=(0.60, 0.30, 0.10),
                report_currency="THB",
            )

            overlay_summary_thb = pd.DataFrame(
                [
                    overlay_results_thb["summary"].loc["S&P 500 daily exposure"],
                    overlay_results_thb["summary"].loc["S&P/Gold/BTC 60/30/10 daily exposure"],
                    overlay_results_thb["summary"].loc["Static HMM daily exposure"],
                    overlay_results_no_momentum_thb["summary"].loc["Static HMM/Gold/BTC 60/30/10 daily exposure"],
                    overlay_results_thb["summary"].loc["Static HMM/Gold/BTC 60/30/10 daily exposure"],
                ],
                index=[
                    "S&P 500 daily exposure",
                    "S&P/Gold/BTC 60/30/10 daily exposure",
                    "Static HMM daily exposure",
                    "Static HMM/Gold/BTC 60/30/10 daily exposure",
                    "Static HMM with momentum/Gold/BTC 60/30/10 daily exposure",
                ],
            )
            overlay_summary_thb
            """
        ),
        md_cell("## THB Overlay Equity Curve Comparison"),
        code_cell(
            """
            overlay_curves_thb = pd.DataFrame(
                {
                    "S&P 500": pd.Series(overlay_results_thb["curves"]["S&P 500"]),
                    "S&P 500 daily exposure": pd.Series(overlay_results_thb["curves"]["S&P 500 daily exposure"]),
                    "S&P/Gold/BTC 60/30/10 daily exposure": pd.Series(overlay_results_thb["curves"]["S&P/Gold/BTC 60/30/10 daily exposure"]),
                    "Static HMM daily exposure": pd.Series(overlay_results_thb["curves"]["Static HMM daily exposure"]),
                    "Static HMM/Gold/BTC 60/30/10 daily exposure": pd.Series(overlay_results_no_momentum_thb["curves"]["Static HMM/Gold/BTC 60/30/10 daily exposure"]),
                    "Static HMM with momentum/Gold/BTC 60/30/10 daily exposure": pd.Series(overlay_results_thb["curves"]["Static HMM/Gold/BTC 60/30/10 daily exposure"]),
                }
            ).ffill()
            fig = go.Figure()
            for column in overlay_curves_thb.columns:
                fig.add_trace(go.Scatter(x=overlay_curves_thb.index, y=overlay_curves_thb[column], mode="lines", name=column))
            fig.update_layout(
                title="THB Equity Curve Comparison: S&P500 vs Overlay Variants incl. Static HMM + Gold/BTC",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                height=600,
            )
            fig.show()
            """
        ),
        md_cell(
            """
            ## Thailand SET100 PIT Equity Sleeve

            This section reads the latest Thailand point-in-time SET100 sleeve outputs. The Thai universe uses
            `universe_mode="set100_pit"`, the local SET100 membership interval file, the Thai extra cache, and
            `^SET.BK` as the benchmark.
            """
        ),
        code_cell(
            """
            thai_metrics_file = paths.result_dir / "thai_set100_pit_metrics.csv"
            thai_universe_file = paths.result_dir / "thai_set100_pit_universe_history.csv"
            thai_members_file = paths.result_dir / "latest_th_hmm_members.csv"
            thai_cache_file = paths.result_dir / "thai_set100_cache_status.csv"

            thai_metrics = pd.read_csv(thai_metrics_file, index_col=0)
            thai_cache = pd.read_csv(thai_cache_file)
            thai_latest_members = pd.read_csv(thai_members_file)
            thai_cache_col = "has_price" if "has_price" in thai_cache.columns else "in_cache"

            thai_cache_status = pd.DataFrame(
                [
                    {
                        "cached_set100_members": int(((thai_cache["kind"] == "set100_member") & thai_cache[thai_cache_col]).sum()),
                        "missing_set100_members": int(((thai_cache["kind"] == "set100_member") & ~thai_cache[thai_cache_col]).sum()),
                        "benchmark_cached": bool(thai_cache.loc[thai_cache["kind"] == "benchmark", thai_cache_col].any()),
                        "universe_history_rows": len(pd.read_csv(thai_universe_file)),
                    }
                ]
            )

            display(thai_metrics)
            display(thai_cache_status)
            display(thai_latest_members.head(30))
            """
        ),
        md_cell(
            """
            ## US + Thailand Blend

            This compares the US Static HMM sleeve, Thailand Static HMM sleeve, Gold, and BTC over the shared
            overlap period. The THB view is the main comparison because it includes currency translation.
            """
        ),
        code_cell(
            """
            blended_summary_thb = pd.read_csv(paths.result_dir / "us_th_gold_btc_blended_summary_thb.csv")
            blended_curves_thb = pd.read_csv(paths.result_dir / "us_th_gold_btc_blended_curves_thb.csv", index_col=0, parse_dates=True)

            fig = go.Figure()
            for column in blended_curves_thb.columns:
                fig.add_trace(go.Scatter(x=blended_curves_thb.index, y=blended_curves_thb[column], mode="lines", name=column))
            fig.update_layout(
                title="THB Equity Curve Comparison: US HMM, Thailand HMM, Gold, BTC",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                height=600,
            )
            fig.show()

            blended_summary_thb
            """
        ),
        md_cell(
            """
            ## US + Thailand Joint Model

            This reads the THB-base experiment where US and Thailand assets are converted to one currency before
            entering the copula model together. It includes two variants: a joint US+TH equity sleeve combined
            with Gold/BTC at 60/30/10, and a single all-asset model that optimizes US stocks, Thai stocks, Gold,
            and BTC together.
            """
        ),
        code_cell(
            """
            joint_model_summary_thb = pd.read_csv(paths.result_dir / "us_th_joint_model_summary_thb.csv")
            joint_model_curves_thb = pd.read_csv(paths.result_dir / "us_th_joint_model_curves_thb.csv", index_col=0, parse_dates=True)

            fig = go.Figure()
            for column in joint_model_curves_thb.columns:
                fig.add_trace(go.Scatter(x=joint_model_curves_thb.index, y=joint_model_curves_thb[column], mode="lines", name=column))
            fig.update_layout(
                title="THB Equity Curve Comparison: Joint US+TH Model Variants",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                height=600,
            )
            fig.show()

            joint_model_summary_thb
            """
        ),
        md_cell(
            """
            ## Interpretation

            สิ่งที่ notebook นี้พยายามพิสูจน์มี 3 จุด:

            1. **Cluster ไม่ได้ถูกตรึงไว้ตลอดเวลา**  
               เราจะเห็น migration ของ asset เมื่อ factor features เปลี่ยนไป

            2. **Dynamic assignment เปลี่ยน covariance forecast ได้จริง**  
               เพราะ factor loadings และ cluster factors ถูกคำนวณใหม่ทุก rebalance

            3. **Backtest output ช่วยตรวจสอบ economic value**  
               ถ้า Dynamic HMM Copula ให้ Sharpe / drawdown / benchmark-relative return ดีกว่า static baseline ก็ถือว่า concept ใช้งานได้ในระดับ PoC

            4. **Universe construction itself is point-in-time**  
               ไม่ใช่คัดจากรายชื่อหุ้นปัจจุบันอย่างเดียว แต่เริ่มจากสมาชิก S&P 500 จริงของวันนั้นก่อน แล้วค่อยจัดอันดับ liquidity

            ถ้าจะขยายต่อให้ใกล้ thesis มากขึ้น งานถัดไปควรเป็น:

            - เปลี่ยน Gaussian copula เป็น skewed-`t` copula
            - ทำ GAS update จาก score ของ log-likelihood โดยตรง
            - ทำ state-space estimation และ model selection ด้วย AIC / conditional log-likelihood
            """
        ),
    ]
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
    return nb


def hydrate_outputs(nb: nbf.NotebookNode) -> None:
    paths = default_paths(ROOT)
    base_feature_flags = {"resid_vol": False, "drawdown": False, "downside_beta": False}
    results = backtest_dynamic_factor_copula(
        start_date="2012-01-01",
        end_date="2026-04-30",
        n_assets=30,
        n_clusters=4,
        lookback_days=756,
        rebalance_freq="ME",
        max_weight=0.08,
        point_in_time_liquid=True,
        universe_mode="sp500_pit",
        feature_flags=base_feature_flags,
        paths=paths,
    )
    results_no_momentum = backtest_dynamic_factor_copula(
        start_date="2012-01-01",
        end_date="2026-04-30",
        n_assets=30,
        n_clusters=4,
        lookback_days=756,
        rebalance_freq="ME",
        max_weight=0.08,
        point_in_time_liquid=True,
        universe_mode="sp500_pit",
        include_momentum=False,
        feature_flags=base_feature_flags,
        paths=paths,
    )
    results_features_only = backtest_dynamic_factor_copula(
        start_date="2012-01-01",
        end_date="2026-04-30",
        n_assets=30,
        n_clusters=4,
        lookback_days=756,
        rebalance_freq="ME",
        max_weight=0.08,
        point_in_time_liquid=True,
        universe_mode="sp500_pit",
        include_momentum_features=True,
        include_momentum_signal=False,
        feature_flags=base_feature_flags,
        paths=paths,
    )
    results_signal_only = backtest_dynamic_factor_copula(
        start_date="2012-01-01",
        end_date="2026-04-30",
        n_assets=30,
        n_clusters=4,
        lookback_days=756,
        rebalance_freq="ME",
        max_weight=0.08,
        point_in_time_liquid=True,
        universe_mode="sp500_pit",
        include_momentum_features=False,
        include_momentum_signal=True,
        feature_flags=base_feature_flags,
        paths=paths,
    )

    metrics = results["metrics"].sort_values("Sharpe", ascending=False)
    nav = results["nav"]

    metrics_file = paths.result_dir / "multi_factor_copula_metrics.csv"
    metrics.to_csv(metrics_file)

    nb.cells[3]["outputs"] = [
        html_output(
            metrics.to_html(float_format=lambda x: f"{x:,.4f}"),
            metrics.to_string(),
        )
    ]
    nb.cells[3]["execution_count"] = 1

    fig, ax = plt.subplots(figsize=(12, 6))
    for name in ["Benchmark", "Equal Weight", "Risk Parity", "Static Copula", "Dynamic HMM Copula"]:
        nav[name].plot(ax=ax, label=name, linewidth=2)
    ax.set_title("Backtest NAV Comparison")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    fig.savefig(paths.result_dir / "multi_factor_copula_nav.png", dpi=150, bbox_inches="tight")
    nb.cells[5]["outputs"] = [png_output(fig)]
    nb.cells[5]["execution_count"] = 2

    display_cols = ["Total Return", "CAGR", "Annual Vol", "Sharpe", "Sortino", "Max Drawdown", "Benchmark Relative Return", "Turnover"]
    perf_html = metrics[display_cols].to_html(
        formatters={
            "Total Return": lambda x: f"{x:.2%}",
            "CAGR": lambda x: f"{x:.2%}",
            "Annual Vol": lambda x: f"{x:.2%}",
            "Sharpe": lambda x: f"{x:.2f}",
            "Sortino": lambda x: f"{x:.2f}",
            "Max Drawdown": lambda x: f"{x:.2%}",
            "Benchmark Relative Return": lambda x: f"{x:.2%}",
            "Turnover": lambda x: f"{x:.2f}",
        }
    )
    perf_text = metrics[display_cols].to_string(float_format=lambda x: f"{x:,.4f}")
    nb.cells[7]["outputs"] = [html_output(perf_html, perf_text)]
    nb.cells[7]["execution_count"] = 3

    assignments = pd.DataFrame(results["dynamic_state"]["assignment_history"]).T.sort_index()
    assignments.index.name = "rebalance_date"
    static_assignments = results["initial_clusters"]["labels"]
    changed_share = assignments.ne(static_assignments, axis=1).mean(axis=1)

    fig, ax = plt.subplots(figsize=(12, 4))
    changed_share.plot(ax=ax, color="#c2410c", linewidth=2)
    ax.set_title("Share of Assets Reassigned vs Initial Clusters")
    ax.set_ylabel("Share")
    ax.set_ylim(0, 1)
    fig.savefig(paths.result_dir / "multi_factor_copula_cluster_migration.png", dpi=150, bbox_inches="tight")
    nb.cells[9]["outputs"] = [
        png_output(fig),
        html_output(changed_share.describe().to_frame("changed_share").to_html(), changed_share.describe().to_string()),
    ]
    nb.cells[9]["execution_count"] = 4

    universe_history = results["universe_history"]
    summary_rows = []
    for date, members in universe_history.items():
        summary_rows.append(
            {
                "rebalance_date": pd.Timestamp(date),
                "selected_assets": len(members),
                "sample_members": ", ".join(members[:10]),
            }
        )
    universe_summary = pd.DataFrame(summary_rows).sort_values("rebalance_date")
    universe_summary.to_csv(paths.result_dir / "sp500_pit_universe_summary.csv", index=False)
    nb.cells[11]["outputs"] = [
        html_output(universe_summary.head(12).to_html(index=False), universe_summary.head(12).to_string(index=False))
    ]
    nb.cells[11]["execution_count"] = 5

    ordered_dates = sorted(universe_history)
    turnover_rows = []
    for prev_date, curr_date in zip(ordered_dates[:-1], ordered_dates[1:]):
        prev_set = set(universe_history[prev_date])
        curr_set = set(universe_history[curr_date])
        entered = curr_set - prev_set
        exited = prev_set - curr_set
        turnover_rows.append(
            {
                "rebalance_date": pd.Timestamp(curr_date),
                "entered_count": len(entered),
                "exited_count": len(exited),
                "membership_turnover": len(entered | exited) / max(len(curr_set), 1),
            }
        )
    turnover_df = pd.DataFrame(turnover_rows)
    turnover_df.to_csv(paths.result_dir / "sp500_pit_universe_turnover.csv", index=False)
    nb.cells[13]["outputs"] = [
        html_output(turnover_df.head(12).to_html(index=False), turnover_df.head(12).to_string(index=False))
    ]
    nb.cells[13]["execution_count"] = 6

    fig, ax = plt.subplots(figsize=(12, 4))
    turnover_df.set_index("rebalance_date")["membership_turnover"].plot(ax=ax, color="#0f766e", linewidth=2)
    ax.set_title("Point-in-Time Universe Membership Turnover")
    ax.set_ylabel("Turnover")
    ax.set_ylim(0, 1)
    fig.savefig(paths.result_dir / "sp500_pit_universe_turnover.png", dpi=150, bbox_inches="tight")
    nb.cells[15]["outputs"] = [png_output(fig)]
    nb.cells[15]["execution_count"] = 7

    latest_date = max(results["dynamic_state"]["posterior_history"])
    latest_posterior = results["dynamic_state"]["posterior_history"][latest_date].copy()
    latest_posterior["assigned_cluster"] = latest_posterior.idxmax(axis=1)
    latest_posterior = latest_posterior.sort_values("assigned_cluster")
    nb.cells[17]["outputs"] = [
        html_output(latest_posterior.to_html(float_format=lambda x: f"{x:,.4f}" if isinstance(x, float) else str(x)), latest_posterior.to_string())
    ]
    nb.cells[17]["execution_count"] = 8

    latest_weight_date = max(results["weights_history"]["Dynamic HMM Copula"])
    recent_weights = pd.DataFrame(
        {
            "dynamic": results["weights_history"]["Dynamic HMM Copula"][latest_weight_date],
            "static": results["weights_history"]["Static Copula"][latest_weight_date],
            "equal_weight": results["weights_history"]["Equal Weight"][latest_weight_date],
        }
    ).fillna(0.0).sort_values("dynamic", ascending=False)
    nb.cells[19]["outputs"] = [
        html_output(recent_weights.head(16).to_html(float_format=lambda x: f"{x:,.4f}"), recent_weights.head(16).to_string())
    ]
    nb.cells[19]["execution_count"] = 9

    from collections import Counter

    counts = Counter()
    for members in universe_history.values():
        counts.update(members)
    frequent_members = pd.DataFrame(
        [{"ticker": ticker, "selected_rebalances": count} for ticker, count in counts.most_common(30)]
    )
    frequent_members.to_csv(paths.result_dir / "sp500_pit_frequent_members.csv", index=False)
    nb.cells[21]["outputs"] = [
        html_output(frequent_members.to_html(index=False), frequent_members.to_string(index=False))
    ]
    nb.cells[21]["execution_count"] = 10

    static_weights = []
    for rebalance_date, weights in sorted(results["weights_history"]["Static Copula"].items()):
        static_weights.append(weights.rename(pd.Timestamp(rebalance_date)))
    static_weights_df = pd.DataFrame(static_weights).fillna(0.0).sort_index(axis=1)
    static_weights_df.to_csv(paths.result_dir / "static_copula_weight_history.csv")
    fig = go.Figure()
    for column in static_weights_df.columns:
        fig.add_trace(
            go.Bar(
                x=static_weights_df.index,
                y=static_weights_df[column],
                name=column,
            )
        )
    fig.update_layout(
        title="Static Copula Weight History",
        barmode="stack",
        xaxis_title="Rebalance Date",
        yaxis_title="Portfolio Weight",
        yaxis_range=[0, 1],
        height=600,
    )
    nb.cells[23]["outputs"] = [plotly_output(fig)]
    nb.cells[23]["execution_count"] = 11

    dynamic_weights = []
    for rebalance_date, weights in sorted(results["weights_history"]["Dynamic HMM Copula"].items()):
        dynamic_weights.append(weights.rename(pd.Timestamp(rebalance_date)))
    dynamic_weights_df = pd.DataFrame(dynamic_weights).fillna(0.0).sort_index(axis=1)
    dynamic_weights_df.to_csv(paths.result_dir / "dynamic_hmm_copula_weight_history.csv")
    fig = go.Figure()
    for column in dynamic_weights_df.columns:
        fig.add_trace(
            go.Bar(
                x=dynamic_weights_df.index,
                y=dynamic_weights_df[column],
                name=column,
            )
        )
    fig.update_layout(
        title="Dynamic HMM Copula Weight History",
        barmode="stack",
        xaxis_title="Rebalance Date",
        yaxis_title="Portfolio Weight",
        yaxis_range=[0, 1],
        height=600,
    )
    nb.cells[25]["outputs"] = [plotly_output(fig)]
    nb.cells[25]["execution_count"] = 12

    risk_parity_weights = []
    for rebalance_date, weights in sorted(results["weights_history"]["Risk Parity"].items()):
        risk_parity_weights.append(weights.rename(pd.Timestamp(rebalance_date)))
    risk_parity_weights_df = pd.DataFrame(risk_parity_weights).fillna(0.0).sort_index(axis=1)
    risk_parity_weights_df.to_csv(paths.result_dir / "risk_parity_weight_history.csv")
    fig = go.Figure()
    for column in risk_parity_weights_df.columns:
        fig.add_trace(
            go.Bar(
                x=risk_parity_weights_df.index,
                y=risk_parity_weights_df[column],
                name=column,
            )
        )
    fig.update_layout(
        title="Risk Parity Weight History",
        barmode="stack",
        xaxis_title="Rebalance Date",
        yaxis_title="Portfolio Weight",
        yaxis_range=[0, 1],
        height=600,
    )
    nb.cells[27]["outputs"] = [plotly_output(fig)]
    nb.cells[27]["execution_count"] = 13

    latest_weight_date = max(results["weights_history"]["Dynamic HMM Copula"])
    latest_weight_compare = pd.DataFrame(
        {
            "Equal Weight": results["weights_history"]["Equal Weight"][latest_weight_date],
            "Risk Parity": results["weights_history"]["Risk Parity"][latest_weight_date],
            "Static Copula": results["weights_history"]["Static Copula"][latest_weight_date],
            "Dynamic HMM Copula": results["weights_history"]["Dynamic HMM Copula"][latest_weight_date],
        }
    ).fillna(0.0)
    latest_weight_compare["max_weight"] = latest_weight_compare.max(axis=1)
    latest_weight_compare = latest_weight_compare.sort_values("max_weight", ascending=False).drop(columns="max_weight")
    latest_weight_compare.to_csv(paths.result_dir / "latest_weight_comparison.csv")
    fig = go.Figure()
    for strategy in latest_weight_compare.columns:
        fig.add_trace(
            go.Bar(
                x=latest_weight_compare.index,
                y=latest_weight_compare[strategy],
                name=strategy,
            )
        )
    fig.update_layout(
        title=f"Latest Weight Comparison Across Optimizers ({pd.Timestamp(latest_weight_date).date()})",
        barmode="group",
        xaxis_title="Asset",
        yaxis_title="Portfolio Weight",
        height=650,
    )
    nb.cells[29]["outputs"] = [
        plotly_output(fig),
        html_output(latest_weight_compare.head(20).to_html(float_format=lambda x: f"{x:,.4f}"), latest_weight_compare.head(20).to_string()),
    ]
    nb.cells[29]["execution_count"] = 14

    overlay_results = build_overlay_comparison(results, paths=paths, mix_weights=(0.60, 0.30, 0.10))
    overlay_results_no_momentum = build_overlay_comparison(results_no_momentum, paths=paths, mix_weights=(0.60, 0.30, 0.10))
    overlay_summary = pd.DataFrame(
        [
            overlay_results["summary"].loc["S&P 500 daily exposure"],
            overlay_results["summary"].loc["S&P/Gold/BTC 60/30/10 daily exposure"],
            overlay_results["summary"].loc["Static HMM daily exposure"],
            overlay_results_no_momentum["summary"].loc["Static HMM/Gold/BTC 60/30/10 daily exposure"],
            overlay_results["summary"].loc["Static HMM/Gold/BTC 60/30/10 daily exposure"],
        ],
        index=[
            "S&P 500 daily exposure",
            "S&P/Gold/BTC 60/30/10 daily exposure",
            "Static HMM daily exposure",
            "Static HMM/Gold/BTC 60/30/10 daily exposure",
            "Static HMM with momentum/Gold/BTC 60/30/10 daily exposure",
        ],
    )
    sp500_curve = pd.Series(overlay_results["curves"]["S&P 500"]).dropna()
    sp500_daily_curve = pd.Series(overlay_results["curves"]["S&P 500 daily exposure"]).dropna()
    static_daily_curve = pd.Series(overlay_results["curves"]["Static HMM daily exposure"]).dropna()
    static_raw_curve = results["nav"]["Static Copula"].reindex(static_daily_curve.index).ffill().dropna()
    static_raw_curve = static_raw_curve / static_raw_curve.iloc[0] * 10_000.0
    daily_exposure_impact = pd.DataFrame(
        [
            compute_port_opt_style_metrics(sp500_daily_curve, risk_free_rate=0.03)
            - compute_port_opt_style_metrics(sp500_curve, risk_free_rate=0.03),
            compute_port_opt_style_metrics(static_daily_curve, risk_free_rate=0.03)
            - compute_port_opt_style_metrics(static_raw_curve, risk_free_rate=0.03),
        ],
        index=[
            "S&P 500 daily exposure minus raw S&P 500",
            "Static HMM daily exposure minus raw Static HMM",
        ],
    )
    daily_exposure_impact["Sharpe Helped"] = daily_exposure_impact["Sharpe"].map(lambda x: "Yes" if x > 0 else "No")
    daily_exposure_impact["Drawdown Helped"] = daily_exposure_impact["Max Drawdown"].map(lambda x: "Yes" if x > 0 else "No")
    overlay_summary.to_csv(paths.result_dir / "overlay_comparison_summary.csv")
    daily_exposure_impact.to_csv(paths.result_dir / "daily_exposure_impact.csv")
    nb.cells[31]["outputs"] = [
        html_output(
            overlay_summary.to_html(
                float_format=lambda x: f"{x:,.4f}"
            ),
            overlay_summary.to_string(),
        ),
        html_output(
            daily_exposure_impact.to_html(float_format=lambda x: f"{x:,.4f}"),
            daily_exposure_impact.to_string(),
        ),
    ]
    nb.cells[31]["execution_count"] = 15

    overlay_curves = pd.DataFrame(
        {
            "S&P 500": pd.Series(overlay_results["curves"]["S&P 500"]),
            "S&P 500 daily exposure": pd.Series(overlay_results["curves"]["S&P 500 daily exposure"]),
            "S&P/Gold/BTC 60/30/10 daily exposure": pd.Series(overlay_results["curves"]["S&P/Gold/BTC 60/30/10 daily exposure"]),
            "Static HMM daily exposure": pd.Series(overlay_results["curves"]["Static HMM daily exposure"]),
            "Static HMM/Gold/BTC 60/30/10 daily exposure": pd.Series(
                overlay_results_no_momentum["curves"]["Static HMM/Gold/BTC 60/30/10 daily exposure"]
            ),
            "Static HMM with momentum/Gold/BTC 60/30/10 daily exposure": pd.Series(
                overlay_results["curves"]["Static HMM/Gold/BTC 60/30/10 daily exposure"]
            ),
        }
    ).ffill()
    overlay_curves = overlay_curves[
        [
            "S&P 500",
            "S&P 500 daily exposure",
            "S&P/Gold/BTC 60/30/10 daily exposure",
            "Static HMM daily exposure",
            "Static HMM/Gold/BTC 60/30/10 daily exposure",
            "Static HMM with momentum/Gold/BTC 60/30/10 daily exposure",
        ]
    ]
    overlay_curves.to_csv(paths.result_dir / "overlay_comparison_curves.csv")
    fig = go.Figure()
    for column in overlay_curves.columns:
        fig.add_trace(go.Scatter(x=overlay_curves.index, y=overlay_curves[column], mode="lines", name=column))
    fig.update_layout(
        title="Equity Curve Comparison: S&P500 vs Overlay Variants incl. Static HMM + Gold/BTC",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        height=600,
    )
    nb.cells[33]["outputs"] = [plotly_output(fig)]
    nb.cells[33]["execution_count"] = 16

    exposure_compare = overlay_results["exposure_compare"].copy()
    exposure_compare.to_csv(paths.result_dir / "overlay_exposure_history.csv")
    fig = go.Figure()
    for column in exposure_compare.columns:
        fig.add_trace(
            go.Scatter(
                x=exposure_compare.index,
                y=exposure_compare[column],
                mode="lines",
                name=column,
                line=dict(width=2.4),
            )
        )
    fig.update_layout(
        title="Daily Exposure History: S&P500 vs 60/30/10 vs Static HMM",
        xaxis_title="Date",
        yaxis_title="Exposure",
        yaxis_range=[0, 1.05],
        height=500,
    )
    nb.cells[35]["outputs"] = [plotly_output(fig)]
    nb.cells[35]["execution_count"] = 17

    overlay_price_file = paths.local_cache_root / "overlay_compare_prices.parquet"
    overlay_prices = pd.read_parquet(overlay_price_file).sort_index()
    overlay_prices.index = pd.to_datetime(overlay_prices.index)
    overlay_prices = overlay_prices.dropna(subset=["SPY", "GC=F", "BTC-USD", "^VIX", "USDTHB=X"]).loc[: "2026-04-29"]
    fx_returns = overlay_prices["USDTHB=X"].pct_change(fill_method=None).fillna(0.0)
    spy = overlay_prices["SPY"]
    gold = overlay_prices["GC=F"]
    btc = overlay_prices["BTC-USD"]
    vix = overlay_prices["^VIX"]
    gold_overlay = compare_apply_returns(
        gold.pct_change(fill_method=None).fillna(0.0),
        compare_trend_exposure(gold, 0.50),
        "USD_STATIC",
        fx_returns,
    )
    btc_overlay = compare_apply_returns(
        btc.pct_change(fill_method=None).fillna(0.0),
        compare_trend_exposure(btc, 0.00),
        "USD_STATIC",
        fx_returns,
    )
    static_hmm_overlay = results["nav"]["Static Copula"].pct_change(fill_method=None).fillna(0.0).reindex(spy.index).fillna(0.0)

    mix_rows = []
    mix_curves = {}
    for mix in [(1.00, 0.00, 0.00), (0.70, 0.20, 0.10), (0.65, 0.25, 0.10), (0.60, 0.30, 0.10)]:
        mix_name = f"{int(mix[0]*100)}/{int(mix[1]*100)}/{int(mix[2]*100)}"
        sleeve = pd.concat(
            {
                "STATIC_HMM_OVERLAY": static_hmm_overlay,
                "GOLD": gold_overlay,
                "BTC": btc_overlay,
            },
            axis=1,
        ).dropna()
        port_ret = compare_rebalanced_portfolio(
            sleeve,
            weights=pd.Series({"STATIC_HMM_OVERLAY": mix[0], "GOLD": mix[1], "BTC": mix[2]}, dtype=float),
        )
        curve = curve_from_returns(port_ret)
        metrics = compute_port_opt_style_metrics(curve, risk_free_rate=0.03)
        mix_rows.append({"Mix": mix_name, **metrics.to_dict()})
        mix_curves[mix_name] = curve
    mix_sweep = pd.DataFrame(mix_rows)
    mix_sweep.to_csv(paths.result_dir / "static_hmm_momentum_mix_sweep.csv", index=False)
    fig = go.Figure()
    for mix_name, curve in mix_curves.items():
        fig.add_trace(go.Scatter(x=curve.index, y=curve.values, mode="lines", name=mix_name))
    fig.update_layout(
        title="Static HMM with Momentum + Gold/BTC Mix Sweep",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        height=550,
    )
    nb.cells[37]["outputs"] = [
        plotly_output(fig),
        html_output(mix_sweep.to_html(index=False, float_format=lambda x: f"{x:,.4f}"), mix_sweep.to_string(index=False)),
    ]
    nb.cells[37]["execution_count"] = 18

    static_hmm_603010_rows = []
    for rebalance_date, weights in sorted(results["weights_history"]["Static Copula"].items()):
        row = (0.60 * weights).copy()
        row.loc["GOLD"] = 0.30
        row.loc["BTC"] = 0.10
        static_hmm_603010_rows.append(row.rename(pd.Timestamp(rebalance_date)))
    static_hmm_603010_df = pd.DataFrame(static_hmm_603010_rows).fillna(0.0).sort_index(axis=1)
    static_hmm_603010_df.to_csv(paths.result_dir / "static_hmm_603010_weight_history.csv")
    fig = go.Figure()
    for column in static_hmm_603010_df.columns:
        fig.add_trace(go.Bar(x=static_hmm_603010_df.index, y=static_hmm_603010_df[column], name=column))
    fig.update_layout(
        title="Static HMM with Momentum/Gold/BTC 60/30/10 Asset Weight History",
        barmode="stack",
        xaxis_title="Rebalance Date",
        yaxis_title="Portfolio Weight",
        yaxis_range=[0, 1],
        height=650,
    )
    nb.cells[39]["outputs"] = [plotly_output(fig)]
    nb.cells[39]["execution_count"] = 19

    attribution_rows = []
    attribution_cases = [
        ("No momentum", results_no_momentum, False, False),
        ("Features only", results_features_only, True, False),
        ("Signal only", results_signal_only, False, True),
        ("Features + Signal", results, True, True),
    ]
    for case_name, case_results, use_feat, use_sig in attribution_cases:
        case_overlay = build_overlay_comparison(case_results, paths=paths, mix_weights=(0.60, 0.30, 0.10))
        metrics = case_overlay["summary"].loc["Static HMM/Gold/BTC 60/30/10 daily exposure"]
        attribution_rows.append(
            {
                "Case": case_name,
                "Momentum Features": use_feat,
                "Momentum Signal": use_sig,
                **metrics.to_dict(),
            }
        )
    attribution_summary = pd.DataFrame(attribution_rows)
    attribution_summary.to_csv(paths.result_dir / "momentum_attribution_summary.csv", index=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=attribution_summary["Case"], y=attribution_summary["CAGR"], name="CAGR"))
    fig.add_trace(go.Bar(x=attribution_summary["Case"], y=attribution_summary["Sharpe"], name="Sharpe", yaxis="y2"))
    fig.update_layout(
        title="Momentum Attribution: Features vs Optimizer Signal",
        barmode="group",
        xaxis_title="Case",
        yaxis=dict(title="CAGR"),
        yaxis2=dict(title="Sharpe", overlaying="y", side="right"),
        height=550,
    )
    nb.cells[41]["outputs"] = [
        plotly_output(fig),
        html_output(attribution_summary.to_html(index=False, float_format=lambda x: f"{x:,.4f}"), attribution_summary.to_string(index=False)),
    ]
    nb.cells[41]["execution_count"] = 20

    lookback_rows = []
    lookback_curves = {}
    for lookback_days in [252, 504, 756]:
        lookback_results = backtest_dynamic_factor_copula(
            start_date="2012-01-01",
            end_date="2026-04-30",
            n_assets=30,
            n_clusters=4,
            lookback_days=lookback_days,
            rebalance_freq="ME",
            max_weight=0.08,
            point_in_time_liquid=True,
            universe_mode="sp500_pit",
            include_momentum=True,
            feature_flags=base_feature_flags,
            paths=paths,
        )
        lookback_overlay = build_overlay_comparison(lookback_results, paths=paths, mix_weights=(0.60, 0.30, 0.10))
        curve_name = "Static HMM/Gold/BTC 60/30/10 daily exposure"
        metrics = lookback_overlay["summary"].loc[curve_name]
        lookback_rows.append({"Lookback Days": lookback_days, **metrics.to_dict()})
        lookback_curves[f"{lookback_days} days"] = lookback_overlay["curves"][curve_name]
    lookback_sweep = pd.DataFrame(lookback_rows)
    lookback_sweep.to_csv(paths.result_dir / "static_hmm_603010_lookback_sweep.csv", index=False)
    fig = go.Figure()
    for label, curve in lookback_curves.items():
        fig.add_trace(go.Scatter(x=curve.index, y=curve.values, mode="lines", name=label))
    fig.update_layout(
        title="Lookback Sweep: Static HMM with Momentum/Gold/BTC 60/30/10",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        height=550,
    )
    nb.cells[43]["outputs"] = [
        plotly_output(fig),
        html_output(lookback_sweep.to_html(index=False, float_format=lambda x: f"{x:,.4f}"), lookback_sweep.to_string(index=False)),
    ]
    nb.cells[43]["execution_count"] = 21

    rebalance_rows = []
    rebalance_curves = {}
    for months in [1, 3, 6]:
        rebalance_overlay = build_overlay_comparison(
            results,
            paths=paths,
            mix_weights=(0.60, 0.30, 0.10),
            strategic_rebalance_months=months,
        )
        curve_name = "Static HMM/Gold/BTC 60/30/10 daily exposure"
        metrics = rebalance_overlay["summary"].loc[curve_name]
        rebalance_rows.append({"Strategic Rebalance (Months)": months, **metrics.to_dict()})
        rebalance_curves[f"{months} month"] = rebalance_overlay["curves"][curve_name]
    rebalance_sweep = pd.DataFrame(rebalance_rows)
    rebalance_sweep.to_csv(paths.result_dir / "static_hmm_603010_rebalance_sweep.csv", index=False)
    fig = go.Figure()
    for label, curve in rebalance_curves.items():
        fig.add_trace(go.Scatter(x=curve.index, y=curve.values, mode="lines", name=label))
    fig.update_layout(
        title="Strategic Rebalance Sweep: Static HMM with Momentum/Gold/BTC 60/30/10",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        height=550,
    )
    nb.cells[45]["outputs"] = [
        plotly_output(fig),
        html_output(rebalance_sweep.to_html(index=False, float_format=lambda x: f"{x:,.4f}"), rebalance_sweep.to_string(index=False)),
    ]
    nb.cells[45]["execution_count"] = 22

    factor_cases = [
        ("All features", {}),
        ("No resid_vol", {"resid_vol": False}),
        ("No drawdown", {"drawdown": False}),
        ("No downside_beta", {"downside_beta": False}),
    ]
    factor_rows = []
    factor_curves = {}
    for case_name, feature_flags in factor_cases:
        factor_results = backtest_dynamic_factor_copula(
            start_date="2012-01-01",
            end_date="2026-04-30",
            n_assets=30,
            n_clusters=4,
            lookback_days=756,
            rebalance_freq="ME",
            max_weight=0.08,
            point_in_time_liquid=True,
            universe_mode="sp500_pit",
            include_momentum=True,
            feature_flags=feature_flags,
            paths=paths,
        )
        factor_overlay = build_overlay_comparison(factor_results, paths=paths, mix_weights=(0.60, 0.30, 0.10))
        curve_name = "Static HMM/Gold/BTC 60/30/10 daily exposure"
        metrics = factor_overlay["summary"].loc[curve_name]
        factor_rows.append({"Case": case_name, **metrics.to_dict()})
        factor_curves[case_name] = factor_overlay["curves"][curve_name]
        del factor_results, factor_overlay
        gc.collect()
    factor_ablation = pd.DataFrame(factor_rows)
    factor_ablation.to_csv(paths.result_dir / "static_hmm_603010_factor_ablation.csv", index=False)
    fig = go.Figure()
    for label, curve in factor_curves.items():
        fig.add_trace(go.Scatter(x=curve.index, y=curve.values, mode="lines", name=label))
    fig.update_layout(
        title="Factor Ablation: Static HMM with Momentum/Gold/BTC 60/30/10",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        height=550,
    )
    nb.cells[47]["outputs"] = [
        plotly_output(fig),
        html_output(factor_ablation.to_html(index=False, float_format=lambda x: f"{x:,.4f}"), factor_ablation.to_string(index=False)),
    ]
    nb.cells[47]["execution_count"] = 23

    signal_rows = []
    signal_curves = {}
    signal_modes = ["mom_21", "mom_63", "blend_21_63", "zscore_63", "rank_63"]
    for signal_mode in signal_modes:
        signal_results = backtest_dynamic_factor_copula(
            start_date="2012-01-01",
            end_date="2026-04-30",
            n_assets=30,
            n_clusters=4,
            lookback_days=756,
            rebalance_freq="ME",
            max_weight=0.08,
            point_in_time_liquid=True,
            universe_mode="sp500_pit",
            include_momentum_features=True,
            include_momentum_signal=True,
            momentum_signal_mode=signal_mode,
            feature_flags=base_feature_flags,
            paths=paths,
        )
        signal_overlay = build_overlay_comparison(signal_results, paths=paths, mix_weights=(0.60, 0.30, 0.10))
        curve_name = "Static HMM/Gold/BTC 60/30/10 daily exposure"
        metrics = signal_overlay["summary"].loc[curve_name]
        signal_rows.append({"Signal Mode": signal_mode, **metrics.to_dict()})
        signal_curves[signal_mode] = signal_overlay["curves"][curve_name]
        del signal_results, signal_overlay
        gc.collect()
    signal_variant_summary = pd.DataFrame(signal_rows)
    signal_variant_summary.to_csv(paths.result_dir / "static_hmm_603010_momentum_signal_variants.csv", index=False)
    fig = go.Figure()
    for label, curve in signal_curves.items():
        fig.add_trace(go.Scatter(x=curve.index, y=curve.values, mode="lines", name=label))
    fig.update_layout(
        title="Momentum Signal Variants: Static HMM with Momentum/Gold/BTC 60/30/10",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        height=550,
    )
    nb.cells[49]["outputs"] = [
        plotly_output(fig),
        html_output(signal_variant_summary.to_html(index=False, float_format=lambda x: f"{x:,.4f}"), signal_variant_summary.to_string(index=False)),
    ]
    nb.cells[49]["execution_count"] = 24

    overlay_results_thb = build_overlay_comparison(
        results,
        paths=paths,
        mix_weights=(0.60, 0.30, 0.10),
        report_currency="THB",
    )
    overlay_results_no_momentum_thb = build_overlay_comparison(
        results_no_momentum,
        paths=paths,
        mix_weights=(0.60, 0.30, 0.10),
        report_currency="THB",
    )
    overlay_summary_thb = pd.DataFrame(
        [
            overlay_results_thb["summary"].loc["S&P 500 daily exposure"],
            overlay_results_thb["summary"].loc["S&P/Gold/BTC 60/30/10 daily exposure"],
            overlay_results_thb["summary"].loc["Static HMM daily exposure"],
            overlay_results_no_momentum_thb["summary"].loc["Static HMM/Gold/BTC 60/30/10 daily exposure"],
            overlay_results_thb["summary"].loc["Static HMM/Gold/BTC 60/30/10 daily exposure"],
        ],
        index=[
            "S&P 500 daily exposure",
            "S&P/Gold/BTC 60/30/10 daily exposure",
            "Static HMM daily exposure",
            "Static HMM/Gold/BTC 60/30/10 daily exposure",
            "Static HMM with momentum/Gold/BTC 60/30/10 daily exposure",
        ],
    )
    overlay_summary_thb.to_csv(paths.result_dir / "overlay_comparison_summary_thb.csv")
    nb.cells[51]["outputs"] = [
        html_output(
            overlay_summary_thb.to_html(float_format=lambda x: f"{x:,.4f}"),
            overlay_summary_thb.to_string(),
        )
    ]
    nb.cells[51]["execution_count"] = 25

    overlay_curves_thb = pd.DataFrame(
        {
            "S&P 500": pd.Series(overlay_results_thb["curves"]["S&P 500"]),
            "S&P 500 daily exposure": pd.Series(overlay_results_thb["curves"]["S&P 500 daily exposure"]),
            "S&P/Gold/BTC 60/30/10 daily exposure": pd.Series(overlay_results_thb["curves"]["S&P/Gold/BTC 60/30/10 daily exposure"]),
            "Static HMM daily exposure": pd.Series(overlay_results_thb["curves"]["Static HMM daily exposure"]),
            "Static HMM/Gold/BTC 60/30/10 daily exposure": pd.Series(
                overlay_results_no_momentum_thb["curves"]["Static HMM/Gold/BTC 60/30/10 daily exposure"]
            ),
            "Static HMM with momentum/Gold/BTC 60/30/10 daily exposure": pd.Series(
                overlay_results_thb["curves"]["Static HMM/Gold/BTC 60/30/10 daily exposure"]
            ),
        }
    ).ffill()
    overlay_curves_thb.to_csv(paths.result_dir / "overlay_comparison_curves_thb.csv")
    fig = go.Figure()
    for column in overlay_curves_thb.columns:
        fig.add_trace(go.Scatter(x=overlay_curves_thb.index, y=overlay_curves_thb[column], mode="lines", name=column))
    fig.update_layout(
        title="THB Equity Curve Comparison: S&P500 vs Overlay Variants incl. Static HMM + Gold/BTC",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        height=600,
    )
    nb.cells[53]["outputs"] = [plotly_output(fig)]
    nb.cells[53]["execution_count"] = 26

    thai_metrics_file = paths.result_dir / "thai_set100_pit_metrics.csv"
    thai_universe_file = paths.result_dir / "thai_set100_pit_universe_history.csv"
    thai_members_file = paths.result_dir / "latest_th_hmm_members.csv"
    thai_cache_file = paths.result_dir / "thai_set100_cache_status.csv"

    thai_metrics = pd.read_csv(thai_metrics_file, index_col=0)
    thai_cache = pd.read_csv(thai_cache_file)
    thai_latest_members = pd.read_csv(thai_members_file)
    thai_universe_history = pd.read_csv(thai_universe_file)
    thai_cache_col = "has_price" if "has_price" in thai_cache.columns else "in_cache"

    thai_cache_status = pd.DataFrame(
        [
            {
                "cached_set100_members": int(((thai_cache["kind"] == "set100_member") & thai_cache[thai_cache_col]).sum()),
                "missing_set100_members": int(((thai_cache["kind"] == "set100_member") & ~thai_cache[thai_cache_col]).sum()),
                "benchmark_cached": bool(thai_cache.loc[thai_cache["kind"] == "benchmark", thai_cache_col].any()),
                "universe_history_rows": len(thai_universe_history),
            }
        ]
    )
    thai_html = (
        "<h4>Thailand SET100 PIT metrics</h4>"
        + thai_metrics.to_html(float_format=lambda x: f"{x:,.4f}")
        + "<h4>Thai cache and universe status</h4>"
        + thai_cache_status.to_html(index=False)
        + "<h4>Latest Thailand HMM members</h4>"
        + thai_latest_members.head(30).to_html(index=False)
    )
    thai_text = "\n\n".join(
        [
            "Thailand SET100 PIT metrics",
            thai_metrics.to_string(),
            "Thai cache and universe status",
            thai_cache_status.to_string(index=False),
            "Latest Thailand HMM members",
            thai_latest_members.head(30).to_string(index=False),
        ]
    )
    nb.cells[55]["outputs"] = [html_output(thai_html, thai_text)]
    nb.cells[55]["execution_count"] = 27

    blended_summary_thb = pd.read_csv(paths.result_dir / "us_th_gold_btc_blended_summary_thb.csv")
    blended_curves_thb = pd.read_csv(paths.result_dir / "us_th_gold_btc_blended_curves_thb.csv", index_col=0, parse_dates=True)
    fig = go.Figure()
    for column in blended_curves_thb.columns:
        fig.add_trace(go.Scatter(x=blended_curves_thb.index, y=blended_curves_thb[column], mode="lines", name=column))
    fig.update_layout(
        title="THB Equity Curve Comparison: US HMM, Thailand HMM, Gold, BTC",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        height=600,
    )
    nb.cells[57]["outputs"] = [
        plotly_output(fig),
        html_output(
            blended_summary_thb.to_html(index=False, float_format=lambda x: f"{x:,.4f}"),
            blended_summary_thb.to_string(index=False),
        ),
    ]
    nb.cells[57]["execution_count"] = 28

    joint_model_summary_thb = pd.read_csv(paths.result_dir / "us_th_joint_model_summary_thb.csv")
    joint_model_curves_thb = pd.read_csv(paths.result_dir / "us_th_joint_model_curves_thb.csv", index_col=0, parse_dates=True)
    fig = go.Figure()
    for column in joint_model_curves_thb.columns:
        fig.add_trace(go.Scatter(x=joint_model_curves_thb.index, y=joint_model_curves_thb[column], mode="lines", name=column))
    fig.update_layout(
        title="THB Equity Curve Comparison: Joint US+TH Model Variants",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        height=600,
    )
    nb.cells[59]["outputs"] = [
        plotly_output(fig),
        html_output(
            joint_model_summary_thb.to_html(index=False, float_format=lambda x: f"{x:,.4f}"),
            joint_model_summary_thb.to_string(index=False),
        ),
    ]
    nb.cells[59]["execution_count"] = 29


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    hydrate_outputs(notebook)
    with NOTEBOOK_FILE.open("w", encoding="utf-8") as handle:
        nbf.write(notebook, handle)
    print(NOTEBOOK_FILE.name)


if __name__ == "__main__":
    main()
