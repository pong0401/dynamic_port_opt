from __future__ import annotations

from pathlib import Path
import sys
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebook"
NOTEBOOK_FILE = NOTEBOOK_DIR / "us_th_tactical_perf_momentum.ipynb"


def md_cell(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code_cell(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md_cell(
            """
            # US/TH Tactical Performance Momentum

            This notebook tests a simple question:

            If Thailand's regime or separately optimized PIT stock sleeve has better recent realized performance than the US side, should the portfolio tactically allocate part of the US core into Thailand?

            The workflow is intentionally separate from the one-model US+TH optimizer:

            - build US stock-only PIT sleeve
            - build Thailand stock-only PIT sleeve
            - compare monthly realized performance against S&P 500 THB and SET benchmark proxies on the same 2005+ window
            - test whether TH outperformance persists using both proxy-regime and sleeve-performance signals
            - sweep tactical entry, exit, and allocation/proportion rules for adding a TH sleeve

            Data note: SET100 PIT membership starts in 2005, so this notebook uses a common `2005-01-01` start. The local Thai cache contains `^SET.BK`, so charts use `SET Index THB proxy` rather than a direct SET100 index ticker.
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import sys
            import importlib

            import numpy as np
            import pandas as pd
            import plotly.graph_objects as go
            import plotly.express as px
            import plotly.io as pio

            ROOT = Path.cwd().resolve()
            while not (ROOT / "src" / "dynamic_factor_copula.py").exists() and ROOT.parent != ROOT:
                ROOT = ROOT.parent
            SRC = ROOT / "src"
            SCRIPTS = ROOT / "scripts"
            for path in [SRC, SCRIPTS]:
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))

            from dynamic_factor_copula import compute_port_opt_style_metrics, default_paths
            import run_us_th_tactical_perf_momentum as tactical
            tactical = importlib.reload(tactical)

            paths = default_paths(ROOT)
            pio.renderers.default = "notebook"

            RUN_BACKTESTS = False
            PREFIX = "us_th_tactical_perf_momentum"

            required_files = [
                paths.result_dir / f"{PREFIX}_comparison_curves_thb.csv",
                paths.result_dir / f"{PREFIX}_data_audit_thb.csv",
                paths.result_dir / f"{PREFIX}_monthly_returns_thb.csv",
                paths.result_dir / f"{PREFIX}_monthly_performance_table_thb.csv",
                paths.result_dir / f"{PREFIX}_persistence_thb.csv",
                paths.result_dir / f"{PREFIX}_tactical_exit_summary_thb.csv",
                paths.result_dir / f"{PREFIX}_tactical_exit_curves_thb.csv",
                paths.result_dir / f"{PREFIX}_tactical_exit_weight_history_thb.csv",
                paths.result_dir / f"{PREFIX}_gold_btc_overlay_summary_thb.csv",
                paths.result_dir / f"{PREFIX}_gold_btc_overlay_curves_thb.csv",
                paths.result_dir / f"{PREFIX}_gold_btc_overlay_weight_history_thb.csv",
                paths.result_dir / f"{PREFIX}_gold_btc_overlay_period_compare_thb.csv",
                paths.result_dir / f"{PREFIX}_final_best_latest_effective_security_weights_thb.csv",
                paths.result_dir / f"{PREFIX}_final_best_latest_effective_sleeve_weights_thb.csv",
                paths.result_dir / f"{PREFIX}_final_best_latest_meta.csv",
                paths.result_dir / f"{PREFIX}_final_model_optimizer_sweep_summary_thb.csv",
                paths.result_dir / f"{PREFIX}_final_model_optimizer_sweep_curves_thb.csv",
                paths.result_dir / f"{PREFIX}_final_model_optimizer_sweep_latest_weights_thb.csv",
                paths.result_dir / f"{PREFIX}_final_model_optimizer_sweep_best_thb.csv",
                paths.result_dir / f"{PREFIX}_gold_btc_overlay_pit_aligned_metrics_thb.csv",
                paths.result_dir / f"{PREFIX}_gold_daily_exposure_sweep_thb.csv",
                paths.result_dir / f"{PREFIX}_gold_daily_exposure_sweep_curves_thb.csv",
                paths.result_dir / f"{PREFIX}_gold_weight_sweep_thb.csv",
                paths.result_dir / f"{PREFIX}_gold_weight_sweep_curves_thb.csv",
                paths.result_dir / f"{PREFIX}_gold_crash_protection_sweep_thb.csv",
                paths.result_dir / f"{PREFIX}_gold_crash_protection_sweep_curves_thb.csv",
                paths.result_dir / f"{PREFIX}_gold_crash_protection_sweep_weight_history_thb.csv",
                paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_summary_thb.csv",
                paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_curves_thb.csv",
                paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_latest_weights_thb.csv",
                paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_period_compare_thb.csv",
                paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_effective_weights_thb.csv",
                paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_group_cap_sweep_summary_thb.csv",
                paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_group_cap_sweep_curves_thb.csv",
                paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_group_cap_sweep_latest_weights_thb.csv",
                paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_group_cap_sweep_period_compare_thb.csv",
                paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_group_cap_sweep_grouped_weight_history_thb.csv",
                paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_asym_group_cap_sweep_summary_thb.csv",
                paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_asym_group_cap_sweep_curves_thb.csv",
                paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_asym_group_cap_sweep_latest_weights_thb.csv",
                paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_asym_group_cap_sweep_period_compare_thb.csv",
                paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_asym_group_cap_sweep_grouped_weight_history_thb.csv",
                paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_asym_group_cap_grid_us70_80_th30_40_summary_thb.csv",
                paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_asym_group_cap_grid_us70_80_th30_40_curves_thb.csv",
                paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_asym_group_cap_grid_us70_80_th30_40_latest_weights_thb.csv",
                paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_asym_group_cap_grid_us70_80_th30_40_period_compare_thb.csv",
                paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_asym_group_cap_grid_us70_80_th30_40_grouped_weight_history_thb.csv",
            ]
            if RUN_BACKTESTS or any(not file.exists() for file in required_files):
                tactical.main()

            comparison_curves = pd.read_csv(paths.result_dir / f"{PREFIX}_comparison_curves_thb.csv", index_col=0, parse_dates=True)
            data_audit = pd.read_csv(paths.result_dir / f"{PREFIX}_data_audit_thb.csv")
            monthly_returns = pd.read_csv(paths.result_dir / f"{PREFIX}_monthly_returns_thb.csv", index_col=0, parse_dates=True)
            monthly_table = pd.read_csv(paths.result_dir / f"{PREFIX}_monthly_performance_table_thb.csv")
            persistence = pd.read_csv(paths.result_dir / f"{PREFIX}_persistence_thb.csv")
            tactical_summary = pd.read_csv(paths.result_dir / f"{PREFIX}_tactical_exit_summary_thb.csv")
            tactical_curves = pd.read_csv(paths.result_dir / f"{PREFIX}_tactical_exit_curves_thb.csv", index_col=0, parse_dates=True)
            tactical_weights = pd.read_csv(paths.result_dir / f"{PREFIX}_tactical_exit_weight_history_thb.csv", index_col=0, parse_dates=True)
            latest_internal_weights = pd.read_csv(paths.result_dir / f"{PREFIX}_latest_internal_weights.csv")
            overlay_summary = pd.read_csv(paths.result_dir / f"{PREFIX}_gold_btc_overlay_summary_thb.csv")
            overlay_curves = pd.read_csv(paths.result_dir / f"{PREFIX}_gold_btc_overlay_curves_thb.csv", index_col=0, parse_dates=True)
            overlay_weights = pd.read_csv(paths.result_dir / f"{PREFIX}_gold_btc_overlay_weight_history_thb.csv", parse_dates=["Date"])
            overlay_period_compare = pd.read_csv(paths.result_dir / f"{PREFIX}_gold_btc_overlay_period_compare_thb.csv")
            final_best_latest_security = pd.read_csv(paths.result_dir / f"{PREFIX}_final_best_latest_effective_security_weights_thb.csv")
            final_best_latest_sleeve = pd.read_csv(paths.result_dir / f"{PREFIX}_final_best_latest_effective_sleeve_weights_thb.csv")
            final_best_latest_meta = pd.read_csv(paths.result_dir / f"{PREFIX}_final_best_latest_meta.csv")
            final_model_optimizer_summary_file = paths.result_dir / f"{PREFIX}_final_model_optimizer_sweep_summary_thb.csv"
            final_model_optimizer_curves_file = paths.result_dir / f"{PREFIX}_final_model_optimizer_sweep_curves_thb.csv"
            final_model_optimizer_latest_file = paths.result_dir / f"{PREFIX}_final_model_optimizer_sweep_latest_weights_thb.csv"
            final_model_optimizer_best_file = paths.result_dir / f"{PREFIX}_final_model_optimizer_sweep_best_thb.csv"
            final_model_optimizer_summary = pd.read_csv(final_model_optimizer_summary_file) if final_model_optimizer_summary_file.exists() else pd.DataFrame()
            final_model_optimizer_curves = (
                pd.read_csv(final_model_optimizer_curves_file, index_col=0, parse_dates=True)
                if final_model_optimizer_curves_file.exists()
                else pd.DataFrame()
            )
            final_model_optimizer_latest = pd.read_csv(final_model_optimizer_latest_file) if final_model_optimizer_latest_file.exists() else pd.DataFrame()
            final_model_optimizer_best = pd.read_csv(final_model_optimizer_best_file) if final_model_optimizer_best_file.exists() else pd.DataFrame()
            overlay_pit_aligned_metrics = pd.read_csv(paths.result_dir / f"{PREFIX}_gold_btc_overlay_pit_aligned_metrics_thb.csv")
            gold_exposure_sweep = pd.read_csv(paths.result_dir / f"{PREFIX}_gold_daily_exposure_sweep_thb.csv")
            gold_exposure_curves = pd.read_csv(paths.result_dir / f"{PREFIX}_gold_daily_exposure_sweep_curves_thb.csv", index_col=0, parse_dates=True)
            gold_weight_sweep = pd.read_csv(paths.result_dir / f"{PREFIX}_gold_weight_sweep_thb.csv")
            gold_weight_curves = pd.read_csv(paths.result_dir / f"{PREFIX}_gold_weight_sweep_curves_thb.csv", index_col=0, parse_dates=True)
            gold_crash_sweep = pd.read_csv(paths.result_dir / f"{PREFIX}_gold_crash_protection_sweep_thb.csv")
            gold_crash_curves = pd.read_csv(paths.result_dir / f"{PREFIX}_gold_crash_protection_sweep_curves_thb.csv", index_col=0, parse_dates=True)
            gold_crash_weights = pd.read_csv(paths.result_dir / f"{PREFIX}_gold_crash_protection_sweep_weight_history_thb.csv", parse_dates=["Date"])
            one_model_summary = pd.read_csv(paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_summary_thb.csv")
            one_model_curves = pd.read_csv(paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_curves_thb.csv", index_col=0, parse_dates=True)
            one_model_latest = pd.read_csv(paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_latest_weights_thb.csv")
            one_model_period_compare = pd.read_csv(paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_period_compare_thb.csv")
            one_model_effective_weights = pd.read_csv(paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_effective_weights_thb.csv", index_col=0, parse_dates=True)
            one_model_group_cap_summary = pd.read_csv(paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_group_cap_sweep_summary_thb.csv")
            one_model_group_cap_curves = pd.read_csv(paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_group_cap_sweep_curves_thb.csv", index_col=0, parse_dates=True)
            one_model_group_cap_latest = pd.read_csv(paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_group_cap_sweep_latest_weights_thb.csv")
            one_model_group_cap_period_compare = pd.read_csv(paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_group_cap_sweep_period_compare_thb.csv")
            one_model_group_cap_grouped_weights = pd.read_csv(paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_group_cap_sweep_grouped_weight_history_thb.csv", parse_dates=["Date"])
            one_model_asym_group_cap_summary = pd.read_csv(paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_asym_group_cap_sweep_summary_thb.csv")
            one_model_asym_group_cap_curves = pd.read_csv(paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_asym_group_cap_sweep_curves_thb.csv", index_col=0, parse_dates=True)
            one_model_asym_group_cap_latest = pd.read_csv(paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_asym_group_cap_sweep_latest_weights_thb.csv")
            one_model_asym_group_cap_period_compare = pd.read_csv(paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_asym_group_cap_sweep_period_compare_thb.csv")
            one_model_asym_group_cap_grouped_weights = pd.read_csv(paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_asym_group_cap_sweep_grouped_weight_history_thb.csv", parse_dates=["Date"])
            one_model_asym_grid_summary = pd.read_csv(paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_asym_group_cap_grid_us70_80_th30_40_summary_thb.csv")
            one_model_asym_grid_curves = pd.read_csv(paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_asym_group_cap_grid_us70_80_th30_40_curves_thb.csv", index_col=0, parse_dates=True)
            one_model_asym_grid_latest = pd.read_csv(paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_asym_group_cap_grid_us70_80_th30_40_latest_weights_thb.csv")
            one_model_asym_grid_period_compare = pd.read_csv(paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_asym_group_cap_grid_us70_80_th30_40_period_compare_thb.csv")
            one_model_asym_grid_grouped_weights = pd.read_csv(paths.result_dir / f"{PREFIX}_one_model_gold30_btc10_th_signal_asym_group_cap_grid_us70_80_th30_40_grouped_weight_history_thb.csv", parse_dates=["Date"])

            def pct_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
                out = frame.copy()
                for column in columns:
                    if column in out:
                        out[column] = out[column].map(lambda value: f"{value:.2%}" if pd.notna(value) else "")
                return out

            def short_label(value: str, max_len: int = 54) -> str:
                text = str(value)
                replacements = {
                    "Tactical TH proxy_regime relative_return binary lb1 cap30 entry0% exit0% hold0 confirm1": "TH tactical best",
                    "Tactical TH/Gold/BTC 60/30/10": "Tactical 60/30/10",
                    "US/Gold/BTC 60/30/10": "US 60/30/10",
                    "asset-level daily exposure": "daily exposure",
                    "no daily exposure": "no exposure gate",
                    "Gold25 crash": "Gold25",
                    "recover-5%": "rec-5%",
                    "no panic": "no panic",
                    "panic-30%/MA200/mom63->0": "panic->0",
                }
                for old, new in replacements.items():
                    text = text.replace(old, new)
                if len(text) <= max_len:
                    return text
                return text[: max_len - 1].rstrip() + "…"

            def add_line(fig: go.Figure, x, y, name: str, **kwargs) -> None:
                full_name = str(name)
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        name=short_label(full_name),
                        customdata=np.repeat(full_name, len(y)),
                        hovertemplate="%{customdata}<br>%{x}<br>%{y:,.2f}<extra></extra>",
                        **kwargs,
                    )
                )

            def add_bar(fig: go.Figure, x, y, name: str, **kwargs) -> None:
                full_name = str(name)
                fig.add_trace(
                    go.Bar(
                        x=x,
                        y=y,
                        name=short_label(full_name),
                        customdata=np.repeat(full_name, len(y)),
                        hovertemplate="%{customdata}<br>%{x}<br>%{y:.2%}<extra></extra>",
                        **kwargs,
                    )
                )

            def balance_legend(fig: go.Figure, *, height: int = 620, bottom: int = 150) -> None:
                fig.update_layout(
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.22,
                        xanchor="left",
                        x=0,
                        traceorder="normal",
                        font=dict(size=11),
                    ),
                    margin=dict(b=bottom),
                    height=height,
                )
            """
        ),
        md_cell(
            """
            ## Step 0 - Common Window Audit

            This audit keeps proxy-regime signals and PIT sleeve implementation on the same timeline.
            """
        ),
        code_cell(
            """
            display(data_audit)
            """
        ),
        md_cell(
            """
            ## Step 1 - Monthly Performance: US vs Thailand

            This section compares the separately optimized sleeves and the benchmark proxies on the same THB timeline.
            """
        ),
        code_cell(
            """
            display_cols = [
                "US PIT optimized sleeve THB",
                "TH PIT optimized sleeve THB",
                "S&P 500 ETF THB",
                "SET Index THB proxy",
            ]

            fig = go.Figure()
            for column in display_cols:
                add_line(fig, comparison_curves.index, comparison_curves[column], column)
            fig.update_layout(
                title="US PIT vs TH PIT vs S&P 500 / SET Proxy",
                yaxis_title="Portfolio Value (THB, rebased to 10,000)",
                template="plotly_white",
                hovermode="x unified",
            )
            balance_legend(fig, height=560, bottom=120)
            fig.show()

            recent_monthly = monthly_returns.tail(18).copy()
            display(pct_frame(recent_monthly.reset_index().rename(columns={"index": "Month"}), display_cols))
            """
        ),
        md_cell(
            """
            ### Latest 12-Month Return

            This table compares the latest 12 monthly returns and the compounded 12-month return for the best tactical strategy against the standalone sleeves and benchmark proxies.
            """
        ),
        code_cell(
            """
            best_strategy = tactical_summary.sort_values(["Sharpe", "CAGR"], ascending=False).iloc[0]["Strategy"]
            tactical_monthly = tactical_curves.resample("ME").last().pct_change(fill_method=None).sort_index()
            latest_12m_returns = (
                pd.concat(
                    [
                        monthly_returns[display_cols],
                        tactical_monthly[[best_strategy]],
                    ],
                    axis=1,
                )
                .sort_index()
                .dropna(how="all")
                .tail(12)
            )
            latest_12m_compound = ((1.0 + latest_12m_returns).prod() - 1.0).rename("Compound Return").reset_index()
            latest_12m_compound.columns = ["Series", "Compound Return"]

            display(pct_frame(latest_12m_returns.reset_index().rename(columns={"index": "Month"}), latest_12m_returns.columns.tolist()))
            display(pct_frame(latest_12m_compound, ["Compound Return"]))
            """
        ),
        md_cell(
            """
            ### US Only vs Best Tactical by Period

            This view compares full-period, 10-year, 5-year, 3-year, and 1-year metrics for the US-only sleeve against the best tactical configuration in this notebook.
            """
        ),
        code_cell(
            """
            from dynamic_factor_copula import compute_port_opt_style_metrics

            best_strategy = tactical_summary.sort_values(["Sharpe", "CAGR"], ascending=False).iloc[0]["Strategy"]
            period_curves = tactical_curves[["US PIT optimized sleeve THB", best_strategy]].dropna(how="all").sort_index()
            end_date = period_curves.index.max()
            period_specs = [
                ("Full period", None),
                ("10Y", 10),
                ("5Y", 5),
                ("3Y", 3),
                ("1Y", 1),
            ]
            label_map = {
                "US PIT optimized sleeve THB": "US only",
                best_strategy: "Best tactical",
            }
            period_rows = []
            for period_name, years in period_specs:
                start_date = period_curves.index.min() if years is None else end_date - pd.DateOffset(years=years)
                for column, label in label_map.items():
                    nav = period_curves[column].dropna()
                    sample = nav.loc[nav.index >= start_date]
                    metrics = compute_port_opt_style_metrics(sample, risk_free_rate=tactical.RISK_FREE_RATE).to_dict()
                    period_rows.append(
                        {
                            "Period": period_name,
                            "Strategy": label,
                            "Start": sample.index.min().date().isoformat(),
                            "End": sample.index.max().date().isoformat(),
                            "Obs": int(sample.shape[0]),
                            **metrics,
                        }
                    )
            period_compare = pd.DataFrame(period_rows)
            period_compare["Sharpe Delta vs US"] = (
                period_compare.pivot(index="Period", columns="Strategy", values="Sharpe")
                .pipe(lambda frame: frame["Best tactical"] - frame["US only"])
                .reindex(period_compare["Period"])
                .to_numpy()
            )
            display(
                pct_frame(
                    period_compare[
                        [
                            "Period",
                            "Strategy",
                            "Start",
                            "End",
                            "CAGR",
                            "Annual Vol",
                            "Sharpe",
                            "Sharpe Delta vs US",
                            "Max Drawdown",
                            "Total Return",
                        ]
                    ],
                    ["CAGR", "Annual Vol", "Max Drawdown", "Total Return"],
                )
            )
            """
        ),
        code_cell(
            """
            long_monthly = monthly_returns.reset_index(names="Month").melt(id_vars="Month", var_name="Series", value_name="Monthly Return")
            fig = px.bar(
                long_monthly.tail(24 * len(display_cols)),
                x="Month",
                y="Monthly Return",
                color="Series",
                barmode="group",
                title="Recent Monthly Returns",
            )
            fig.update_layout(template="plotly_white", yaxis_tickformat=".0%", hovermode="x unified")
            balance_legend(fig, height=560, bottom=120)
            fig.show()

            rolling_cols = [
                "Month",
                "Series",
                "Monthly Return",
                "Rolling 3M Return",
                "Rolling 6M Return",
                "Rolling 3M Sharpe",
                "Rolling 6M Sharpe",
            ]
            display(monthly_table[rolling_cols].tail(4 * 12))
            """
        ),
        code_cell(
            """
            display(
                latest_internal_weights.loc[latest_internal_weights["Internal Weight"] > 0]
                .sort_values(["Sleeve", "Internal Weight"], ascending=[True, False])
                .head(80)
            )
            """
        ),
        md_cell(
            """
            ## Step 2 - Does TH Momentum Persist?

            The signal is measured at month end and shifted one month before judging forward returns. This avoids using the same month to both detect and benefit from the signal.
            """
        ),
        code_cell(
            """
            persistence_view = persistence.sort_values(["Forward Months", "Average Forward TH-US Return"], ascending=[True, False])
            display(
                pct_frame(
                    persistence_view.head(30),
                    ["Hit Rate", "Average Forward TH-US Return", "Median Forward TH-US Return", "All Months Average TH-US Return"],
                )
            )

            fig = px.scatter(
                persistence,
                x="Lookback Months",
                y="Average Forward TH-US Return",
                color="Signal",
                symbol="Signal Source",
                facet_col="Forward Months",
                size="Signal Count",
                title="Forward TH-US Return After TH Momentum Signals",
            )
            fig.update_layout(template="plotly_white", yaxis_tickformat=".0%")
            balance_legend(fig, height=560, bottom=120)
            fig.show()
            """
        ),
        md_cell(
            """
            ## Step 3 - Tactical TH Entry/Exit Strategy

            The tactical portfolio starts from the US PIT optimized sleeve as core. Thailand weight is added only when the rolling signal is active.

            Exit rules swept:

            - entry mode: relative return, relative Sharpe, return and Sharpe, return or Sharpe, relative return with positive TH, positive TH return, score 3-of-5
            - allocation method: binary, return-spread scaled, Sharpe-spread scaled, inverse-vol, return-spread + inverse-vol, score-scaled
            - lookback: 1, 2, 3, 6, 12 months
            - TH cap: 5%, 10%, 15%, 20%, 25%, 30%
            - entry threshold: 0%, 1%, 2%
            - exit threshold: -2%, -1%, 0%
            - minimum hold: 0, 1, 2, 3 months
            - exit confirmation: 1 or 2 months
            """
        ),
        code_cell(
            """
            summary_cols = [
                "Strategy",
                "Signal Source",
                "Signal Mode",
                "Allocation Method",
                "Lookback Months",
                "TH Weight Cap",
                "Entry Threshold",
                "Exit Threshold",
                "Min Hold Months",
                "Exit Confirm Months",
                "CAGR",
                "Annual Vol",
                "Sharpe",
                "Max Drawdown",
                "Average TH Weight",
                "TH On Months",
            ]
            best_tactical = tactical_summary.sort_values(["Sharpe", "CAGR"], ascending=False).head(25)
            display(
                pct_frame(
                    best_tactical[summary_cols],
                    ["TH Weight Cap", "Entry Threshold", "Exit Threshold", "CAGR", "Annual Vol", "Max Drawdown", "Average TH Weight"],
                )
            )
            """
        ),
        code_cell(
            """
            fig = go.Figure()
            for column in tactical_curves.columns[:12]:
                add_line(fig, tactical_curves.index, tactical_curves[column], column)
            fig.update_layout(
                title="Best Tactical Exit Strategies vs US Core",
                yaxis_title="Portfolio Value (THB)",
                template="plotly_white",
                hovermode="x unified",
            )
            balance_legend(fig, height=640, bottom=170)
            fig.show()

            if not tactical_weights.empty:
                best_strategy = tactical_summary.sort_values(["Sharpe", "CAGR"], ascending=False).iloc[0]["Strategy"]
                fig = go.Figure()
                add_line(fig, tactical_weights.index, tactical_weights[best_strategy], "TH Weight")
                fig.update_layout(
                    title=f"Best Tactical TH Weight: {short_label(best_strategy, 80)}",
                    yaxis_title="TH Weight",
                    template="plotly_white",
                    yaxis_tickformat=".0%",
                    hovermode="x unified",
                )
                balance_legend(fig, height=520, bottom=100)
                fig.show()
            """
        ),
        md_cell(
            """
            ### Best Tactical Sleeve Weight History

            The stacked bar chart follows the PIT reselect notebook style and shows the month-end allocation between the US core sleeve and the tactical Thailand sleeve.
            """
        ),
        code_cell(
            """
            if not tactical_weights.empty:
                best_strategy = tactical_summary.sort_values(["Sharpe", "CAGR"], ascending=False).iloc[0]["Strategy"]
                th_weight_monthly = tactical_weights[best_strategy].resample("ME").last().dropna().clip(0.0, 1.0)
                sleeve_weight_history = pd.DataFrame(
                    {
                        "US Core Sleeve": 1.0 - th_weight_monthly,
                        "TH Tactical Sleeve": th_weight_monthly,
                    }
                )

                fig = go.Figure()
                for column in sleeve_weight_history.columns:
                    add_bar(fig, sleeve_weight_history.index, sleeve_weight_history[column], column)
                fig.update_layout(
                    title=f"Best Tactical Sleeve Weight History: {short_label(best_strategy, 80)}",
                    barmode="stack",
                    xaxis_title="Month End",
                    yaxis_title="Portfolio Weight",
                    yaxis_range=[0, 1],
                    yaxis_tickformat=".0%",
                    template="plotly_white",
                    height=520,
                    hovermode="x unified",
                )
                balance_legend(fig, height=560, bottom=120)
                fig.show()

                display(pct_frame(sleeve_weight_history.tail(24).reset_index().rename(columns={"index": "Month"}), sleeve_weight_history.columns.tolist()))
            """
        ),
        code_cell(
            """
            fixed = tactical_summary[tactical_summary["Signal Mode"].eq("fixed")].copy()
            best_by_mode = (
                tactical_summary.sort_values(["Sharpe", "CAGR"], ascending=False)
                .groupby(["Signal Source", "Signal Mode", "Allocation Method"], as_index=False)
                .head(1)
                .sort_values("Sharpe", ascending=False)
            )
            compare = pd.concat([best_by_mode, fixed], ignore_index=True).drop_duplicates("Strategy")
            display(
                pct_frame(
                    compare[summary_cols].sort_values("Sharpe", ascending=False),
                    ["TH Weight Cap", "Entry Threshold", "Exit Threshold", "CAGR", "Annual Vol", "Max Drawdown", "Average TH Weight"],
                )
            )
            """
        ),
        md_cell(
            """
            ## Step 4 - Best Entry Mode and Allocation Method

            These views summarize the sweep by separating the entry decision from the sizing decision.
            """
        ),
        code_cell(
            """
            tactical_only = tactical_summary[
                ~tactical_summary["Signal Mode"].isin(["baseline", "fixed"])
            ].copy()

            best_entry_mode = (
                tactical_only.sort_values(["Sharpe", "CAGR"], ascending=False)
                .groupby(["Signal Source", "Signal Mode"], as_index=False)
                .head(1)
                .sort_values("Sharpe", ascending=False)
            )
            display(
                pct_frame(
                    best_entry_mode[summary_cols].head(30),
                    ["TH Weight Cap", "Entry Threshold", "Exit Threshold", "CAGR", "Annual Vol", "Max Drawdown", "Average TH Weight"],
                )
            )

            best_alloc_method = (
                tactical_only.sort_values(["Sharpe", "CAGR"], ascending=False)
                .groupby(["Signal Source", "Allocation Method"], as_index=False)
                .head(1)
                .sort_values("Sharpe", ascending=False)
            )
            display(
                pct_frame(
                    best_alloc_method[summary_cols].head(30),
                    ["TH Weight Cap", "Entry Threshold", "Exit Threshold", "CAGR", "Annual Vol", "Max Drawdown", "Average TH Weight"],
                )
            )

            heat = (
                tactical_only.groupby(["Signal Mode", "Allocation Method"])["Sharpe"]
                .max()
                .reset_index()
                .pivot(index="Signal Mode", columns="Allocation Method", values="Sharpe")
            )
            fig = px.imshow(
                heat,
                text_auto=".2f",
                aspect="auto",
                title="Best Sharpe by Entry Mode and Allocation Method",
                color_continuous_scale="Viridis",
            )
            fig.update_layout(template="plotly_white")
            fig.show()
            """
        ),
        md_cell(
            """
            ## Step 5 - Tactical TH + Gold/BTC Overlay

            This confirm comparison adds the repo baseline overlay mix to the US/TH tactical equity sleeve:

            - Equity sleeve: 60%
            - Gold: 30%
            - BTC: 10%
            - Tactical TH uses the best notebook rule, with TH weight applied inside the 60% equity sleeve
            - Asset-level daily exposure variant: US uses SPY MA300 below 50%, TH uses SET MA200 below 0%, Gold uses MA50 below 100%, BTC uses MA50 below 0%
            """
        ),
        code_cell(
            """
            overlay_cols = [
                "Strategy",
                "Start",
                "End",
                "CAGR",
                "Annual Vol",
                "Sharpe",
                "Max Drawdown",
                "Average US Equity Weight",
                "Average TH Equity Weight",
                "Average Gold Weight",
                "Average BTC Weight",
                "Average Cash / Reduced Exposure Weight",
            ]
            display(
                pct_frame(
                    overlay_summary[overlay_cols].sort_values("Sharpe", ascending=False),
                    [
                        "CAGR",
                        "Annual Vol",
                        "Max Drawdown",
                        "Average US Equity Weight",
                        "Average TH Equity Weight",
                        "Average Gold Weight",
                        "Average BTC Weight",
                        "Average Cash / Reduced Exposure Weight",
                    ],
                )
            )

            fig = go.Figure()
            for column in overlay_curves.columns:
                add_line(fig, overlay_curves.index, overlay_curves[column], column)
            fig.update_layout(
                title="US/TH Tactical + Gold/BTC Overlay Curves",
                yaxis_title="Portfolio Value (THB)",
                template="plotly_white",
                hovermode="x unified",
            )
            balance_legend(fig, height=640, bottom=170)
            fig.show()
            """
        ),
        code_cell(
            """
            period_cols = [
                "Period",
                "Strategy",
                "Start",
                "End",
                "CAGR",
                "Annual Vol",
                "Sharpe",
                "Max Drawdown",
                "Total Return",
            ]
            period_order = ["Full period", "10Y", "5Y", "3Y", "1Y"]
            period_view = overlay_period_compare.copy()
            period_view["Period"] = pd.Categorical(period_view["Period"], categories=period_order, ordered=True)
            period_view = period_view.sort_values(["Period", "Sharpe"], ascending=[True, False])
            display(
                pct_frame(
                    period_view[period_cols],
                    ["CAGR", "Annual Vol", "Max Drawdown", "Total Return"],
                )
            )

            best_overlay_strategy = overlay_summary.sort_values(["Sharpe", "CAGR"], ascending=False).iloc[0]["Strategy"]
            best_overlay_weights = (
                overlay_weights.loc[overlay_weights["Strategy"].eq(best_overlay_strategy)]
                .set_index("Date")
                .sort_index()
                [["US Equity", "TH Equity", "Gold", "BTC", "Cash / Reduced Exposure"]]
                .resample("ME")
                .last()
                .dropna(how="all")
            )
            fig = go.Figure()
            for column in best_overlay_weights.columns:
                add_bar(fig, best_overlay_weights.index, best_overlay_weights[column], column)
            fig.update_layout(
                title=f"Best Overlay Effective Weight History: {short_label(best_overlay_strategy, 80)}",
                barmode="stack",
                xaxis_title="Month End",
                yaxis_title="Portfolio Weight",
                yaxis_range=[0, 1],
                yaxis_tickformat=".0%",
                template="plotly_white",
                height=560,
                hovermode="x unified",
            )
            balance_legend(fig, height=600, bottom=130)
            fig.show()

            display(pct_frame(best_overlay_weights.tail(24).reset_index().rename(columns={"Date": "Month"}), best_overlay_weights.columns.tolist()))
            """
        ),
        md_cell(
            """
            ### Gold Daily Exposure Sweep

            This test keeps the final tactical US/TH + Gold/BTC framework but varies only Gold's daily exposure rule. US, TH, and BTC use the selected daily-exposure rules from Step 5.
            """
        ),
        code_cell(
            """
            gold_cols = [
                "Strategy",
                "Gold MA Period",
                "Gold Below Exposure",
                "CAGR",
                "Annual Vol",
                "Sharpe",
                "Max Drawdown",
                "Average Gold Weight",
                "Average Cash / Reduced Exposure Weight",
            ]
            display(
                pct_frame(
                    gold_exposure_sweep[gold_cols].head(25),
                    [
                        "Gold Below Exposure",
                        "CAGR",
                        "Annual Vol",
                        "Max Drawdown",
                        "Average Gold Weight",
                        "Average Cash / Reduced Exposure Weight",
                    ],
                )
            )

            heat = gold_exposure_sweep.pivot_table(
                index="Gold MA Period",
                columns="Gold Below Exposure",
                values="Sharpe",
                aggfunc="max",
            )
            fig = px.imshow(
                heat,
                text_auto=".2f",
                aspect="auto",
                title="Gold Daily Exposure Sweep: Sharpe",
                color_continuous_scale="Viridis",
            )
            fig.update_layout(template="plotly_white")
            fig.show()

            fig = go.Figure()
            for column in gold_exposure_curves.columns[:8]:
                add_line(fig, gold_exposure_curves.index, gold_exposure_curves[column], column)
            fig.update_layout(
                title="Top Gold Daily Exposure Sweep Curves",
                yaxis_title="Portfolio Value (THB)",
                template="plotly_white",
                hovermode="x unified",
            )
            balance_legend(fig, height=640, bottom=170)
            fig.show()
            """
        ),
        md_cell(
            """
            ### Gold Weight Sweep

            This sweep keeps BTC at 10% and shifts the remaining capital between equity and Gold:

            - Gold 20% / Equity 70% / BTC 10%
            - Gold 25% / Equity 65% / BTC 10%
            - Gold 30% / Equity 60% / BTC 10%
            """
        ),
        code_cell(
            """
            gold_weight_cols = [
                "Strategy",
                "Equity Weight",
                "Gold Weight",
                "BTC Weight",
                "Daily Exposure Mode",
                "CAGR",
                "Annual Vol",
                "Sharpe",
                "Max Drawdown",
                "Average Gold Weight",
                "Average Cash / Reduced Exposure Weight",
            ]
            display(
                pct_frame(
                    gold_weight_sweep[gold_weight_cols].sort_values(["Sharpe", "CAGR"], ascending=False),
                    [
                        "Equity Weight",
                        "Gold Weight",
                        "BTC Weight",
                        "CAGR",
                        "Annual Vol",
                        "Max Drawdown",
                        "Average Gold Weight",
                        "Average Cash / Reduced Exposure Weight",
                    ],
                )
            )

            fig = px.bar(
                gold_weight_sweep,
                x="Gold Weight",
                y="Sharpe",
                color="Daily Exposure Mode",
                barmode="group",
                title="Gold Weight Sweep: Sharpe",
            )
            fig.update_layout(template="plotly_white", xaxis_tickformat=".0%")
            balance_legend(fig, height=560, bottom=120)
            fig.show()

            fig = go.Figure()
            for column in gold_weight_curves.columns:
                add_line(fig, gold_weight_curves.index, gold_weight_curves[column], column)
            fig.update_layout(
                title="Gold Weight Sweep Curves",
                yaxis_title="Portfolio Value (THB)",
                template="plotly_white",
                hovermode="x unified",
            )
            balance_legend(fig, height=640, bottom=170)
            fig.show()
            """
        ),
        md_cell(
            """
            ### Gold Crash Protection Sweep

            This sweep uses the selected Gold-weight mix from the sweep above, `Equity/Gold/BTC 65/25/10`, and tests a stateful Gold drawdown rule:

            - measure Gold drawdown from a rolling high
            - cut Gold to a warning exposure after the warning threshold
            - cut Gold further after the crash threshold
            - restore Gold only after drawdown recovers enough

            The signal is shifted by one trading day to avoid look-ahead.
            """
        ),
        code_cell(
            """
            gold_crash_cols = [
                "Strategy",
                "Equity Weight",
                "Gold Weight",
                "BTC Weight",
                "DD Window",
                "Warn Drawdown",
                "Crash Drawdown",
                "Warn Exposure",
                "Crash Exposure",
                "Recovery Drawdown",
                "Panic Drawdown",
                "Panic MA Period",
                "Panic Momentum Period",
                "CAGR",
                "Annual Vol",
                "Sharpe",
                "Max Drawdown",
                "Average Gold Weight",
                "Average Cash / Reduced Exposure Weight",
                "Gold Reduced Days",
                "Gold Zero Days",
            ]
            display(
                pct_frame(
                    gold_crash_sweep[gold_crash_cols].head(25),
                    [
                        "Equity Weight",
                        "Gold Weight",
                        "BTC Weight",
                        "Warn Drawdown",
                        "Crash Drawdown",
                        "Warn Exposure",
                        "Crash Exposure",
                        "Recovery Drawdown",
                        "Panic Drawdown",
                        "CAGR",
                        "Annual Vol",
                        "Max Drawdown",
                        "Average Gold Weight",
                        "Average Cash / Reduced Exposure Weight",
                    ],
                )
            )

            fig = px.scatter(
                gold_crash_sweep.head(80),
                x="Average Gold Weight",
                y="Sharpe",
                color="Max Drawdown",
                size="Gold Reduced Days",
                hover_name="Strategy",
                hover_data=[
                    "CAGR",
                    "Annual Vol",
                    "DD Window",
                    "Warn Drawdown",
                    "Crash Drawdown",
                    "Recovery Drawdown",
                    "Panic Drawdown",
                    "Gold Zero Days",
                ],
                title="Gold Crash Protection Sweep: Sharpe vs Average Gold Weight",
                color_continuous_scale="RdYlGn",
            )
            fig.update_layout(template="plotly_white", xaxis_tickformat=".0%")
            fig.show()

            fig = go.Figure()
            for column in gold_crash_curves.columns:
                add_line(fig, gold_crash_curves.index, gold_crash_curves[column], column)
            fig.update_layout(
                title="Top Gold Crash Protection Curves",
                yaxis_title="Portfolio Value (THB)",
                template="plotly_white",
                hovermode="x unified",
            )
            balance_legend(fig, height=660, bottom=190)
            fig.show()

            best_gold_crash_strategy = gold_crash_sweep.sort_values(["Sharpe", "CAGR"], ascending=False).iloc[0]["Strategy"]
            if best_gold_crash_strategy not in set(gold_crash_weights["Strategy"]):
                best_gold_crash_strategy = gold_crash_weights["Strategy"].drop_duplicates().iloc[0]
            best_gold_crash_weights = (
                gold_crash_weights
                .loc[gold_crash_weights["Strategy"].eq(best_gold_crash_strategy)]
                .set_index("Date")
                [["US Equity", "TH Equity", "Gold", "BTC", "Cash / Reduced Exposure"]]
                .resample("ME")
                .last()
                .dropna(how="all")
            )
            fig = go.Figure()
            for column in best_gold_crash_weights.columns:
                add_bar(fig, best_gold_crash_weights.index, best_gold_crash_weights[column], column)
            fig.update_layout(
                title=f"Gold Crash Protection Effective Weight History: {short_label(best_gold_crash_strategy, 80)}",
                barmode="stack",
                xaxis_title="Month End",
                yaxis_title="Portfolio Weight",
                yaxis_range=[0, 1],
                yaxis_tickformat=".0%",
                template="plotly_white",
                hovermode="x unified",
            )
            balance_legend(fig, height=600, bottom=130)
            fig.show()

            display(
                pct_frame(
                    best_gold_crash_weights.tail(24).reset_index().rename(columns={"Date": "Month"}),
                    best_gold_crash_weights.columns.tolist(),
                )
            )
            """
        ),
        md_cell(
            """
            ## One-Model Optimizer Test: Stocks + Gold/BTC With TH Signal

            This test replaces fixed overlay allocation with a single optimizer. At each monthly rebalance:

            - US PIT top 30 stocks are always eligible
            - TH PIT top 30 stocks are eligible only when the best tactical TH signal is on
            - Gold and BTC are always eligible
            - caps are stock `8%`, Gold `30%`, BTC `10%`

            The second row applies the selected asset-level daily exposure gates after the optimizer weights are set.
            """
        ),
        code_cell(
            """
            one_model_cols = [
                "Strategy",
                "Start",
                "End",
                "CAGR",
                "Annual Vol",
                "Sharpe",
                "Max Drawdown",
                "Average US Stock Weight",
                "Average TH Stock Weight",
                "Average Gold Weight",
                "Average BTC Weight",
                "Average Cash / Reduced Exposure Weight",
                "Stock Cap",
                "Gold Cap",
                "BTC Cap",
            ]
            display(
                pct_frame(
                    one_model_summary[one_model_cols].sort_values(["Sharpe", "CAGR"], ascending=False),
                    [
                        "CAGR",
                        "Annual Vol",
                        "Max Drawdown",
                        "Average US Stock Weight",
                        "Average TH Stock Weight",
                        "Average Gold Weight",
                        "Average BTC Weight",
                        "Average Cash / Reduced Exposure Weight",
                        "Stock Cap",
                        "Gold Cap",
                        "BTC Cap",
                    ],
                )
            )

            fig = go.Figure()
            for column in one_model_curves.columns:
                add_line(fig, one_model_curves.index, one_model_curves[column], column)
            fig.update_layout(
                title="One-Model Optimizer Curves",
                yaxis_title="Portfolio Value (THB)",
                template="plotly_white",
                hovermode="x unified",
            )
            balance_legend(fig, height=600, bottom=130)
            fig.show()

            period_order = ["Full period", "10Y", "5Y", "3Y", "1Y"]
            one_model_period_view = one_model_period_compare.copy()
            one_model_period_view["Period"] = pd.Categorical(one_model_period_view["Period"], categories=period_order, ordered=True)
            one_model_period_view = one_model_period_view.sort_values(["Period", "Sharpe"], ascending=[True, False])
            display(
                pct_frame(
                    one_model_period_view[
                        ["Period", "Strategy", "Start", "End", "CAGR", "Annual Vol", "Sharpe", "Max Drawdown", "Total Return"]
                    ],
                    ["CAGR", "Annual Vol", "Max Drawdown", "Total Return"],
                )
            )

            one_model_monthly_weights = one_model_effective_weights.resample("ME").last().fillna(0.0)
            one_model_grouped_weights = pd.DataFrame(
                {
                    "US Stocks": one_model_monthly_weights[
                        [
                            column
                            for column in one_model_monthly_weights.columns
                            if column not in ["GC=F", "BTC-USD", "Cash / Reduced Exposure"] and not column.endswith(".BK")
                        ]
                    ].sum(axis=1),
                    "TH Stocks": one_model_monthly_weights[
                        [column for column in one_model_monthly_weights.columns if column.endswith(".BK")]
                    ].sum(axis=1),
                    "Gold": one_model_monthly_weights["GC=F"] if "GC=F" in one_model_monthly_weights else 0.0,
                    "BTC": one_model_monthly_weights["BTC-USD"] if "BTC-USD" in one_model_monthly_weights else 0.0,
                    "Cash / Reduced Exposure": (
                        one_model_monthly_weights["Cash / Reduced Exposure"]
                        if "Cash / Reduced Exposure" in one_model_monthly_weights
                        else 0.0
                    ),
                }
            ).clip(lower=0.0)

            fig = go.Figure()
            for column in one_model_grouped_weights.columns:
                add_bar(fig, one_model_grouped_weights.index, one_model_grouped_weights[column], column)
            fig.update_layout(
                title="One-Model Effective Weight History",
                barmode="stack",
                xaxis_title="Month End",
                yaxis_title="Portfolio Weight",
                yaxis_range=[0, 1],
                yaxis_tickformat=".0%",
                template="plotly_white",
                hovermode="x unified",
            )
            balance_legend(fig, height=600, bottom=130)
            fig.show()

            display(
                pct_frame(
                    one_model_grouped_weights.tail(24).reset_index().rename(columns={"index": "Month"}),
                    one_model_grouped_weights.columns.tolist(),
                )
            )

            display(
                pct_frame(
                    one_model_latest.head(50),
                    ["Effective Weight"],
                )
            )
            """
        ),
        md_cell(
            """
            ### One-Model US/TH Group Cap Sweep

            This sweep adds portfolio-level group caps on total US stock weight and total TH stock weight. Tested caps are `40%`, `50%`, and `60%`; the no-group-cap case is kept for comparison.

            When TH signal is off, low US group caps can leave unavoidable cash because US + Gold + BTC caps may not sum to 100%.
            """
        ),
        code_cell(
            """
            group_cap_cols = [
                "Strategy",
                "Case",
                "Mode",
                "Group Cap",
                "CAGR",
                "Annual Vol",
                "Sharpe",
                "Max Drawdown",
                "Average US Stock Weight",
                "Average TH Stock Weight",
                "Average Gold Weight",
                "Average BTC Weight",
                "Average Cash / Reduced Exposure Weight",
            ]
            display(
                pct_frame(
                    one_model_group_cap_summary[group_cap_cols].sort_values(["Sharpe", "CAGR"], ascending=False),
                    [
                        "Group Cap",
                        "CAGR",
                        "Annual Vol",
                        "Max Drawdown",
                        "Average US Stock Weight",
                        "Average TH Stock Weight",
                        "Average Gold Weight",
                        "Average BTC Weight",
                        "Average Cash / Reduced Exposure Weight",
                    ],
                )
            )

            fig = go.Figure()
            for column in one_model_group_cap_curves.columns:
                add_line(fig, one_model_group_cap_curves.index, one_model_group_cap_curves[column], column)
            fig.update_layout(
                title="One-Model Group Cap Sweep Curves",
                yaxis_title="Portfolio Value (THB)",
                template="plotly_white",
                hovermode="x unified",
            )
            balance_legend(fig, height=660, bottom=190)
            fig.show()

            best_group_case = one_model_group_cap_summary.sort_values(["Sharpe", "CAGR"], ascending=False).iloc[0]["Case"]
            best_group_weights = (
                one_model_group_cap_grouped_weights
                .loc[one_model_group_cap_grouped_weights["Case"].eq(best_group_case)]
                .set_index("Date")
                [["US Stocks", "TH Stocks", "Gold", "BTC", "Cash / Reduced Exposure"]]
                .sort_index()
            )
            fig = go.Figure()
            for column in best_group_weights.columns:
                add_bar(fig, best_group_weights.index, best_group_weights[column], column)
            fig.update_layout(
                title=f"One-Model Group Cap Effective Weight History: {best_group_case}",
                barmode="stack",
                xaxis_title="Month End",
                yaxis_title="Portfolio Weight",
                yaxis_range=[0, 1],
                yaxis_tickformat=".0%",
                template="plotly_white",
                hovermode="x unified",
            )
            balance_legend(fig, height=600, bottom=130)
            fig.show()

            display(
                pct_frame(
                    best_group_weights.tail(24).reset_index().rename(columns={"Date": "Month"}),
                    best_group_weights.columns.tolist(),
                )
            )

            display(
                pct_frame(
                    one_model_group_cap_latest.loc[one_model_group_cap_latest["Case"].eq(best_group_case)].head(50),
                    ["Effective Weight"],
                )
            )
            """
        ),
        md_cell(
            """
            ### One-Model Asymmetric US/TH Group Cap Sweep

            This sweep keeps TH stock weight capped at `50%`, then tests US stock group caps of `70%`, `80%`, and `90%`.
            """
        ),
        code_cell(
            """
            asym_cols = [
                "Strategy",
                "Case",
                "Mode",
                "US Group Cap",
                "TH Group Cap",
                "CAGR",
                "Annual Vol",
                "Sharpe",
                "Max Drawdown",
                "Average US Stock Weight",
                "Average TH Stock Weight",
                "Average Gold Weight",
                "Average BTC Weight",
                "Average Cash / Reduced Exposure Weight",
            ]
            display(
                pct_frame(
                    one_model_asym_group_cap_summary[asym_cols].sort_values(["Sharpe", "CAGR"], ascending=False),
                    [
                        "US Group Cap",
                        "TH Group Cap",
                        "CAGR",
                        "Annual Vol",
                        "Max Drawdown",
                        "Average US Stock Weight",
                        "Average TH Stock Weight",
                        "Average Gold Weight",
                        "Average BTC Weight",
                        "Average Cash / Reduced Exposure Weight",
                    ],
                )
            )

            fig = go.Figure()
            for column in one_model_asym_group_cap_curves.columns:
                add_line(fig, one_model_asym_group_cap_curves.index, one_model_asym_group_cap_curves[column], column)
            fig.update_layout(
                title="One-Model Asymmetric Group Cap Sweep Curves",
                yaxis_title="Portfolio Value (THB)",
                template="plotly_white",
                hovermode="x unified",
            )
            balance_legend(fig, height=640, bottom=170)
            fig.show()

            best_asym_case = one_model_asym_group_cap_summary.sort_values(["Sharpe", "CAGR"], ascending=False).iloc[0]["Case"]
            best_asym_weights = (
                one_model_asym_group_cap_grouped_weights
                .loc[one_model_asym_group_cap_grouped_weights["Case"].eq(best_asym_case)]
                .set_index("Date")
                [["US Stocks", "TH Stocks", "Gold", "BTC", "Cash / Reduced Exposure"]]
                .sort_index()
            )
            fig = go.Figure()
            for column in best_asym_weights.columns:
                add_bar(fig, best_asym_weights.index, best_asym_weights[column], column)
            fig.update_layout(
                title=f"One-Model Asymmetric Group Cap Effective Weight History: {best_asym_case}",
                barmode="stack",
                xaxis_title="Month End",
                yaxis_title="Portfolio Weight",
                yaxis_range=[0, 1],
                yaxis_tickformat=".0%",
                template="plotly_white",
                hovermode="x unified",
            )
            balance_legend(fig, height=600, bottom=130)
            fig.show()

            display(
                pct_frame(
                    best_asym_weights.tail(24).reset_index().rename(columns={"Date": "Month"}),
                    best_asym_weights.columns.tolist(),
                )
            )

            display(
                pct_frame(
                    one_model_asym_group_cap_latest.loc[one_model_asym_group_cap_latest["Case"].eq(best_asym_case)].head(50),
                    ["Effective Weight"],
                )
            )
            """
        ),
        md_cell(
            """
            ### One-Model Asymmetric Grid: US 70/80, TH 30/40

            This narrower grid tests US stock group caps of `70%` and `80%`, with TH stock group caps of `30%` and `40%`.
            """
        ),
        code_cell(
            """
            display(
                pct_frame(
                    one_model_asym_grid_summary[asym_cols].sort_values(["Sharpe", "CAGR"], ascending=False),
                    [
                        "US Group Cap",
                        "TH Group Cap",
                        "CAGR",
                        "Annual Vol",
                        "Max Drawdown",
                        "Average US Stock Weight",
                        "Average TH Stock Weight",
                        "Average Gold Weight",
                        "Average BTC Weight",
                        "Average Cash / Reduced Exposure Weight",
                    ],
                )
            )

            fig = go.Figure()
            for column in one_model_asym_grid_curves.columns:
                add_line(fig, one_model_asym_grid_curves.index, one_model_asym_grid_curves[column], column)
            fig.update_layout(
                title="One-Model Asymmetric Grid Curves: US 70/80, TH 30/40",
                yaxis_title="Portfolio Value (THB)",
                template="plotly_white",
                hovermode="x unified",
            )
            balance_legend(fig, height=640, bottom=170)
            fig.show()

            best_asym_grid_case = one_model_asym_grid_summary.sort_values(["Sharpe", "CAGR"], ascending=False).iloc[0]["Case"]
            best_asym_grid_weights = (
                one_model_asym_grid_grouped_weights
                .loc[one_model_asym_grid_grouped_weights["Case"].eq(best_asym_grid_case)]
                .set_index("Date")
                [["US Stocks", "TH Stocks", "Gold", "BTC", "Cash / Reduced Exposure"]]
                .sort_index()
            )
            fig = go.Figure()
            for column in best_asym_grid_weights.columns:
                add_bar(fig, best_asym_grid_weights.index, best_asym_grid_weights[column], column)
            fig.update_layout(
                title=f"One-Model Asymmetric Grid Effective Weight History: {best_asym_grid_case}",
                barmode="stack",
                xaxis_title="Month End",
                yaxis_title="Portfolio Weight",
                yaxis_range=[0, 1],
                yaxis_tickformat=".0%",
                template="plotly_white",
                hovermode="x unified",
            )
            balance_legend(fig, height=600, bottom=130)
            fig.show()

            display(
                pct_frame(
                    best_asym_grid_weights.tail(24).reset_index().rename(columns={"Date": "Month"}),
                    best_asym_grid_weights.columns.tolist(),
                )
            )

            display(
                pct_frame(
                    one_model_asym_grid_latest.loc[one_model_asym_grid_latest["Case"].eq(best_asym_grid_case)].head(50),
                    ["Effective Weight"],
                )
            )
            """
        ),
        md_cell(
            """
            ## Final Model and Optimizer Sweep

            This test compares the final US/TH/Gold/BTC candidate as one optimizer across model families and optimizer objectives. It includes the multi-factor copula models:

            - Equal Weight
            - Risk Parity
            - Static Copula
            - Dynamic HMM Copula

            Optimizer objectives tested: mean-variance, minimum-volatility momentum tilt, max-Sharpe momentum, and risk-parity momentum tilt.
            """
        ),
        code_cell(
            """
            final_model_cols = [
                "Strategy",
                "Model Family",
                "Objective",
                "Start",
                "End",
                "CAGR",
                "Annual Vol",
                "Sharpe",
                "Max Drawdown",
                "Total Return",
                "US Cap",
                "TH Cap",
                "Gold Cap",
                "BTC Cap",
            ]
            if final_model_optimizer_summary.empty:
                print(
                    "Final model/optimizer sweep output is missing. "
                    "Run scripts/run_us_th_tactical_final_model_optimizer_sweep.py to generate it."
                )
            else:
                display(
                    pct_frame(
                        final_model_optimizer_summary[final_model_cols].sort_values(["Sharpe", "CAGR"], ascending=False),
                        ["CAGR", "Annual Vol", "Max Drawdown", "Total Return", "US Cap", "TH Cap", "Gold Cap", "BTC Cap"],
                    )
                )

                best_final_model = final_model_optimizer_summary.sort_values(["Sharpe", "CAGR"], ascending=False).iloc[0]["Strategy"]
                if best_final_model in final_model_optimizer_curves:
                    compare_cols = [best_final_model]
                    selected_gold_crash = "Tactical TH/Gold/BTC 65/25/10 Gold crash protection"
                    if selected_gold_crash in gold_crash_curves:
                        compare_cols.append(selected_gold_crash)
                    fig = go.Figure()
                    for column in compare_cols:
                        source = final_model_optimizer_curves if column in final_model_optimizer_curves else gold_crash_curves
                        add_line(fig, source.index, source[column], short_label(column, 80))
                    fig.update_layout(
                        title="Final Strategy vs Model/Optimizer Sweep Leader",
                        template="plotly_white",
                        yaxis_title="Portfolio Value",
                        hovermode="x unified",
                    )
                    balance_legend(fig, height=560, bottom=130)
                    fig.show()

                if not final_model_optimizer_latest.empty and not final_model_optimizer_best.empty:
                    best_strategy = str(final_model_optimizer_best.iloc[0]["Strategy"])
                    display(
                        pct_frame(
                            final_model_optimizer_latest
                            .loc[final_model_optimizer_latest["Strategy"].eq(best_strategy)]
                            .sort_values("Portfolio Weight", ascending=False)
                            .head(60),
                            ["Portfolio Weight"],
                        )
                    )
            """
        ),
        md_cell(
            """
            ## Final Selected Best Sharpe Strategy

            Selected final strategy for this notebook:

            `Tactical TH/Gold/BTC 65/25/10 Gold crash protection`

            The latest weight tables below come from `scripts/recheck_us_th_tactical_gold_btc_latest_weights.py`, which refreshes the latest common close and reruns the US and TH PIT sleeves before applying the selected tactical and daily-exposure rules.
            """
        ),
        code_cell(
            """
            final_summary_cols = [
                "Strategy",
                "Start",
                "End",
                "Equity Weight",
                "Gold Weight",
                "BTC Weight",
                "Panic Drawdown",
                "CAGR",
                "Annual Vol",
                "Sharpe",
                "Max Drawdown",
                "Average Gold Weight",
                "Average Cash / Reduced Exposure Weight",
                "Gold Reduced Days",
                "Gold Zero Days",
            ]
            final_selected = gold_crash_sweep.sort_values(["Sharpe", "CAGR"], ascending=False).head(1)
            final_strategy = final_selected.iloc[0]["Strategy"]
            aligned_curve = gold_crash_curves[final_strategy].loc["2018-01-02":"2026-04-29"].dropna()
            aligned_metric = compute_port_opt_style_metrics(aligned_curve, risk_free_rate=tactical.RISK_FREE_RATE).to_frame().T
            aligned_metric.insert(0, "Strategy", final_strategy)
            aligned_metric.insert(1, "Start", aligned_curve.index.min().date().isoformat())
            aligned_metric.insert(2, "End", aligned_curve.index.max().date().isoformat())
            aligned_cols = [
                "Strategy",
                "Start",
                "End",
                "CAGR",
                "Annual Vol",
                "Sharpe",
                "Max Drawdown",
                "Total Return",
            ]
            display(
                pct_frame(
                    aligned_metric[aligned_cols],
                    ["CAGR", "Annual Vol", "Max Drawdown", "Total Return"],
                )
            )
            display(
                pct_frame(
                    final_selected[final_summary_cols],
                    [
                        "CAGR",
                        "Annual Vol",
                        "Equity Weight",
                        "Gold Weight",
                        "BTC Weight",
                        "Panic Drawdown",
                        "Max Drawdown",
                        "Average Gold Weight",
                        "Average Cash / Reduced Exposure Weight",
                    ],
                )
            )

            latest_strategy = str(final_best_latest_meta.iloc[0]["Strategy"]) if not final_best_latest_meta.empty else ""
            if latest_strategy == "Final Best Sharpe Tactical TH/Gold/BTC 65/25/10 Gold crash protection":
                display(final_best_latest_meta)
                display(
                    pct_frame(
                        final_best_latest_sleeve[
                            ["Sleeve", "Effective Weight", "Raw Sleeve Weight", "Daily Exposure", "Date"]
                        ],
                        ["Effective Weight", "Raw Sleeve Weight", "Daily Exposure"],
                    )
                )
                display(
                    pct_frame(
                        final_best_latest_security[
                            [
                                "Asset",
                                "Sleeve",
                                "Effective Weight",
                                "Internal Weight",
                                "Raw Sleeve Weight",
                                "Daily Exposure",
                                "Sleeve Multiplier",
                                "Internal Weight Date",
                                "Date",
                            ]
                        ],
                        ["Effective Weight", "Internal Weight", "Raw Sleeve Weight", "Daily Exposure", "Sleeve Multiplier"],
                    )
                )
            else:
                print(
                    "Latest weight files are not for the selected Gold25 crash-protection strategy. "
                    "Run scripts/recheck_us_th_tactical_gold_btc_latest_weights.py after yfinance rate limit clears."
                )
                display(final_best_latest_meta)
            """
        ),
        md_cell(
            """
            ## Files

            Generated outputs:

            - `result/us_th_tactical_perf_momentum_comparison_curves_thb.csv`
            - `result/us_th_tactical_perf_momentum_data_audit_thb.csv`
            - `result/us_th_tactical_perf_momentum_monthly_returns_thb.csv`
            - `result/us_th_tactical_perf_momentum_monthly_performance_table_thb.csv`
            - `result/us_th_tactical_perf_momentum_persistence_thb.csv`
            - `result/us_th_tactical_perf_momentum_tactical_exit_summary_thb.csv`
            - `result/us_th_tactical_perf_momentum_tactical_exit_curves_thb.csv`
            - `result/us_th_tactical_perf_momentum_tactical_exit_weight_history_thb.csv`
            - `result/us_th_tactical_perf_momentum_gold_btc_overlay_summary_thb.csv`
            - `result/us_th_tactical_perf_momentum_gold_btc_overlay_curves_thb.csv`
            - `result/us_th_tactical_perf_momentum_gold_btc_overlay_weight_history_thb.csv`
            - `result/us_th_tactical_perf_momentum_gold_btc_overlay_period_compare_thb.csv`
            - `result/us_th_tactical_perf_momentum_final_best_latest_effective_security_weights_thb.csv`
            - `result/us_th_tactical_perf_momentum_final_best_latest_effective_sleeve_weights_thb.csv`
            - `result/us_th_tactical_perf_momentum_final_best_latest_meta.csv`
            - `result/us_th_tactical_perf_momentum_final_model_optimizer_sweep_summary_thb.csv`
            - `result/us_th_tactical_perf_momentum_final_model_optimizer_sweep_curves_thb.csv`
            - `result/us_th_tactical_perf_momentum_final_model_optimizer_sweep_latest_weights_thb.csv`
            - `result/us_th_tactical_perf_momentum_final_model_optimizer_sweep_best_thb.csv`
            - `result/us_th_tactical_perf_momentum_gold_daily_exposure_sweep_thb.csv`
            - `result/us_th_tactical_perf_momentum_gold_daily_exposure_sweep_curves_thb.csv`
            - `result/us_th_tactical_perf_momentum_gold_weight_sweep_thb.csv`
            - `result/us_th_tactical_perf_momentum_gold_weight_sweep_curves_thb.csv`
            - `result/us_th_tactical_perf_momentum_gold_crash_protection_sweep_thb.csv`
            - `result/us_th_tactical_perf_momentum_gold_crash_protection_sweep_curves_thb.csv`
            - `result/us_th_tactical_perf_momentum_gold_crash_protection_sweep_weight_history_thb.csv`
            - `result/us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_summary_thb.csv`
            - `result/us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_curves_thb.csv`
            - `result/us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_latest_weights_thb.csv`
            - `result/us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_period_compare_thb.csv`
            - `result/us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_group_cap_sweep_summary_thb.csv`
            - `result/us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_group_cap_sweep_curves_thb.csv`
            - `result/us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_group_cap_sweep_latest_weights_thb.csv`
            - `result/us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_group_cap_sweep_period_compare_thb.csv`
            - `result/us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_group_cap_sweep_grouped_weight_history_thb.csv`
            - `result/us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_asym_group_cap_sweep_summary_thb.csv`
            - `result/us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_asym_group_cap_sweep_curves_thb.csv`
            - `result/us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_asym_group_cap_sweep_latest_weights_thb.csv`
            - `result/us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_asym_group_cap_sweep_period_compare_thb.csv`
            - `result/us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_asym_group_cap_sweep_grouped_weight_history_thb.csv`
            - `result/us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_asym_group_cap_grid_us70_80_th30_40_summary_thb.csv`
            - `result/us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_asym_group_cap_grid_us70_80_th30_40_curves_thb.csv`
            - `result/us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_asym_group_cap_grid_us70_80_th30_40_latest_weights_thb.csv`
            - `result/us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_asym_group_cap_grid_us70_80_th30_40_period_compare_thb.csv`
            - `result/us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_asym_group_cap_grid_us70_80_th30_40_grouped_weight_history_thb.csv`
            """
        ),
    ]
    return nb


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    nbf.write(nb, NOTEBOOK_FILE)
    print(f"Wrote {NOTEBOOK_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
