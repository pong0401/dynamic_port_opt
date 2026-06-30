from __future__ import annotations

from pathlib import Path
import sys
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebook"
NOTEBOOK_FILE = NOTEBOOK_DIR / "etf_universe_ex_thailand_backtest.ipynb"


def md_cell(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code_cell(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md_cell(
            """
            # ETF Universe ex-Thailand Backtest

            Purpose: test the requested ETF universe as a global ex-Thailand extension to the existing US/TH tactical framework.

            Strategies in this notebook:

            - `One-model US/ETF cap 70% / TH cap 30% with daily exposure`
            - `US/TH/ETF tactical final best Sharpe 65/25/10 with Gold crash protection`

            Timing note: monthly optimizer weights are formed from trailing data through each rebalance date, then applied to the next holding period. Daily exposure gates are lagged inside the reused signal helpers.
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
            import plotly.io as pio

            ROOT = Path.cwd().resolve()
            while not (ROOT / "src" / "dynamic_factor_copula.py").exists() and ROOT.parent != ROOT:
                ROOT = ROOT.parent
            SRC = ROOT / "src"
            SCRIPTS = ROOT / "scripts"
            for path in [SRC, SCRIPTS]:
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))

            from dynamic_factor_copula import default_paths
            import run_etf_universe_ex_thailand_backtest as etf_bt
            etf_bt = importlib.reload(etf_bt)

            paths = default_paths(ROOT)
            pio.renderers.default = "notebook"
            RUN_BACKTEST = False
            PREFIX = etf_bt.OUTPUT_PREFIX

            required = [
                paths.result_dir / f"{PREFIX}_summary_thb.csv",
                paths.result_dir / f"{PREFIX}_curves_thb.csv",
                paths.result_dir / f"{PREFIX}_latest_weights_thb.csv",
                paths.result_dir / f"{PREFIX}_coverage.csv",
                paths.result_dir / f"{PREFIX}_universe_history_thb.csv",
                paths.result_dir / f"{PREFIX}_period_compare_thb.csv",
                paths.result_dir / f"{PREFIX}_daily_exposure_history.csv",
            ]
            if RUN_BACKTEST or any(not file.exists() for file in required):
                etf_bt.main()

            summary = pd.read_csv(paths.result_dir / f"{PREFIX}_summary_thb.csv")
            curves = pd.read_csv(paths.result_dir / f"{PREFIX}_curves_thb.csv", index_col=0, parse_dates=True)
            latest_weights = pd.read_csv(paths.result_dir / f"{PREFIX}_latest_weights_thb.csv")
            coverage = pd.read_csv(paths.result_dir / f"{PREFIX}_coverage.csv")
            universe_history = pd.read_csv(paths.result_dir / f"{PREFIX}_universe_history_thb.csv")
            period_compare = pd.read_csv(paths.result_dir / f"{PREFIX}_period_compare_thb.csv")
            exposure = pd.read_csv(paths.result_dir / f"{PREFIX}_daily_exposure_history.csv", index_col=0, parse_dates=True)

            def pct_cols(frame, columns):
                out = frame.copy()
                for column in columns:
                    if column in out:
                        out[column] = out[column].map(lambda value: f"{value:.2%}" if pd.notna(value) else "")
                return out
            """
        ),
        md_cell(
            """
            ## ETF Universe
            """
        ),
        code_cell(
            """
            pd.DataFrame({"ETF": etf_bt.ETF_UNIVERSE_EX_THAILAND})
            """
        ),
        md_cell(
            """
            ## Data Coverage
            """
        ),
        code_cell(
            """
            coverage
            """
        ),
        md_cell(
            """
            ## Strategy Summary
            """
        ),
        code_cell(
            """
            metric_cols = [
                "CAGR",
                "Annual Vol",
                "Sharpe",
                "Max Drawdown",
                "Average US Stock Weight",
                "Average ETF Weight",
                "Average TH Stock Weight",
                "Average Gold Weight",
                "Average BTC Weight",
                "Average Cash / Reduced Exposure Weight",
            ]
            display_cols = ["Strategy", "Start", "End"] + metric_cols
            pct_cols(summary[display_cols], [c for c in metric_cols if c != "Sharpe"])
            """
        ),
        md_cell(
            """
            ## Equity Curves
            """
        ),
        code_cell(
            """
            fig = go.Figure()
            for column in curves.columns:
                fig.add_trace(go.Scatter(x=curves.index, y=curves[column], mode="lines", name=column))
            fig.update_layout(
                title="ETF universe strategy curves (THB)",
                yaxis_title="Portfolio value",
                xaxis_title="Date",
                legend=dict(orientation="h", y=-0.25),
                margin=dict(t=60, b=100),
                height=560,
            )
            fig.show()
            """
        ),
        md_cell(
            """
            ## Period Comparison
            """
        ),
        code_cell(
            """
            period_cols = ["Period", "Strategy", "Start", "End", "CAGR", "Annual Vol", "Sharpe", "Max Drawdown"]
            pct_cols(period_compare[period_cols], ["CAGR", "Annual Vol", "Max Drawdown"])
            """
        ),
        md_cell(
            """
            ## Latest Effective Weights
            """
        ),
        code_cell(
            """
            view = latest_weights.loc[latest_weights["Effective Weight"].abs().gt(0.0001)].copy()
            view["Effective Weight"] = view["Effective Weight"].map(lambda value: f"{value:.2%}")
            view[["Strategy", "Date", "Sleeve", "Asset", "Effective Weight"]]
            """
        ),
        md_cell(
            """
            ## Grouped Latest Weights
            """
        ),
        code_cell(
            """
            grouped_latest = (
                latest_weights.groupby(["Strategy", "Date", "Sleeve"], as_index=False)["Effective Weight"]
                .sum()
                .sort_values(["Strategy", "Effective Weight"], ascending=[True, False])
            )
            grouped_latest["Effective Weight"] = grouped_latest["Effective Weight"].map(lambda value: f"{value:.2%}")
            grouped_latest
            """
        ),
        md_cell(
            """
            ## Universe History
            """
        ),
        code_cell(
            """
            universe_history.tail(12)
            """
        ),
        md_cell(
            """
            ## Daily Exposure History
            """
        ),
        code_cell(
            """
            fig = go.Figure()
            for column in exposure.columns:
                fig.add_trace(go.Scatter(x=exposure.index, y=exposure[column], mode="lines", name=column))
            fig.update_layout(
                title="Lagged daily exposure gates",
                yaxis_title="Exposure",
                xaxis_title="Date",
                legend=dict(orientation="h", y=-0.25),
                margin=dict(t=60, b=100),
                height=420,
            )
            fig.show()
            """
        ),
        md_cell(
            """
            ## Output Files
            """
        ),
        code_cell(
            """
            pd.DataFrame(
                {
                    "File": [
                        f"result/{PREFIX}_summary_thb.csv",
                        f"result/{PREFIX}_curves_thb.csv",
                        f"result/{PREFIX}_latest_weights_thb.csv",
                        f"result/{PREFIX}_coverage.csv",
                        f"result/{PREFIX}_universe_history_thb.csv",
                        f"result/{PREFIX}_period_compare_thb.csv",
                        f"result/{PREFIX}_daily_exposure_history.csv",
                    ]
                }
            )
            """
        ),
    ]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    return nb


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    nbf.write(nb, NOTEBOOK_FILE)
    print(f"Wrote {NOTEBOOK_FILE.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
