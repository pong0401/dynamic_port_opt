from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebook"
NOTEBOOK_FILE = NOTEBOOK_DIR / "etf_momentum_prefilter_backtest.ipynb"


def md_cell(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code_cell(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md_cell(
            """
            # ETF Monthly Momentum Pre-Filter Backtest

            Purpose: test a monthly ETF rebalance pipeline where momentum filtering happens before the portfolio optimizer.

            Default universe: `SPMO`, `MTUM`, `SCHG`, `XLK`, `EWT`, `EWY`, `EWJ`, `INDA`, `MCHI`.

            Timing rule: signals are calculated using prices available on or before each month-end rebalance date. Weights generated on that date are applied only from the next trading day through the next rebalance window.
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import sys
            import importlib

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
            import run_etf_momentum_prefilter_backtest as etf_mom
            etf_mom = importlib.reload(etf_mom)

            paths = default_paths(ROOT)
            pio.renderers.default = "notebook"
            RUN_BACKTEST = False
            PREFIX = etf_mom.OUTPUT_PREFIX

            required = [
                paths.result_dir / f"{PREFIX}_rebalance_table.csv",
                paths.result_dir / f"{PREFIX}_momentum_history.csv",
                paths.result_dir / f"{PREFIX}_weight_history.csv",
                paths.result_dir / f"{PREFIX}_selected_history.csv",
                paths.result_dir / f"{PREFIX}_portfolio_returns.csv",
                paths.result_dir / f"{PREFIX}_equity_curve.csv",
                paths.result_dir / f"{PREFIX}_performance_metrics.csv",
            ]
            if RUN_BACKTEST or any(not file.exists() for file in required):
                etf_mom.main()

            rebalance_table = pd.read_csv(paths.result_dir / f"{PREFIX}_rebalance_table.csv", parse_dates=["rebalance_date", "next_rebalance_date"])
            momentum_history = pd.read_csv(paths.result_dir / f"{PREFIX}_momentum_history.csv", parse_dates=["rebalance_date"])
            weight_history = pd.read_csv(paths.result_dir / f"{PREFIX}_weight_history.csv", parse_dates=["rebalance_date"])
            selected_history = pd.read_csv(paths.result_dir / f"{PREFIX}_selected_history.csv", parse_dates=["rebalance_date"])
            portfolio_returns = pd.read_csv(paths.result_dir / f"{PREFIX}_portfolio_returns.csv", index_col=0, parse_dates=True)
            equity_curve = pd.read_csv(paths.result_dir / f"{PREFIX}_equity_curve.csv", index_col=0, parse_dates=True)
            performance_metrics = pd.read_csv(paths.result_dir / f"{PREFIX}_performance_metrics.csv")

            config = etf_mom.ETFMomentumConfig()

            def pct_cols(frame, columns):
                out = frame.copy()
                for column in columns:
                    if column in out:
                        out[column] = out[column].map(lambda value: f"{value:.2%}" if pd.notna(value) else "")
                return out
            """
        ),
        md_cell("## Config"),
        code_cell(
            """
            pd.DataFrame(
                [
                    {"Parameter": "Universe", "Value": ", ".join(config.universe)},
                    {"Parameter": "Rebalance", "Value": config.rebalance_frequency},
                    {"Parameter": "Top N", "Value": config.top_n},
                    {"Parameter": "Min N", "Value": config.min_n},
                    {"Parameter": "SMA Filter", "Value": f"Close > SMA{config.sma_window}"},
                    {"Parameter": "Positive Momentum", "Value": "3M and 6M returns > 0"},
                    {"Parameter": "Max ETF Weight", "Value": f"{config.max_weight_per_asset:.0%}"},
                    {"Parameter": "Fallback", "Value": config.fallback_asset},
                ]
            )
            """
        ),
        md_cell("## Performance Summary"),
        code_cell(
            """
            pct_cols(
                performance_metrics,
                ["Total Return", "CAGR", "Annual Vol", "Max Drawdown", "Hit Rate", "Monthly Win Rate", "Exposure Percentage"],
            )
            """
        ),
        md_cell("## Equity Curve"),
        code_cell(
            """
            fig = go.Figure()
            for column in equity_curve.columns:
                fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve[column], mode="lines", name=column))
            fig.update_layout(
                title="ETF momentum pre-filter portfolio",
                yaxis_title="Portfolio value",
                xaxis_title="Date",
                legend=dict(orientation="h", y=-0.25),
                margin=dict(t=60, b=90),
                height=520,
            )
            fig.show()
            """
        ),
        md_cell("## Latest Weights"),
        code_cell(
            """
            latest_date = weight_history["rebalance_date"].max()
            latest_weights = weight_history.loc[weight_history["rebalance_date"].eq(latest_date)].copy()
            latest_weights = latest_weights.sort_values("weight", ascending=False)
            latest_weights["weight"] = latest_weights["weight"].map(lambda value: f"{value:.2%}")
            latest_weights
            """
        ),
        md_cell("## Monthly Selected ETFs"),
        code_cell(
            """
            selected_history.tail(18)
            """
        ),
        md_cell("## Latest Momentum Ranking"),
        code_cell(
            """
            latest_momentum_date = momentum_history["rebalance_date"].max()
            latest_momentum = momentum_history.loc[momentum_history["rebalance_date"].eq(latest_momentum_date)].copy()
            display_cols = [
                "rebalance_date",
                "ETF",
                "price",
                "ret_1m",
                "ret_3m",
                "ret_6m",
                "ret_12m",
                "sma200",
                "pass_ma200",
                "pass_momentum",
                "momentum_score",
                "selected",
            ]
            pct_cols(latest_momentum[display_cols], ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "momentum_score"])
            """
        ),
        md_cell("## Selection Count"),
        code_cell(
            """
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=selected_history["rebalance_date"],
                    y=selected_history["selected_count"],
                    name="Selected ETFs",
                )
            )
            fig.update_layout(
                title="Monthly selected ETF count",
                yaxis_title="Count",
                xaxis_title="Rebalance date",
                height=420,
            )
            fig.show()
            """
        ),
        md_cell("## Rebalance Audit"),
        code_cell(
            """
            audit = rebalance_table.tail(18).copy()
            audit["portfolio_return_next_period"] = audit["portfolio_return_next_period"].map(lambda value: f"{value:.2%}")
            audit["turnover"] = audit["turnover"].map(lambda value: f"{value:.2%}")
            audit["cash_weight"] = audit["cash_weight"].map(lambda value: f"{value:.2%}")
            audit[["rebalance_date", "next_rebalance_date", "selected_etfs", "selected_count", "cash_weight", "portfolio_return_next_period", "turnover"]]
            """
        ),
    ]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    return nb


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), NOTEBOOK_FILE)
    print(f"Wrote {NOTEBOOK_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
