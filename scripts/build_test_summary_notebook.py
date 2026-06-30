from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import nbformat as nbf
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynamic_factor_copula import default_paths  # noqa: E402


NOTEBOOK_DIR = ROOT / "notebook"
NOTEBOOK_FILE = NOTEBOOK_DIR / "test_summary_by_step.ipynb"
REGISTRY_FILE = ROOT / "result" / "test_config_registry.csv"


def md_cell(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code_cell(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def build_registry() -> pd.DataFrame:
    rows = [
        {
            "Step": 0,
            "Experiment": "S&P 500 buy and hold",
            "Mode": "on-the-fly from cached benchmark",
            "Source": "source cache benchmark.parquet",
            "Universe": "SPY benchmark series",
            "Lookback Days": "",
            "Rebalance": "none",
            "Daily Exposure": "No",
            "Gold/BTC Mix": "No",
            "Report Currency": "USD",
            "Notes": "Computed from cached benchmark price series inside the summary notebook.",
        },
        {
            "Step": 1,
            "Experiment": "US PIT equity sleeve baseline",
            "Mode": "precomputed",
            "Source": "result/multi_factor_copula_metrics.csv",
            "Universe": "sp500_pit, top 30 liquid names",
            "Lookback Days": 756,
            "Rebalance": "ME",
            "Daily Exposure": "No",
            "Gold/BTC Mix": "No",
            "Report Currency": "USD",
            "Notes": "4 clusters, max_weight 0.08, momentum on, mom_63, feature flags resid_vol/drawdown/downside_beta off.",
        },
        {
            "Step": 4,
            "Experiment": "US overlay lookback sweep",
            "Mode": "precomputed",
            "Source": "result/static_hmm_603010_lookback_sweep.csv",
            "Universe": "sp500_pit, top 30 liquid names",
            "Lookback Days": "252 / 504 / 756",
            "Rebalance": "ME",
            "Daily Exposure": "Yes",
            "Gold/BTC Mix": "60/30/10",
            "Report Currency": "USD",
            "Notes": "4 clusters, max_weight 0.08, momentum on, overlay strategic rebalance held at notebook baseline.",
        },
        {
            "Step": 4,
            "Experiment": "US overlay strategic rebalance sweep",
            "Mode": "precomputed",
            "Source": "result/static_hmm_603010_rebalance_sweep.csv",
            "Universe": "sp500_pit, top 30 liquid names",
            "Lookback Days": 756,
            "Rebalance": "overlay rebalance 1 / 3 / 6 months",
            "Daily Exposure": "Yes",
            "Gold/BTC Mix": "60/30/10",
            "Report Currency": "USD",
            "Notes": "Equity sleeve still monthly; sweep is the strategic overlay rebalance cadence.",
        },
        {
            "Step": 2,
            "Experiment": "Thailand PIT equity sleeve baseline",
            "Mode": "precomputed",
            "Source": "result/thai_set100_pit_metrics.csv",
            "Universe": "set100_pit, top 30 liquid names",
            "Lookback Days": 504,
            "Rebalance": "ME",
            "Daily Exposure": "No",
            "Gold/BTC Mix": "No",
            "Report Currency": "THB",
            "Notes": "^SET.BK benchmark, no vol proxy, 4 clusters, max_weight 0.08, momentum on, mom_63.",
        },
        {
            "Step": 2,
            "Experiment": "US + TH fixed-weight stock-only blends",
            "Mode": "precomputed",
            "Source": "result/us_th_stocks_only_blended_summary_thb.csv",
            "Universe": "US Static HMM sleeve + TH Static HMM sleeve",
            "Lookback Days": 504,
            "Rebalance": "ME sleeve, 1M strategic mix",
            "Daily Exposure": "No",
            "Gold/BTC Mix": "No",
            "Report Currency": "THB",
            "Notes": "Fixed-weight stock-only blends such as 100/0, 85/15, 70/30, 50/50, 30/70, and 0/100.",
        },
        {
            "Step": 2,
            "Experiment": "US + TH joint stocks-only model",
            "Mode": "precomputed",
            "Source": "result/us_th_joint_stocks_only_pit_reselect_summary_thb.csv",
            "Universe": "US and TH equities in one THB model",
            "Lookback Days": 504,
            "Rebalance": "ME sleeve",
            "Daily Exposure": "No",
            "Gold/BTC Mix": "No",
            "Report Currency": "THB",
            "Notes": "Joint model with US and TH equities only, full PIT reselect every rebalance, no Gold/BTC overlay.",
        },
        {
            "Step": 3,
            "Experiment": "Static HMM + Gold/BTC mix sweep",
            "Mode": "precomputed",
            "Source": "result/static_hmm_momentum_mix_sweep.csv",
            "Universe": "US Static HMM sleeve",
            "Lookback Days": 756,
            "Rebalance": "3M strategic mix",
            "Daily Exposure": "No",
            "Gold/BTC Mix": "100/0/0, 70/20/10, 65/25/10, 60/30/10",
            "Report Currency": "USD",
            "Notes": "Momentum-enabled sleeve mix sweep before the later confirmed 504d/1M update.",
        },
        {
            "Step": 4,
            "Experiment": "US/TH Gold/BTC blend family",
            "Mode": "precomputed",
            "Source": "result/us_th_gold_btc_blended_summary_thb.csv",
            "Universe": "US and TH sleeves with Gold/BTC overlay",
            "Lookback Days": 504,
            "Rebalance": "daily exposure + strategic mix",
            "Daily Exposure": "Yes",
            "Gold/BTC Mix": "US/TH/Gold/BTC fixed allocations",
            "Report Currency": "THB",
            "Notes": "Blend family that already includes the daily exposure overlay on the equity and side assets.",
        },
        {
            "Step": 4,
            "Experiment": "US/TH stocks-only vs with Gold/BTC side-trigger comparison",
            "Mode": "precomputed",
            "Source": "result/us_th_stocks_only_vs_gold_btc_side_trigger_pit_reselect_comparison_thb.csv",
            "Universe": "US + TH equity sleeves with side-trigger risk shift",
            "Lookback Days": 504,
            "Rebalance": "daily exposure + fee/slippage test",
            "Daily Exposure": "Yes",
            "Gold/BTC Mix": "compare none vs 60/30/10 overlay",
            "Report Currency": "THB",
            "Notes": "This file isolates how adding Gold/BTC changes the stock-only side-trigger portfolio after full PIT reselect every rebalance.",
        },
        {
            "Step": 4,
            "Experiment": "Confirmed best US overlay config",
            "Mode": "precomputed",
            "Source": "result/joint_confirm_603010_504d_1m_overlay_summary_usd.csv",
            "Universe": "sp500_pit, top 30 liquid names",
            "Lookback Days": 504,
            "Rebalance": "1M strategic mix",
            "Daily Exposure": "Yes",
            "Gold/BTC Mix": "60/30/10",
            "Report Currency": "USD / THB",
            "Notes": "Current preferred confirmed baseline from best_config_latest.json.",
        },
        {
            "Step": "3B",
            "Experiment": "US/TH all-asset static cap sweep",
            "Mode": "precomputed",
            "Source": "result/us_th_all_asset_cap_sweep_pit_reselect_summary_thb.csv",
            "Universe": "US and TH equities plus Gold and BTC in one static model",
            "Lookback Days": 504,
            "Rebalance": "ME sleeve",
            "Daily Exposure": "No",
            "Gold/BTC Mix": "In-model, capped by asset group",
            "Report Currency": "THB",
            "Notes": "Cap sweep over US equity, TH equity, Gold, and BTC max weights with full PIT reselect every rebalance.",
        },
        {
            "Step": "3C",
            "Experiment": "Top-5 all-asset cap cases objective sweep",
            "Mode": "precomputed",
            "Source": "result/us_th_all_asset_cap_top5_objective_pit_reselect_sweep_thb.csv",
            "Universe": "US and TH equities plus Gold and BTC in one static model",
            "Lookback Days": 504,
            "Rebalance": "ME sleeve",
            "Daily Exposure": "No",
            "Gold/BTC Mix": "In-model, capped by asset group",
            "Report Currency": "THB",
            "Notes": "Objective sweep on the top 5 cap cases by Sharpe from the PIT-reselected all-asset cap sweep.",
        },
        {
            "Step": "3D",
            "Experiment": "Best cap config asset-count sweep with PIT reselect",
            "Mode": "precomputed",
            "Source": "result/us_th_best_cap_asset_count_pit_reselect_sweep_summary_thb.csv",
            "Universe": "US and TH equities plus Gold and BTC in one static model",
            "Lookback Days": 504,
            "Rebalance": "ME sleeve",
            "Daily Exposure": "No",
            "Gold/BTC Mix": "In-model, capped by asset group",
            "Report Currency": "THB",
            "Notes": "Keeps the best cap configuration fixed and changes the US/TH stock counts under full PIT reselect every rebalance.",
        },
        {
            "Step": 4,
            "Experiment": "Per-asset daily exposure test",
            "Mode": "on-the-fly from cached overlay prices",
            "Source": "data/cache/dynamic_factor_copula/overlay_compare_prices.parquet",
            "Universe": "SPY, Gold, BTC",
            "Lookback Days": "200-day trend rules",
            "Rebalance": "daily",
            "Daily Exposure": "Yes, lagged 1 day",
            "Gold/BTC Mix": "No",
            "Report Currency": "USD",
            "Notes": "Notebook computes raw vs daily-exposure metrics directly from cached prices with no lookahead.",
        },
    ]
    registry = pd.DataFrame(rows)
    registry.to_csv(REGISTRY_FILE, index=False)
    return registry


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md_cell(
            """
            # Test Summary By Step

            This notebook is a map of the main tests already present in the repo. It is designed to reduce config confusion:

            - use precomputed results where they already exist
            - state clearly which config generated each result family
            - compute only lightweight checks on top of cached data
            - keep daily exposure explicitly non-lookahead
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import sys
            import json

            import numpy as np
            import pandas as pd
            import plotly.graph_objects as go

            ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebook" else Path.cwd().resolve()
            SRC = ROOT / "src"
            if str(SRC) not in sys.path:
                sys.path.insert(0, str(SRC))

            from dynamic_factor_copula import (
                compare_apply_returns,
                compare_sp_exposure,
                compare_trend_exposure,
                compute_port_opt_style_metrics,
                curve_from_returns,
                default_paths,
                load_overlay_compare_prices,
            )

            pd.set_option("display.max_colwidth", 120)
            pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
            paths = default_paths(ROOT)

            def read_csv_clean(path, index_col=None):
                df = pd.read_csv(path, index_col=index_col)
                if "Unnamed: 0" in df.columns:
                    df = df.rename(columns={"Unnamed: 0": "Strategy"})
                return df

            def yes_no(value):
                return "Yes" if value else "No"

            def display_path(path):
                try:
                    return str(path.relative_to(ROOT))
                except ValueError:
                    return str(path)
            """
        ),
        md_cell(
            """
            ## Precompute Status

            This first section answers two questions:

            1. Which result families are already saved as precomputed files?
            2. Which families are still computed on the fly inside this summary notebook?
            """
        ),
        code_cell(
            """
            tracked_files = [
                ("Main US sleeve metrics", paths.result_dir / "multi_factor_copula_metrics.csv", "precomputed"),
                ("US lookback sweep", paths.result_dir / "static_hmm_603010_lookback_sweep.csv", "precomputed"),
                ("US rebalance sweep", paths.result_dir / "static_hmm_603010_rebalance_sweep.csv", "precomputed"),
                ("US mix sweep", paths.result_dir / "static_hmm_momentum_mix_sweep.csv", "precomputed"),
                ("US overlay summary", paths.result_dir / "overlay_comparison_summary.csv", "precomputed"),
                ("Confirmed best US overlay", paths.result_dir / "joint_confirm_603010_504d_1m_overlay_summary_usd.csv", "precomputed"),
                ("Thailand PIT sleeve", paths.result_dir / "thai_set100_pit_metrics.csv", "precomputed"),
                ("US/TH stocks-only blend", paths.result_dir / "us_th_stocks_only_blended_summary_thb.csv", "precomputed"),
                ("US/TH stocks-only joint model", paths.result_dir / "us_th_joint_stocks_only_pit_reselect_summary_thb.csv", "precomputed"),
                ("US/TH Gold/BTC blend", paths.result_dir / "us_th_gold_btc_blended_summary_thb.csv", "precomputed"),
                ("US/TH Gold/BTC joint model", paths.result_dir / "us_th_joint_model_summary_thb.csv", "precomputed"),
                ("US/TH objective sweep", paths.result_dir / "us_th_joint_model_objective_sweep_thb.csv", "precomputed"),
                ("US/TH asset-count sweep", paths.result_dir / "us_th_best_cap_asset_count_pit_reselect_sweep_summary_thb.csv", "precomputed"),
                ("US/TH all-asset static caps", paths.result_dir / "us_th_all_asset_static_caps_pit_reselect_summary_thb.csv", "precomputed"),
                ("US/TH all-asset cap sweep", paths.result_dir / "us_th_all_asset_cap_sweep_pit_reselect_summary_thb.csv", "precomputed"),
                ("US/TH top-5 cap objective sweep", paths.result_dir / "us_th_all_asset_cap_top5_objective_pit_reselect_sweep_thb.csv", "precomputed"),
                ("US/TH stocks-only vs Gold/BTC comparison", paths.result_dir / "us_th_stocks_only_vs_gold_btc_side_trigger_pit_reselect_comparison_thb.csv", "precomputed"),
                ("Config registry", paths.result_dir / "test_config_registry.csv", "precomputed"),
                ("SPY buy and hold metrics", paths.source_cache_root / "benchmark.parquet", "computed in notebook from cache"),
                ("Per-asset daily exposure test", paths.local_cache_root / "overlay_compare_prices.parquet", "computed in notebook from cache"),
            ]

            precompute_status = pd.DataFrame(
                [
                    {
                        "Artifact": name,
                        "Path": display_path(path),
                        "Status": mode,
                        "Exists": yes_no(path.exists()),
                        "Last Modified": pd.Timestamp(path.stat().st_mtime, unit="s") if path.exists() else pd.NaT,
                    }
                    for name, path, mode in tracked_files
                ]
            )
            precompute_status
            """
        ),
        md_cell(
            """
            ## Config Registry

            This is the compact registry for the experiment families used in this notebook. If a result is in this table, we treat it as part of the supported summary path rather than an orphan CSV.
            """
        ),
        code_cell(
            """
            config_registry = pd.read_csv(paths.result_dir / "test_config_registry.csv")
            config_registry
            """
        ),
        md_cell(
            """
            ## Step 0 - S&P Buy And Hold

            This is the benchmark reference. We compute it directly from the cached benchmark price series so the notebook always has a clean baseline even if no dedicated summary CSV exists for SPY buy and hold.
            """
        ),
        code_cell(
            """
            benchmark = pd.read_parquet(paths.source_cache_root / "benchmark.parquet").rename(columns={"value": "benchmark"})
            benchmark.index = pd.to_datetime(benchmark.index)
            benchmark = benchmark["benchmark"].sort_index().loc["2012-01-01":"2026-04-30"].ffill()

            def metrics_from_price_window(series, start, end, label):
                window = series.loc[start:end].dropna()
                returns = window.pct_change(fill_method=None).fillna(0.0)
                curve = curve_from_returns(returns)
                row = compute_port_opt_style_metrics(curve, risk_free_rate=0.03).to_dict()
                row["Strategy"] = label
                row["Start"] = window.index.min().date().isoformat()
                row["End"] = window.index.max().date().isoformat()
                return row

            spx_windows = pd.DataFrame(
                [
                    metrics_from_price_window(benchmark, "2012-01-01", "2026-04-30", "S&P 500 buy and hold (full)"),
                    metrics_from_price_window(benchmark, "2016-01-04", "2026-04-29", "S&P 500 buy and hold (overlay window)"),
                    metrics_from_price_window(benchmark, "2017-12-29", "2026-04-29", "S&P 500 buy and hold (US+TH joint window)"),
                ]
            ).set_index("Strategy")
            spx_windows
            """
        ),
        md_cell(
            """
            ## Step 1 - US PIT Static vs Dynamic HMM

            This section is sleeve-only. It summarizes the main US point-in-time equity sleeve against S&P 500 buy and hold. It does not include Gold or BTC. The current precomputed family keeps `n_clusters=4`; there is no separate cluster-count sweep artifact in the repo today, so cluster count is treated as fixed rather than optimized from a saved sweep.
            """
        ),
        code_cell(
            """
            us_metrics = read_csv_clean(paths.result_dir / "multi_factor_copula_metrics.csv").set_index("Strategy")
            us_vs_spx = pd.concat(
                [
                    spx_windows.loc[["S&P 500 buy and hold (full)"]][["CAGR", "Annual Vol", "Sharpe", "Max Drawdown"]],
                    us_metrics[["CAGR", "Annual Vol", "Sharpe", "Max Drawdown", "Turnover"]],
                ],
                axis=0,
            )
            display(us_vs_spx)
            """
        ),
        md_cell(
            """
            ## Step 2 - Add Thailand

            This section answers the Thailand expansion in three layers:

            - Thailand-only PIT sleeve performance
            - fixed-weight US/TH stock-only blends
            - US and Thailand together inside one joint stocks-only model
            """
        ),
        code_cell(
            """
            thai_metrics = read_csv_clean(paths.result_dir / "thai_set100_pit_metrics.csv").set_index("Strategy")
            us_th_sleeve_compare = pd.concat(
                {
                    "US PIT sleeve": us_metrics[["CAGR", "Sharpe", "Max Drawdown", "Turnover"]],
                    "TH PIT sleeve": thai_metrics[["CAGR", "Sharpe", "Max Drawdown", "Turnover"]],
                },
                axis=1,
            )

            us_th_stock_blend = pd.read_csv(paths.result_dir / "us_th_stocks_only_blended_summary_thb.csv").sort_values("Sharpe", ascending=False)
            us_th_joint_stocks_only = pd.read_csv(paths.result_dir / "us_th_joint_stocks_only_pit_reselect_summary_thb.csv").sort_values("Sharpe", ascending=False)

            fixed_base = us_th_stock_blend.loc[us_th_stock_blend["Strategy"] == "US/TH stocks only 100/0"].iloc[0]
            fixed_help = us_th_stock_blend.copy()
            fixed_help["CAGR Delta vs 100/0"] = fixed_help["CAGR"] - float(fixed_base["CAGR"])
            fixed_help["Sharpe Delta vs 100/0"] = fixed_help["Sharpe"] - float(fixed_base["Sharpe"])
            fixed_help["Max DD Delta vs 100/0"] = fixed_help["Max Drawdown"] - float(fixed_base["Max Drawdown"])
            fixed_help["Thailand Helped?"] = fixed_help["Sharpe Delta vs 100/0"].map(lambda x: "Yes" if x > 0 else "No")

            joint_base_map = {
                "Static": us_th_joint_stocks_only.loc[us_th_joint_stocks_only["Strategy"] == "Joint US-only stocks Static Copula PIT reselect"].iloc[0],
                "Dynamic": us_th_joint_stocks_only.loc[us_th_joint_stocks_only["Strategy"] == "Joint US-only stocks Dynamic HMM Copula PIT reselect"].iloc[0],
            }
            joint_help_rows = []
            for _, row in us_th_joint_stocks_only.iterrows():
                model_key = "Dynamic" if "Dynamic" in row["Strategy"] else "Static"
                base = joint_base_map[model_key]
                joint_help_rows.append(
                    {
                        "Strategy": row["Strategy"],
                        "Model": model_key,
                        "CAGR": row["CAGR"],
                        "Sharpe": row["Sharpe"],
                        "Max Drawdown": row["Max Drawdown"],
                        "CAGR Delta vs US-only joint": row["CAGR"] - float(base["CAGR"]),
                        "Sharpe Delta vs US-only joint": row["Sharpe"] - float(base["Sharpe"]),
                        "Max DD Delta vs US-only joint": row["Max Drawdown"] - float(base["Max Drawdown"]),
                        "Thailand Helped?": "Yes" if (row["Sharpe"] - float(base["Sharpe"])) > 0 else "No",
                    }
                )
            joint_help = pd.DataFrame(joint_help_rows)

            display(us_th_sleeve_compare)
            display(us_th_stock_blend)
            display(us_th_joint_stocks_only)
            display(fixed_help[["Strategy", "CAGR Delta vs 100/0", "Sharpe Delta vs 100/0", "Max DD Delta vs 100/0", "Thailand Helped?"]])
            display(joint_help[["Strategy", "CAGR Delta vs US-only joint", "Sharpe Delta vs US-only joint", "Max DD Delta vs US-only joint", "Thailand Helped?"]])
            """
        ),
        md_cell(
            """
            ## Step 3 - Add Gold and BTC

            This section is only for the plain allocation question: how much Gold and BTC should be added to the base US sleeve before introducing the daily exposure overlay logic.

            This mix sweep uses the US Static HMM sleeve only, with no Thailand sleeve included.

            It intentionally stays narrow:

            - the older US Static HMM mix sweep

            The all-assets-in-one-model capped portfolio is shown in its own max-weight section next, and the broader overlay families that already depend on daily exposure are shown in Step 4.
            """
        ),
        code_cell(
            """
            mix_sweep = pd.read_csv(paths.result_dir / "static_hmm_momentum_mix_sweep.csv").sort_values("Sharpe", ascending=False)

            display(mix_sweep)
            """
        ),
        md_cell(
            """
            ## Step 3B - Asset Max Weight

            This section isolates the all-assets-in-one-model static portfolio for US stocks, Thailand stocks, Gold, and BTC.

            The key idea here is not daily exposure. It is asset-level weight caps:

            - default equity max weight = 8%
            - Gold max weight = 30%
            - BTC max weight = 10%

            This is a one-model static portfolio with capped weights, so it belongs in its own section rather than being mixed into the plain Gold/BTC allocation sweep.
            """
        ),
        code_cell(
            """
            all_asset_static_caps = pd.read_csv(paths.result_dir / "us_th_all_asset_static_caps_pit_reselect_summary_thb.csv").rename(columns={"Unnamed: 0": "Strategy"})
            all_asset_cap_sweep = pd.read_csv(paths.result_dir / "us_th_all_asset_cap_sweep_pit_reselect_summary_thb.csv").sort_values("Sharpe", ascending=False)
            all_asset_static_caps_latest = pd.read_csv(paths.result_dir / "us_th_all_asset_static_caps_pit_reselect_latest_weights_thb.csv")
            all_asset_cap_sweep_latest = pd.read_csv(paths.result_dir / "us_th_all_asset_cap_sweep_pit_reselect_latest_weights_thb.csv")
            all_asset_static_caps.insert(1, "Daily Exposure", "No")
            all_asset_cap_sweep.insert(1, "Daily Exposure", "No")
            best_cap_case = "US6/TH6/Gold30/BTC10"
            best_cap_last_weight = (
                all_asset_cap_sweep_latest.loc[all_asset_cap_sweep_latest["Case"] == best_cap_case]
                .loc[lambda df: df["Portfolio Weight"] > 0]
                .sort_values("Portfolio Weight", ascending=False)
                .reset_index(drop=True)
            )

            asset_cap_rules = pd.DataFrame(
                [
                    {"Asset Group": "US Equity", "Max Weight": 0.08, "Notes": "Default per-asset cap"},
                    {"Asset Group": "TH Equity", "Max Weight": 0.08, "Notes": "Default per-asset cap"},
                    {"Asset Group": "Gold", "Max Weight": 0.30, "Notes": "Explicit override cap"},
                    {"Asset Group": "BTC", "Max Weight": 0.10, "Notes": "Explicit override cap"},
                ]
            )

            display(asset_cap_rules)
            display(all_asset_static_caps)
            display(all_asset_cap_sweep)
            display(best_cap_last_weight)
            display(all_asset_static_caps_latest)
            """
        ),
        md_cell(
            """
            ## Step 3C - Top 5 Cap Cases by Objective

            This section takes the top 5 all-asset cap cases by Sharpe and reruns each case across the objective menu:

            - mean_variance
            - max_sharpe_mom
            - min_vol_mom_tilt
            - risk_parity_mom_tilt

            The goal is to separate two questions:

            - which cap structure is best
            - which objective is best once that cap structure is fixed
            """
        ),
        code_cell(
            """
            all_asset_cap_top5_objective = pd.read_csv(paths.result_dir / "us_th_all_asset_cap_top5_objective_pit_reselect_sweep_thb.csv")
            all_asset_cap_top5_best = pd.read_csv(paths.result_dir / "us_th_all_asset_cap_top5_objective_pit_reselect_best_by_case_thb.csv")
            all_asset_cap_top5_latest = pd.read_csv(paths.result_dir / "us_th_all_asset_cap_top5_objective_pit_reselect_latest_weights_thb.csv")
            best_top5_strategy = all_asset_cap_top5_best.sort_values("Sharpe", ascending=False).iloc[0]["Strategy"]
            best_top5_last_weight = (
                all_asset_cap_top5_latest.loc[all_asset_cap_top5_latest["Strategy"] == best_top5_strategy]
                .loc[lambda df: df["Portfolio Weight"] > 0]
                .sort_values("Portfolio Weight", ascending=False)
                .reset_index(drop=True)
            )

            display(all_asset_cap_top5_best)
            display(all_asset_cap_top5_objective.sort_values(["Top 5 Rank", "Sharpe"], ascending=[True, False]))
            display(best_top5_last_weight)
            """
        ),
        md_cell(
            """
            ## Step 3D - Best Cap Config with More Stock Selections

            This section keeps the best cap-weight setup fixed and only changes how many US and Thailand stocks are allowed into the all-asset model.

            Fixed best-cap configuration:

            - objective = `mean_variance`
            - US equity cap = 6%
            - TH equity cap = 6%
            - Gold cap = 40%
            - BTC cap = 10%

            Stock selection rule:

            - this repo does **not** select by market cap
            - it reselects the US and Thailand stock universes at **every rebalance date**
            - inside each rebalance window it ranks candidates by median `price * volume`
            - it requires strong history coverage before a name is eligible

            This is the strict level-3 version of the test:

            - point-in-time membership is refreshed every rebalance
            - liquidity ranking is refreshed every rebalance
            - then the all-asset static capped model is rebuilt on that current PIT universe

            The sweep below tests whether expanding the reselected US/TH stock lists from `30/30` to `40/40`, `50/50`, and `100/100` improves the best cap-weight portfolio.
            """
        ),
        code_cell(
            """
            best_cap_asset_count_sweep = pd.read_csv(paths.result_dir / "us_th_best_cap_asset_count_pit_reselect_sweep_summary_thb.csv").sort_values("Sharpe", ascending=False)
            best_cap_asset_count_latest = pd.read_csv(paths.result_dir / "us_th_best_cap_asset_count_pit_reselect_sweep_latest_weights_thb.csv")
            best_cap_asset_count_winner = best_cap_asset_count_sweep.iloc[0]["Strategy"]
            best_cap_asset_count_last_weight = (
                best_cap_asset_count_latest.loc[best_cap_asset_count_latest["Strategy"] == best_cap_asset_count_winner]
                .loc[lambda df: df["Portfolio Weight"] > 0]
                .sort_values("Portfolio Weight", ascending=False)
                .reset_index(drop=True)
            )

            stock_selection_rule = pd.DataFrame(
                [
                    {"Rule": "Primary rank metric", "Value": "Median(price * volume) inside each rebalance lookback window"},
                    {"Rule": "Availability filter", "Value": "At least 90% non-null price history inside the rebalance lookback window"},
                    {"Rule": "Refresh cadence", "Value": "Full PIT reselect every rebalance"},
                    {"Rule": "Uses market cap?", "Value": "No"},
                ]
            )

            display(stock_selection_rule)
            display(best_cap_asset_count_sweep)
            display(best_cap_asset_count_last_weight)
            """
        ),
        md_cell(
            """
            ## Step 4 - Daily Exposure

            Daily exposure in this repo must be non-lookahead:

            - the signal is observed from today's close and today's fully known state
            - that signal affects tomorrow's exposure
            - we do not let today's close change today's return

            The helper functions in `src/dynamic_factor_copula.py` now lag close-based signals by one session before applying them to returns.

            This section contains all summary tables that depend on daily exposure overlays, including:

            - the confirmed best US overlay summary
            - US/TH Gold/BTC blend rows
            - direct stocks-only vs with-Gold/BTC side-trigger comparison

            The primary US overlay table shown here is the confirmed best config:

            - equity sleeve lookback = 504 trading days
            - overlay strategic rebalance = 1 month

            The older legacy `756d / 3M` summary is intentionally not shown here to keep the section focused on the current preferred baseline.
            """
        ),
        code_cell(
            """
            confirmed_us = read_csv_clean(paths.result_dir / "joint_confirm_603010_504d_1m_overlay_summary_usd.csv").set_index("Strategy")
            us_th_gold_btc_blend = pd.read_csv(paths.result_dir / "us_th_gold_btc_blended_summary_thb.csv").sort_values("Sharpe", ascending=False)
            stocks_only_vs_gold_btc = pd.read_csv(paths.result_dir / "us_th_stocks_only_vs_gold_btc_side_trigger_pit_reselect_comparison_thb.csv").sort_values("Sharpe", ascending=False)
            best_config = json.load(open(paths.result_dir / "best_config_latest.json", encoding="utf-8"))

            overlay_prices = load_overlay_compare_prices(
                paths,
                start_date="2016-01-01",
                end_date="2026-04-29",
                tickers=["SPY", "GC=F", "BTC-USD", "^VIX", "USDTHB=X"],
            ).dropna(subset=["SPY", "GC=F", "BTC-USD", "^VIX"])

            spy = overlay_prices["SPY"]
            gold = overlay_prices["GC=F"]
            btc = overlay_prices["BTC-USD"]
            vix = overlay_prices["^VIX"]
            zero_fx = pd.Series(0.0, index=overlay_prices.index, dtype=float)

            def daily_exposure_delta(label, raw_returns, daily_returns):
                raw_curve = curve_from_returns(raw_returns)
                daily_curve = curve_from_returns(daily_returns)
                raw_metrics = compute_port_opt_style_metrics(raw_curve, risk_free_rate=0.03)
                daily_metrics = compute_port_opt_style_metrics(daily_curve, risk_free_rate=0.03)
                delta = (daily_metrics - raw_metrics).to_dict()
                delta["Asset"] = label
                delta["Sharpe Helped"] = yes_no(delta["Sharpe"] > 0)
                delta["Drawdown Helped"] = yes_no(delta["Max Drawdown"] > 0)
                return delta

            spy_returns = spy.pct_change(fill_method=None).fillna(0.0)
            gold_returns = gold.pct_change(fill_method=None).fillna(0.0)
            btc_returns = btc.pct_change(fill_method=None).fillna(0.0)

            spy_daily = compare_apply_returns(spy_returns, compare_sp_exposure(spy, vix), "USD_STATIC", zero_fx)
            gold_daily = compare_apply_returns(gold_returns, compare_trend_exposure(gold, 0.50), "USD_STATIC", zero_fx)
            btc_daily = compare_apply_returns(btc_returns, compare_trend_exposure(btc, 0.00), "USD_STATIC", zero_fx)

            asset_daily_exposure_test = pd.DataFrame(
                [
                    daily_exposure_delta("SPY", spy_returns, spy_daily),
                    daily_exposure_delta("Gold", gold_returns, gold_daily),
                    daily_exposure_delta("BTC", btc_returns, btc_daily),
                ]
            ).set_index("Asset")

            confirmed_us_display = confirmed_us.copy()
            confirmed_us_display.insert(
                0,
                "Momentum",
                [
                    "No",
                    "No",
                    "Yes",
                    "Yes",
                ],
            )

            us_th_gold_btc_blend_display = us_th_gold_btc_blend.copy()
            us_th_gold_btc_blend_display.insert(1, "Momentum", "Yes")

            stocks_only_vs_gold_btc_display = stocks_only_vs_gold_btc.copy()
            stocks_only_vs_gold_btc_display.insert(1, "Momentum", "Yes")

            asset_daily_exposure_test_display = asset_daily_exposure_test.copy()
            asset_daily_exposure_test_display.insert(0, "Momentum", "No")

            display(confirmed_us_display)
            display(us_th_gold_btc_blend_display)
            display(stocks_only_vs_gold_btc_display)
            display(asset_daily_exposure_test_display)
            display(pd.DataFrame([best_config["best_confirmed_joint_config"]["equity_sleeve"]]))
            """
        ),
        md_cell(
            """
            ## Best Sharpe

            This is the compact scoreboard for the highest-Sharpe result in each headline category after the latest reruns.
            """
        ),
        code_cell(
            """
            best_sharpe_snapshot = pd.DataFrame(
                [
                    {
                        "Category": "Buy and hold",
                        "Strategy": "S&P 500 buy and hold (full)",
                        "Sharpe": float(spx_windows.loc["S&P 500 buy and hold (full)", "Sharpe"]),
                        "CAGR": float(spx_windows.loc["S&P 500 buy and hold (full)", "CAGR"]),
                        "Source": "benchmark cache",
                    },
                    {
                        "Category": "US stock",
                        "Strategy": us_metrics["Sharpe"].idxmax(),
                        "Sharpe": float(us_metrics["Sharpe"].max()),
                        "CAGR": float(us_metrics.loc[us_metrics["Sharpe"].idxmax(), "CAGR"]),
                        "Source": "result/multi_factor_copula_metrics.csv",
                    },
                    {
                        "Category": "Momentum",
                        "Strategy": f"Static HMM mix {mix_sweep.iloc[0]['Mix']}",
                        "Sharpe": float(mix_sweep.iloc[0]["Sharpe"]),
                        "CAGR": float(mix_sweep.iloc[0]["CAGR"]),
                        "Source": "result/static_hmm_momentum_mix_sweep.csv",
                    },
                    {
                        "Category": "Daily exposure",
                        "Strategy": confirmed_us["Sharpe"].idxmax(),
                        "Sharpe": float(confirmed_us["Sharpe"].max()),
                        "CAGR": float(confirmed_us.loc[confirmed_us["Sharpe"].idxmax(), "CAGR"]),
                        "Source": "result/joint_confirm_603010_504d_1m_overlay_summary_usd.csv",
                    },
                    {
                        "Category": "All",
                        "Strategy": all_asset_cap_sweep.iloc[0]["Strategy"],
                        "Sharpe": float(all_asset_cap_sweep.iloc[0]["Sharpe"]),
                        "CAGR": float(all_asset_cap_sweep.iloc[0]["CAGR"]),
                        "Source": "result/us_th_all_asset_cap_sweep_pit_reselect_summary_thb.csv",
                    },
                    {
                        "Category": "Top-5 cap objective winner",
                        "Strategy": all_asset_cap_top5_best.sort_values("Sharpe", ascending=False).iloc[0]["Strategy"],
                        "Sharpe": float(all_asset_cap_top5_best["Sharpe"].max()),
                        "CAGR": float(all_asset_cap_top5_best.sort_values("Sharpe", ascending=False).iloc[0]["CAGR"]),
                        "Source": "result/us_th_all_asset_cap_top5_objective_pit_reselect_best_by_case_thb.csv",
                    },
                ]
            )
            best_sharpe_snapshot
            """
        ),
        md_cell(
            """
            ## Step 5 - Summary

            This final section condenses the best currently saved answers from the repo. It is not meant to replace the detailed sections above; it is the short scoreboard for deciding what to trust first.
            """
        ),
        code_cell(
            """
            summary_scoreboard = pd.DataFrame(
                [
                    {
                        "Question": "Best saved US PIT sleeve family",
                        "Answer": "Static Copula is slightly ahead of Dynamic HMM in the saved US PIT baseline file.",
                        "Evidence": "result/multi_factor_copula_metrics.csv",
                    },
                    {
                        "Question": "Current preferred confirmed US overlay config",
                        "Answer": "Static HMM with momentum + Gold/BTC 60/30/10, 504d lookback, 1M strategic rebalance",
                        "Evidence": "result/best_config_latest.json",
                    },
                    {
                        "Question": "Best saved plain Gold/BTC mix in the older US sweep",
                        "Answer": mix_sweep.iloc[0]["Mix"],
                        "Evidence": "result/static_hmm_momentum_mix_sweep.csv",
                    },
                    {
                        "Question": "Best saved US/TH fixed-weight blend in THB by Sharpe",
                        "Answer": us_th_stock_blend.iloc[0]["Strategy"],
                        "Evidence": "result/us_th_stocks_only_blended_summary_thb.csv",
                    },
                    {
                        "Question": "Best saved US/TH joint stocks-only model row in THB by Sharpe",
                        "Answer": us_th_joint_stocks_only.iloc[0]["Strategy"],
                        "Evidence": "result/us_th_joint_stocks_only_pit_reselect_summary_thb.csv",
                    },
                    {
                        "Question": "Does Thailand help in the stock-only fixed-weight blends?",
                        "Answer": "No in the saved tests; 100/0 remains the best Sharpe and CAGR among the stock-only blend rows.",
                        "Evidence": "result/us_th_stocks_only_blended_summary_thb.csv",
                    },
                    {
                        "Question": "Does Thailand help in the stock-only joint model?",
                        "Answer": "No in the saved control test; the US-only joint rows beat the US+TH joint rows on both CAGR and Sharpe.",
                        "Evidence": "result/us_th_joint_stocks_only_pit_reselect_summary_thb.csv",
                    },
                    {
                        "Question": "What happens when Gold/BTC is added to the stock-only side-trigger portfolio?",
                        "Answer": stocks_only_vs_gold_btc.iloc[0]["Strategy"],
                        "Evidence": "result/us_th_stocks_only_vs_gold_btc_side_trigger_pit_reselect_comparison_thb.csv",
                    },
                    {
                        "Question": "Best saved US/TH Gold/BTC overlay blend in THB by Sharpe",
                        "Answer": us_th_gold_btc_blend.iloc[0]["Strategy"],
                        "Evidence": "result/us_th_gold_btc_blended_summary_thb.csv",
                    },
                    {
                        "Question": "Cluster-count sweep available?",
                        "Answer": "No saved cluster-count sweep found; current precomputed families use 4 clusters.",
                        "Evidence": "config registry + existing result files",
                    },
                    {
                        "Question": "Daily exposure asset tests fully precomputed?",
                        "Answer": "No; SPY/Gold/BTC asset-level tests are computed here from cached prices, while sleeve-level overlay rows are precomputed.",
                        "Evidence": "overlay cache + overlay summary files",
                    },
                ]
            )
            summary_scoreboard
            """
        ),
    ]
    return nb


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    build_registry()
    notebook = build_notebook()
    with NOTEBOOK_FILE.open("w", encoding="utf-8") as handle:
        nbf.write(notebook, handle)
    print(NOTEBOOK_FILE.name)
    print(REGISTRY_FILE.name)


if __name__ == "__main__":
    main()
