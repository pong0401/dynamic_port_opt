from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


NOTEBOOK_DIR = ROOT / "notebook"
NOTEBOOK_FILE = NOTEBOOK_DIR / "best_param_by_step.ipynb"


def md_cell(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code_cell(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md_cell(
            """
            # Best Param By Step

            This notebook reruns the simple benchmark path from cache and searches parameters step by step:

            1. S&P buy-and-hold performance
            2. S&P / Gold / BTC allocation sweep
            3. Daily exposure parameter sweep by asset

            Daily exposure signals are **lagged one trading session** before returns are multiplied, so a close-based signal from day `t` can only affect day `t+1`.
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import sys
            import numpy as np
            import pandas as pd
            import plotly.graph_objects as go
            from IPython.display import HTML, display
            from plotly.subplots import make_subplots

            pd.set_option("display.max_columns", None)
            pd.set_option("display.max_rows", 200)
            pd.set_option("display.max_colwidth", None)
            pd.set_option("display.width", 240)

            ROOT = Path.cwd().resolve()
            while not (ROOT / "src" / "dynamic_factor_copula.py").exists() and ROOT.parent != ROOT:
                ROOT = ROOT.parent
            SRC = ROOT / "src"
            if str(SRC) not in sys.path:
                sys.path.insert(0, str(SRC))

            from dynamic_factor_copula import (
                compare_rebalanced_portfolio,
                compute_port_opt_style_metrics,
                curve_from_returns,
                default_paths,
                lag_close_signal_to_next_session,
                load_cached_market_data,
                load_overlay_compare_prices,
            )

            paths = default_paths(ROOT)
            paths.result_dir.mkdir(parents=True, exist_ok=True)

            START_DATE = "2016-01-01"
            END_DATE = "2026-04-29"
            RISK_FREE_RATE = 0.03
            INITIAL_VALUE = 10_000.0

            def metrics_from_returns(returns: pd.Series, label: str) -> dict:
                curve = curve_from_returns(returns.fillna(0.0), initial=INITIAL_VALUE)
                row = compute_port_opt_style_metrics(curve, risk_free_rate=RISK_FREE_RATE).to_dict()
                row["Strategy"] = label
                row["Start"] = curve.dropna().index.min().date().isoformat()
                row["End"] = curve.dropna().index.max().date().isoformat()
                return row

            def trend_exposure(price: pd.Series, ma_period: int, below_exposure: float) -> pd.Series:
                price = price.astype(float).sort_index().ffill()
                min_periods = max(20, int(ma_period * 0.20))
                ma = price.rolling(ma_period, min_periods=min_periods).mean()
                signal = pd.Series(1.0, index=price.index, dtype=float)
                signal.loc[price < ma] = below_exposure
                signal.loc[ma.isna()] = 1.0
                return lag_close_signal_to_next_session(signal, initial=1.0)

            def apply_exposure(price: pd.Series, ma_period: int, below_exposure: float) -> pd.DataFrame:
                price = price.astype(float).sort_index().ffill()
                returns = price.pct_change(fill_method=None).fillna(0.0)
                exposure = trend_exposure(price, ma_period=ma_period, below_exposure=below_exposure)
                overlay_returns = returns.mul(exposure.reindex(returns.index).ffill().fillna(1.0))
                return pd.DataFrame(
                    {
                        "Price": price,
                        "Raw Return": returns,
                        "Daily Exposure": exposure,
                        "Overlay Return": overlay_returns,
                        "Overlay Growth": curve_from_returns(overlay_returns, initial=INITIAL_VALUE),
                    }
                )

            def drawdown_exposure(
                price: pd.Series,
                warn_drawdown: float = -0.08,
                crash_drawdown: float = -0.15,
                warn_exposure: float = 0.50,
                crash_exposure: float = 0.25,
            ) -> pd.Series:
                price = price.astype(float).sort_index().ffill()
                drawdown = price / price.cummax() - 1.0
                signal = pd.Series(1.0, index=price.index, dtype=float)
                signal.loc[drawdown <= warn_drawdown] = warn_exposure
                signal.loc[drawdown <= crash_drawdown] = crash_exposure
                return lag_close_signal_to_next_session(signal, initial=1.0)

            def apply_drawdown_exposure(
                price: pd.Series,
                warn_drawdown: float,
                crash_drawdown: float,
                warn_exposure: float,
                crash_exposure: float,
            ) -> pd.DataFrame:
                price = price.astype(float).sort_index().ffill()
                returns = price.pct_change(fill_method=None).fillna(0.0)
                exposure = drawdown_exposure(
                    price,
                    warn_drawdown=warn_drawdown,
                    crash_drawdown=crash_drawdown,
                    warn_exposure=warn_exposure,
                    crash_exposure=crash_exposure,
                )
                overlay_returns = returns.mul(exposure.reindex(returns.index).ffill().fillna(1.0))
                return pd.DataFrame(
                    {
                        "Price": price,
                        "Drawdown": price / price.cummax() - 1.0,
                        "Raw Return": returns,
                        "Daily Exposure": exposure,
                        "Overlay Return": overlay_returns,
                        "Overlay Growth": curve_from_returns(overlay_returns, initial=INITIAL_VALUE),
                    }
                )

            BASE_ALLOCATION_ASSETS = ["SPY", "Gold", "BTC", "BIL", "IEF", "VXUS", "TIP"]
            MANAGED_FUTURES_ASSETS = ["DBMF", "KMLM"]
            EXTENDED_ALLOCATION_ASSETS = [
                "SPY",
                "DBMF",
                "KMLM",
                "CTA",
                "WTMF",
                "BIL",
                "SGOV",
                "TBIL",
                "BTAL",
                "SH",
                "PSQ",
                "XLP",
                "XLU",
                "XLV",
                "GLD",
                "VIXY",
                "BTC",
                "ETH",
            ]
            ALLOCATION_ASSETS = BASE_ALLOCATION_ASSETS.copy()
            ALLOCATION_COLUMNS = {
                "SPY": "SPY",
                "Gold": "Gold",
                "BTC": "BTC",
                "BIL": "BIL",
                "IEF": "IEF",
                "VXUS": "VXUS",
                "TIP": "TIP",
                "DBMF": "DBMF",
                "KMLM": "KMLM",
            }
            MAX_WEIGHT = {
                "SPY": 0.70,
                "Gold": 0.40,
                "BTC": 0.10,
                "BIL": 0.50,
                "IEF": 0.30,
                "VXUS": 0.40,
                "TIP": 0.30,
                "DBMF": 0.30,
                "KMLM": 0.30,
                "CTA": 0.30,
                "WTMF": 0.30,
                "SGOV": 0.60,
                "TBIL": 0.60,
                "BTAL": 0.20,
                "SH": 0.15,
                "PSQ": 0.15,
                "XLP": 0.30,
                "XLU": 0.30,
                "XLV": 0.30,
                "GLD": 0.40,
                "VIXY": 0.05,
                "ETH": 0.05,
            }

            def active_weight_items(row: pd.Series, assets: list[str], min_weight: float = 0.005) -> list[tuple[str, float]]:
                items = []
                for asset in assets:
                    weight_col = f"{asset} Weight"
                    if weight_col not in row:
                        continue
                    weight = float(row[weight_col])
                    if weight > min_weight:
                        items.append((asset, weight))
                return items

            def format_weight_strategy(prefix: str, row: pd.Series, assets: list[str]) -> str:
                items = active_weight_items(row, assets)
                if not items:
                    return prefix
                asset_label = "/".join(asset for asset, _weight in items)
                weight_label = "/".join(str(int(round(weight * 100))) for _asset, weight in items)
                return f"{prefix} {asset_label} {weight_label}"

            def format_weight_config(row: pd.Series, assets: list[str]) -> str:
                items = active_weight_items(row, assets)
                if not items:
                    return "No active allocation weights."
                return ", ".join(f"{asset}={weight:.0%}" for asset, weight in items)

            def display_full_table(df: pd.DataFrame, caption: str | None = None):
                caption_html = f"<caption>{caption}</caption>" if caption else ""
                html = df.to_html(index=True, escape=True)
                html = html.replace("<table", f"<table>{caption_html}", 1)
                return display(
                    HTML(
                        f'''
                        <style>
                        .full-width-table {{
                            max-width: 100%;
                            overflow-x: auto;
                            margin: 8px 0 18px 0;
                        }}
                        .full-width-table table {{
                            border-collapse: collapse;
                            width: max-content;
                            min-width: 100%;
                            font-size: 13px;
                            line-height: 1.35;
                        }}
                        .full-width-table caption {{
                            caption-side: top;
                            text-align: left;
                            font-weight: 700;
                            padding: 6px 0;
                        }}
                        .full-width-table th,
                        .full-width-table td {{
                            border: 1px solid #d9dee7;
                            padding: 7px 9px;
                            vertical-align: top;
                            white-space: normal;
                        }}
                        .full-width-table th {{
                            background: #f5f7fb;
                        }}
                        .full-width-table td:nth-child(4),
                        .full-width-table td:nth-child(5) {{
                            min-width: 300px;
                            max-width: 560px;
                        }}
                        </style>
                        <div class="full-width-table">{html}</div>
                        '''
                    )
                )

            def generate_capped_weight_grid(max_weight: dict[str, float], step: float = 0.05) -> list[pd.Series]:
                assets = list(max_weight)
                units_total = int(round(1.0 / step))
                cap_units = {asset: int(round(cap / step)) for asset, cap in max_weight.items()}
                rows = []

                def walk(idx: int, remaining: int, current: dict[str, int]) -> None:
                    asset = assets[idx]
                    if idx == len(assets) - 1:
                        if 0 <= remaining <= cap_units[asset]:
                            row = current.copy()
                            row[asset] = remaining
                            rows.append(pd.Series({name: units * step for name, units in row.items()}, dtype=float))
                        return
                    for units in range(min(cap_units[asset], remaining) + 1):
                        current[asset] = units
                        walk(idx + 1, remaining - units, current)
                    current.pop(asset, None)

                walk(0, units_total, {})
                return rows

            def run_allocation_sweep(
                returns: pd.DataFrame,
                strategy_prefix: str,
                output_prefix: str,
                allocation_assets: list[str] | None = None,
                max_weight: dict[str, float] | None = None,
                step: float = 0.05,
                batch_size: int = 1_000,
            ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
                allocation_assets = allocation_assets or ALLOCATION_ASSETS
                max_weight = max_weight or {asset: MAX_WEIGHT[asset] for asset in allocation_assets}
                returns = returns.reindex(columns=allocation_assets).dropna()
                weight_rows = generate_capped_weight_grid(max_weight, step=step)
                weight_grid = pd.DataFrame(weight_rows).reindex(columns=allocation_assets).astype(float)

                def monthly_returns_for_grid(grid: pd.DataFrame) -> pd.DataFrame:
                    weights_np = grid.to_numpy()
                    portfolio_blocks = []
                    for _period, block in returns.groupby(returns.index.to_period("M")):
                        block = block.reindex(columns=allocation_assets).fillna(0.0)
                        asset_growth = (1.0 + block).cumprod().to_numpy()
                        values = asset_growth @ weights_np.T
                        previous = np.vstack([np.ones((1, values.shape[1])), values[:-1]])
                        block_returns = values / np.maximum(previous, 1e-12) - 1.0
                        portfolio_blocks.append(
                            pd.DataFrame(block_returns, index=block.index, columns=grid.index)
                        )
                    return pd.concat(portfolio_blocks, axis=0).sort_index()

                def streaming_metrics(grid: pd.DataFrame) -> pd.DataFrame:
                    weights_np = grid.to_numpy()
                    n_cols = len(grid)
                    nav = np.full(n_cols, INITIAL_VALUE, dtype=float)
                    first_nav = np.full(n_cols, np.nan, dtype=float)
                    peak = np.full(n_cols, INITIAL_VALUE, dtype=float)
                    max_drawdown = np.zeros(n_cols, dtype=float)
                    sum_ret = np.zeros(n_cols, dtype=float)
                    sumsq_ret = np.zeros(n_cols, dtype=float)
                    neg_count = np.zeros(n_cols, dtype=float)
                    neg_sum = np.zeros(n_cols, dtype=float)
                    neg_sumsq = np.zeros(n_cols, dtype=float)
                    hit_count = np.zeros(n_cols, dtype=float)
                    observed_returns = 0
                    nav_points = 0

                    for _period, block in returns.groupby(returns.index.to_period("M")):
                        block = block.reindex(columns=allocation_assets).fillna(0.0)
                        asset_growth = (1.0 + block).cumprod().to_numpy()
                        values = asset_growth @ weights_np.T
                        previous = np.vstack([np.ones((1, values.shape[1])), values[:-1]])
                        block_returns = values / np.maximum(previous, 1e-12) - 1.0
                        for day_returns in block_returns:
                            nav *= 1.0 + day_returns
                            if nav_points == 0:
                                first_nav[:] = nav
                            else:
                                sum_ret += day_returns
                                sumsq_ret += day_returns * day_returns
                                hit_count += day_returns > 0.0
                                neg_mask = day_returns < 0.0
                                neg_count += neg_mask
                                neg_sum += np.where(neg_mask, day_returns, 0.0)
                                neg_sumsq += np.where(neg_mask, day_returns * day_returns, 0.0)
                                observed_returns += 1
                            peak = np.maximum(peak, nav)
                            max_drawdown = np.minimum(max_drawdown, nav / np.maximum(peak, 1e-12) - 1.0)
                            nav_points += 1

                    years = max(nav_points / 252.0, 1.0 / 252.0)
                    count = max(observed_returns, 1)
                    mean_return = sum_ret / count
                    variance = np.maximum(sumsq_ret / count - mean_return * mean_return, 0.0)
                    annual_vol = np.sqrt(variance) * np.sqrt(252.0)
                    sharpe = ((mean_return * 252.0) - RISK_FREE_RATE) / np.maximum(annual_vol, 1e-8)
                    neg_mean = np.divide(neg_sum, neg_count, out=np.zeros_like(neg_sum), where=neg_count > 0)
                    neg_var = np.divide(neg_sumsq, neg_count, out=np.zeros_like(neg_sumsq), where=neg_count > 0) - neg_mean * neg_mean
                    downside = np.sqrt(np.maximum(neg_var, 0.0)) * np.sqrt(252.0)
                    sortino = ((mean_return * 252.0) - RISK_FREE_RATE) / np.maximum(downside, 1e-8)
                    total_return = nav / np.maximum(first_nav, 1e-12) - 1.0
                    cagr = np.power(nav, 1.0 / years) / np.power(np.maximum(first_nav, 1e-12), 1.0 / years) - 1.0
                    hit_rate = hit_count / count

                    metric_rows = pd.DataFrame(
                        {
                            "Weight Index": grid.index,
                            "Total Return": total_return,
                            "CAGR": cagr,
                            "Annual Vol": annual_vol,
                            "Sharpe": sharpe,
                            "Sortino": sortino,
                            "Max Drawdown": max_drawdown,
                            "Hit Rate": hit_rate,
                            "Start": returns.index.min().date().isoformat(),
                            "End": returns.index.max().date().isoformat(),
                        }
                    )
                    for asset in allocation_assets:
                        metric_rows[f"{asset} Weight"] = grid[asset].to_numpy()
                    return metric_rows

                rows = []
                for start in range(0, len(weight_grid), batch_size):
                    grid_chunk = weight_grid.iloc[start : start + batch_size]
                    rows.append(streaming_metrics(grid_chunk))

                sweep = pd.concat(rows, ignore_index=True)
                labels = []
                for _, row in sweep.iterrows():
                    weights_label = "/".join(
                        str(int(round(row[f"{asset} Weight"] * 100))) for asset in allocation_assets
                    )
                    labels.append(f"{strategy_prefix} {weights_label}")
                sweep.insert(0, "Strategy", labels)
                sweep = sweep.sort_values("Sharpe", ascending=False)
                sweep.to_csv(paths.result_dir / f"{output_prefix}_allocation_sweep.csv", index=False)
                best_sharpe = sweep.sort_values("Sharpe", ascending=False).head(10)
                best_drawdown = sweep.sort_values("Max Drawdown", ascending=False).head(10)
                labels = list(dict.fromkeys([best_sharpe.iloc[0]["Strategy"], best_drawdown.iloc[0]["Strategy"]]))
                winner_indexes = [
                    int(sweep.loc[sweep["Strategy"].eq(label), "Weight Index"].iloc[0])
                    for label in labels
                ]
                winner_returns = monthly_returns_for_grid(weight_grid.loc[winner_indexes])
                curves = {
                    label: curve_from_returns(winner_returns[idx], initial=INITIAL_VALUE)
                    for label, idx in zip(labels, winner_indexes)
                }
                pd.DataFrame({label: curves[label] for label in labels}).to_csv(
                    paths.result_dir / f"{output_prefix}_best_curves.csv"
                )
                return best_sharpe, best_drawdown, curves

            def generate_random_capped_weight_grid(
                allocation_assets: list[str],
                max_weight: dict[str, float],
                n_candidates: int,
                rng: np.random.Generator,
            ) -> pd.DataFrame:
                caps = np.array([max_weight[asset] for asset in allocation_assets], dtype=float)
                if caps.sum() < 1.0:
                    raise ValueError("Max weights must allow at least 100% allocation.")

                rows = np.empty((n_candidates, len(allocation_assets)), dtype=float)
                alpha = np.full(len(allocation_assets), 0.65, dtype=float)
                for idx in range(n_candidates):
                    weights = rng.dirichlet(alpha)
                    for _ in range(10):
                        over = weights > caps
                        if not over.any():
                            break
                        excess = float((weights[over] - caps[over]).sum())
                        weights[over] = caps[over]
                        room = np.maximum(caps - weights, 0.0)
                        room[over] = 0.0
                        room_total = float(room.sum())
                        if room_total <= 1e-12:
                            break
                        weights += excess * room / room_total
                    weights = np.minimum(weights, caps)
                    weights = weights / max(float(weights.sum()), 1e-12)
                    rows[idx] = weights

                return pd.DataFrame(rows, columns=allocation_assets)

            def run_memory_aware_allocation_search(
                returns: pd.DataFrame,
                strategy_prefix: str,
                output_prefix: str,
                allocation_assets: list[str],
                max_weight: dict[str, float],
                n_candidates: int = 60_000,
                batch_size: int = 1_000,
                keep_top: int = 250,
                seed: int = 42,
            ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
                returns = returns.reindex(columns=allocation_assets).dropna()
                rng = np.random.default_rng(seed)

                def monthly_returns_for_grid(grid: pd.DataFrame) -> pd.DataFrame:
                    weights_np = grid.to_numpy()
                    portfolio_blocks = []
                    for _period, block in returns.groupby(returns.index.to_period("M")):
                        block = block.reindex(columns=allocation_assets).fillna(0.0)
                        asset_growth = (1.0 + block).cumprod().to_numpy()
                        values = asset_growth @ weights_np.T
                        previous = np.vstack([np.ones((1, values.shape[1])), values[:-1]])
                        block_returns = values / np.maximum(previous, 1e-12) - 1.0
                        portfolio_blocks.append(pd.DataFrame(block_returns, index=block.index, columns=grid.index))
                    return pd.concat(portfolio_blocks, axis=0).sort_index()

                def streaming_metrics(grid: pd.DataFrame) -> pd.DataFrame:
                    weights_np = grid.to_numpy()
                    n_cols = len(grid)
                    nav = np.full(n_cols, INITIAL_VALUE, dtype=float)
                    first_nav = np.full(n_cols, np.nan, dtype=float)
                    peak = np.full(n_cols, INITIAL_VALUE, dtype=float)
                    max_drawdown = np.zeros(n_cols, dtype=float)
                    sum_ret = np.zeros(n_cols, dtype=float)
                    sumsq_ret = np.zeros(n_cols, dtype=float)
                    neg_count = np.zeros(n_cols, dtype=float)
                    neg_sum = np.zeros(n_cols, dtype=float)
                    neg_sumsq = np.zeros(n_cols, dtype=float)
                    hit_count = np.zeros(n_cols, dtype=float)
                    observed_returns = 0
                    nav_points = 0

                    for _period, block in returns.groupby(returns.index.to_period("M")):
                        block = block.reindex(columns=allocation_assets).fillna(0.0)
                        asset_growth = (1.0 + block).cumprod().to_numpy()
                        values = asset_growth @ weights_np.T
                        previous = np.vstack([np.ones((1, values.shape[1])), values[:-1]])
                        block_returns = values / np.maximum(previous, 1e-12) - 1.0
                        for day_returns in block_returns:
                            nav *= 1.0 + day_returns
                            if nav_points == 0:
                                first_nav[:] = nav
                            else:
                                sum_ret += day_returns
                                sumsq_ret += day_returns * day_returns
                                hit_count += day_returns > 0.0
                                neg_mask = day_returns < 0.0
                                neg_count += neg_mask
                                neg_sum += np.where(neg_mask, day_returns, 0.0)
                                neg_sumsq += np.where(neg_mask, day_returns * day_returns, 0.0)
                                observed_returns += 1
                            peak = np.maximum(peak, nav)
                            max_drawdown = np.minimum(max_drawdown, nav / np.maximum(peak, 1e-12) - 1.0)
                            nav_points += 1

                    years = max(nav_points / 252.0, 1.0 / 252.0)
                    count = max(observed_returns, 1)
                    mean_return = sum_ret / count
                    variance = np.maximum(sumsq_ret / count - mean_return * mean_return, 0.0)
                    annual_vol = np.sqrt(variance) * np.sqrt(252.0)
                    sharpe = ((mean_return * 252.0) - RISK_FREE_RATE) / np.maximum(annual_vol, 1e-8)
                    neg_mean = np.divide(neg_sum, neg_count, out=np.zeros_like(neg_sum), where=neg_count > 0)
                    neg_var = np.divide(neg_sumsq, neg_count, out=np.zeros_like(neg_sumsq), where=neg_count > 0) - neg_mean * neg_mean
                    downside = np.sqrt(np.maximum(neg_var, 0.0)) * np.sqrt(252.0)
                    sortino = ((mean_return * 252.0) - RISK_FREE_RATE) / np.maximum(downside, 1e-8)
                    total_return = nav / np.maximum(first_nav, 1e-12) - 1.0
                    cagr = np.power(nav, 1.0 / years) / np.power(np.maximum(first_nav, 1e-12), 1.0 / years) - 1.0
                    hit_rate = hit_count / count

                    metric_rows = pd.DataFrame(
                        {
                            "Weight Index": grid.index,
                            "Total Return": total_return,
                            "CAGR": cagr,
                            "Annual Vol": annual_vol,
                            "Sharpe": sharpe,
                            "Sortino": sortino,
                            "Max Drawdown": max_drawdown,
                            "Hit Rate": hit_rate,
                            "Start": returns.index.min().date().isoformat(),
                            "End": returns.index.max().date().isoformat(),
                        }
                    )
                    for asset in allocation_assets:
                        metric_rows[f"{asset} Weight"] = grid[asset].to_numpy()
                    return metric_rows

                top_sharpe = pd.DataFrame()
                top_drawdown = pd.DataFrame()
                for start in range(0, n_candidates, batch_size):
                    size = min(batch_size, n_candidates - start)
                    grid_chunk = generate_random_capped_weight_grid(allocation_assets, max_weight, size, rng)
                    grid_chunk.index = range(start, start + size)
                    metrics = streaming_metrics(grid_chunk)
                    labels = []
                    for _, row in metrics.iterrows():
                        labels.append(format_weight_strategy(strategy_prefix, row, allocation_assets))
                    metrics.insert(0, "Strategy", labels)

                    top_sharpe = (
                        pd.concat([top_sharpe, metrics], ignore_index=True)
                        .sort_values("Sharpe", ascending=False)
                        .head(keep_top)
                    )
                    top_drawdown = (
                        pd.concat([top_drawdown, metrics], ignore_index=True)
                        .sort_values("Max Drawdown", ascending=False)
                        .head(keep_top)
                    )

                top_sharpe.to_csv(paths.result_dir / f"{output_prefix}_top_sharpe.csv", index=False)
                top_drawdown.to_csv(paths.result_dir / f"{output_prefix}_top_drawdown.csv", index=False)
                pd.concat(
                    [
                        top_sharpe.head(25).assign(Winner_Type="Best Sharpe"),
                        top_drawdown.head(25).assign(Winner_Type="Best Drawdown"),
                    ],
                    ignore_index=True,
                ).to_csv(paths.result_dir / f"{output_prefix}_summary.csv", index=False)

                winners = list(
                    dict.fromkeys(
                        [
                            top_sharpe.iloc[0]["Strategy"],
                            top_drawdown.iloc[0]["Strategy"],
                        ]
                    )
                )
                winner_rows = []
                for label in winners:
                    source = pd.concat([top_sharpe, top_drawdown], ignore_index=True)
                    row = source.loc[source["Strategy"].eq(label)].iloc[0]
                    winner_rows.append({asset: float(row[f"{asset} Weight"]) for asset in allocation_assets})
                winner_grid = pd.DataFrame(winner_rows, index=winners).reindex(columns=allocation_assets)
                winner_returns = monthly_returns_for_grid(winner_grid)
                curves = {
                    label: curve_from_returns(winner_returns[label], initial=INITIAL_VALUE)
                    for label in winners
                }
                pd.DataFrame({label: curves[label] for label in winners}).to_csv(
                    paths.result_dir / f"{output_prefix}_best_curves.csv"
                )
                return top_sharpe.head(10), top_drawdown.head(10), curves
            """
        ),
        md_cell(
            """
            ## Step 1 - S&P Buy And Hold Performance

            Loads the cached benchmark series from `port_opt_advance` and computes S&P buy-and-hold metrics over the overlay test window.
            """
        ),
        code_cell(
            """
            cached = load_cached_market_data(paths)
            spx = cached["benchmark"]["benchmark"].sort_index().loc[START_DATE:END_DATE].ffill()

            spx_returns = spx.pct_change(fill_method=None).fillna(0.0)
            spx_curve = curve_from_returns(spx_returns, initial=INITIAL_VALUE).rename("S&P 500 Buy Hold")
            spx_metrics = pd.DataFrame([metrics_from_returns(spx_returns, "S&P 500 buy and hold")]).set_index("Strategy")

            spx_curve.to_frame().to_csv(paths.result_dir / "best_param_step1_sp500_buy_hold_curve.csv")
            spx_metrics.to_csv(paths.result_dir / "best_param_step1_sp500_buy_hold_metrics.csv")
            spx_metrics
            """
        ),
        md_cell(
            """
            ## Step 2 - Multi-Asset Best Allocation

            This step uses buy-and-hold asset returns with monthly strategic rebalance. No daily exposure is applied here.

            Base asset universe:

            - `SPY`
            - `Gold`
            - `BTC`
            - `BIL`
            - `IEF`
            - `VXUS`
            - `TIP`

            Managed futures is tested in Step 2B using the best Step 2 allocation as one `Core` sleeve, because `DBMF` and `KMLM` have shorter and different available histories.

            Max weights:

            - `SPY`: 70%
            - `Gold`: 40%
            - `BTC`: 10%
            - `BIL`: 50%
            - `IEF`: 30%
            - `VXUS`: 40%
            - `TIP`: 30%
            The grid uses 5% increments and weights must sum to 100%.

            The notebook reports two winners: best Sharpe and best max drawdown.
            """
        ),
        code_cell(
            """
            overlay_prices = load_overlay_compare_prices(
                paths,
                start_date=START_DATE,
                end_date=END_DATE,
                tickers=[
                    "SPY",
                    "GC=F",
                    "BTC-USD",
                    "BIL",
                    "IEF",
                    "VXUS",
                    "TIP",
                    "DBMF",
                    "KMLM",
                    "CTA",
                    "WTMF",
                    "SGOV",
                    "TBIL",
                    "BTAL",
                    "SH",
                    "PSQ",
                    "XLP",
                    "XLU",
                    "XLV",
                    "GLD",
                    "VIXY",
                    "ETH-USD",
                ],
            )

            allocation_prices = overlay_prices.rename(
                columns={
                    "SPY": "SPY",
                    "GC=F": "Gold",
                    "BTC-USD": "BTC",
                    "BIL": "BIL",
                    "IEF": "IEF",
                    "VXUS": "VXUS",
                    "TIP": "TIP",
                    "DBMF": "DBMF",
                    "KMLM": "KMLM",
                    "CTA": "CTA",
                    "WTMF": "WTMF",
                    "SGOV": "SGOV",
                    "TBIL": "TBIL",
                    "BTAL": "BTAL",
                    "SH": "SH",
                    "PSQ": "PSQ",
                    "XLP": "XLP",
                    "XLU": "XLU",
                    "XLV": "XLV",
                    "GLD": "GLD",
                    "VIXY": "VIXY",
                    "ETH-USD": "ETH",
                }
            )
            asset_returns = allocation_prices.pct_change(fill_method=None).where(allocation_prices.notna())

            best_allocation_sharpe, best_allocation_drawdown, allocation_curves = run_allocation_sweep(
                asset_returns[BASE_ALLOCATION_ASSETS],
                strategy_prefix="SPY/Gold/BTC/BIL/IEF/VXUS/TIP",
                output_prefix="best_param_step2_multi_asset",
                allocation_assets=BASE_ALLOCATION_ASSETS,
                max_weight={asset: MAX_WEIGHT[asset] for asset in BASE_ALLOCATION_ASSETS},
                step=0.05,
            )

            display(best_allocation_sharpe)
            display(best_allocation_drawdown)
            """
        ),
        md_cell(
            """
            ## Step 2C - Extended Defensive Asset Allocation

            This section adds a wider defensive/hedge universe alongside `SPY`:

            - Managed futures / trend following: `DBMF`, `KMLM`, `CTA`, `WTMF`
            - Cash / T-Bill: `BIL`, `SGOV`, `TBIL`
            - Market-neutral / anti-beta: `BTAL`
            - Inverse equity hedge: `SH`, `PSQ`
            - Defensive sectors: `XLP`, `XLU`, `XLV`
            - Gold ETF: `GLD`
            - Volatility ETF: `VIXY`
            - Crypto: `BTC`, `ETH`

            Because this universe is too large for an exhaustive 5% grid, it uses a memory-aware random capped search. Candidate portfolios are evaluated in batches and the notebook only keeps the top-ranked rows by Sharpe and drawdown, plus the best port-growth curves.
            """
        ),
        code_cell(
            """
            EXTENDED_MIN_DATA_YEARS = 10.0
            extended_asset_coverage = []
            for asset in EXTENDED_ALLOCATION_ASSETS:
                if asset not in allocation_prices.columns:
                    extended_asset_coverage.append(
                        {"Asset": asset, "First Date": None, "Last Date": None, "Years": 0.0, "Included": False, "Reason": "missing"}
                    )
                    continue
                valid_price = allocation_prices[asset].dropna()
                if valid_price.empty:
                    extended_asset_coverage.append(
                        {"Asset": asset, "First Date": None, "Last Date": None, "Years": 0.0, "Included": False, "Reason": "no data"}
                    )
                    continue
                years = (valid_price.index.max() - valid_price.index.min()).days / 365.25
                included = years >= EXTENDED_MIN_DATA_YEARS
                extended_asset_coverage.append(
                    {
                        "Asset": asset,
                        "First Date": valid_price.index.min().date().isoformat(),
                        "Last Date": valid_price.index.max().date().isoformat(),
                        "Years": years,
                        "Included": included,
                        "Reason": "included" if included else f"less than {EXTENDED_MIN_DATA_YEARS:.0f} years",
                    }
                )

            extended_asset_coverage = pd.DataFrame(extended_asset_coverage)
            extended_asset_coverage.to_csv(
                paths.result_dir / "best_param_step2c_extended_defensive_asset_coverage.csv",
                index=False,
            )
            available_extended_assets = extended_asset_coverage.loc[
                extended_asset_coverage["Included"], "Asset"
            ].tolist()
            EXTENDED_MAX_WEIGHT = {asset: MAX_WEIGHT[asset] for asset in available_extended_assets}
            EXTENDED_SEARCH_CANDIDATES = 60_000
            EXTENDED_SEARCH_BATCH_SIZE = 1_000
            EXTENDED_SEARCH_KEEP_TOP = 250

            extended_returns = asset_returns[available_extended_assets].dropna()
            best_extended_sharpe, best_extended_drawdown, extended_allocation_curves = run_memory_aware_allocation_search(
                extended_returns,
                strategy_prefix="Extended defensive allocation",
                output_prefix="best_param_step2c_extended_defensive_allocation",
                allocation_assets=available_extended_assets,
                max_weight=EXTENDED_MAX_WEIGHT,
                n_candidates=EXTENDED_SEARCH_CANDIDATES,
                batch_size=EXTENDED_SEARCH_BATCH_SIZE,
                keep_top=EXTENDED_SEARCH_KEEP_TOP,
                seed=20260607,
            )

            extended_search_settings = pd.DataFrame(
                [
                    {"Field": "Candidates", "Value": EXTENDED_SEARCH_CANDIDATES},
                    {"Field": "Batch Size", "Value": EXTENDED_SEARCH_BATCH_SIZE},
                    {"Field": "Top Rows Kept", "Value": EXTENDED_SEARCH_KEEP_TOP},
                    {"Field": "Seed", "Value": 20260607},
                    {"Field": "Min Data Years", "Value": EXTENDED_MIN_DATA_YEARS},
                    {"Field": "Assets Used", "Value": ", ".join(available_extended_assets)},
                    {
                        "Field": "Assets Excluded",
                        "Value": ", ".join(
                            extended_asset_coverage.loc[
                                ~extended_asset_coverage["Included"], "Asset"
                            ].tolist()
                        ),
                    },
                    {"Field": "Available Start", "Value": extended_returns.index.min().date().isoformat()},
                    {"Field": "Available End", "Value": extended_returns.index.max().date().isoformat()},
                ]
                + [{"Field": f"{asset} Max Weight", "Value": EXTENDED_MAX_WEIGHT[asset]} for asset in available_extended_assets]
            )
            extended_search_settings.to_csv(
                paths.result_dir / "best_param_step2c_extended_defensive_allocation_settings.csv",
                index=False,
            )

            display_full_table(extended_asset_coverage, "Step 2C asset data coverage")
            display_full_table(extended_search_settings, "Step 2C search settings")
            display_full_table(best_extended_sharpe, "Step 2C best Sharpe")
            display_full_table(best_extended_drawdown, "Step 2C best drawdown")
            """
        ),
        md_cell(
            """
            ## Step 2D - S&P vs Base Allocation vs Extended Allocation

            Compares port growth and metrics for:

            1. `S&P 500 buy and hold`
            2. Best Step 2 allocation across `SPY`, Gold, BTC, and T-Bills/bonds
            3. Best Step 2C extended defensive allocation

            The table intentionally keeps all metric and weight columns visible.
            """
        ),
        code_cell(
            """
            base_allocation_label = best_allocation_sharpe.iloc[0]["Strategy"]
            extended_allocation_label = best_extended_sharpe.iloc[0]["Strategy"]

            step2d_curves = pd.concat(
                [
                    spx_curve.rename("1. S&P 500 buy and hold"),
                    allocation_curves[base_allocation_label].rename(
                        f"2. Base allocation - {format_weight_strategy('Monthly allocation', best_allocation_sharpe.iloc[0], BASE_ALLOCATION_ASSETS)}"
                    ),
                    extended_allocation_curves[extended_allocation_label].rename(
                        f"3. Extended allocation - {format_weight_strategy('Extended defensive allocation', best_extended_sharpe.iloc[0], available_extended_assets)}"
                    ),
                ],
                axis=1,
            ).sort_index()
            step2d_curves.to_csv(paths.result_dir / "best_param_step2d_sp500_base_vs_extended_curves.csv")

            metric_columns = ["Strategy", "CAGR", "Annual Vol", "Sharpe", "Sortino", "Max Drawdown", "Hit Rate", "Start", "End"]
            weight_columns = sorted(
                {
                    column
                    for source in [best_allocation_sharpe, best_extended_sharpe]
                    for column in source.columns
                    if column.endswith(" Weight")
                }
            )
            step2d_metric_rows = [
                spx_metrics.reset_index().iloc[0].to_dict(),
                best_allocation_sharpe.iloc[0].to_dict(),
                best_extended_sharpe.iloc[0].to_dict(),
            ]
            step2d_metrics = pd.DataFrame(step2d_metric_rows)
            step2d_metrics["Strategy"] = [
                "1. S&P 500 buy and hold",
                f"2. {format_weight_strategy('Monthly allocation', best_allocation_sharpe.iloc[0], BASE_ALLOCATION_ASSETS)}",
                f"3. {format_weight_strategy('Extended defensive allocation', best_extended_sharpe.iloc[0], available_extended_assets)}",
            ]
            step2d_metrics = step2d_metrics.reindex(columns=metric_columns + weight_columns)
            step2d_metrics.to_csv(paths.result_dir / "best_param_step2d_sp500_base_vs_extended_metrics.csv", index=False)

            step2d_fig = go.Figure()
            for column in step2d_curves.columns:
                step2d_fig.add_trace(
                    go.Scatter(
                        x=step2d_curves.index,
                        y=step2d_curves[column],
                        mode="lines",
                        name=column,
                        connectgaps=False,
                        hovertemplate="%{x|%Y-%m-%d}<br>Portfolio growth: %{y:,.0f}<extra></extra>",
                    )
                )
            step2d_fig.update_layout(
                title="Port Growth - S&P Buy Hold vs Base Allocation vs Extended Allocation",
                template="plotly_white",
                height=560,
                yaxis_title="Portfolio growth",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
                margin=dict(l=60, r=30, t=80, b=155),
            )
            step2d_fig.write_html(
                paths.result_dir / "best_param_step2d_sp500_base_vs_extended_chart.html",
                include_plotlyjs="cdn",
            )

            display_full_table(step2d_metrics, "S&P vs base allocation vs extended allocation metrics")
            step2d_fig.show()
            """
        ),
        md_cell(
            """
            ## Step 2B - Managed Futures Allocation Windows

            This section takes the best Sharpe allocation from Step 2 and treats it as one monthly rebalanced `Core` sleeve.

            Then it tests managed futures one at a time:

            - `Core` + `DBMF`
            - `Core` + `KMLM`

            Each family uses its own available overlap window after dropping rows where the managed futures ETF has no price. This avoids forcing the long-history base allocation to start only when the shortest ETF begins.
            """
        ),
        code_cell(
            """
            core_weight_row = best_allocation_sharpe.iloc[0]
            core_weights = pd.Series(
                {asset: float(core_weight_row[f"{asset} Weight"]) for asset in BASE_ALLOCATION_ASSETS},
                dtype=float,
                name="Core Weight",
            )
            core_returns = compare_rebalanced_portfolio(
                asset_returns[BASE_ALLOCATION_ASSETS].dropna(),
                core_weights,
                rebalance_months=1,
            ).rename("Core")

            core_config = pd.DataFrame(
                [{"Field": f"{asset} Weight", "Value": float(core_weights[asset])} for asset in BASE_ALLOCATION_ASSETS]
                + [
                    {"Field": "Source Strategy", "Value": core_weight_row["Strategy"]},
                    {"Field": "Source Sharpe", "Value": float(core_weight_row["Sharpe"])},
                    {"Field": "Source CAGR", "Value": float(core_weight_row["CAGR"])},
                    {"Field": "Source Max Drawdown", "Value": float(core_weight_row["Max Drawdown"])},
                ]
            )
            core_config.to_csv(paths.result_dir / "best_param_step2b_core_config.csv")

            managed_futures_results = []
            managed_futures_best_curves = {}

            for mf_asset in MANAGED_FUTURES_ASSETS:
                mf_assets = ["Core", mf_asset]
                mf_returns = pd.concat(
                    [
                        core_returns,
                        asset_returns[mf_asset].rename(mf_asset),
                    ],
                    axis=1,
                ).dropna()
                if mf_returns.empty:
                    continue
                best_mf_sharpe, best_mf_drawdown, mf_curves = run_allocation_sweep(
                    mf_returns,
                    strategy_prefix=f"Step2Core/{mf_asset}",
                    output_prefix=f"best_param_step2b_core_{mf_asset.lower()}",
                    allocation_assets=mf_assets,
                    max_weight={"Core": 1.00, mf_asset: MAX_WEIGHT[mf_asset]},
                    step=0.05,
                )
                best_mf_sharpe = best_mf_sharpe.copy()
                best_mf_drawdown = best_mf_drawdown.copy()
                best_mf_sharpe["Family"] = f"Base + {mf_asset}"
                best_mf_sharpe["Winner Type"] = "Best Sharpe"
                best_mf_drawdown["Family"] = f"Base + {mf_asset}"
                best_mf_drawdown["Winner Type"] = "Best Drawdown"
                managed_futures_results.extend([best_mf_sharpe.head(5), best_mf_drawdown.head(5)])
                managed_futures_best_curves.update(mf_curves)

            managed_futures_summary = (
                pd.concat(managed_futures_results, ignore_index=True)
                if managed_futures_results
                else pd.DataFrame()
            )
            if not managed_futures_summary.empty:
                for weight_col in ["Core Weight", "DBMF Weight", "KMLM Weight"]:
                    if weight_col not in managed_futures_summary.columns:
                        managed_futures_summary[weight_col] = 0.0
                    managed_futures_summary[weight_col] = managed_futures_summary[weight_col].fillna(0.0)
            managed_futures_summary.to_csv(paths.result_dir / "best_param_step2b_managed_futures_summary.csv", index=False)
            managed_futures_summary
            """
        ),
        md_cell(
            """
            ## Step 3 - Daily Exposure Parameter Sweep By Asset

            For each asset, the notebook searches a simple trend exposure rule:

            - compute moving average using `ma_period`
            - if price is below the moving average, use `below_exposure`
            - otherwise use exposure `1.0`
            - shift the exposure by one trading session before applying it to returns

            The sweep reports the best Sharpe and the best max drawdown for each asset.
            """
        ),
        code_cell(
            """
            MA_PERIODS = [50, 75, 100, 150, 200, 250, 300]
            BELOW_EXPOSURES = [0.00, 0.25, 0.50, 0.65, 0.80, 1.00]
            ASSET_MAP = {
                "S&P 500": "SPY",
                "Gold": "GOLD",
                "BTC": "BTC",
            }

            exposure_rows = []
            exposure_curves = {}
            exposure_histories = {}

            for asset_label, column in ASSET_MAP.items():
                price = overlay_prices[{"SPY": "SPY", "GOLD": "GC=F", "BTC": "BTC-USD"}[column]].dropna()
                raw_returns = price.pct_change(fill_method=None).fillna(0.0)
                raw_row = metrics_from_returns(raw_returns, f"{asset_label} raw buy hold")
                raw_row["Asset"] = asset_label
                raw_row["MA Period"] = 0
                raw_row["Below Exposure"] = 1.0
                raw_row["Rule"] = "Raw"
                exposure_rows.append(raw_row)

                for ma_period in MA_PERIODS:
                    for below_exposure in BELOW_EXPOSURES:
                        strategy = apply_exposure(price, ma_period=ma_period, below_exposure=below_exposure)
                        label = f"{asset_label} MA{ma_period} below{below_exposure:.2f}"
                        row = compute_port_opt_style_metrics(strategy["Overlay Growth"], risk_free_rate=RISK_FREE_RATE).to_dict()
                        row["Strategy"] = label
                        row["Asset"] = asset_label
                        row["MA Period"] = ma_period
                        row["Below Exposure"] = below_exposure
                        row["Rule"] = "Trend exposure, lag 1 session"
                        row["Start"] = strategy.dropna().index.min().date().isoformat()
                        row["End"] = strategy.dropna().index.max().date().isoformat()
                        exposure_rows.append(row)
                        exposure_curves[label] = strategy["Overlay Growth"]
                        exposure_histories[label] = strategy["Daily Exposure"]

            exposure_sweep = pd.DataFrame(exposure_rows)
            exposure_sweep.to_csv(paths.result_dir / "best_param_step3_daily_exposure_asset_sweep.csv", index=False)

            best_exposure_by_sharpe = (
                exposure_sweep.loc[exposure_sweep["Rule"] != "Raw"]
                .sort_values(["Asset", "Sharpe"], ascending=[True, False])
                .groupby("Asset", as_index=False)
                .head(1)
                .sort_values("Asset")
            )
            best_exposure_by_drawdown = (
                exposure_sweep.loc[exposure_sweep["Rule"] != "Raw"]
                .sort_values(["Asset", "Max Drawdown"], ascending=[True, False])
                .groupby("Asset", as_index=False)
                .head(1)
                .sort_values("Asset")
            )

            best_exposure_by_sharpe.to_csv(paths.result_dir / "best_param_step3_daily_exposure_best_sharpe_by_asset.csv", index=False)
            best_exposure_by_drawdown.to_csv(paths.result_dir / "best_param_step3_daily_exposure_best_drawdown_by_asset.csv", index=False)

            best_exposure_labels = list(dict.fromkeys(
                best_exposure_by_sharpe["Strategy"].tolist() + best_exposure_by_drawdown["Strategy"].tolist()
            ))
            pd.DataFrame({label: exposure_curves[label] for label in best_exposure_labels}).to_csv(
                paths.result_dir / "best_param_step3_daily_exposure_best_curves.csv"
            )
            pd.DataFrame({label: exposure_histories[label] for label in best_exposure_labels}).to_csv(
                paths.result_dir / "best_param_step3_daily_exposure_best_exposure_history.csv"
            )

            display(best_exposure_by_sharpe)
            display(best_exposure_by_drawdown)
            """
        ),
        md_cell(
            """
            ## Step 3A - S&P Buy Hold vs Daily Exposure

            Compares the Step 1 S&P buy-and-hold path against the best-Sharpe S&P daily exposure rule from Step 3.
            """
        ),
        code_cell(
            """
            sp_exposure_row = best_exposure_by_sharpe.loc[
                best_exposure_by_sharpe["Asset"].eq("S&P 500")
            ].iloc[0]
            sp_exposure_label = sp_exposure_row["Strategy"]
            sp_daily_exposure_curve = exposure_curves[sp_exposure_label].rename(
                f"S&P 500 daily exposure ({sp_exposure_label})"
            )

            sp_compare_curves = pd.concat(
                [
                    spx_curve.rename("S&P 500 buy and hold"),
                    sp_daily_exposure_curve,
                ],
                axis=1,
            ).dropna()
            sp_compare_curves.to_csv(paths.result_dir / "best_param_step3a_sp500_buy_hold_vs_daily_exposure_curves.csv")

            sp_compare_metrics = pd.DataFrame(
                [
                    spx_metrics.reset_index().iloc[0].to_dict(),
                    sp_exposure_row.to_dict(),
                ]
            )
            sp_compare_metrics["Strategy"] = [
                "S&P 500 buy and hold",
                sp_daily_exposure_curve.name,
            ]
            sp_compare_metrics = sp_compare_metrics[
                ["Strategy", "CAGR", "Annual Vol", "Sharpe", "Sortino", "Max Drawdown", "Hit Rate", "Start", "End"]
            ]
            sp_compare_metrics.to_csv(
                paths.result_dir / "best_param_step3a_sp500_buy_hold_vs_daily_exposure_metrics.csv",
                index=False,
            )

            sp_compare_fig = go.Figure()
            for column in sp_compare_curves.columns:
                sp_compare_fig.add_trace(
                    go.Scatter(
                        x=sp_compare_curves.index,
                        y=sp_compare_curves[column],
                        mode="lines",
                        name=column,
                        hovertemplate="%{x|%Y-%m-%d}<br>Portfolio growth: %{y:,.0f}<extra></extra>",
                    )
                )
            sp_compare_fig.update_layout(
                title="S&P 500 Port Growth - Buy Hold vs Daily Exposure",
                template="plotly_white",
                height=520,
                yaxis_title="Portfolio growth",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=60, r=30, t=80, b=55),
            )
            sp_compare_fig.write_html(
                paths.result_dir / "best_param_step3a_sp500_buy_hold_vs_daily_exposure_chart.html",
                include_plotlyjs="cdn",
            )

            display(sp_compare_metrics)
            sp_compare_fig.show()
            """
        ),
        md_cell(
            """
            ## Step 3B - Daily Exposure Allocation Using Best Asset Signals

            This optional follow-up uses the best Sharpe signal for each risky asset, then reruns the multi-asset allocation sweep using the exposed returns.

            `BIL`, `IEF`, `VXUS`, and `TIP` are kept as raw buy-and-hold sleeve returns in this section.
            """
        ),
        code_cell(
            """
            best_signal_returns = {}
            best_signal_config = {}
            raw_column_map = {"S&P 500": "SPY", "Gold": "GC=F", "BTC": "BTC-USD"}

            for _, row in best_exposure_by_sharpe.iterrows():
                asset_label = row["Asset"]
                price = overlay_prices[raw_column_map[asset_label]].dropna()
                strategy = apply_exposure(
                    price,
                    ma_period=int(row["MA Period"]),
                    below_exposure=float(row["Below Exposure"]),
                )
                output_col = {"S&P 500": "SPY", "Gold": "GOLD", "BTC": "BTC"}[asset_label]
                best_signal_returns[output_col] = strategy["Overlay Return"]
                best_signal_config[output_col] = {
                    "Asset": asset_label,
                    "MA Period": int(row["MA Period"]),
                    "Below Exposure": float(row["Below Exposure"]),
                    "Sharpe": float(row["Sharpe"]),
                    "Max Drawdown": float(row["Max Drawdown"]),
                }

            exposed_returns = pd.DataFrame(best_signal_returns).dropna().rename(columns={"GOLD": "Gold"})
            for raw_asset in ["BIL", "IEF", "VXUS", "TIP"]:
                exposed_returns[raw_asset] = asset_returns[raw_asset].reindex(exposed_returns.index).fillna(0.0)
            pd.DataFrame(best_signal_config).T.to_csv(paths.result_dir / "best_param_step3b_best_signal_config_used.csv")

            best_exposed_sharpe, best_exposed_drawdown, exposed_allocation_curves = run_allocation_sweep(
                exposed_returns[BASE_ALLOCATION_ASSETS],
                strategy_prefix="Exposed SPY/Gold/BTC + BIL/IEF/VXUS/TIP",
                output_prefix="best_param_step3b_daily_exposure_multi_asset",
                allocation_assets=BASE_ALLOCATION_ASSETS,
                max_weight={asset: MAX_WEIGHT[asset] for asset in BASE_ALLOCATION_ASSETS},
                step=0.05,
            )

            display(pd.DataFrame(best_signal_config).T)
            display(best_exposed_sharpe)
            display(best_exposed_drawdown)
            """
        ),
        md_cell(
            """
            ## Step 3C - Gold Drawdown Exposure Test

            Step 3B selected `Gold MA50 below100%`, which means Gold exposure is never reduced by the trend rule.

            This section keeps the same SPY/BTC daily exposure signals and tests Gold drawdown caps instead. It first tests the original `35/30/10/25` mix, then reruns the allocation sweep using the best fixed-mix Gold drawdown rule.
            """
        ),
        code_cell(
            """
            GOLD_DRAWDOWN_RULES = [
                {"Warn Drawdown": -0.08, "Crash Drawdown": -0.15, "Warn Exposure": 0.65, "Crash Exposure": 0.25},
                {"Warn Drawdown": -0.08, "Crash Drawdown": -0.15, "Warn Exposure": 0.50, "Crash Exposure": 0.25},
                {"Warn Drawdown": -0.08, "Crash Drawdown": -0.20, "Warn Exposure": 0.50, "Crash Exposure": 0.25},
                {"Warn Drawdown": -0.10, "Crash Drawdown": -0.20, "Warn Exposure": 0.65, "Crash Exposure": 0.25},
                {"Warn Drawdown": -0.10, "Crash Drawdown": -0.20, "Warn Exposure": 0.50, "Crash Exposure": 0.25},
                {"Warn Drawdown": -0.12, "Crash Drawdown": -0.20, "Warn Exposure": 0.65, "Crash Exposure": 0.25},
                {"Warn Drawdown": -0.12, "Crash Drawdown": -0.20, "Warn Exposure": 0.50, "Crash Exposure": 0.25},
                {"Warn Drawdown": -0.12, "Crash Drawdown": -0.25, "Warn Exposure": 0.50, "Crash Exposure": 0.00},
            ]
            FIXED_DAILY_EXPOSURE_WEIGHTS = pd.Series(
                {"SPY": 0.35, "Gold": 0.30, "BTC": 0.10, "BIL": 0.25},
                dtype=float,
            )

            gold_price = overlay_prices["GC=F"].dropna()
            gold_drawdown_rows = []
            gold_drawdown_returns = {}
            gold_drawdown_exposures = {}
            gold_drawdown_fixed_curves = {}

            base_fixed_returns = compare_rebalanced_portfolio(
                exposed_returns[FIXED_DAILY_EXPOSURE_WEIGHTS.index.tolist()].dropna(),
                FIXED_DAILY_EXPOSURE_WEIGHTS,
                rebalance_months=1,
            )
            base_fixed_curve = curve_from_returns(base_fixed_returns, initial=INITIAL_VALUE).rename(
                "Original daily exposure allocation SPY/Gold/BTC/BIL 35/30/10/25"
            )
            base_fixed_row = metrics_from_returns(
                base_fixed_returns,
                "Original daily exposure allocation SPY/Gold/BTC/BIL 35/30/10/25",
            )
            base_fixed_row.update(
                {
                    "Rule": "Original Gold trend rule",
                    "Warn Drawdown": np.nan,
                    "Crash Drawdown": np.nan,
                    "Warn Exposure": np.nan,
                    "Crash Exposure": np.nan,
                }
            )
            gold_drawdown_rows.append(base_fixed_row)
            gold_drawdown_fixed_curves[base_fixed_row["Strategy"]] = base_fixed_curve

            for rule in GOLD_DRAWDOWN_RULES:
                gold_dd = apply_drawdown_exposure(
                    gold_price,
                    warn_drawdown=rule["Warn Drawdown"],
                    crash_drawdown=rule["Crash Drawdown"],
                    warn_exposure=rule["Warn Exposure"],
                    crash_exposure=rule["Crash Exposure"],
                )
                label = (
                    "Gold DD "
                    f"warn{rule['Warn Drawdown']:.0%}->{rule['Warn Exposure']:.0%} "
                    f"crash{rule['Crash Drawdown']:.0%}->{rule['Crash Exposure']:.0%}"
                )
                candidate_returns = exposed_returns.copy()
                candidate_returns["Gold"] = gold_dd["Overlay Return"].reindex(candidate_returns.index).fillna(0.0)
                fixed_returns = compare_rebalanced_portfolio(
                    candidate_returns[FIXED_DAILY_EXPOSURE_WEIGHTS.index.tolist()].dropna(),
                    FIXED_DAILY_EXPOSURE_WEIGHTS,
                    rebalance_months=1,
                )
                fixed_curve = curve_from_returns(fixed_returns, initial=INITIAL_VALUE).rename(
                    f"Daily exposure allocation with {label}"
                )
                row = metrics_from_returns(fixed_returns, f"Daily exposure allocation with {label}")
                row.update(rule)
                row["Rule"] = label
                gold_drawdown_rows.append(row)
                gold_drawdown_returns[label] = candidate_returns
                gold_drawdown_exposures[label] = gold_dd["Daily Exposure"]
                gold_drawdown_fixed_curves[row["Strategy"]] = fixed_curve

            gold_drawdown_sweep = pd.DataFrame(gold_drawdown_rows).sort_values("Sharpe", ascending=False)
            gold_drawdown_sweep.to_csv(paths.result_dir / "best_param_step3c_gold_drawdown_fixed_mix_sweep.csv", index=False)
            pd.DataFrame(gold_drawdown_fixed_curves).to_csv(
                paths.result_dir / "best_param_step3c_gold_drawdown_fixed_mix_curves.csv"
            )
            if gold_drawdown_exposures:
                pd.DataFrame(gold_drawdown_exposures).to_csv(
                    paths.result_dir / "best_param_step3c_gold_drawdown_exposure_history.csv"
                )

            best_gold_drawdown_rule = gold_drawdown_sweep.loc[
                gold_drawdown_sweep["Rule"].ne("Original Gold trend rule")
            ].iloc[0]
            best_gold_drawdown_label = best_gold_drawdown_rule["Rule"]
            gold_drawdown_exposed_returns = gold_drawdown_returns[best_gold_drawdown_label]
            best_gold_drawdown_fixed_curve = gold_drawdown_fixed_curves[
                best_gold_drawdown_rule["Strategy"]
            ].rename("Best Gold-DD fixed 35/30/10/25")
            best_gold_drawdown_fixed_curve.to_frame().to_csv(
                paths.result_dir / "best_param_step3c_gold_drawdown_best_fixed_curve.csv"
            )

            best_gold_dd_allocation_sharpe, best_gold_dd_allocation_drawdown, gold_dd_allocation_curves = run_allocation_sweep(
                gold_drawdown_exposed_returns[BASE_ALLOCATION_ASSETS],
                strategy_prefix="Gold-DD daily-exposure allocation",
                output_prefix="best_param_step3c_gold_drawdown_daily_exposure_multi_asset",
                allocation_assets=BASE_ALLOCATION_ASSETS,
                max_weight={asset: MAX_WEIGHT[asset] for asset in BASE_ALLOCATION_ASSETS},
                step=0.05,
            )

            gold_drawdown_summary_curves = {
                "Original fixed 35/30/10/25": base_fixed_curve,
                f"Best Gold-DD fixed 35/30/10/25 - {best_gold_drawdown_label}": gold_drawdown_fixed_curves[best_gold_drawdown_rule["Strategy"]],
                f"Best Gold-DD optimized - {best_gold_dd_allocation_sharpe.iloc[0]['Strategy']}": next(iter(gold_dd_allocation_curves.values())),
            }
            pd.DataFrame(gold_drawdown_summary_curves).to_csv(
                paths.result_dir / "best_param_step3c_gold_drawdown_summary_curves.csv"
            )

            gold_drawdown_fig = go.Figure()
            for label, curve in gold_drawdown_summary_curves.items():
                gold_drawdown_fig.add_trace(
                    go.Scatter(
                        x=curve.index,
                        y=curve,
                        mode="lines",
                        name=label,
                        hovertemplate="%{x|%Y-%m-%d}<br>Portfolio growth: %{y:,.0f}<extra></extra>",
                    )
                )
            gold_drawdown_fig.update_layout(
                title="Gold Drawdown Exposure Test - Port Growth",
                template="plotly_white",
                height=560,
                yaxis_title="Portfolio growth",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
                margin=dict(l=60, r=30, t=80, b=160),
            )
            gold_drawdown_fig.write_html(
                paths.result_dir / "best_param_step3c_gold_drawdown_summary_chart.html",
                include_plotlyjs="cdn",
            )

            display_full_table(gold_drawdown_sweep, "Gold drawdown fixed 35/30/10/25 sweep")
            display_full_table(best_gold_dd_allocation_sharpe, "Gold drawdown optimized allocation best Sharpe")
            display_full_table(best_gold_dd_allocation_drawdown, "Gold drawdown optimized allocation best drawdown")
            gold_drawdown_fig.show()
            """
        ),
        md_cell(
            """
            ## Final Snapshot

            Compact view of the key winners from each step.
            """
        ),
        code_cell(
            """
            def row_period(row: pd.Series) -> str:
                start = row.get("Start", START_DATE)
                end = row.get("End", END_DATE)
                return f"{start} to {end}"

            def metric_dict(row: pd.Series) -> dict:
                return {
                    "Total Return": float(row.get("Total Return", np.nan)),
                    "CAGR": float(row["CAGR"]),
                    "Annual Vol": float(row.get("Annual Vol", np.nan)),
                    "Sharpe": float(row["Sharpe"]),
                    "Sortino": float(row.get("Sortino", np.nan)),
                    "Max Drawdown": float(row["Max Drawdown"]),
                    "Hit Rate": float(row.get("Hit Rate", np.nan)),
                }

            step1_row = spx_metrics.iloc[0]
            step2_row = best_allocation_sharpe.iloc[0]
            step3b_row = best_exposed_sharpe.iloc[0]
            step3c_fixed_row = None
            step3c_optimized_row = None
            extended_pruned_row = None

            if "gold_drawdown_sweep" in globals() and not gold_drawdown_sweep.empty:
                step3c_fixed_row = gold_drawdown_sweep.loc[
                    gold_drawdown_sweep["Rule"].ne("Original Gold trend rule")
                ].sort_values("Sharpe", ascending=False).iloc[0]
            if "best_gold_dd_allocation_sharpe" in globals() and not best_gold_dd_allocation_sharpe.empty:
                step3c_optimized_row = best_gold_dd_allocation_sharpe.iloc[0]

            if "best_extended_sharpe" in globals() and not best_extended_sharpe.empty:
                extended_source_row = best_extended_sharpe.iloc[0]
                prune_threshold = 0.03
                pruned_weights = pd.Series(
                    {
                        asset: float(extended_source_row.get(f"{asset} Weight", 0.0))
                        for asset in available_extended_assets
                    },
                    dtype=float,
                )
                pruned_weights = pruned_weights[pruned_weights >= prune_threshold]
                pruned_weights = pruned_weights / pruned_weights.sum()
                pruned_returns = compare_rebalanced_portfolio(
                    extended_returns[pruned_weights.index.tolist()].dropna(),
                    pruned_weights,
                    rebalance_months=1,
                ).rename("Extended defensive allocation pruned 3pct")
                pruned_curve = curve_from_returns(pruned_returns, initial=INITIAL_VALUE).rename(
                    "Extended defensive allocation pruned 3pct"
                )
                pruned_curve.to_frame().to_csv(
                    paths.result_dir / "best_param_step2c_extended_defensive_pruned_3pct_curve.csv"
                )
                pruned_weights.rename("Weight").to_frame().to_csv(
                    paths.result_dir / "best_param_step2c_extended_defensive_pruned_3pct_weights.csv"
                )
                extended_pruned_row = pd.Series(metrics_from_returns(pruned_returns, "Extended defensive allocation pruned 3pct"))
                for asset, weight in pruned_weights.items():
                    extended_pruned_row[f"{asset} Weight"] = float(weight)

            snapshot_rows = [
                {
                    "Step": "1. S&P buy hold",
                    "Winner Type": "Benchmark Sharpe",
                    "Strategy": "S&P 500 buy and hold",
                    "Config": "SPX benchmark buy-and-hold; no allocation sweep; no daily exposure.",
                    "Total Return": float(step1_row["Total Return"]),
                    "Sharpe": float(step1_row["Sharpe"]),
                    "CAGR": float(step1_row["CAGR"]),
                    "Max Drawdown": float(step1_row["Max Drawdown"]),
                    "Precompute Port Growth File": str((paths.result_dir / "best_param_step1_sp500_buy_hold_curve.csv").relative_to(ROOT)),
                    "Precompute Period": row_period(step1_row),
                },
                {
                    "Step": "2. Allocation",
                    "Winner Type": "Best Sharpe",
                    "Strategy": format_weight_strategy("Monthly allocation", step2_row, BASE_ALLOCATION_ASSETS),
                    "Config": (
                        "Monthly rebalanced buy-and-hold allocation; "
                        f"active weights: {format_weight_config(step2_row, BASE_ALLOCATION_ASSETS)}; "
                        "tested assets: SPY, Gold, BTC, BIL, IEF, VXUS, TIP; "
                        "step=5%; max weights SPY=70%, Gold=40%, BTC=10%, BIL=50%, IEF=30%, VXUS=40%, TIP=30%."
                    ),
                    "Total Return": float(step2_row["Total Return"]),
                    "Sharpe": float(step2_row["Sharpe"]),
                    "CAGR": float(step2_row["CAGR"]),
                    "Max Drawdown": float(step2_row["Max Drawdown"]),
                    "Precompute Port Growth File": str((paths.result_dir / "best_param_step2_multi_asset_best_curves.csv").relative_to(ROOT)),
                    "Precompute Period": row_period(step2_row),
                },
            ]
            if "best_extended_sharpe" in globals() and not best_extended_sharpe.empty:
                extended_row = best_extended_sharpe.iloc[0]
                snapshot_rows.append(
                    {
                        "Step": "2C. Extended allocation",
                        "Winner Type": "Best Sharpe",
                        "Strategy": format_weight_strategy(
                            "Extended defensive allocation",
                            extended_row,
                            available_extended_assets,
                        ),
                        "Config": (
                            "Memory-aware random capped search across SPY plus managed futures, T-Bills, anti-beta, "
                            "inverse equity hedges, defensive sectors, GLD, volatility, BTC, and ETH; "
                            f"active weights: {format_weight_config(extended_row, available_extended_assets)}; "
                            f"candidates={EXTENDED_SEARCH_CANDIDATES:,}; batch size={EXTENDED_SEARCH_BATCH_SIZE:,}; "
                            f"top rows kept={EXTENDED_SEARCH_KEEP_TOP:,}; seed=20260607."
                        ),
                        "Total Return": float(extended_row["Total Return"]),
                        "Sharpe": float(extended_row["Sharpe"]),
                        "CAGR": float(extended_row["CAGR"]),
                        "Max Drawdown": float(extended_row["Max Drawdown"]),
                        "Precompute Port Growth File": str(
                            (paths.result_dir / "best_param_step2c_extended_defensive_allocation_best_curves.csv").relative_to(ROOT)
                        ),
                        "Precompute Period": row_period(extended_row),
                    }
                )
                if extended_pruned_row is not None:
                    pruned_assets = pruned_weights.index.tolist()
                    snapshot_rows.append(
                        {
                            "Step": "2C. Extended >=3%",
                            "Winner Type": "Pruned Weights",
                            "Strategy": format_weight_strategy(
                                "Extended defensive allocation >=3%",
                                extended_pruned_row,
                                pruned_assets,
                            ),
                            "Config": (
                                "Derived from the best Step 2C allocation by removing assets below 3% weight, "
                                "renormalizing the remaining weights, and monthly rebalancing; "
                                f"active weights: {format_weight_config(extended_pruned_row, pruned_assets)}."
                            ),
                            "Total Return": float(extended_pruned_row["Total Return"]),
                            "Sharpe": float(extended_pruned_row["Sharpe"]),
                            "CAGR": float(extended_pruned_row["CAGR"]),
                            "Max Drawdown": float(extended_pruned_row["Max Drawdown"]),
                            "Precompute Port Growth File": str(
                                (paths.result_dir / "best_param_step2c_extended_defensive_pruned_3pct_curve.csv").relative_to(ROOT)
                            ),
                            "Precompute Period": row_period(extended_pruned_row),
                        }
                    )
            if "managed_futures_summary" in globals() and not managed_futures_summary.empty:
                mf_row = managed_futures_summary.loc[
                    managed_futures_summary["Winner Type"].eq("Best Sharpe")
                ].sort_values("Sharpe", ascending=False).iloc[0]
                mf_file = "best_param_step2b_core_dbmf_best_curves.csv"
                if float(mf_row.get("KMLM Weight", 0.0)) > 0.0:
                    mf_file = "best_param_step2b_core_kmlm_best_curves.csv"
                snapshot_rows.append(
                    {
                        "Step": "2B. Managed futures",
                        "Winner Type": "Best Sharpe",
                        "Strategy": format_weight_strategy("Managed futures overlay", mf_row, ["Core", "DBMF", "KMLM"]),
                        "Config": (
                            "Monthly rebalanced blend of Step 2 best-Sharpe Core sleeve plus one managed-futures ETF; "
                            f"active weights: {format_weight_config(mf_row, ['Core', 'DBMF', 'KMLM'])}; "
                            f"Core source: {format_weight_strategy('Monthly allocation', step2_row, BASE_ALLOCATION_ASSETS)}; "
                            "DBMF and KMLM are tested separately on their own overlap windows; managed-futures max weight=30%."
                        ),
                        "Total Return": float(mf_row["Total Return"]),
                        "Sharpe": float(mf_row["Sharpe"]),
                        "CAGR": float(mf_row["CAGR"]),
                        "Max Drawdown": float(mf_row["Max Drawdown"]),
                        "Precompute Port Growth File": str((paths.result_dir / mf_file).relative_to(ROOT)),
                        "Precompute Period": row_period(mf_row),
                    }
                )

            signal_text = "; ".join(
                f"{asset}: MA{cfg['MA Period']}, below={cfg['Below Exposure']:.0%}"
                for asset, cfg in best_signal_config.items()
            )
            snapshot_rows.append(
                {
                    "Step": "3B. Daily exposure allocation",
                    "Winner Type": "Best Sharpe",
                    "Strategy": format_weight_strategy("Daily-exposure allocation", step3b_row, BASE_ALLOCATION_ASSETS),
                    "Config": (
                        "Best-Sharpe daily exposure signal applied to SPY, Gold, and BTC, then monthly allocation sweep; "
                        f"daily exposure signals: {signal_text}; "
                        f"active weights: {format_weight_config(step3b_row, BASE_ALLOCATION_ASSETS)}; "
                        "daily exposure uses lagged close signal by one trading session to avoid lookahead."
                    ),
                    "Total Return": float(step3b_row["Total Return"]),
                    "Sharpe": float(step3b_row["Sharpe"]),
                    "CAGR": float(step3b_row["CAGR"]),
                    "Max Drawdown": float(step3b_row["Max Drawdown"]),
                    "Precompute Port Growth File": str((paths.result_dir / "best_param_step3b_daily_exposure_multi_asset_best_curves.csv").relative_to(ROOT)),
                    "Precompute Period": row_period(step3b_row),
                }
            )
            if step3c_fixed_row is not None:
                snapshot_rows.append(
                    {
                        "Step": "3C. Gold DD fixed",
                        "Winner Type": "Best Sharpe",
                        "Strategy": "Gold-DD fixed daily-exposure allocation SPY/Gold/BTC/BIL 35/30/10/25",
                        "Config": (
                            "Uses the same fixed allocation as Step 3B, but replaces Gold's no-op trend exposure "
                            f"with drawdown exposure rule: {step3c_fixed_row['Rule']}."
                        ),
                        "Total Return": float(step3c_fixed_row["Total Return"]),
                        "Sharpe": float(step3c_fixed_row["Sharpe"]),
                        "CAGR": float(step3c_fixed_row["CAGR"]),
                        "Max Drawdown": float(step3c_fixed_row["Max Drawdown"]),
                        "Precompute Port Growth File": str(
                            (paths.result_dir / "best_param_step3c_gold_drawdown_best_fixed_curve.csv").relative_to(ROOT)
                        ),
                        "Precompute Period": row_period(step3c_fixed_row),
                    }
                )
            final_snapshot = pd.DataFrame(snapshot_rows)
            final_snapshot.to_csv(paths.result_dir / "best_param_by_step_final_snapshot.csv", index=False)

            md_lines = [
                "# Best Param By Step - Port Opt Advance Handoff",
                "",
                "This file lists only the best-Sharpe strategy from each final notebook family. Assets with zero allocation are intentionally omitted from strategy names and active-weight configs.",
                "",
            ]
            for row in snapshot_rows:
                md_lines.extend(
                    [
                        f"## {row['Step']}: {row['Strategy']}",
                        "",
                        f"- Strategy name: `{row['Strategy']}`",
                        f"- Winner type: `{row['Winner Type']}`",
                        f"- Config: {row['Config']}",
                        "- Metrics:",
                        f"  - Total Return: {row['Total Return']:.4f}",
                        f"  - CAGR: {row['CAGR']:.4f}",
                        f"  - Sharpe: {row['Sharpe']:.4f}",
                        f"  - Max Drawdown: {row['Max Drawdown']:.4f}",
                        f"- Precompute port growth file path: `{row['Precompute Port Growth File']}`",
                        f"- Precompute period: `{row['Precompute Period']}`",
                        "",
                    ]
                )
            doc_dir = ROOT / "doc"
            doc_dir.mkdir(parents=True, exist_ok=True)
            (doc_dir / "BEST_PARAM_BY_STEP_PORT_OPT_ADVANCE.md").write_text("\\n".join(md_lines), encoding="utf-8")

            sp_md_lines = [
                "# Best Param S&P Port Opt Advance",
                "",
                "This handoff keeps the fixed Gold drawdown variant and omits the optimized Gold-DD duplicate because the optimized sweep selected the same `35/30/10/25` weights and produced the same metrics.",
                "",
                "## Selected Gold-DD Variant",
                "",
            ]
            if step3c_fixed_row is not None:
                sp_md_lines.extend(
                    [
                        "- Selected strategy: `Gold-DD fixed daily-exposure allocation SPY/Gold/BTC/BIL 35/30/10/25`",
                        f"- Gold rule: `{step3c_fixed_row['Rule']}`",
                        "- Reason: fixed and optimized Gold-DD results are identical, so the fixed version is simpler and avoids duplicate reporting.",
                        "- Metrics:",
                        f"  - Total Return: {float(step3c_fixed_row['Total Return']):.4f}",
                        f"  - CAGR: {float(step3c_fixed_row['CAGR']):.4f}",
                        f"  - Sharpe: {float(step3c_fixed_row['Sharpe']):.4f}",
                        f"  - Max Drawdown: {float(step3c_fixed_row['Max Drawdown']):.4f}",
                        f"- Precompute port growth file path: `{(paths.result_dir / 'best_param_step3c_gold_drawdown_best_fixed_curve.csv').relative_to(ROOT)}`",
                        f"- Precompute period: `{row_period(step3c_fixed_row)}`",
                        "",
                    ]
                )
            sp_md_lines.extend(
                [
                    "## Final Snapshot",
                    "",
                    "| Strategy | Total Return | Max Drawdown | Sharpe | CAGR | Period |",
                    "|---|---:|---:|---:|---:|---|",
                ]
            )
            for _, row in final_snapshot.iterrows():
                sp_md_lines.append(
                    f"| {row['Strategy']} | {row['Total Return']:.4f} | {row['Max Drawdown']:.4f} | {row['Sharpe']:.4f} | {row['CAGR']:.4f} | {row['Precompute Period']} |"
                )
            sp_md_lines.extend(
                [
                    "",
                    "## Output Files",
                    "",
                    "- `result/best_param_by_step_final_snapshot.csv`",
                    "- `result/best_param_by_step_final_snapshot_port_growth_chart.html`",
                    "- `result/best_param_step3c_gold_drawdown_summary_chart.html`",
                    "- `result/best_param_step3c_gold_drawdown_fixed_mix_sweep.csv`",
                ]
            )
            sp_md_text = "\\n".join(sp_md_lines)
            (doc_dir / "BEST_PARAM_S&P_PORT_OPT_ADVANCE.md").write_text(sp_md_text, encoding="utf-8")
            (doc_dir / "BEST_PARAM_SP_PORT_OPT_ADVANCE.md").write_text(sp_md_text, encoding="utf-8")
            final_snapshot_display = final_snapshot.drop(columns=["Step", "Winner Type"])
            final_snapshot_display = final_snapshot_display[
                [
                    "Strategy",
                    "Config",
                    "Total Return",
                    "Max Drawdown",
                    "Sharpe",
                    "CAGR",
                    "Precompute Port Growth File",
                    "Precompute Period",
                ]
            ]
            display_full_table(final_snapshot_display, "Final snapshot")

            final_growth_curves = {}
            for _, row in final_snapshot.iterrows():
                curve_path = ROOT / Path(str(row["Precompute Port Growth File"]))
                curve = pd.read_csv(curve_path, index_col=0, parse_dates=True)
                curve_series = curve.iloc[:, 0].astype(float)
                final_growth_curves[f"{row['Step']} - {row['Strategy']}"] = curve_series

            final_growth_curves = pd.DataFrame(final_growth_curves).sort_index()
            final_growth_curves.to_csv(paths.result_dir / "best_param_by_step_final_snapshot_port_growth_curves.csv")

            final_growth_fig = go.Figure()
            for column in final_growth_curves.columns:
                final_growth_fig.add_trace(
                    go.Scatter(
                        x=final_growth_curves.index,
                        y=final_growth_curves[column],
                        mode="lines",
                        name=column,
                        hovertemplate="%{x|%Y-%m-%d}<br>Portfolio growth: %{y:,.0f}<extra></extra>",
                    )
                )
            final_growth_fig.update_layout(
                title="Best Param By Step - Final Snapshot Port Growth",
                template="plotly_white",
                height=560,
                yaxis_title="Portfolio growth",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
                margin=dict(l=60, r=30, t=80, b=155),
            )
            final_growth_fig.write_html(
                paths.result_dir / "best_param_by_step_final_snapshot_port_growth_chart.html",
                include_plotlyjs="cdn",
            )
            final_growth_fig.show()

            final_metric_fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.12,
                subplot_titles=("Sharpe by step", "CAGR and drawdown by step"),
            )
            hover_data = final_snapshot[["Strategy", "Config", "Precompute Period"]].to_numpy()
            final_metric_fig.add_trace(
                go.Bar(
                    name="Sharpe",
                    x=final_snapshot["Step"],
                    y=final_snapshot["Sharpe"],
                    marker_color="#2563eb",
                    customdata=hover_data,
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Sharpe: %{y:.4f}<br>"
                        "Strategy: %{customdata[0]}<br>"
                        "Period: %{customdata[2]}"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )
            final_metric_fig.add_trace(
                go.Bar(
                    name="CAGR",
                    x=final_snapshot["Step"],
                    y=final_snapshot["CAGR"],
                    marker_color="#16a34a",
                    customdata=hover_data,
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "CAGR: %{y:.2%}<br>"
                        "Strategy: %{customdata[0]}<br>"
                        "Period: %{customdata[2]}"
                        "<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )
            final_metric_fig.add_trace(
                go.Bar(
                    name="Max Drawdown",
                    x=final_snapshot["Step"],
                    y=final_snapshot["Max Drawdown"],
                    marker_color="#dc2626",
                    customdata=hover_data,
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Max Drawdown: %{y:.2%}<br>"
                        "Strategy: %{customdata[0]}<br>"
                        "Period: %{customdata[2]}"
                        "<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )
            final_metric_fig.update_layout(
                title="Best Param By Step - Final Snapshot Metrics",
                template="plotly_white",
                barmode="group",
                height=650,
                legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0),
                margin=dict(l=60, r=30, t=80, b=130),
            )
            final_metric_fig.update_yaxes(title_text="Sharpe", row=1, col=1)
            final_metric_fig.update_yaxes(title_text="Return / Drawdown", tickformat=".0%", row=2, col=1)
            final_metric_fig.write_html(paths.result_dir / "best_param_by_step_final_snapshot_chart.html", include_plotlyjs="cdn")
            final_metric_fig.show()
            None
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
