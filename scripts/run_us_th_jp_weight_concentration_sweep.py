from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dynamic_factor_copula import default_paths  # noqa: E402
from run_us_th_jp_allocation_models import (  # noqa: E402
    ALLOCATION_PROFILES,
    PREFIX as BASE_PREFIX,
    _asset_exposure_variants,
    _build_japan_pit_sleeve,
    _evaluate_index_signal_models,
    _load_japan_signal_price,
    _load_overlay_assets,
    _read_us_th_returns,
)


PREFIX = "us_th_jp_weight_concentration_sweep"
JP_SELECTED_COUNTS = [20, 15, 10]
FOCUS_EXPOSURES = [
    "daily exposure all assets + gold drawdown 252d warn10 crash20",
    "weekly exposure all assets + gold drawdown 252d warn10 crash20",
]


def _build_asset_returns_and_signals(
    paths,
    us_th_returns: pd.DataFrame,
    jp_selected_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    jp_returns, jp_index, jp_internal_weights, jp_price_end = _build_japan_pit_sleeve(
        paths,
        us_th_returns.index,
        selected_count=jp_selected_count,
    )
    common_index = us_th_returns.index.intersection(jp_returns.dropna().index).sort_values()
    jp_holding_mask = jp_internal_weights.sum(axis=1).gt(0.0)
    if not jp_holding_mask.any():
        raise RuntimeError("Japan PIT sleeve has no active holdings.")
    common_index = common_index[
        (common_index >= jp_holding_mask.idxmax())
        & (common_index <= jp_price_end)
    ]
    us_th = us_th_returns.reindex(common_index).fillna(0.0)
    jp = jp_returns.reindex(common_index).fillna(0.0)
    overlay_prices, signal_prices = _load_overlay_assets(paths, common_index)
    signal_prices["JP Equity"] = _load_japan_signal_price(paths, common_index, jp_index)
    asset_returns = pd.DataFrame(
        {
            "US Equity": us_th["US PIT optimized sleeve THB"],
            "TH Equity": us_th["TH PIT optimized sleeve THB"],
            "JP Equity": jp,
            "Gold": overlay_prices["Gold"].pct_change(fill_method=None).fillna(0.0),
            "BTC": overlay_prices["BTC"].pct_change(fill_method=None).fillna(0.0),
        },
        index=common_index,
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return asset_returns, signal_prices, jp_internal_weights.reindex(common_index).fillna(0.0), pd.Timestamp(jp_price_end)


def main() -> None:
    paths = default_paths(ROOT)
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    us_th_returns = _read_us_th_returns(paths)

    rows: list[dict[str, object]] = []
    curves: dict[str, pd.Series] = {}
    weight_frames: list[pd.DataFrame] = []
    latest_rows: list[dict[str, object]] = []

    for jp_count in JP_SELECTED_COUNTS:
        asset_returns, signal_prices, jp_internal_weights, _ = _build_asset_returns_and_signals(
            paths,
            us_th_returns,
            jp_selected_count=jp_count,
        )
        exposure_variants = {
            name: frame.reindex(asset_returns.index).ffill().fillna(1.0).clip(0.0, 1.0)
            for name, frame in _asset_exposure_variants(signal_prices).items()
            if name in FOCUS_EXPOSURES
        }
        latest_internal_jp_max = float(jp_internal_weights.iloc[-1].max()) if not jp_internal_weights.empty else 0.0
        for profile in ALLOCATION_PROFILES:
            new_rows, new_curves, new_weights = _evaluate_index_signal_models(
                asset_returns,
                signal_prices,
                profile,
                exposure_variants,
            )
            for row in new_rows:
                row["JP Selected Count"] = jp_count
                row["Latest JP Internal Max Weight"] = latest_internal_jp_max
                row["Approx Latest JP Stock Max Effective Weight"] = (
                    row.get("Average JP Weight", 0.0) * latest_internal_jp_max
                )
            rows.extend(new_rows)
            for name, curve in new_curves.items():
                curves[f"JP{jp_count} {name}"] = curve
            for frame in new_weights:
                frame = frame.copy()
                frame["JP Selected Count"] = jp_count
                weight_frames.append(frame)

                latest = frame.sort_values("Date").tail(1)
                if not latest.empty:
                    latest_row = latest.iloc[0]
                    latest_rows.append(
                        {
                            "Strategy": latest_row["Strategy"],
                            "JP Selected Count": jp_count,
                            "Date": latest_row["Date"],
                            "Latest US Weight": latest_row.get("US Equity", 0.0),
                            "Latest TH Weight": latest_row.get("TH Equity", 0.0),
                            "Latest JP Weight": latest_row.get("JP Equity", 0.0),
                            "Latest Gold Weight": latest_row.get("Gold", 0.0),
                            "Latest BTC Weight": latest_row.get("BTC", 0.0),
                            "Latest Cash Weight": latest_row.get("Cash / Reduced Exposure", 0.0),
                            "Latest JP Internal Max Weight": latest_internal_jp_max,
                            "Latest JP Stock Max Effective Weight": latest_row.get("JP Equity", 0.0)
                            * latest_internal_jp_max,
                        }
                    )

    summary = pd.DataFrame(rows)
    focus = summary.loc[
        summary["Strategy"].str.contains("Index signal leaves inactive equity in cash", regex=False)
        & ~summary["Strategy"].str.contains("no JP", regex=False)
        & (
            summary["Strategy"].str.contains("gold drawdown 252d", regex=False)
            | ~summary["Strategy"].str.contains("+", regex=False)
        )
    ].copy()
    focus = focus.sort_values(["Sharpe", "CAGR"], ascending=False)

    summary.sort_values(["Sharpe", "CAGR"], ascending=False).to_csv(
        paths.result_dir / f"{PREFIX}_summary_thb.csv",
        index=False,
    )
    focus.to_csv(paths.result_dir / f"{PREFIX}_focus_summary_thb.csv", index=False)
    pd.DataFrame(curves).dropna(how="all").to_csv(paths.result_dir / f"{PREFIX}_curves_thb.csv")
    pd.concat(weight_frames, ignore_index=True).to_csv(
        paths.result_dir / f"{PREFIX}_weight_history_thb.csv",
        index=False,
    )
    pd.DataFrame(latest_rows).to_csv(paths.result_dir / f"{PREFIX}_latest_weights_thb.csv", index=False)

    print(focus.head(30).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nBase output prefix: {BASE_PREFIX}")


if __name__ == "__main__":
    main()
