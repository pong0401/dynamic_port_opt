from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for path in [SCRIPTS, SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dynamic_factor_copula import default_paths  # noqa: E402
import run_spy_gold_btc_bil_adaptive_gold_country_winner as adaptive  # noqa: E402
import run_spy_gold_btc_bil_balanced_plan_refine as refine  # noqa: E402
import run_spy_gold_btc_bil_country_etf_sweep as country  # noqa: E402

OUTPUT_PREFIX = "spy_gold_btc_bil_balanced_plan_refine_v2"


def main() -> None:
    paths = default_paths(ROOT)
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    raw = country.load_prices()
    prices = country.asset_prices(raw).ffill()
    returns = prices.pct_change(fill_method=None).where(prices.notna()).fillna(0.0)
    candidates = tuple(asset for asset in country.COUNTRY_ETFS if asset in prices and prices[asset].dropna().shape[0] >= 2520)
    rule = adaptive.GoldRule(
        "dd252_warn8_crash20_half",
        "dd_simple",
        warn_dd=-0.08,
        crash_dd=-0.20,
        warn_exposure=0.50,
        crash_exposure=0.50,
    )
    configs: list[refine.Config] = []
    for trigger in ["spy_below_ma200_or_dd8", "spy_below_ma300_or_dd10"]:
        for boost in [0.10, 0.1125, 0.125, 0.1375, 0.15]:
            for bucket in [0.05, 0.06, 0.07, 0.075, 0.08, 0.09]:
                name = f"balanced_refine_v2 boost{boost:.2%}_{trigger} country{bucket:.1%}_top2"
                configs.append(refine.Config(name, rule, trigger, boost, bucket, 2))

    curves = {}
    summaries = []
    selections = []
    latest_frames = []
    for config in configs:
        curve, sel, latest, exposure = refine.run_variant(prices, returns, config, candidates)
        curves[config.name] = curve
        summaries.append(refine.metrics_row(curve, config, sel, latest, exposure, candidates))
        selections.append(sel)
        latest_frames.append(latest)

    summary = pd.DataFrame(summaries).sort_values(["Sharpe", "CAGR"], ascending=False)
    summary.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_summary.csv", index=False)
    pd.DataFrame(curves).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_curves.csv")
    pd.concat(selections, ignore_index=True).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_selection_history.csv", index=False)
    pd.concat(latest_frames, ignore_index=True).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_latest_weights.csv", index=False)
    cols = [
        "Strategy",
        "CAGR",
        "Annual Vol",
        "Sharpe",
        "Max Drawdown",
        "Gold Boost Active Rate",
        "Average Gold Exposure",
        "Latest Gold Weight",
        "Latest BIL Weight",
        "Latest Country Assets",
        "Latest Country Weight",
    ]
    print(summary[cols].head(40).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
