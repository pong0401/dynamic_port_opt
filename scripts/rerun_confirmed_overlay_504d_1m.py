from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynamic_factor_copula import (  # noqa: E402
    backtest_dynamic_factor_copula,
    build_overlay_comparison,
    default_paths,
)


FEATURE_FLAGS = {"resid_vol": False, "drawdown": False, "downside_beta": False}
MIX_WEIGHTS = (0.60, 0.30, 0.10)


def main() -> None:
    paths = default_paths(ROOT)
    results = backtest_dynamic_factor_copula(
        start_date="2012-01-01",
        end_date="2026-04-30",
        n_assets=30,
        n_clusters=4,
        lookback_days=504,
        rebalance_freq="ME",
        max_weight=0.08,
        point_in_time_liquid=True,
        universe_mode="sp500_pit",
        include_momentum_features=True,
        include_momentum_signal=True,
        momentum_signal_mode="mom_63",
        feature_flags=FEATURE_FLAGS,
        paths=paths,
    )

    usd = build_overlay_comparison(
        results,
        paths=paths,
        mix_weights=MIX_WEIGHTS,
        strategic_rebalance_months=1,
        report_currency="USD",
    )
    thb = build_overlay_comparison(
        results,
        paths=paths,
        mix_weights=MIX_WEIGHTS,
        strategic_rebalance_months=1,
        report_currency="THB",
    )

    usd_summary = usd["summary"].copy()
    thb_summary = thb["summary"].copy()
    usd_curves = pd.DataFrame(usd["curves"]).sort_index()
    thb_curves = pd.DataFrame(thb["curves"]).sort_index()

    usd_summary.to_csv(paths.result_dir / "joint_confirm_603010_504d_1m_overlay_summary_usd.csv")
    thb_summary.to_csv(paths.result_dir / "joint_confirm_603010_504d_1m_overlay_summary_thb.csv")
    usd_curves.to_csv(paths.result_dir / "joint_confirm_603010_504d_1m_overlay_curves_usd.csv")
    thb_curves.to_csv(paths.result_dir / "joint_confirm_603010_504d_1m_overlay_curves_thb.csv")

    print("USD summary")
    print(usd_summary.to_string())
    print("\nTHB summary")
    print(thb_summary.to_string())


if __name__ == "__main__":
    main()
