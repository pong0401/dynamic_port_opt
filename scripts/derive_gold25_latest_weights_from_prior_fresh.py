from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_us_th_tactical_gold_crash_protection_sweep import _gold_crash_exposure  # noqa: E402
from run_us_th_tactical_perf_momentum import RESULT_PREFIX, _close_trend_exposure  # noqa: E402


FINAL_PREFIX = "us_th_tactical_perf_momentum_final_best"
STRATEGY = "Final Best Sharpe Tactical TH/Gold/BTC 65/25/10 Gold crash protection"
SELECTED_MIX = {"Equity": 0.65, "Gold": 0.25, "BTC": 0.10}
OVERLAY_TICKERS = ["SPY", "^VIX", "GC=F", "BTC-USD", "USDTHB=X", "^SET.BK"]


def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    closes = {}
    for ticker in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker in raw.columns.get_level_values(0):
                closes[ticker] = raw[ticker]["Close"].dropna()
            elif "Close" in raw.columns.get_level_values(0) and ticker in raw["Close"].columns:
                closes[ticker] = raw["Close"][ticker].dropna()
        elif "Close" in raw.columns and len(tickers) == 1:
            closes[ticker] = raw["Close"].dropna()
    return pd.DataFrame(closes).sort_index()


def main() -> None:
    result_dir = ROOT / "result"
    yf.set_tz_cache_location(str(ROOT / ".yfinance"))
    raw = yf.download(
        OVERLAY_TICKERS,
        start="2023-01-01",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=False,
    )
    overlay = _extract_close(raw, OVERLAY_TICKERS).ffill()
    required = ["SPY", "^VIX", "GC=F", "BTC-USD", "USDTHB=X", "^SET.BK"]
    missing = [ticker for ticker in required if ticker not in overlay.columns]
    if missing:
        raise RuntimeError(f"Missing fresh overlay tickers: {missing}")
    common = overlay[required].dropna()
    if common.empty:
        raise RuntimeError("No common fresh overlay close.")
    as_of = pd.Timestamp(common.index.max())

    old_security = pd.read_csv(result_dir / f"{FINAL_PREFIX}_latest_effective_security_weights_thb.csv")
    old_meta = pd.read_csv(result_dir / f"{FINAL_PREFIX}_latest_meta.csv")
    th_tactical_weight = float(old_meta.iloc[0]["TH Tactical Weight Inside Equity Sleeve"])
    raw_sleeve = pd.Series(
        {
            "US Equity": SELECTED_MIX["Equity"] * (1.0 - th_tactical_weight),
            "TH Equity": SELECTED_MIX["Equity"] * th_tactical_weight,
            "Gold": SELECTED_MIX["Gold"],
            "BTC": SELECTED_MIX["BTC"],
        },
        dtype=float,
    )

    exposure = pd.Series(
        {
            "US Equity": float(_close_trend_exposure(overlay["SPY"], 300, 0.50).loc[:as_of].iloc[-1]),
            "TH Equity": float(_close_trend_exposure(overlay["^SET.BK"], 200, 0.00).loc[:as_of].iloc[-1]),
            "Gold": float(
                _gold_crash_exposure(
                    overlay["GC=F"],
                    dd_window=252,
                    warn_dd=-0.08,
                    crash_dd=-0.20,
                    warn_exposure=0.50,
                    crash_exposure=0.50,
                    recovery_dd=-0.05,
                    panic_dd=-0.30,
                    panic_ma_period=200,
                    panic_mom_period=63,
                )
                .loc[:as_of]
                .iloc[-1]
            ),
            "BTC": float(_close_trend_exposure(overlay["BTC-USD"], 50, 0.00).loc[:as_of].iloc[-1]),
        },
        dtype=float,
    )
    effective_sleeve = raw_sleeve.mul(exposure)
    effective_sleeve["Cash / Reduced Exposure"] = max(0.0, 1.0 - float(effective_sleeve.sum()))

    frames = []
    for sleeve in ["US Equity", "TH Equity"]:
        source = old_security.loc[old_security["Sleeve"].eq(sleeve)].copy()
        total_internal = source["Internal Weight"].sum()
        if total_internal > 0:
            source["Effective Weight"] = source["Internal Weight"] / total_internal * effective_sleeve[sleeve]
            source["Sleeve Multiplier"] = effective_sleeve[sleeve]
            frames.append(source)
    frames.append(
        pd.DataFrame(
            [
                {
                    "Asset": "GC=F",
                    "Internal Weight": 1.0,
                    "Sleeve": "Gold",
                    "Sleeve Multiplier": effective_sleeve["Gold"],
                    "Effective Weight": effective_sleeve["Gold"],
                    "Internal Weight Date": as_of.date().isoformat(),
                    "Date": as_of.date().isoformat(),
                },
                {
                    "Asset": "BTC-USD",
                    "Internal Weight": 1.0,
                    "Sleeve": "BTC",
                    "Sleeve Multiplier": effective_sleeve["BTC"],
                    "Effective Weight": effective_sleeve["BTC"],
                    "Internal Weight Date": as_of.date().isoformat(),
                    "Date": as_of.date().isoformat(),
                },
                {
                    "Asset": "Cash / Reduced Exposure",
                    "Internal Weight": 1.0,
                    "Sleeve": "Cash / Reduced Exposure",
                    "Sleeve Multiplier": effective_sleeve["Cash / Reduced Exposure"],
                    "Effective Weight": effective_sleeve["Cash / Reduced Exposure"],
                    "Internal Weight Date": as_of.date().isoformat(),
                    "Date": as_of.date().isoformat(),
                },
            ]
        )
    )
    security = pd.concat(frames, ignore_index=True)
    security["Strategy"] = STRATEGY
    security["Raw Sleeve Weight"] = security["Sleeve"].map(raw_sleeve).fillna(security["Sleeve Multiplier"])
    security["Daily Exposure"] = security["Sleeve"].map(exposure).fillna(1.0)
    security["Effective Weight %"] = security["Effective Weight"].mul(100.0)
    security = security.loc[
        security["Effective Weight"].abs().gt(1e-12)
        | security["Sleeve"].isin({"Gold", "BTC", "Cash / Reduced Exposure"})
    ].sort_values("Effective Weight", ascending=False)

    sleeve = effective_sleeve.rename("Effective Weight").reset_index().rename(columns={"index": "Sleeve"})
    sleeve["Raw Sleeve Weight"] = sleeve["Sleeve"].map(raw_sleeve).fillna(0.0)
    sleeve["Daily Exposure"] = sleeve["Sleeve"].map(exposure).fillna(1.0)
    sleeve["Date"] = as_of.date().isoformat()
    sleeve["Strategy"] = STRATEGY
    sleeve["Effective Weight %"] = sleeve["Effective Weight"].mul(100.0)

    gold_price = overlay["GC=F"].loc[:as_of].iloc[-1]
    gold_dd = gold_price / overlay["GC=F"].rolling(252, min_periods=63).max().loc[:as_of].iloc[-1] - 1.0
    meta = pd.DataFrame(
        [
            {
                "Date": as_of.date().isoformat(),
                "Strategy": STRATEGY,
                "Tactical Rule": "proxy_regime relative_return binary lb1 cap30 entry0 exit0 hold0 confirm1",
                "Overlay Mix": "Equity/Gold/BTC 65/25/10",
                "Daily Exposure": "US SPY MA300 below50%; TH SET MA200 below0%; Gold DD252 warn-8%->50%, crash-20%->50%, panic-30%+belowMA200+mom63<0->0%, recover-5%; BTC MA50 below0%",
                "TH Tactical Weight Inside Equity Sleeve": th_tactical_weight,
                "US Sleeve Internal Weight Date": old_meta.iloc[0].get("US Sleeve Internal Weight Date", ""),
                "TH Sleeve Internal Weight Date": old_meta.iloc[0].get("TH Sleeve Internal Weight Date", ""),
                "BTC Price Source": "yfinance fresh BTC-USD overlay-only refresh",
                "BTC Price": float(overlay["BTC-USD"].loc[:as_of].iloc[-1]),
                "BTC MA50": float(overlay["BTC-USD"].rolling(50, min_periods=20).mean().loc[:as_of].iloc[-1]),
                "BTC Daily Exposure": float(exposure["BTC"]),
                "Gold Price": float(gold_price),
                "Gold DD252": float(gold_dd),
                "Gold Daily Exposure": float(exposure["Gold"]),
                "Timing Note": "Overlay tickers refreshed from yfinance; US/TH internal weights reused from the last successful fresh PIT rerun and rescaled to 65/25/10.",
            }
        ]
    )

    security.to_csv(result_dir / f"{FINAL_PREFIX}_latest_effective_security_weights_thb.csv", index=False)
    sleeve.to_csv(result_dir / f"{FINAL_PREFIX}_latest_effective_sleeve_weights_thb.csv", index=False)
    meta.to_csv(result_dir / f"{FINAL_PREFIX}_latest_meta.csv", index=False)
    print(meta.to_string(index=False))
    print(sleeve.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(security[["Asset", "Sleeve", "Effective Weight", "Internal Weight", "Raw Sleeve Weight", "Daily Exposure"]].head(50).to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
