from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    import yfinance as yf
except Exception as exc:  # pragma: no cover
    raise RuntimeError("yfinance is required for the ETF universe backtest") from exc


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in [SRC, SCRIPTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dynamic_factor_copula import (  # noqa: E402
    EPSILON,
    build_momentum_signal,
    compute_feature_table,
    compute_port_opt_style_metrics,
    curve_from_returns,
    default_paths,
    get_set100_members_as_of,
    get_sp500_members_as_of,
    monthly_rebalance_dates,
    select_point_in_time_universe,
)
from run_us_th_joint_model import END_DATE, FEATURE_FLAGS, LOOKBACK_DAYS, N_CLUSTERS  # noqa: E402
from run_us_th_tactical_gold_crash_protection_sweep import _gold_crash_exposure  # noqa: E402
from run_us_th_tactical_gold_exposure_sweep import _load_overlay_inputs_from_cache  # noqa: E402
from run_us_th_tactical_one_model import (  # noqa: E402
    BTC_CAP,
    GOLD_CAP,
    INITIAL_VALUE,
    STOCK_CAP,
    TH_ASSETS,
    US_ASSETS,
    _load_full_us_th_overlay_panel_from_cache,
    _load_tactical_th_signal,
    _period_compare,
)
from run_us_th_tactical_perf_momentum import RISK_FREE_RATE, _close_trend_exposure  # noqa: E402
from us_th_pit_reselect_utils import drop_duplicate_share_classes, load_full_us_th_thb_panel  # noqa: E402


OUTPUT_PREFIX = "etf_universe_ex_thailand"
CASH_ASSET = "Cash / Reduced Exposure"
OVERLAY_ASSETS = ["GC=F", "BTC-USD"]
ETF_UNIVERSE_EX_THAILAND = [
    "SPY",
    "SPMO",
    "MTUM",
    "SCHG",
    "QQQ",
    "SMH",
    "XLK",
    "EWJ",
    "DXJ",
    "200A.T",
    "2644.T",
    "1615.T",
    "VGK",
    "EUFN",
    "SHLD",
    "DFNS.L",
    "MCHI",
    "KWEB",
    "CQQQ",
    "CHIQ",
    "EWT",
    "FLTW",
    "EWY",
    "FLKR",
    "091160.KS",
    "396500.KS",
    "INDA",
    "FLIN",
    "INQQ",
    "SMIN",
    "EPI",
]
FX_TICKERS = ["USDTHB=X", "JPYTHB=X", "KRWTHB=X", "GBPTHB=X"]
US_GROUP_CAP = 0.70
TH_GROUP_CAP = 0.30
FINAL_BEST_MIX = {"Equity": 0.65, "Gold": 0.25, "BTC": 0.10}
MAX_ETF_DAILY_MOVE = 0.80


def _ticker_currency(ticker: str) -> tuple[str, float]:
    if ticker.endswith(".T"):
        return "JPYTHB=X", 1.0
    if ticker.endswith(".KS"):
        return "KRWTHB=X", 1.0
    if ticker.endswith(".L"):
        return "GBPTHB=X", 0.01
    return "USDTHB=X", 1.0


def _extract_close_volume(raw: pd.DataFrame, tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.xs("Close", axis=1, level=1)
        volume = raw["Volume"] if "Volume" in raw.columns.get_level_values(0) else raw.xs("Volume", axis=1, level=1)
    else:
        close = raw
        volume = pd.DataFrame(index=raw.index)
    close = close.reindex(columns=tickers).sort_index()
    volume = volume.reindex(columns=tickers).sort_index().fillna(0.0)
    close.index = pd.to_datetime(close.index)
    volume.index = pd.to_datetime(volume.index)
    return close, volume


def _clean_price_spikes(prices: pd.DataFrame, max_abs_return: float = MAX_ETF_DAILY_MOVE) -> pd.DataFrame:
    cleaned = prices.astype(float).sort_index().copy()
    for column in cleaned.columns:
        series = cleaned[column]
        values = []
        last_valid = np.nan
        changed = False
        for value in series.to_numpy(dtype=float):
            if np.isnan(value) or value <= 0.0:
                values.append(np.nan)
                continue
            if not np.isnan(last_valid) and abs(value / last_valid - 1.0) > max_abs_return:
                values.append(last_valid)
                changed = True
                continue
            values.append(value)
            last_valid = value
        if changed:
            cleaned[column] = pd.Series(values, index=series.index).ffill()
    return cleaned


def _load_etf_panel(start_date: str, end_date: str, index: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = default_paths(ROOT)
    cache_prices = paths.local_cache_root / f"{OUTPUT_PREFIX}_prices_thb.parquet"
    cache_volumes = paths.local_cache_root / f"{OUTPUT_PREFIX}_volumes.parquet"
    cache_raw = paths.local_cache_root / f"{OUTPUT_PREFIX}_raw_prices.parquet"
    tickers = ETF_UNIVERSE_EX_THAILAND + FX_TICKERS

    if cache_prices.exists() and cache_volumes.exists() and cache_raw.exists():
        prices_thb = pd.read_parquet(cache_prices)
        volumes = pd.read_parquet(cache_volumes)
        raw_prices = pd.read_parquet(cache_raw)
        prices_thb.index = pd.to_datetime(prices_thb.index)
        volumes.index = pd.to_datetime(volumes.index)
        raw_prices.index = pd.to_datetime(raw_prices.index)
        if (
            all(t in prices_thb.columns for t in ETF_UNIVERSE_EX_THAILAND)
            and prices_thb.index.min() <= pd.Timestamp(start_date)
            and prices_thb.index.max() >= pd.Timestamp(end_date) - pd.Timedelta(days=7)
        ):
            prices_thb = _clean_price_spikes(prices_thb)
            raw_prices = _clean_price_spikes(raw_prices)
            prices_thb.to_parquet(cache_prices)
            raw_prices.to_parquet(cache_raw)
            return (
                prices_thb.reindex(index).ffill(),
                volumes.reindex(index).fillna(0.0),
                raw_prices.reindex(index).ffill(),
            )

    yf.set_tz_cache_location(str(paths.local_cache_root / ".yfinance"))
    raw = yf.download(
        tickers,
        start=start_date,
        end=(pd.Timestamp(end_date) + pd.Timedelta(days=1)).date().isoformat(),
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=False,
    )
    close, volumes = _extract_close_volume(raw, tickers)
    raw_prices = _clean_price_spikes(close.reindex(columns=ETF_UNIVERSE_EX_THAILAND).ffill())
    prices_thb = pd.DataFrame(index=close.index)
    for ticker in ETF_UNIVERSE_EX_THAILAND:
        fx_ticker, scale = _ticker_currency(ticker)
        prices_thb[ticker] = raw_prices[ticker].mul(scale).mul(close[fx_ticker].ffill())
    prices_thb = _clean_price_spikes(prices_thb.sort_index().ffill())
    volumes = volumes.reindex(columns=ETF_UNIVERSE_EX_THAILAND).fillna(0.0)
    prices_thb.to_parquet(cache_prices)
    volumes.to_parquet(cache_volumes)
    raw_prices.to_parquet(cache_raw)
    return prices_thb.reindex(index).ffill(), volumes.reindex(index).fillna(0.0), raw_prices.reindex(index).ffill()


def _load_panel_with_etfs() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str], list[str], list[str], pd.DataFrame]:
    prices, volumes, benchmark, vol_proxy, us_all, th_all = load_full_us_th_thb_panel(
        include_overlay_assets=True,
        overlay_asset_tickers=OVERLAY_ASSETS,
        end_date=END_DATE,
    )
    etf_prices, etf_volumes, etf_raw_prices = _load_etf_panel(str(prices.index.min().date()), END_DATE, prices.index)
    etf_prices = etf_prices.loc[:, etf_prices.notna().sum().gt(260)]
    etf_volumes = etf_volumes.reindex(columns=etf_prices.columns).fillna(0.0)
    etfs = etf_prices.columns.tolist()
    prices = pd.concat([prices, etf_prices], axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated(keep="last")].sort_index().ffill()
    volumes = pd.concat([volumes, etf_volumes], axis=1)
    volumes = volumes.loc[:, ~volumes.columns.duplicated(keep="last")].reindex(prices.index).fillna(0.0)
    return prices, volumes, benchmark.reindex(prices.index).ffill(), vol_proxy.reindex(prices.index).ffill(), us_all, th_all, etfs, etf_raw_prices


def _optimize_with_group_caps(
    cov: pd.DataFrame,
    momentum_signal: pd.Series,
    us_group_assets: list[str],
    th_assets: list[str],
    caps: dict[str, float],
    us_cap: float = US_GROUP_CAP,
    th_cap: float = TH_GROUP_CAP,
    add_cash: bool = True,
) -> pd.Series:
    base_assets = cov.index.tolist()
    cash_cap = max(0.0, 1.0 - (us_cap + th_cap + GOLD_CAP + BTC_CAP))
    assets = base_assets + ([CASH_ASSET] if add_cash and cash_cap > EPSILON else [])
    cov2 = cov.reindex(index=assets, columns=assets).fillna(0.0)
    if CASH_ASSET in cov2.index:
        cov2.loc[CASH_ASSET, CASH_ASSET] = EPSILON
    mu = momentum_signal.reindex(assets).fillna(0.0)
    if mu.dropna().shape[0] > 2:
        mu = mu.clip(mu.quantile(0.10), mu.quantile(0.90))

    cap_series = pd.Series({asset: caps.get(asset, STOCK_CAP) for asset in assets}, dtype=float)
    if CASH_ASSET in cap_series.index:
        cap_series.loc[CASH_ASSET] = max(cash_cap, EPSILON)
    if float(cap_series.sum()) < 1.0 - EPSILON:
        raise RuntimeError(f"Infeasible caps: {cap_series.sum():.4f}")

    cov_matrix = cov2.to_numpy(dtype=float)
    mu_vec = mu.to_numpy(dtype=float)
    x0 = (cap_series / cap_series.sum()).to_numpy(dtype=float)
    bounds = [(0.0, float(cap_series.loc[asset])) for asset in assets]
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
    us_idx = [assets.index(asset) for asset in us_group_assets if asset in assets]
    th_idx = [assets.index(asset) for asset in th_assets if asset in assets]
    if us_idx:
        constraints.append({"type": "ineq", "fun": lambda x, idx=us_idx: us_cap - float(np.sum(x[idx]))})
    if th_idx:
        constraints.append({"type": "ineq", "fun": lambda x, idx=th_idx: th_cap - float(np.sum(x[idx]))})

    def objective(x: np.ndarray) -> float:
        variance = float(x @ cov_matrix @ x)
        expected = float(mu_vec @ x)
        concentration = float(np.sum(np.square(x)))
        cash_penalty = 0.01 * float(x[assets.index(CASH_ASSET)]) if CASH_ASSET in assets else 0.0
        return 0.5 * 8.0 * variance - expected + 0.02 * concentration + cash_penalty

    result = minimize(objective, x0=x0, bounds=bounds, constraints=constraints, method="SLSQP")
    weights = pd.Series(result.x if result.success else x0, index=assets).clip(lower=0.0)
    return weights / weights.sum()


def _metrics_row(curve: pd.Series, strategy: str, weights: pd.DataFrame, etfs: list[str]) -> dict[str, object]:
    clean = curve.dropna()
    us_stock_cols = [c for c in weights.columns if c not in OVERLAY_ASSETS + [CASH_ASSET] and not c.endswith(".BK") and c not in etfs]
    th_cols = [c for c in weights.columns if c.endswith(".BK")]
    row = compute_port_opt_style_metrics(clean, risk_free_rate=RISK_FREE_RATE).to_dict()
    row.update(
        {
            "Strategy": strategy,
            "Start": clean.index.min().date().isoformat(),
            "End": clean.index.max().date().isoformat(),
            "Average US Stock Weight": float(weights[us_stock_cols].sum(axis=1).mean()) if us_stock_cols else 0.0,
            "Average ETF Weight": float(weights[[c for c in etfs if c in weights]].sum(axis=1).mean()) if etfs else 0.0,
            "Average TH Stock Weight": float(weights[th_cols].sum(axis=1).mean()) if th_cols else 0.0,
            "Average Gold Weight": float(weights["GC=F"].mean() if "GC=F" in weights else 0.0),
            "Average BTC Weight": float(weights["BTC-USD"].mean() if "BTC-USD" in weights else 0.0),
            "Average Cash / Reduced Exposure Weight": float(weights[CASH_ASSET].mean() if CASH_ASSET in weights else 0.0),
        }
    )
    return row


def main() -> None:
    paths = default_paths(ROOT)
    prices, volumes, benchmark, vol_proxy, us_all, th_all, etfs, etf_raw_prices = _load_panel_with_etfs()
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    benchmark_ret = benchmark.pct_change(fill_method=None).rename("benchmark")
    vol_proxy_ret = vol_proxy.pct_change(fill_method=None).rename("vol_proxy")
    schedule = monthly_rebalance_dates(prices.index, lookback_days=LOOKBACK_DAYS, freq="ME")
    best_signal_name, th_signal = _load_tactical_th_signal(prices.index)

    all_asset_weights = pd.DataFrame(index=prices.index, columns=list(prices.columns) + [CASH_ASSET], dtype=float)
    final_mix_weights = pd.DataFrame(index=prices.index, columns=list(prices.columns) + [CASH_ASSET], dtype=float)
    selected_rows = []

    for idx, rebalance_date in enumerate(schedule[:-1]):
        next_date = schedule[idx + 1]
        loc = prices.index.get_loc(rebalance_date)
        train_index = prices.index[max(0, loc - LOOKBACK_DAYS + 1) : loc + 1]
        test_index = prices.index[(prices.index > rebalance_date) & (prices.index <= next_date)]
        if len(test_index) == 0:
            continue

        us_pool = [ticker for ticker in get_sp500_members_as_of(rebalance_date, paths) if ticker in us_all and ticker in prices.columns]
        us_pool = drop_duplicate_share_classes(us_pool)
        th_is_on = float(th_signal.loc[:rebalance_date].iloc[-1]) > 1e-12
        th_pool = [ticker for ticker in get_set100_members_as_of(rebalance_date, paths) if ticker in th_all and ticker in prices.columns] if th_is_on else []
        us_selected = select_point_in_time_universe(prices.loc[train_index], volumes.loc[train_index], us_pool, n_assets=US_ASSETS)
        th_selected = select_point_in_time_universe(prices.loc[train_index], volumes.loc[train_index], th_pool, n_assets=TH_ASSETS) if th_is_on else []
        etf_selected = [
            ticker
            for ticker in etfs
            if ticker in prices.columns and prices.loc[train_index, ticker].dropna().shape[0] >= max(260, int(0.70 * len(train_index)))
        ]

        current_assets = list(dict.fromkeys(us_selected + th_selected + etf_selected + OVERLAY_ASSETS))
        train_returns = returns.reindex(train_index)[current_assets].dropna(axis=1, thresh=max(int(0.85 * len(train_index)), 60))
        if train_returns.shape[1] < max(N_CLUSTERS + 2, 6):
            continue
        current_assets = train_returns.columns.tolist()
        feature_table = compute_feature_table(
            train_returns,
            benchmark_ret.reindex(train_index),
            vol_proxy_ret.reindex(train_index),
            prices.reindex(train_index)[current_assets],
            include_momentum_features=True,
            feature_flags=FEATURE_FLAGS,
        )
        if feature_table.empty:
            continue
        momentum_signal = build_momentum_signal(feature_table, mode="mom_63")

        us_group_assets = [asset for asset in current_assets if asset in us_selected or asset in etfs]
        th_group_assets = [asset for asset in current_assets if asset in th_selected]
        caps = {asset: STOCK_CAP for asset in current_assets}
        caps.update({"GC=F": GOLD_CAP, "BTC-USD": BTC_CAP})
        cov = train_returns.cov().reindex(index=current_assets, columns=current_assets).fillna(0.0)
        weights_all = _optimize_with_group_caps(cov, momentum_signal, us_group_assets, th_group_assets, caps)

        equity_assets = [asset for asset in current_assets if asset not in OVERLAY_ASSETS]
        equity_returns = train_returns[equity_assets]
        equity_cov = equity_returns.cov().reindex(index=equity_assets, columns=equity_assets).fillna(0.0)
        equity_signal = momentum_signal.reindex(equity_assets).fillna(0.0)
        equity_caps = {asset: STOCK_CAP for asset in equity_assets}
        equity_weights = _optimize_with_group_caps(
            equity_cov,
            equity_signal,
            [asset for asset in equity_assets if asset in us_group_assets],
            [asset for asset in equity_assets if asset in th_group_assets],
            equity_caps,
            add_cash=False,
        )
        weights_final = equity_weights.mul(FINAL_BEST_MIX["Equity"])
        weights_final.loc["GC=F"] = FINAL_BEST_MIX["Gold"]
        weights_final.loc["BTC-USD"] = FINAL_BEST_MIX["BTC"]
        weights_final = weights_final / weights_final.sum()

        for target, weights in [(all_asset_weights, weights_all), (final_mix_weights, weights_final)]:
            target.loc[test_index, :] = 0.0
            target.loc[test_index, weights.index] = np.tile(weights.to_numpy(dtype=float), (len(test_index), 1))

        selected_rows.append(
            {
                "Date": rebalance_date.date().isoformat(),
                "TH Signal On": th_is_on,
                "US Count": len([asset for asset in current_assets if asset in us_selected]),
                "ETF Count": len([asset for asset in current_assets if asset in etfs]),
                "TH Count": len([asset for asset in current_assets if asset in th_selected]),
                "US/ETF Weight Strategy 1": float(weights_all.reindex(us_group_assets).fillna(0.0).sum()),
                "TH Weight Strategy 1": float(weights_all.reindex(th_group_assets).fillna(0.0).sum()),
                "Gold Weight Strategy 1": float(weights_all.get("GC=F", 0.0)),
                "BTC Weight Strategy 1": float(weights_all.get("BTC-USD", 0.0)),
                "Assets": ",".join(current_assets),
            }
        )

    raw_all_weights = all_asset_weights.ffill().fillna(0.0)
    raw_final_weights = final_mix_weights.ffill().fillna(0.0)
    start_idx = raw_all_weights.sum(axis=1).gt(0.0)
    index = raw_all_weights.index[start_idx]
    raw_all_weights = raw_all_weights.reindex(index).fillna(0.0)
    raw_final_weights = raw_final_weights.reindex(index).fillna(0.0)

    _, signal_prices = _load_overlay_inputs_from_cache(index)
    exposure = pd.DataFrame(
        {
            "US/ETF": _close_trend_exposure(signal_prices["US Equity"], 300, 0.50),
            "TH": _close_trend_exposure(signal_prices["TH Equity"], 200, 0.00),
            "GC=F": _gold_crash_exposure(
                signal_prices["Gold"],
                dd_window=252,
                warn_dd=-0.08,
                crash_dd=-0.20,
                warn_exposure=0.50,
                crash_exposure=0.50,
                recovery_dd=-0.05,
                panic_dd=-0.30,
                panic_ma_period=200,
                panic_mom_period=63,
            ),
            "BTC-USD": _close_trend_exposure(signal_prices["BTC"], 50, 0.00),
        },
        index=index,
    ).ffill().fillna(1.0).clip(0.0, 1.0)

    def apply_exposure(weights: pd.DataFrame) -> pd.DataFrame:
        effective = weights.reindex(index).ffill().fillna(0.0).copy()
        for column in effective.columns:
            if column == CASH_ASSET:
                continue
            if column == "GC=F":
                effective[column] *= exposure["GC=F"]
            elif column == "BTC-USD":
                effective[column] *= exposure["BTC-USD"]
            elif column.endswith(".BK"):
                effective[column] *= exposure["TH"]
            else:
                effective[column] *= exposure["US/ETF"]
        effective[CASH_ASSET] = (1.0 - effective.drop(columns=[CASH_ASSET], errors="ignore").sum(axis=1)).clip(lower=0.0)
        return effective

    effective_all = apply_exposure(raw_all_weights)
    effective_final = apply_exposure(raw_final_weights)
    aligned_returns = returns.reindex(index).fillna(0.0)

    def make_curve(weights: pd.DataFrame, name: str) -> pd.Series:
        asset_cols = [column for column in weights.columns if column in aligned_returns.columns]
        strategy_returns = aligned_returns[asset_cols].mul(weights[asset_cols], axis=1).sum(axis=1)
        return curve_from_returns(strategy_returns, initial=INITIAL_VALUE).rename(name)

    strategy_1 = "One-model US/ETF cap 70% / TH cap 30% with daily exposure"
    strategy_2 = "US/TH/ETF tactical final best Sharpe 65/25/10 with Gold crash protection"
    curves = pd.concat([make_curve(effective_all, strategy_1), make_curve(effective_final, strategy_2)], axis=1).dropna(how="all")
    summary = pd.DataFrame(
        [
            _metrics_row(curves[strategy_1], strategy_1, effective_all, etfs),
            _metrics_row(curves[strategy_2], strategy_2, effective_final, etfs),
        ]
    ).sort_values(["Sharpe", "CAGR"], ascending=False)
    summary["ETF Universe"] = ", ".join(ETF_UNIVERSE_EX_THAILAND)
    summary["Tactical TH Signal"] = best_signal_name
    summary["US/ETF Group Cap"] = US_GROUP_CAP
    summary["TH Group Cap"] = TH_GROUP_CAP

    latest_frames = []
    for strategy, weights in [(strategy_1, effective_all), (strategy_2, effective_final)]:
        latest = weights.iloc[-1].rename("Effective Weight").reset_index().rename(columns={"index": "Asset"})
        latest = latest.loc[latest["Effective Weight"].abs().gt(1e-12)].sort_values("Effective Weight", ascending=False)
        latest.insert(0, "Strategy", strategy)
        latest["Date"] = weights.index.max().date().isoformat()
        latest["Sleeve"] = latest["Asset"].map(
            lambda asset: "Gold" if asset == "GC=F" else "BTC" if asset == "BTC-USD" else "Cash" if asset == CASH_ASSET else "TH Stock" if str(asset).endswith(".BK") else "ETF" if asset in etfs else "US Stock"
        )
        latest_frames.append(latest)

    coverage = pd.DataFrame(
        {
            "Ticker": ETF_UNIVERSE_EX_THAILAND,
            "Available": [ticker in etfs for ticker in ETF_UNIVERSE_EX_THAILAND],
            "First Date": [etf_raw_prices[ticker].dropna().index.min().date().isoformat() if ticker in etf_raw_prices and not etf_raw_prices[ticker].dropna().empty else "" for ticker in ETF_UNIVERSE_EX_THAILAND],
            "Last Date": [etf_raw_prices[ticker].dropna().index.max().date().isoformat() if ticker in etf_raw_prices and not etf_raw_prices[ticker].dropna().empty else "" for ticker in ETF_UNIVERSE_EX_THAILAND],
            "Observations": [int(etf_raw_prices[ticker].dropna().shape[0]) if ticker in etf_raw_prices else 0 for ticker in ETF_UNIVERSE_EX_THAILAND],
        }
    )

    curves.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_curves_thb.csv")
    summary.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_summary_thb.csv", index=False)
    effective_all.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_strategy1_effective_weights_thb.csv")
    effective_final.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_strategy2_effective_weights_thb.csv")
    pd.concat(latest_frames, ignore_index=True).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_latest_weights_thb.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_universe_history_thb.csv", index=False)
    _period_compare(curves).to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_period_compare_thb.csv", index=False)
    coverage.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_coverage.csv", index=False)
    exposure.to_csv(paths.result_dir / f"{OUTPUT_PREFIX}_daily_exposure_history.csv")

    cols = [
        "Strategy",
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
    print(summary[cols].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
