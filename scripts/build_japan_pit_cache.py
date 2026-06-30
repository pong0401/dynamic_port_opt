from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynamic_factor_copula import default_paths  # noqa: E402


API_ROOT = "https://api.jquants.com"
DEFAULT_MARKETS = ("Prime", "Standard", "Growth")
DEFAULT_TOP_N = 20
DEFAULT_LIQUIDITY_LOOKBACK = 60
DEFAULT_MIN_DAYS_TRADED = 50
DEFAULT_SLEEP_SECONDS = 1.0
DEFAULT_KEY_FILE = ROOT / "key" / "jquant.key"
RATE_LIMIT_BACKOFF_SECONDS = (10, 30, 60)


@dataclass(frozen=True)
class JapanPitConfig:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    requested_years: int
    fallback_years: int
    top_n: int
    liquidity_lookback: int
    min_days_traded: int
    markets: tuple[str, ...]
    sleep_seconds: float
    max_rebalances: Optional[int]
    flush_daily_cache_every: int
    dry_run: bool


class JQuantsClient:
    def __init__(
        self,
        id_token: Optional[str] = None,
        refresh_token: Optional[str] = None,
        email: Optional[str] = None,
        password: Optional[str] = None,
        api_root: str = API_ROOT,
        sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
        daily_bars_cache_path: Optional[Path] = None,
        flush_daily_cache_every: int = 5,
    ) -> None:
        self.api_root = api_root.rstrip("/")
        self.sleep_seconds = sleep_seconds
        self.id_token = id_token or os.getenv("JQUANTS_ID_TOKEN")
        self.api_key = os.getenv("JQUANTS_API_KEY") or _read_secret_file(DEFAULT_KEY_FILE)
        self.refresh_token = (
            refresh_token
            or os.getenv("JQUANTS_REFRESH_TOKEN")
        )
        self._daily_bars_by_date: dict[str, pd.DataFrame] = {}
        self._daily_bars_cache_path = daily_bars_cache_path
        self._flush_daily_cache_every = max(1, flush_daily_cache_every)
        self._new_daily_bar_dates = 0
        self._load_daily_bars_cache()
        self.email = email or os.getenv("JQUANTS_EMAIL") or os.getenv("JQUANTS_MAILADDRESS")
        self.password = password or os.getenv("JQUANTS_PASSWORD")

    def authenticate(self) -> None:
        if self.id_token or self.api_key:
            return
        if not self.refresh_token:
            if not self.email or not self.password:
                raise RuntimeError(
                    "J-Quants credentials are missing. Set JQUANTS_ID_TOKEN, "
                    "or JQUANTS_REFRESH_TOKEN/JQUANTS_API_KEY, or JQUANTS_EMAIL/JQUANTS_PASSWORD."
                )
            payload = {"mailaddress": self.email, "password": self.password}
            response = self._request_json(
                "/v2/token/auth_user",
                method="POST",
                payload=payload,
                requires_auth=False,
            )
            self.refresh_token = response.get("refreshToken")
            if not self.refresh_token:
                raise RuntimeError("J-Quants auth_user did not return refreshToken.")
        response = self._request_json(
            "/v2/token/auth_refresh",
            params={"refreshtoken": self.refresh_token},
            requires_auth=False,
        )
        self.id_token = response.get("idToken")
        if not self.id_token:
            raise RuntimeError("J-Quants auth_refresh did not return idToken.")

    def get_calendar(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        data = self._get_paginated(
            "/v2/markets/calendar",
            params={"from": _date_param(start), "to": _date_param(end)},
            preferred_keys=("calendar", "market_calendar", "trading_calendar"),
        )
        return pd.DataFrame(data)

    def get_master(self, date: pd.Timestamp) -> pd.DataFrame:
        data = self._get_paginated(
            "/v2/equities/master",
            params={"date": _date_param(date)},
            preferred_keys=("master", "info", "listed_info", "listedIssue"),
        )
        return pd.DataFrame(data)

    def get_daily_bars(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        data = self._get_paginated(
            "/v2/equities/bars/daily",
            params={"from": _date_param(start), "to": _date_param(end)},
            preferred_keys=("daily_bars", "daily_quotes", "prices", "bars"),
        )
        return pd.DataFrame(data)

    def get_daily_bars_for_dates(self, dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
        frames = []
        for date in dates:
            key = _date_param(pd.Timestamp(date))
            if key in self._daily_bars_by_date:
                frames.append(self._daily_bars_by_date[key])
                continue
            data = self._get_paginated(
                "/v2/equities/bars/daily",
                params={"date": key},
                preferred_keys=("daily_bars", "daily_quotes", "prices", "bars"),
            )
            if data:
                frame = pd.DataFrame(data)
                self._daily_bars_by_date[key] = frame
                frames.append(frame)
                self._new_daily_bar_dates += 1
                if self._new_daily_bar_dates >= self._flush_daily_cache_every:
                    self.flush_daily_bars_cache()
            else:
                self._daily_bars_by_date[key] = pd.DataFrame()
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def flush_daily_bars_cache(self) -> None:
        if not self._daily_bars_cache_path or not self._daily_bars_by_date:
            return
        frames = [frame for frame in self._daily_bars_by_date.values() if not frame.empty]
        if not frames:
            return
        cache = pd.concat(frames, ignore_index=True)
        if "Date" in cache.columns:
            cache["Date"] = pd.to_datetime(cache["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        if "Code" in cache.columns:
            cache["Code"] = cache["Code"].astype(str).str.strip()
        if "Date" in cache.columns and "Code" in cache.columns:
            cache = cache.dropna(subset=["Date", "Code"]).drop_duplicates(["Date", "Code"])
        self._daily_bars_cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache.to_parquet(self._daily_bars_cache_path, index=False)
        self._new_daily_bar_dates = 0

    def _load_daily_bars_cache(self) -> None:
        if not self._daily_bars_cache_path or not self._daily_bars_cache_path.exists():
            return
        cache = pd.read_parquet(self._daily_bars_cache_path)
        if cache.empty or "Date" not in cache.columns:
            return
        cache_dates = pd.to_datetime(cache["Date"], errors="coerce")
        for date, frame in cache.assign(Date=cache_dates).dropna(subset=["Date"]).groupby("Date"):
            self._daily_bars_by_date[_date_param(pd.Timestamp(date))] = frame.copy()

    def _get_paginated(
        self,
        path: str,
        params: dict[str, str],
        preferred_keys: tuple[str, ...],
    ) -> list[dict]:
        rows: list[dict] = []
        page_params = dict(params)
        while True:
            payload = self._request_json(path, params=page_params)
            rows.extend(_extract_rows(payload, preferred_keys))
            pagination_key = payload.get("pagination_key") or payload.get("paginationKey")
            if not pagination_key:
                break
            page_params["pagination_key"] = pagination_key
        return rows

    def _request_json(
        self,
        path: str,
        method: str = "GET",
        params: Optional[dict[str, str]] = None,
        payload: Optional[dict[str, str]] = None,
        requires_auth: bool = True,
    ) -> dict:
        if requires_auth and not self.id_token:
            self.authenticate()
        url = f"{self.api_root}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if requires_auth:
            if self.api_key:
                headers["x-api-key"] = self.api_key
            else:
                headers["Authorization"] = f"Bearer {self.id_token}"
        raw = self._open_with_retries(url, body, headers, method, path)
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)
        return json.loads(raw)

    def _open_with_retries(
        self,
        url: str,
        body: Optional[bytes],
        headers: dict[str, str],
        method: str,
        path: str,
    ) -> str:
        for attempt in range(len(RATE_LIMIT_BACKOFF_SECONDS) + 1):
            request = urllib.request.Request(url, data=body, headers=headers, method=method)
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    return response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code == 429 and attempt < len(RATE_LIMIT_BACKOFF_SECONDS):
                    wait = RATE_LIMIT_BACKOFF_SECONDS[attempt]
                    print(f"Rate limited on {path}; sleeping {wait}s before retry {attempt + 1}.")
                    time.sleep(wait)
                    continue
                subscription = _parse_subscription_window(detail)
                if subscription:
                    start, end = subscription
                    raise RuntimeError(
                        f"J-Quants request failed {exc.code} for {path}: subscription covers "
                        f"{start} to {end}; adjust --start-date/--end-date or use --years 2 for the free plan."
                    ) from exc
                raise RuntimeError(f"J-Quants request failed {exc.code} for {path}: {detail}") from exc
        raise RuntimeError(f"J-Quants request failed for {path}: exhausted retries.")


def _date_param(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def _read_secret_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def _parse_subscription_window(detail: str) -> Optional[tuple[str, str]]:
    match = re.search(r"(\d{4}-\d{2}-\d{2})\s*~\s*(\d{4}-\d{2}-\d{2})", detail)
    if not match:
        return None
    return match.group(1), match.group(2)


def _extract_rows(payload: dict, preferred_keys: Iterable[str]) -> list[dict]:
    for key in preferred_keys:
        value = payload.get(key)
        if isinstance(value, list):
            return value
    for value in payload.values():
        if isinstance(value, list):
            return value
    return []


def _normalize_date_column(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    date_col = _first_existing(frame, ("Date", "date", "BusinessDate", "business_date"))
    if date_col is None:
        raise ValueError(f"Could not find date column in {list(frame.columns)}")
    frame["Date"] = pd.to_datetime(frame[date_col], errors="coerce")
    return frame.dropna(subset=["Date"])


def _first_existing(frame: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    columns = set(frame.columns)
    for name in candidates:
        if name in columns:
            return name
    return None


def _trading_days_from_calendar(calendar: pd.DataFrame) -> pd.DatetimeIndex:
    calendar = _normalize_date_column(calendar)
    if calendar.empty:
        return pd.DatetimeIndex([])
    business_col = _first_existing(
        calendar,
        (
            "is_tse_business_day",
            "IsTSEBusinessDay",
            "isTSEBusinessDay",
            "HolDiv",
            "HolidayDivision",
            "holiday_division",
        ),
    )
    if business_col is None:
        return pd.DatetimeIndex(calendar["Date"].drop_duplicates().sort_values())
    values = calendar[business_col]
    if values.dtype == bool:
        mask = values
    elif business_col in {"HolDiv", "HolidayDivision", "holiday_division"}:
        mask = values.astype(str).str.strip().eq("1")
    else:
        normalized = values.astype(str).str.strip().str.lower()
        mask = normalized.isin(("true", "1", "businessday", "business_day"))
    return pd.DatetimeIndex(calendar.loc[mask, "Date"].drop_duplicates().sort_values())


def _month_end_signal_dates(trading_days: pd.DatetimeIndex) -> list[pd.Timestamp]:
    if trading_days.empty:
        return []
    return pd.Series(trading_days, index=trading_days).resample("ME").last().dropna().tolist()


def _next_trading_day(trading_days: pd.DatetimeIndex, signal_date: pd.Timestamp) -> Optional[pd.Timestamp]:
    future = trading_days[trading_days > pd.Timestamp(signal_date)]
    if future.empty:
        return None
    return pd.Timestamp(future[0])


def _last_n_trading_days(
    trading_days: pd.DatetimeIndex,
    signal_date: pd.Timestamp,
    count: int,
) -> pd.DatetimeIndex:
    dates = trading_days[trading_days <= pd.Timestamp(signal_date)]
    return dates[-count:]


def _normalize_master(master: pd.DataFrame) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame(columns=["Code", "CoName", "CoNameEn", "MktNm", "S33Nm", "S17Nm", "ScaleCat"])
    frame = master.copy()
    rename_map = {
        "code": "Code",
        "LocalCode": "Code",
        "CompanyName": "CoName",
        "CompanyNameEnglish": "CoNameEn",
        "MarketCodeName": "MktNm",
        "MarketName": "MktNm",
        "Sector33CodeName": "S33Nm",
        "Sector17CodeName": "S17Nm",
        "ScaleCategory": "ScaleCat",
    }
    frame = frame.rename(columns={old: new for old, new in rename_map.items() if old in frame.columns})
    if "Code" not in frame.columns:
        raise ValueError(f"Could not find Code column in master columns: {list(frame.columns)}")
    for col in ["CoName", "CoNameEn", "MktNm", "S33Nm", "S17Nm", "ScaleCat"]:
        if col not in frame.columns:
            frame[col] = ""
    frame["Code"] = frame["Code"].astype(str).str.strip()
    return frame


def _normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame(columns=["Date", "Code", "Close", "Volume", "TradingValue"])
    frame = _normalize_date_column(bars)
    rename_map = {
        "code": "Code",
        "LocalCode": "Code",
        "AdjustmentClose": "AdjClose",
        "AdjustedClose": "AdjClose",
        "AdjustmentVolume": "AdjVolume",
        "Volume": "Volume",
        "TradingVolume": "Volume",
        "TurnoverValue": "TradingValue",
        "TradingValue": "TradingValue",
    }
    frame = frame.rename(columns={old: new for old, new in rename_map.items() if old in frame.columns})
    if "Code" not in frame.columns:
        raise ValueError(f"Could not find Code column in daily bar columns: {list(frame.columns)}")
    close_col = _first_existing(frame, ("AdjClose", "AdjustmentClose", "Close", "C"))
    volume_col = _first_existing(frame, ("AdjVolume", "Vo", "Volume", "TradingVolume", "V"))
    value_col = _first_existing(frame, ("Va", "TradingValue", "TurnoverValue"))
    if close_col is None:
        raise ValueError(f"Could not find close column in daily bar columns: {list(frame.columns)}")
    frame["Code"] = frame["Code"].astype(str).str.strip()
    frame["Close"] = pd.to_numeric(frame[close_col], errors="coerce")
    frame["Volume"] = pd.to_numeric(frame[volume_col], errors="coerce") if volume_col else 0.0
    if value_col:
        frame["TradingValue"] = pd.to_numeric(frame[value_col], errors="coerce")
    else:
        frame["TradingValue"] = frame["Close"] * frame["Volume"]
    return frame.dropna(subset=["Date", "Code"])[["Date", "Code", "Close", "Volume", "TradingValue"]]


def _filter_master(master: pd.DataFrame, markets: tuple[str, ...]) -> pd.DataFrame:
    frame = _normalize_master(master)
    if frame.empty:
        return frame
    if "MktNm" not in frame.columns:
        return frame
    allowed = {_normalize_market_name(market) for market in markets}
    market_names = frame["MktNm"].map(_normalize_market_name)
    return frame.loc[market_names.isin(allowed)].copy()


def _normalize_market_name(value: object) -> str:
    normalized = str(value).strip().lower().replace(" market", "")
    aliases = {
        "プライム": "prime",
        "スタンダード": "standard",
        "グロース": "growth",
    }
    return aliases.get(normalized, normalized)


def _rank_liquidity(
    bars: pd.DataFrame,
    universe: pd.DataFrame,
    signal_date: pd.Timestamp,
    entry_date: Optional[pd.Timestamp],
    top_n: int,
    min_days_traded: int,
) -> pd.DataFrame:
    if bars.empty or universe.empty:
        return pd.DataFrame()
    enriched = bars.merge(
        universe[["Code", "CoName", "CoNameEn", "MktNm", "S33Nm", "S17Nm", "ScaleCat"]],
        on="Code",
        how="inner",
    )
    if enriched.empty:
        return pd.DataFrame()
    grouped = (
        enriched.sort_values(["Code", "Date"])
        .groupby("Code")
        .agg(
            avg_trading_value_20d=("TradingValue", lambda values: values.tail(20).mean()),
            avg_trading_value_60d=("TradingValue", "mean"),
            days_traded=("TradingValue", "count"),
            last_close=("Close", "last"),
            company=("CoName", "last"),
            company_en=("CoNameEn", "last"),
            market=("MktNm", "last"),
            sector_33=("S33Nm", "last"),
            sector_17=("S17Nm", "last"),
            scale_category=("ScaleCat", "last"),
        )
        .reset_index()
    )
    grouped = grouped.loc[
        (grouped["days_traded"] >= min_days_traded)
        & (grouped["avg_trading_value_60d"] > 0)
    ].copy()
    grouped = grouped.sort_values("avg_trading_value_60d", ascending=False).head(top_n).copy()
    grouped["rank"] = range(1, len(grouped) + 1)
    grouped["signal_date"] = pd.Timestamp(signal_date).date().isoformat()
    grouped["entry_date"] = pd.Timestamp(entry_date).date().isoformat() if entry_date is not None else ""
    return grouped[
        [
            "signal_date",
            "entry_date",
            "rank",
            "Code",
            "company",
            "company_en",
            "market",
            "sector_33",
            "sector_17",
            "scale_category",
            "avg_trading_value_20d",
            "avg_trading_value_60d",
            "days_traded",
            "last_close",
        ]
    ]


def _persist_price_volume_cache(paths, bars: pd.DataFrame, tickers: Optional[Iterable[str]] = None) -> tuple[int, int]:
    if bars.empty:
        return (0, 0)
    if tickers is not None:
        selected = {str(ticker).strip() for ticker in tickers if str(ticker).strip()}
        bars = bars.loc[bars["Code"].astype(str).str.strip().isin(selected)].copy()
        if bars.empty:
            return (0, 0)
    prices = bars.pivot_table(index="Date", columns="Code", values="Close", aggfunc="last").sort_index()
    volumes = bars.pivot_table(index="Date", columns="Code", values="Volume", aggfunc="last").sort_index()
    paths.local_cache_root.mkdir(parents=True, exist_ok=True)
    japan_price_file = paths.local_cache_root / "japan_selected_prices.parquet"
    japan_volume_file = paths.local_cache_root / "japan_selected_volumes.parquet"
    if japan_price_file.exists():
        prices = pd.read_parquet(japan_price_file).combine_first(prices)
    if japan_volume_file.exists():
        volumes = pd.read_parquet(japan_volume_file).combine_first(volumes)
    prices.sort_index().to_parquet(japan_price_file)
    volumes.sort_index().to_parquet(japan_volume_file)
    return (len(prices.columns), len(volumes.columns))


def _read_checkpoint(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def _write_atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _append_checkpoint(frame: pd.DataFrame, path: Path, subset: list[str]) -> pd.DataFrame:
    prior = _read_checkpoint(path)
    combined = pd.concat([prior, frame], ignore_index=True) if not prior.empty else frame.copy()
    if not combined.empty:
        combined = combined.drop_duplicates(subset, keep="last")
    _write_atomic_parquet(combined, path)
    return combined


def _completed_signal_dates(paths) -> set[str]:
    checkpoint = paths.local_cache_root / "japan_pit_universe_history.parquet"
    if not checkpoint.exists():
        return set()
    history = pd.read_parquet(checkpoint)
    if history.empty or "signal_date" not in history.columns:
        return set()
    return set(history["signal_date"].dropna().astype(str).unique().tolist())


def _choose_window(config: JapanPitConfig, client: JQuantsClient) -> tuple[pd.Timestamp, pd.Timestamp, int, bool, pd.DataFrame, pd.DatetimeIndex]:
    attempts = [
        (config.requested_years, config.start_date),
        (config.fallback_years, config.end_date - pd.DateOffset(years=config.fallback_years)),
    ]
    seen: set[int] = set()
    for years, start in attempts:
        if years in seen:
            continue
        seen.add(years)
        calendar = client.get_calendar(start, config.end_date)
        trading_days = _trading_days_from_calendar(calendar)
        if len(trading_days) >= years * 200:
            return (pd.Timestamp(start), config.end_date, years, years != config.requested_years, calendar, trading_days)
    years, start = attempts[-1]
    calendar = client.get_calendar(start, config.end_date)
    trading_days = _trading_days_from_calendar(calendar)
    return (pd.Timestamp(start), config.end_date, years, years != config.requested_years, calendar, trading_days)


def _collect_rebalance_data(
    client: JQuantsClient,
    config: JapanPitConfig,
    trading_days: pd.DatetimeIndex,
    paths,
) -> tuple[list[pd.Timestamp], list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], list[dict[str, str]]]:
    signals = _month_end_signal_dates(trading_days)
    signals = [signal for signal in signals if _next_trading_day(trading_days, signal) is not None]
    if config.max_rebalances:
        signals = signals[-config.max_rebalances :]
    completed = _completed_signal_dates(paths)

    all_master_frames: list[pd.DataFrame] = []
    all_bars_frames: list[pd.DataFrame] = []
    universe_rows: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []

    for idx, signal_date in enumerate(signals, start=1):
        signal_key = pd.Timestamp(signal_date).date().isoformat()
        if signal_key in completed:
            print(f"{idx}/{len(signals)} {signal_key} skipped: checkpoint exists")
            continue
        entry_date = _next_trading_day(trading_days, signal_date)
        lookback_dates = _last_n_trading_days(trading_days, signal_date, config.liquidity_lookback)
        if len(lookback_dates) < config.min_days_traded:
            continue
        try:
            master = client.get_master(signal_date)
            universe = _filter_master(master, config.markets)
            universe["signal_date"] = pd.Timestamp(signal_date).date().isoformat()
            bars = _normalize_bars(client.get_daily_bars_for_dates(lookback_dates))
            selected = _rank_liquidity(
                bars=bars,
                universe=universe,
                signal_date=signal_date,
                entry_date=entry_date,
                top_n=config.top_n,
                min_days_traded=config.min_days_traded,
            )
            all_master_frames.append(universe)
            all_bars_frames.append(bars)
            if not selected.empty:
                universe_rows.append(selected)
                _append_checkpoint(
                    selected,
                    paths.local_cache_root / "japan_pit_universe_history.parquet",
                    subset=["signal_date", "Code"],
                )
                _write_atomic_csv(
                    pd.read_parquet(paths.local_cache_root / "japan_pit_universe_history.parquet"),
                    paths.result_dir / "japan_pit_universe_history.csv",
                )
                completed.add(signal_key)
            if not universe.empty:
                _append_checkpoint(
                    universe,
                    paths.local_cache_root / "japan_master_history.parquet",
                    subset=["signal_date", "Code"],
                )
            client.flush_daily_bars_cache()
            print(
                f"{idx}/{len(signals)} {pd.Timestamp(signal_date).date()} "
                f"universe={len(universe)} selected={len(selected)}"
            )
        except Exception as exc:
            failures.append({"signal_date": pd.Timestamp(signal_date).date().isoformat(), "error": str(exc)})
            print(f"{idx}/{len(signals)} {pd.Timestamp(signal_date).date()} failed: {exc}")
    return signals, all_master_frames, all_bars_frames, universe_rows, failures


def build_japan_pit_cache(config: JapanPitConfig) -> dict[str, object]:
    paths = default_paths(ROOT)
    paths.local_cache_root.mkdir(parents=True, exist_ok=True)
    paths.result_dir.mkdir(parents=True, exist_ok=True)

    client = JQuantsClient(
        sleep_seconds=config.sleep_seconds,
        daily_bars_cache_path=paths.local_cache_root / "japan_daily_bars.parquet",
        flush_daily_cache_every=config.flush_daily_cache_every,
    )
    if config.dry_run:
        status = {
            "dry_run": True,
            "requested_years": config.requested_years,
            "fallback_years": config.fallback_years,
            "start_date": config.start_date.date().isoformat(),
            "end_date": config.end_date.date().isoformat(),
            "markets": ",".join(config.markets),
            "top_n": config.top_n,
            "liquidity_lookback": config.liquidity_lookback,
            "min_days_traded": config.min_days_traded,
        }
        pd.DataFrame([status]).to_csv(paths.result_dir / "japan_pit_cache_dry_run_status.csv", index=False)
        return status

    client.authenticate()
    start, end, loaded_years, fallback_used, calendar, trading_days = _choose_window(config, client)
    calendar = _normalize_date_column(calendar)
    calendar.to_parquet(paths.local_cache_root / "japan_calendar.parquet", index=False)

    signals, all_master_frames, all_bars_frames, universe_rows, failures = _collect_rebalance_data(
        client=client,
        config=config,
        trading_days=trading_days,
        paths=paths,
    )
    completed_after_collect = _completed_signal_dates(paths)
    completed_in_window = {
        pd.Timestamp(signal).date().isoformat()
        for signal in signals
        if pd.Timestamp(signal).date().isoformat() in completed_after_collect
    }
    min_success = int(len(signals) * 0.80)
    should_fallback = (
        not fallback_used
        and config.fallback_years != config.requested_years
        and len(signals) > 0
        and len(completed_in_window) < min_success
    )
    if should_fallback:
        print(
            f"Only {len(universe_rows)}/{len(signals)} requested-window rebalances succeeded; "
            f"retrying with {config.fallback_years}Y fallback."
        )
        start = config.end_date - pd.DateOffset(years=config.fallback_years)
        calendar = client.get_calendar(start, config.end_date)
        trading_days = _trading_days_from_calendar(calendar)
        calendar = _normalize_date_column(calendar)
        calendar.to_parquet(paths.local_cache_root / "japan_calendar.parquet", index=False)
        loaded_years = config.fallback_years
        fallback_used = True
        signals, all_master_frames, all_bars_frames, universe_rows, failures = _collect_rebalance_data(
            client=client,
            config=config,
            trading_days=trading_days,
            paths=paths,
        )
        completed_after_collect = _completed_signal_dates(paths)
        completed_in_window = {
            pd.Timestamp(signal).date().isoformat()
            for signal in signals
            if pd.Timestamp(signal).date().isoformat() in completed_after_collect
        }

    master_checkpoint = paths.local_cache_root / "japan_master_history.parquet"
    universe_checkpoint = paths.local_cache_root / "japan_pit_universe_history.parquet"
    master_history = _read_checkpoint(master_checkpoint)
    if all_master_frames:
        new_master = pd.concat(all_master_frames, ignore_index=True)
        master_history = pd.concat([master_history, new_master], ignore_index=True) if not master_history.empty else new_master
        master_history = master_history.drop_duplicates(["signal_date", "Code"], keep="last")
    bars_history = pd.concat(all_bars_frames, ignore_index=True).drop_duplicates(["Date", "Code"]) if all_bars_frames else pd.DataFrame()
    top_history = _read_checkpoint(universe_checkpoint)
    if universe_rows:
        new_top = pd.concat(universe_rows, ignore_index=True)
        top_history = pd.concat([top_history, new_top], ignore_index=True) if not top_history.empty else new_top
        top_history = top_history.drop_duplicates(["signal_date", "Code"], keep="last")
    client.flush_daily_bars_cache()

    _write_atomic_parquet(master_history, master_checkpoint)
    _write_atomic_parquet(top_history, universe_checkpoint)
    _write_atomic_csv(top_history, paths.result_dir / "japan_pit_universe_history.csv")
    failure_file = paths.result_dir / "japan_pit_cache_failures.csv"
    if failures:
        pd.DataFrame(failures).to_csv(failure_file, index=False)
    elif failure_file.exists():
        failure_file.unlink()

    available_tickers = sorted(top_history["Code"].dropna().astype(str).unique().tolist()) if not top_history.empty else []
    price_cols, volume_cols = _persist_price_volume_cache(paths, bars_history, tickers=available_tickers)
    status = {
        "dry_run": False,
        "requested_years": config.requested_years,
        "loaded_years": loaded_years,
        "fallback_used": fallback_used,
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "trading_days": len(trading_days),
        "rebalance_count": len(signals),
        "completed_rebalances": len(completed_in_window),
        "new_rebalances_processed": len(universe_rows),
        "failed_rebalances": len(failures),
        "available_tickers": len(available_tickers),
        "extra_price_columns_after_write": price_cols,
        "extra_volume_columns_after_write": volume_cols,
        "markets": ",".join(config.markets),
        "top_n": config.top_n,
        "liquidity_lookback": config.liquidity_lookback,
        "min_days_traded": config.min_days_traded,
    }
    pd.DataFrame([status]).to_csv(paths.result_dir / "japan_pit_cache_status.csv", index=False)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Japan PIT liquidity universe cache from J-Quants.")
    parser.add_argument("--years", type=int, default=10, help="Preferred history length in years.")
    parser.add_argument("--fallback-years", type=int, default=5, help="Fallback history length when preferred history is unavailable.")
    parser.add_argument("--start-date", default=None, help="Optional explicit start date, YYYY-MM-DD.")
    parser.add_argument("--end-date", default=pd.Timestamp.today().date().isoformat(), help="End date, YYYY-MM-DD.")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="Top liquid stocks per rebalance.")
    parser.add_argument("--liquidity-lookback", type=int, default=DEFAULT_LIQUIDITY_LOOKBACK, help="Trading-day liquidity lookback.")
    parser.add_argument("--min-days-traded", type=int, default=DEFAULT_MIN_DAYS_TRADED, help="Minimum bars in liquidity lookback.")
    parser.add_argument("--markets", nargs="+", default=list(DEFAULT_MARKETS), help="J-Quants market names to include.")
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS, help="Delay between API requests.")
    parser.add_argument("--max-rebalances", type=int, default=None, help="Limit to latest N rebalance dates for testing.")
    parser.add_argument("--flush-daily-cache-every", type=int, default=5, help="Persist downloaded daily bars after this many new dates.")
    parser.add_argument("--dry-run", action="store_true", help="Write status/config only; do not call J-Quants.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    end_date = pd.Timestamp(args.end_date).normalize()
    start_date = (
        pd.Timestamp(args.start_date).normalize()
        if args.start_date
        else end_date - pd.DateOffset(years=args.years) + pd.Timedelta(days=1)
    )
    config = JapanPitConfig(
        start_date=start_date,
        end_date=end_date,
        requested_years=args.years,
        fallback_years=args.fallback_years,
        top_n=args.top_n,
        liquidity_lookback=args.liquidity_lookback,
        min_days_traded=args.min_days_traded,
        markets=tuple(args.markets),
        sleep_seconds=args.sleep_seconds,
        max_rebalances=args.max_rebalances,
        flush_daily_cache_every=args.flush_daily_cache_every,
        dry_run=args.dry_run,
    )
    status = build_japan_pit_cache(config)
    print(pd.Series(status).to_string())


if __name__ == "__main__":
    main()
