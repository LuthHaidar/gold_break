from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RecentIntrabarConfig:
    ticker: str = "GC=F"
    period: str = "7d"
    interval: str = "1m"
    cache_dir: Path = Path("cache") / "yfinance_intrabar_recent"
    session_gap_threshold_minutes: int = 30

    def cache_key(self) -> str:
        safe_ticker = (
            self.ticker.replace("=", "_")
            .replace("/", "_")
            .replace("^", "_")
            .replace(":", "_")
        )
        return f"{safe_ticker}_{self.period}_{self.interval}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cache_dir"] = str(self.cache_dir)
        return payload


def recent_intrabar_config_frame(config: RecentIntrabarConfig) -> pd.DataFrame:
    payload = config.to_dict()
    return pd.DataFrame({"parameter": list(payload.keys()), "value": list(payload.values())})


def recent_intrabar_cache_paths(config: RecentIntrabarConfig) -> dict[str, Path]:
    cache_key = config.cache_key()
    return {
        "bars_csv": config.cache_dir / f"{cache_key}.csv",
        "metadata_json": config.cache_dir / f"{cache_key}.metadata.json",
    }


def _normalize_yfinance_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("yfinance returned no rows for the requested recent intraday sample.")
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1)
        else:
            df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    timestamp_column = df.columns[0]
    normalized_map = {
        str(column).lower().replace(" ", "_"): column for column in df.columns
    }
    required_columns = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adj_close": "adj_close",
        "volume": "volume",
    }
    rename_map = {timestamp_column: "ts_event"}
    for source_name, target_name in required_columns.items():
        source_column = normalized_map.get(source_name)
        if source_column is None:
            if source_name == "adj_close" and normalized_map.get("close") is not None:
                continue
            raise ValueError(f"Missing required yfinance column: {source_name}")
        rename_map[source_column] = target_name
    df = df.rename(columns=rename_map)
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]
    ts_event = pd.to_datetime(df["ts_event"], errors="raise")
    if ts_event.dt.tz is None:
        ts_event = ts_event.dt.tz_localize("UTC")
    else:
        ts_event = ts_event.dt.tz_convert("UTC")
    df["ts_event"] = ts_event
    df["ticker"] = ticker
    numeric_columns = ["open", "high", "low", "close", "adj_close", "volume"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["ts_event", "open", "high", "low", "close", "volume"]).copy()
    return df[
        ["ts_event", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    ].sort_values("ts_event").reset_index(drop=True)


def _load_cached_intraday_data(config: RecentIntrabarConfig) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    cache_paths = recent_intrabar_cache_paths(config)
    if not cache_paths["bars_csv"].exists() or not cache_paths["metadata_json"].exists():
        return None
    bars = pd.read_csv(cache_paths["bars_csv"])
    bars["ts_event"] = pd.to_datetime(bars["ts_event"], utc=True, errors="raise")
    metadata = json.loads(cache_paths["metadata_json"].read_text(encoding="utf-8"))
    metadata["cache_hit"] = True
    return bars, metadata


def _write_intraday_cache(
    config: RecentIntrabarConfig,
    bars: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    cache_paths = recent_intrabar_cache_paths(config)
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    bars.to_csv(cache_paths["bars_csv"], index=False)
    cache_paths["metadata_json"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _import_yfinance() -> Any:
    try:
        import yfinance as yf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for the recent intrabar notebook. Install it with `pip install yfinance`."
        ) from exc
    return yf


def load_recent_intraday_data(
    config: RecentIntrabarConfig,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not force_refresh:
        cached = _load_cached_intraday_data(config)
        if cached is not None:
            return cached

    yf = _import_yfinance()
    raw = yf.download(
        tickers=config.ticker,
        period=config.period,
        interval=config.interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    bars = _normalize_yfinance_frame(raw, config.ticker)
    metadata = {
        "source": "yfinance",
        "ticker": config.ticker,
        "period": config.period,
        "interval": config.interval,
        "rows": int(bars.shape[0]),
        "downloaded_at_utc": pd.Timestamp.utcnow().isoformat(),
        "ts_min": bars["ts_event"].min().isoformat() if not bars.empty else None,
        "ts_max": bars["ts_event"].max().isoformat() if not bars.empty else None,
        "cache_hit": False,
    }
    _write_intraday_cache(config, bars, metadata)
    return bars, metadata


def assign_intraday_sessions(
    minute_bars: pd.DataFrame,
    session_gap_threshold_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if minute_bars.empty:
        empty_sessions = pd.DataFrame(
            columns=[
                "session_id",
                "session_open_ts",
                "session_close_ts",
                "session_date",
                "session_bars",
                "session_high",
                "session_low",
                "session_open",
                "session_close",
                "session_volume",
            ]
        )
        return empty_sessions, minute_bars.copy()
    df = minute_bars.sort_values("ts_event").reset_index(drop=True).copy()
    gap_threshold = pd.Timedelta(minutes=session_gap_threshold_minutes)
    session_start = df["ts_event"].diff().isna() | df["ts_event"].diff().gt(gap_threshold)
    df["session_id"] = session_start.cumsum().astype(int) - 1
    session_table = (
        df.groupby("session_id", as_index=False)
        .agg(
            session_open_ts=("ts_event", "min"),
            session_close_ts=("ts_event", "max"),
            session_bars=("ts_event", "count"),
            session_open=("open", "first"),
            session_high=("high", "max"),
            session_low=("low", "min"),
            session_close=("close", "last"),
            session_volume=("volume", "sum"),
        )
        .sort_values("session_id")
        .reset_index(drop=True)
    )
    session_table["session_date"] = session_table["session_open_ts"].dt.normalize()
    df = df.merge(
        session_table[["session_id", "session_open_ts", "session_close_ts", "session_date"]],
        on="session_id",
        how="left",
        validate="many_to_one",
    )
    df["session_open_bar"] = df["ts_event"].eq(df["session_open_ts"])
    df["session_close_bar"] = df["ts_event"].eq(df["session_close_ts"])
    return session_table, df.reset_index(drop=True)


def resample_intraday_to_hourly(minute_bars: pd.DataFrame) -> pd.DataFrame:
    if minute_bars.empty:
        return pd.DataFrame(
            columns=[
                "session_id",
                "session_date",
                "ts_event",
                "minute_count",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
            ]
        )
    df = minute_bars.copy()
    df["ts_event_hour"] = df["ts_event"].dt.floor("h")
    hourly = (
        df.groupby(["session_id", "session_date", "ts_event_hour"], as_index=False)
        .agg(
            minute_count=("ts_event", "count"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            adj_close=("adj_close", "last"),
            volume=("volume", "sum"),
        )
        .rename(columns={"ts_event_hour": "ts_event"})
        .sort_values(["ts_event", "session_id"])
        .reset_index(drop=True)
    )
    return hourly


def build_recent_intrabar_context(
    config: RecentIntrabarConfig,
    force_refresh: bool = False,
) -> dict[str, Any]:
    minute_bars, metadata = load_recent_intraday_data(config, force_refresh=force_refresh)
    session_table, session_bars = assign_intraday_sessions(
        minute_bars=minute_bars,
        session_gap_threshold_minutes=config.session_gap_threshold_minutes,
    )
    hourly_bars = resample_intraday_to_hourly(session_bars)
    return {
        "config": config,
        "metadata": metadata,
        "cache_paths": recent_intrabar_cache_paths(config),
        "minute_bars": session_bars,
        "session_table": session_table,
        "hourly_bars": hourly_bars,
    }
