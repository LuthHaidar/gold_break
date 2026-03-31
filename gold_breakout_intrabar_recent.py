from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VALID_TP_SL_MODES = ("symmetric_fixed", "atr_based", "prior_day_range")
TICK_SIZE = 0.10
TICK_VALUE = 10.00
CONTRACT_MULTIPLIER = 100.0


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

@dataclass
class IntrabarPositionState:
    trade_id: int
    session_id: int
    session_date: pd.Timestamp
    direction: str
    entry_ts: pd.Timestamp
    entry_bar_index: int
    entry_price: float
    tp_price: float
    initial_sl_price: float
    current_trail_stop: float
    highest_favorable_price: float
    lowest_favorable_price: float
    max_price_seen: float
    min_price_seen: float
    bars_held: int = 0


def recent_intrabar_config_frame(config: RecentIntrabarConfig) -> pd.DataFrame:
    payload = config.to_dict()
    return pd.DataFrame({"parameter": list(payload.keys()), "value": list(payload.values())})


def recent_intrabar_replay_config_frame(config: RecentIntrabarReplayConfig) -> pd.DataFrame:
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


def _wilder_atr(true_range: pd.Series, period: int) -> pd.Series:
    values = true_range.astype(float).to_numpy()
    atr = np.full(values.shape, np.nan, dtype=float)
    if len(values) < period:
        return pd.Series(atr, index=true_range.index)
    atr[period - 1] = np.nanmean(values[:period])
    for index in range(period, len(values)):
        atr[index] = ((atr[index - 1] * (period - 1)) + values[index]) / period
    return pd.Series(atr, index=true_range.index)


def _compute_sl_distance(
    tp_sl_mode: str,
    fixed_ticks: int,
    atr_multiplier_tpsl: float,
    k: float,
    atr_value: float,
    prev_session_range: float,
) -> float:
    if tp_sl_mode == "symmetric_fixed":
        return fixed_ticks * TICK_SIZE
    if tp_sl_mode == "atr_based":
        return atr_multiplier_tpsl * atr_value if pd.notna(atr_value) else np.nan
    if tp_sl_mode == "prior_day_range":
        return k * prev_session_range if pd.notna(prev_session_range) else np.nan
    raise ValueError(f"Unsupported tp_sl_mode: {tp_sl_mode}")


def _prepare_intrabar_strategy_bars(
    minute_bars: pd.DataFrame,
    replay_config: RecentIntrabarReplayConfig,
) -> pd.DataFrame:
    if replay_config.tp_sl_mode not in VALID_TP_SL_MODES:
        raise ValueError(f"Unsupported tp_sl_mode: {replay_config.tp_sl_mode}")
    if minute_bars.empty:
        return minute_bars.copy()
    df = minute_bars.sort_values("ts_event").reset_index(drop=True).copy()
    session_stats = (
        df.groupby(["session_id", "session_date"], as_index=False)
        .agg(
            session_high=("high", "max"),
            session_low=("low", "min"),
        )
        .sort_values("session_id")
        .reset_index(drop=True)
    )
    session_stats["prev_session_high"] = session_stats["session_high"].shift()
    session_stats["prev_session_low"] = session_stats["session_low"].shift()
    session_stats["prev_session_range"] = session_stats["prev_session_high"] - session_stats["prev_session_low"]
    df = df.merge(
        session_stats[["session_id", "prev_session_high", "prev_session_low", "prev_session_range"]],
        on="session_id",
        how="left",
        validate="many_to_one",
    )
    previous_close = df["close"].shift()
    true_range_components = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    )
    df["true_range"] = true_range_components.max(axis=1)
    df["atr_unshifted"] = _wilder_atr(df["true_range"], replay_config.atr_period)
    df["atr"] = df["atr_unshifted"].shift(1)
    df["sl_distance_price"] = df.apply(
        lambda row: _compute_sl_distance(
            tp_sl_mode=replay_config.tp_sl_mode,
            fixed_ticks=replay_config.fixed_ticks,
            atr_multiplier_tpsl=replay_config.atr_multiplier_tpsl,
            k=replay_config.k,
            atr_value=row["atr"],
            prev_session_range=row["prev_session_range"],
        ),
        axis=1,
    )
    df["bar_index"] = df.groupby("session_id").cumcount().astype(int)
    return df.reset_index(drop=True)


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


def _evaluate_position_on_minute(
    position: IntrabarPositionState,
    bar: pd.Series,
    replay_config: RecentIntrabarReplayConfig,
) -> dict[str, Any] | None:
    active_trail_stop = float(position.current_trail_stop)
    active_initial_sl = float(position.initial_sl_price)
    trail_is_active = not np.isclose(active_trail_stop, active_initial_sl)
    if position.direction == "long":
        tp_hit = float(bar["high"]) >= position.tp_price
        stop_hit = float(bar["low"]) <= active_trail_stop
        if tp_hit and stop_hit:
            exit_reason = "trail_stop" if trail_is_active else "initial_sl"
            return {
                "position": position,
                "exit_price": active_trail_stop - TICK_SIZE,
                "exit_reason": exit_reason,
                "exit_fill_basis": "stop_minus_tick_exit",
            }
        if stop_hit:
            exit_reason = "trail_stop" if trail_is_active else "initial_sl"
            return {
                "position": position,
                "exit_price": active_trail_stop - TICK_SIZE,
                "exit_reason": exit_reason,
                "exit_fill_basis": "stop_minus_tick_exit",
            }
        if tp_hit:
            return {
                "position": position,
                "exit_price": position.tp_price,
                "exit_reason": "tp",
                "exit_fill_basis": "tp_limit_exit",
            }
    else:
        tp_hit = float(bar["low"]) <= position.tp_price
        stop_hit = float(bar["high"]) >= active_trail_stop
        if tp_hit and stop_hit:
            exit_reason = "trail_stop" if trail_is_active else "initial_sl"
            return {
                "position": position,
                "exit_price": active_trail_stop + TICK_SIZE,
                "exit_reason": exit_reason,
                "exit_fill_basis": "stop_plus_tick_exit",
            }
        if stop_hit:
            exit_reason = "trail_stop" if trail_is_active else "initial_sl"
            return {
                "position": position,
                "exit_price": active_trail_stop + TICK_SIZE,
                "exit_reason": exit_reason,
                "exit_fill_basis": "stop_plus_tick_exit",
            }
        if tp_hit:
            return {
                "position": position,
                "exit_price": position.tp_price,
                "exit_reason": "tp",
                "exit_fill_basis": "tp_limit_exit",
            }

    position.max_price_seen = max(position.max_price_seen, float(bar["high"]))
    position.min_price_seen = min(position.min_price_seen, float(bar["low"]))
    atr_value = bar["atr"]
    if pd.notna(atr_value):
        if position.direction == "long":
            position.highest_favorable_price = max(position.highest_favorable_price, float(bar["high"]))
            new_trail = position.highest_favorable_price - (replay_config.atr_multiplier * float(atr_value))
            position.current_trail_stop = max(position.current_trail_stop, new_trail)
        else:
            position.lowest_favorable_price = min(position.lowest_favorable_price, float(bar["low"]))
            new_trail = position.lowest_favorable_price + (replay_config.atr_multiplier * float(atr_value))
            position.current_trail_stop = min(position.current_trail_stop, new_trail)
    return None


def _position_pnl_usd(direction: str, entry_price: float, exit_price: float) -> float:
    if direction == "long":
        return (exit_price - entry_price) * CONTRACT_MULTIPLIER
    return (entry_price - exit_price) * CONTRACT_MULTIPLIER


def _close_intrabar_position(
    position: IntrabarPositionState,
    exit_ts: pd.Timestamp,
    exit_price: float,
    exit_reason: str,
    exit_fill_basis: str,
) -> dict[str, Any]:
    pnl_usd = _position_pnl_usd(position.direction, position.entry_price, exit_price)
    if position.direction == "long":
        mfe_price = max(position.max_price_seen - position.entry_price, 0.0)
        mae_price = max(position.entry_price - position.min_price_seen, 0.0)
    else:
        mfe_price = max(position.entry_price - position.min_price_seen, 0.0)
        mae_price = max(position.max_price_seen - position.entry_price, 0.0)
    return {
        "trade_id": position.trade_id,
        "session_id": position.session_id,
        "session_date": position.session_date,
        "direction": position.direction,
        "entry_ts": position.entry_ts,
        "entry_price": position.entry_price,
        "tp_price": position.tp_price,
        "initial_sl_price": position.initial_sl_price,
        "exit_ts": exit_ts,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "exit_fill_basis": exit_fill_basis,
        "bars_held": position.bars_held,
        "pnl_usd": pnl_usd,
        "mfe_price": mfe_price,
        "mae_price": mae_price,
    }


def _open_intrabar_position(
    trade_id: int,
    session_id: int,
    session_date: pd.Timestamp,
    direction: str,
    bar: pd.Series,
    entry_price: float,
    sl_distance_price: float,
) -> IntrabarPositionState:
    if direction == "long":
        tp_price = entry_price + sl_distance_price
        initial_sl_price = entry_price - sl_distance_price
        highest_favorable_price = entry_price
        lowest_favorable_price = entry_price
    else:
        tp_price = entry_price - sl_distance_price
        initial_sl_price = entry_price + sl_distance_price
        highest_favorable_price = entry_price
        lowest_favorable_price = entry_price
    return IntrabarPositionState(
        trade_id=trade_id,
        session_id=session_id,
        session_date=pd.Timestamp(session_date),
        direction=direction,
        entry_ts=pd.Timestamp(bar["ts_event"]),
        entry_bar_index=int(bar["bar_index"]),
        entry_price=float(entry_price),
        tp_price=float(tp_price),
        initial_sl_price=float(initial_sl_price),
        current_trail_stop=float(initial_sl_price),
        highest_favorable_price=float(highest_favorable_price),
        lowest_favorable_price=float(lowest_favorable_price),
        max_price_seen=float(entry_price),
        min_price_seen=float(entry_price),
        bars_held=0,
    )


def simulate_intrabar_diagnostic(
    context: dict[str, Any],
    replay_config: RecentIntrabarReplayConfig,
) -> dict[str, Any]:
    minute_bars = _prepare_intrabar_strategy_bars(context["minute_bars"], replay_config)
    session_summary_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    trade_id_counter = 1

    for session_id, session_bars in minute_bars.groupby("session_id", sort=True):
        session_bars = session_bars.reset_index(drop=True)
        first_bar = session_bars.iloc[0]
        session_date = pd.Timestamp(first_bar["session_date"])
        buy_stop = float(first_bar["prev_session_high"]) if pd.notna(first_bar["prev_session_high"]) else np.nan
        sell_stop = float(first_bar["prev_session_low"]) if pd.notna(first_bar["prev_session_low"]) else np.nan
        sl_distance_price = float(first_bar["sl_distance_price"]) if pd.notna(first_bar["sl_distance_price"]) else np.nan
        if pd.isna(buy_stop) or pd.isna(sell_stop) or pd.isna(sl_distance_price) or sl_distance_price <= 0:
            continue

        positions: list[IntrabarPositionState] = []
        long_triggered = False
        short_triggered = False
        session_trade_rows: list[dict[str, Any]] = []
        session_event_rows: list[dict[str, Any]] = []

        for _, bar in session_bars.iterrows():
            ts_event = pd.Timestamp(bar["ts_event"])
            bar_events: list[str] = []

            existing_positions = [position for position in positions if position.entry_bar_index < int(bar["bar_index"])]
            exit_payloads: list[dict[str, Any]] = []
            for position in existing_positions:
                position.bars_held += 1
                exit_payload = _evaluate_position_on_minute(position, bar, replay_config)
                if exit_payload is not None:
                    exit_payloads.append(exit_payload)

            closed_trade_ids: set[int] = set()
            for exit_payload in exit_payloads:
                trade_row = _close_intrabar_position(
                    position=exit_payload["position"],
                    exit_ts=ts_event,
                    exit_price=float(exit_payload["exit_price"]),
                    exit_reason=str(exit_payload["exit_reason"]),
                    exit_fill_basis=str(exit_payload["exit_fill_basis"]),
                )
                session_trade_rows.append(trade_row)
                session_event_rows.append(
                    {
                        "session_id": session_id,
                        "session_date": session_date,
                        "ts_event": ts_event,
                        "trade_id": trade_row["trade_id"],
                        "direction": trade_row["direction"],
                        "event_type": "exit",
                        "event_reason": trade_row["exit_reason"],
                        "price": trade_row["exit_price"],
                    }
                )
                bar_events.append(f"{trade_row['direction']} exit: {trade_row['exit_reason']}")
                closed_trade_ids.add(trade_row["trade_id"])
            if closed_trade_ids:
                positions = [position for position in positions if position.trade_id not in closed_trade_ids]

            if not long_triggered:
                long_entry_price: float | None = None
                if float(bar["open"]) > buy_stop:
                    long_entry_price = float(bar["open"])
                elif float(bar["high"]) >= buy_stop:
                    long_entry_price = buy_stop + TICK_SIZE
                if long_entry_price is not None:
                    long_position = _open_intrabar_position(
                        trade_id=trade_id_counter,
                        session_id=int(session_id),
                        session_date=session_date,
                        direction="long",
                        bar=bar,
                        entry_price=long_entry_price,
                        sl_distance_price=sl_distance_price,
                    )
                    trade_id_counter += 1
                    positions.append(long_position)
                    long_triggered = True
                    session_event_rows.append(
                        {
                            "session_id": session_id,
                            "session_date": session_date,
                            "ts_event": ts_event,
                            "trade_id": long_position.trade_id,
                            "direction": "long",
                            "event_type": "entry",
                            "event_reason": "buy_stop_trigger",
                            "price": long_entry_price,
                        }
                    )
                    bar_events.append("long entry")
                    if not replay_config.allow_independent_long_short:
                        short_triggered = True

            if not short_triggered:
                short_entry_price: float | None = None
                if float(bar["open"]) < sell_stop:
                    short_entry_price = float(bar["open"])
                elif float(bar["low"]) <= sell_stop:
                    short_entry_price = sell_stop - TICK_SIZE
                if short_entry_price is not None:
                    short_position = _open_intrabar_position(
                        trade_id=trade_id_counter,
                        session_id=int(session_id),
                        session_date=session_date,
                        direction="short",
                        bar=bar,
                        entry_price=short_entry_price,
                        sl_distance_price=sl_distance_price,
                    )
                    trade_id_counter += 1
                    positions.append(short_position)
                    short_triggered = True
                    session_event_rows.append(
                        {
                            "session_id": session_id,
                            "session_date": session_date,
                            "ts_event": ts_event,
                            "trade_id": short_position.trade_id,
                            "direction": "short",
                            "event_type": "entry",
                            "event_reason": "sell_stop_trigger",
                            "price": short_entry_price,
                        }
                    )
                    bar_events.append("short entry")
                    if not replay_config.allow_independent_long_short:
                        long_triggered = True

            if bool(bar["session_close_bar"]) and replay_config.close_positions_at_session_end:
                remaining_positions = list(positions)
                for position in remaining_positions:
                    trade_row = _close_intrabar_position(
                        position=position,
                        exit_ts=ts_event,
                        exit_price=float(bar["close"]),
                        exit_reason="session_close",
                        exit_fill_basis="session_close_exit",
                    )
                    session_trade_rows.append(trade_row)
                    session_event_rows.append(
                        {
                            "session_id": session_id,
                            "session_date": session_date,
                            "ts_event": ts_event,
                            "trade_id": trade_row["trade_id"],
                            "direction": trade_row["direction"],
                            "event_type": "exit",
                            "event_reason": trade_row["exit_reason"],
                            "price": trade_row["exit_price"],
                        }
                    )
                    bar_events.append(f"{trade_row['direction']} exit: session_close")
                positions = []

            active_long = next((position for position in positions if position.direction == "long"), None)
            active_short = next((position for position in positions if position.direction == "short"), None)
            frame_rows.append(
                {
                    "session_id": session_id,
                    "session_date": session_date,
                    "ts_event": ts_event,
                    "bar_index": int(bar["bar_index"]),
                    "open": float(bar["open"]),
                    "high": float(bar["high"]),
                    "low": float(bar["low"]),
                    "close": float(bar["close"]),
                    "buy_stop": buy_stop,
                    "sell_stop": sell_stop,
                    "long_order_active": not long_triggered,
                    "short_order_active": not short_triggered,
                    "long_position_open": active_long is not None,
                    "short_position_open": active_short is not None,
                    "long_active_trail_stop": active_long.current_trail_stop if active_long is not None else np.nan,
                    "short_active_trail_stop": active_short.current_trail_stop if active_short is not None else np.nan,
                    "long_tp": active_long.tp_price if active_long is not None else np.nan,
                    "short_tp": active_short.tp_price if active_short is not None else np.nan,
                    "event_text": "; ".join(bar_events),
                }
            )

        trade_rows.extend(session_trade_rows)
        event_rows.extend(session_event_rows)
        session_summary_rows.append(
            {
                "session_id": int(session_id),
                "session_date": session_date,
                "prev_session_high": buy_stop,
                "prev_session_low": sell_stop,
                "sl_distance_price": sl_distance_price,
                "allow_independent_long_short": replay_config.allow_independent_long_short,
                "trade_count": len(session_trade_rows),
                "entry_count": sum(1 for row in session_event_rows if row["event_type"] == "entry"),
                "exit_count": sum(1 for row in session_event_rows if row["event_type"] == "exit"),
                "net_pnl_usd": float(sum(row["pnl_usd"] for row in session_trade_rows)),
            }
        )

    session_summary = pd.DataFrame(session_summary_rows)
    trade_log = pd.DataFrame(trade_rows)
    event_log = pd.DataFrame(event_rows)
    frame_state = pd.DataFrame(frame_rows)
    return {
        "replay_config": replay_config,
        "minute_bars": minute_bars,
        "session_summary": session_summary,
        "trade_log": trade_log,
        "event_log": event_log,
        "frame_state": frame_state,
    }


def _dedupe_legend(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    deduped_handles: list[Any] = []
    deduped_labels: list[str] = []
    seen: set[str] = set()
    for handle, label in zip(handles, labels):
        if not label or label in seen:
            continue
        seen.add(label)
        deduped_handles.append(handle)
        deduped_labels.append(label)
    if deduped_handles:
        ax.legend(
            deduped_handles,
            deduped_labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )


def _plot_intrabar_candles(ax: plt.Axes, visible_bars: pd.DataFrame) -> None:
    if visible_bars.empty:
        return
    for _, row in visible_bars.iterrows():
        x_value = int(row["bar_index"])
        open_price = float(row["open"])
        close_price = float(row["close"])
        high_price = float(row["high"])
        low_price = float(row["low"])
        candle_color = "#1b9e77" if close_price >= open_price else "#d95f02"
        ax.vlines(x_value, low_price, high_price, color=candle_color, linewidth=1.0, alpha=0.9)
        body_low = min(open_price, close_price)
        body_height = max(abs(close_price - open_price), 0.01)
        ax.add_patch(
            plt.Rectangle(
                (x_value - 0.35, body_low),
                0.7,
                body_height,
                facecolor=candle_color,
                edgecolor=candle_color,
                alpha=0.65,
            )
        )


def _render_intrabar_session_frame(
    ax: plt.Axes,
    diagnostic: dict[str, Any],
    session_id: int,
    end_bar_index: int | None = None,
) -> None:
    minute_bars = diagnostic["minute_bars"]
    frame_state = diagnostic["frame_state"]
    event_log = diagnostic["event_log"]
    trade_log = diagnostic["trade_log"]
    replay_config = diagnostic["replay_config"]

    session_bars = (
        minute_bars.loc[minute_bars["session_id"].eq(session_id)]
        .sort_values("bar_index")
        .reset_index(drop=True)
    )
    session_frames = (
        frame_state.loc[frame_state["session_id"].eq(session_id)]
        .sort_values("bar_index")
        .reset_index(drop=True)
    )
    if session_bars.empty or session_frames.empty:
        raise ValueError(f"No intrabar replay data is available for session_id={session_id}.")

    if end_bar_index is None:
        end_bar_index = int(session_frames["bar_index"].max())

    visible_bars = (
        session_bars.loc[session_bars["bar_index"].le(end_bar_index)]
        .sort_values("bar_index")
        .reset_index(drop=True)
    )
    visible_frames = (
        session_frames.loc[session_frames["bar_index"].le(end_bar_index)]
        .sort_values("bar_index")
        .reset_index(drop=True)
    )
    if visible_bars.empty or visible_frames.empty:
        raise ValueError(f"No visible bars remain for session_id={session_id} at end_bar_index={end_bar_index}.")

    current_frame = visible_frames.iloc[-1]
    session_date = pd.Timestamp(current_frame["session_date"]).date()
    min_x = int(visible_bars["bar_index"].min())
    max_x = int(visible_bars["bar_index"].max())

    ax.clear()
    _plot_intrabar_candles(ax, visible_bars)

    ax.hlines(
        float(current_frame["buy_stop"]),
        min_x,
        max_x,
        colors="#1f77b4",
        linestyles="--",
        linewidth=1.4,
        label="Buy Stop",
    )
    ax.hlines(
        float(current_frame["sell_stop"]),
        min_x,
        max_x,
        colors="#d62728",
        linestyles="--",
        linewidth=1.4,
        label="Sell Stop",
    )

    for column_name, color, label, style in [
        ("long_active_trail_stop", "#2ca02c", "Long Active Stop", "-"),
        ("short_active_trail_stop", "#9467bd", "Short Active Stop", "-"),
        ("long_tp", "#17becf", "Long TP", ":"),
        ("short_tp", "#8c564b", "Short TP", ":"),
    ]:
        series = visible_frames[["bar_index", column_name]].dropna()
        if not series.empty:
            ax.plot(
                series["bar_index"],
                series[column_name],
                color=color,
                linestyle=style,
                linewidth=1.5,
                label=label,
            )

    session_events = event_log.loc[event_log["session_id"].eq(session_id)].copy()
    if not session_events.empty:
        session_events["ts_event"] = pd.to_datetime(session_events["ts_event"], utc=True, errors="coerce")
        session_events = session_events.merge(
            session_bars[["ts_event", "bar_index"]],
            on="ts_event",
            how="left",
            validate="many_to_one",
        )
        session_events = session_events.loc[session_events["bar_index"].le(end_bar_index)]
        if not session_events.empty:
            long_entries = session_events.loc[
                session_events["event_type"].eq("entry") & session_events["direction"].eq("long")
            ]
            short_entries = session_events.loc[
                session_events["event_type"].eq("entry") & session_events["direction"].eq("short")
            ]
            exits = session_events.loc[session_events["event_type"].eq("exit")]
            if not long_entries.empty:
                ax.scatter(
                    long_entries["bar_index"],
                    long_entries["price"],
                    marker="^",
                    s=70,
                    color="#1b9e77",
                    edgecolor="black",
                    linewidth=0.5,
                    label="Long Entry",
                    zorder=5,
                )
            if not short_entries.empty:
                ax.scatter(
                    short_entries["bar_index"],
                    short_entries["price"],
                    marker="v",
                    s=70,
                    color="#d95f02",
                    edgecolor="black",
                    linewidth=0.5,
                    label="Short Entry",
                    zorder=5,
                )
            if not exits.empty:
                ax.scatter(
                    exits["bar_index"],
                    exits["price"],
                    marker="X",
                    s=70,
                    color="black",
                    linewidth=0.5,
                    label="Exit",
                    zorder=6,
                )

    session_trades = trade_log.loc[trade_log["session_id"].eq(session_id)]
    net_pnl_usd = float(session_trades["pnl_usd"].sum()) if not session_trades.empty else 0.0
    trade_count = int(session_trades.shape[0])
    event_text = str(current_frame.get("event_text", "")).strip()
    annotation_lines = [
        f"Mode: {'Independent long/short' if replay_config.allow_independent_long_short else 'Single-side only'}",
        f"Visible bars: {int(current_frame['bar_index']) + 1}",
        f"Trades closed: {trade_count}",
        f"Net PnL: ${net_pnl_usd:,.2f}",
    ]
    if event_text:
        annotation_lines.append(f"Last bar events: {event_text}")
    ax.text(
        0.01,
        0.99,
        "\n".join(annotation_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#bbbbbb"},
    )

    tick_count = min(8, visible_bars.shape[0])
    if tick_count > 1:
        tick_positions = np.linspace(min_x, max_x, num=tick_count, dtype=int)
        tick_labels = []
        for tick_position in tick_positions:
            matching_rows = visible_bars.loc[visible_bars["bar_index"].eq(tick_position), "ts_event"]
            if matching_rows.empty:
                tick_labels.append(str(tick_position))
            else:
                tick_labels.append(pd.Timestamp(matching_rows.iloc[0]).strftime("%m-%d\n%H:%M"))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)

    ax.set_title(
        f"GC Recent Intrabar Replay | Session {session_id} | {session_date} | Up to bar {end_bar_index}"
    )
    ax.set_xlabel("Minute Bar")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.2)
    _dedupe_legend(ax)


def plot_intrabar_session_replay(
    diagnostic: dict[str, Any],
    session_id: int,
    figsize: tuple[float, float] = (15.0, 7.5),
) -> plt.Figure:
    figure, ax = plt.subplots(figsize=figsize)
    _render_intrabar_session_frame(ax, diagnostic, session_id, end_bar_index=None)
    figure.tight_layout()
    return figure