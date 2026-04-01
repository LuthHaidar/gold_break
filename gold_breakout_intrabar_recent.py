from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

import gold_breakout_backtest as historical_backtest


VALID_TP_SL_MODES = ("symmetric_fixed", "atr_based", "prior_day_range")
VALID_WALK_FORWARD_OBJECTIVES = ("sharpe_ratio", "total_return_pct", "calmar_ratio")
TICK_SIZE = 0.10
TICK_VALUE = 10.00
CONTRACT_MULTIPLIER = 100.0
ROUND_TURN_COST = 24.74


@dataclass(frozen=True)
class RecentIntrabarConfig:
    csv_path: Path = Path("glbx-mdp3-20250330-20260329.ohlcv-1m.csv")
    backtest_start_session_date: str = "2025-03-30"
    max_ohlc_violations: int = 0
    min_session_volume: int = 0
    accepted_rtypes: tuple[int, ...] = (33, 34)

    def resolved_start_session_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.backtest_start_session_date, tz="UTC").normalize()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["csv_path"] = str(self.csv_path)
        return payload


@dataclass(frozen=True)
class RecentIntrabarReplayConfig:
    tp_sl_mode: str = "prior_day_range"
    fixed_ticks: int = 100
    atr_multiplier_tpsl: float = 2.0
    k: float = 0.50
    atr_period: int = 14
    atr_multiplier: float = 1.5
    starting_capital: float = 100_000.0
    risk_free_rate: float = 0.0
    round_turn_cost: float = ROUND_TURN_COST
    allow_independent_long_short: bool = True
    close_positions_at_session_end: bool = True
    frame_step: int = 5
    animation_interval_ms: int = 120
    sensitivity_atr_periods: tuple[int, ...] = (7, 14, 21)
    sensitivity_atr_multipliers: tuple[float, ...] = (0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0)
    sensitivity_ks: tuple[float, ...] = (0.10, 0.15, 0.25, 0.50, 0.75, 1.00)
    sensitivity_tp_sl_modes: tuple[str, ...] = VALID_TP_SL_MODES
    walk_forward_objective: str = "sharpe_ratio"
    walk_forward_train_months: int = 6
    walk_forward_test_months: int = 1
    walk_forward_step_months: int = 1
    walk_forward_atr_periods: tuple[int, ...] = (14,)
    walk_forward_tp_sl_modes: tuple[str, ...] = ("prior_day_range",)
    walk_forward_atr_multipliers: tuple[float, ...] = (0.50, 0.75, 1.0, 1.5)
    walk_forward_ks: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecentIntrabarResult:
    config: RecentIntrabarReplayConfig
    data: pd.DataFrame
    session_table: pd.DataFrame
    trade_log: pd.DataFrame
    roll_events: pd.DataFrame
    order_cancellations: pd.DataFrame
    floor_events: pd.DataFrame
    margin_flags: pd.DataFrame
    skipped_sessions: pd.DataFrame
    event_log: pd.DataFrame
    equity_curve: pd.DataFrame
    session_equity: pd.DataFrame
    performance_summary: pd.Series
    direction_summary: pd.DataFrame
    annual_summary: pd.DataFrame
    exit_breakdown: pd.DataFrame
    benchmark_curves: pd.DataFrame
    benchmark_summary: pd.DataFrame
    validation_results: pd.DataFrame
    diagnostics: dict[str, Any]
    frame_state: pd.DataFrame
    session_summary: pd.DataFrame


@dataclass
class RecentIntrabarWalkForwardResult:
    config: RecentIntrabarReplayConfig
    fold_summary: pd.DataFrame
    optimization_results: pd.DataFrame
    parameter_stability: pd.DataFrame
    oos_equity_curve: pd.DataFrame
    oos_session_equity: pd.DataFrame
    oos_trade_log: pd.DataFrame
    diagnostics: dict[str, Any]


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


def make_default_recent_intrabar_replay_config(**overrides: Any) -> RecentIntrabarReplayConfig:
    base = RecentIntrabarReplayConfig()
    return replace(base, **overrides)


def make_default_recent_intrabar_config(**overrides: Any) -> RecentIntrabarConfig:
    base = RecentIntrabarConfig()
    return replace(base, **overrides)


def recent_intrabar_config_frame(config: RecentIntrabarConfig) -> pd.DataFrame:
    payload = config.to_dict()
    return pd.DataFrame({"parameter": list(payload.keys()), "value": list(payload.values())})


def recent_intrabar_replay_config_frame(config: RecentIntrabarReplayConfig) -> pd.DataFrame:
    payload = config.to_dict()
    return pd.DataFrame({"parameter": list(payload.keys()), "value": list(payload.values())})


def _validate_replay_config(config: RecentIntrabarReplayConfig) -> None:
    if config.tp_sl_mode not in VALID_TP_SL_MODES:
        raise ValueError(f"Unsupported tp_sl_mode: {config.tp_sl_mode}")
    if config.walk_forward_objective not in VALID_WALK_FORWARD_OBJECTIVES:
        raise ValueError(f"Unsupported walk_forward_objective: {config.walk_forward_objective}")
    if config.fixed_ticks <= 0:
        raise ValueError("fixed_ticks must be positive.")
    if config.atr_multiplier_tpsl <= 0:
        raise ValueError("atr_multiplier_tpsl must be positive.")
    if config.k <= 0:
        raise ValueError("k must be positive.")
    if config.atr_period <= 0:
        raise ValueError("atr_period must be positive.")
    if config.atr_multiplier <= 0:
        raise ValueError("atr_multiplier must be positive.")
    if config.starting_capital <= 0:
        raise ValueError("starting_capital must be positive.")
    if config.round_turn_cost < 0:
        raise ValueError("round_turn_cost cannot be negative.")
    if config.frame_step <= 0:
        raise ValueError("frame_step must be positive.")
    if config.animation_interval_ms <= 0:
        raise ValueError("animation_interval_ms must be positive.")
    if any(period <= 0 for period in config.sensitivity_atr_periods):
        raise ValueError("All sensitivity_atr_periods values must be positive.")
    if any(multiplier <= 0 for multiplier in config.sensitivity_atr_multipliers):
        raise ValueError("All sensitivity_atr_multipliers values must be positive.")
    if any(k_value <= 0 for k_value in config.sensitivity_ks):
        raise ValueError("All sensitivity_ks values must be positive.")
    if any(mode not in VALID_TP_SL_MODES for mode in config.sensitivity_tp_sl_modes):
        raise ValueError("sensitivity_tp_sl_modes contains an unsupported mode.")
    if config.walk_forward_train_months <= 0:
        raise ValueError("walk_forward_train_months must be positive.")
    if config.walk_forward_test_months <= 0:
        raise ValueError("walk_forward_test_months must be positive.")
    if config.walk_forward_step_months <= 0:
        raise ValueError("walk_forward_step_months must be positive.")
    if config.walk_forward_step_months < config.walk_forward_test_months:
        raise ValueError("walk_forward_step_months must be at least the walk-forward test horizon in months.")
    if any(period <= 0 for period in config.walk_forward_atr_periods):
        raise ValueError("All walk_forward_atr_periods values must be positive.")
    if any(mode not in VALID_TP_SL_MODES for mode in config.walk_forward_tp_sl_modes):
        raise ValueError("walk_forward_tp_sl_modes contains an unsupported mode.")
    if any(multiplier <= 0 for multiplier in config.walk_forward_atr_multipliers):
        raise ValueError("All walk_forward_atr_multipliers values must be positive.")
    if any(k_value <= 0 for k_value in config.walk_forward_ks):
        raise ValueError("All walk_forward_ks values must be positive.")


def _validate_intrabar_config(config: RecentIntrabarConfig) -> None:
    if not config.csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {config.csv_path}")
    if config.max_ohlc_violations < 0:
        raise ValueError("max_ohlc_violations cannot be negative.")
    if config.min_session_volume < 0:
        raise ValueError("min_session_volume cannot be negative.")
    if not config.accepted_rtypes:
        raise ValueError("accepted_rtypes cannot be empty.")


def recent_intrabar_data_paths(config: RecentIntrabarConfig) -> dict[str, Path]:
    return {"csv_path": config.csv_path}


def recent_intrabar_cache_paths(config: RecentIntrabarConfig) -> dict[str, Path]:
    return recent_intrabar_data_paths(config)


def summarize_intrabar_audit(audit: dict[str, Any]) -> pd.DataFrame:
    return historical_backtest.summarize_audit(audit)


def _proxy_backtest_config(config: RecentIntrabarConfig, atr_period: int = 14) -> historical_backtest.BacktestConfig:
    return historical_backtest.make_default_config(
        csv_path=config.csv_path,
        backtest_start_session_date=config.backtest_start_session_date,
        max_ohlc_violations=config.max_ohlc_violations,
        min_session_volume=config.min_session_volume,
        atr_period=atr_period,
    )


def _load_local_minute_csv(config: RecentIntrabarConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    _validate_intrabar_config(config)
    raw = historical_backtest._load_csv(_proxy_backtest_config(config))
    accepted_mask = raw["rtype"].isin(config.accepted_rtypes)
    accepted_rows = raw.loc[accepted_mask].copy()
    if accepted_rows.empty:
        raise ValueError(
            f"No rows matched accepted_rtypes={config.accepted_rtypes} in {config.csv_path.name}."
        )
    accepted_rows["rtype"] = 34
    metadata = {
        "source": "local_csv",
        "csv_path": str(config.csv_path),
        "rows_raw": int(raw.shape[0]),
        "rows_after_rtype_filter": int(accepted_rows.shape[0]),
        "accepted_rtypes": list(config.accepted_rtypes),
        "ts_min": accepted_rows["ts_event"].min().isoformat() if not accepted_rows.empty else None,
        "ts_max": accepted_rows["ts_event"].max().isoformat() if not accepted_rows.empty else None,
        "backtest_start_session_date": config.backtest_start_session_date,
    }
    return accepted_rows, metadata


def _build_intrabar_minute_bars_from_continuous(continuous_data: pd.DataFrame) -> pd.DataFrame:
    if continuous_data.empty:
        return pd.DataFrame(
            columns=[
                "ts_event",
                "ticker",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
                "session_id",
                "session_date",
                "session_open_ts",
                "session_close_ts",
                "session_open_bar",
                "session_close_bar",
                "dominant_contract",
                "adj_factor",
            ]
        )
    minute_bars = pd.DataFrame(
        {
            "ts_event": continuous_data["ts_event"],
            "ticker": continuous_data["dominant_contract"],
            "open": continuous_data["adj_open"],
            "high": continuous_data["adj_high"],
            "low": continuous_data["adj_low"],
            "close": continuous_data["adj_close"],
            "adj_close": continuous_data["adj_close"],
            "volume": continuous_data["volume"],
            "session_id": continuous_data["session_id"],
            "session_date": continuous_data["session_date"],
            "session_open_ts": continuous_data["session_open_ts"],
            "session_close_ts": continuous_data["session_close_ts"],
            "session_open_bar": continuous_data["session_open_bar"],
            "session_close_bar": continuous_data["session_close_bar"],
            "dominant_contract": continuous_data["dominant_contract"],
            "adj_factor": continuous_data["adj_factor"],
        }
    )
    return minute_bars.sort_values("ts_event").reset_index(drop=True)


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


def forward_fill_intraday_hourly_gaps(
    hourly_bars: pd.DataFrame,
    session_table: pd.DataFrame,
) -> pd.DataFrame:
    if session_table.empty:
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
                "forward_filled",
            ]
        )

    session_frames: list[pd.DataFrame] = []
    for session_row in session_table.itertuples(index=False):
        session_hourly = (
            hourly_bars.loc[hourly_bars["session_id"].eq(session_row.session_id)]
            .sort_values("ts_event")
            .reset_index(drop=True)
        )
        session_index = pd.date_range(
            start=pd.Timestamp(session_row.session_open_ts).floor("h"),
            end=pd.Timestamp(session_row.session_close_ts).floor("h"),
            freq="h",
            tz="UTC",
        )
        session_grid = pd.DataFrame({"ts_event": session_index})
        merged = session_grid.merge(
            session_hourly,
            on="ts_event",
            how="left",
            validate="one_to_one",
        )
        merged["session_id"] = int(session_row.session_id)
        merged["session_date"] = pd.Timestamp(session_row.session_date)
        merged["forward_filled"] = merged["open"].isna()
        merged["adj_close"] = merged["adj_close"].ffill().bfill()
        for column in ["close", "open", "high", "low"]:
            merged[column] = merged[column].fillna(merged["adj_close"])
        merged["minute_count"] = merged["minute_count"].fillna(0).astype(int)
        merged["volume"] = merged["volume"].fillna(0.0)
        session_frames.append(
            merged[
                [
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
                    "forward_filled",
                ]
            ]
        )
    return pd.concat(session_frames, ignore_index=True) if session_frames else pd.DataFrame()


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
    _validate_replay_config(replay_config)
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


def prepare_recent_intrabar_feature_data(
    context: dict[str, Any],
    replay_config: RecentIntrabarReplayConfig,
) -> pd.DataFrame:
    return _prepare_intrabar_strategy_bars(context["minute_bars"], replay_config)


def preview_intrabar_signal_inputs(
    context: dict[str, Any],
    replay_config: RecentIntrabarReplayConfig,
    rows: int = 15,
) -> pd.DataFrame:
    feature_data = prepare_recent_intrabar_feature_data(context, replay_config)
    if feature_data.empty:
        return pd.DataFrame()
    preview = (
        feature_data.loc[feature_data["session_open_bar"]]
        .copy()
        .sort_values("ts_event")
        .reset_index(drop=True)
    )
    preview["buy_stop"] = preview["prev_session_high"]
    preview["sell_stop"] = preview["prev_session_low"]
    columns = [
        "ts_event",
        "session_id",
        "session_date",
        "buy_stop",
        "sell_stop",
        "prev_session_range",
        "atr",
        "sl_distance_price",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    available = [column for column in columns if column in preview.columns]
    return preview[available].head(rows)


def build_recent_intrabar_context(
    config: RecentIntrabarConfig,
    force_refresh: bool = False,
) -> dict[str, Any]:
    del force_refresh
    raw, metadata = _load_local_minute_csv(config)
    audit: dict[str, Any] = {"counts": {}, "frames": {}, "lists": {}}
    proxy_config = _proxy_backtest_config(config)
    clean = historical_backtest._clean_outright_data(raw, proxy_config, audit)
    session_table_all, clean = historical_backtest._assign_sessions(clean)
    audit["counts"]["session_count_all_outrights"] = int(session_table_all.shape[0])
    audit["frames"]["low_volume_sessions_all_outrights"] = historical_backtest._flag_low_volume_sessions(
        session_table_all,
        config.min_session_volume,
    )
    dominant_table, continuous = historical_backtest._build_continuous_series(clean, session_table_all, audit)
    continuous = historical_backtest._attach_session_features(continuous)
    filtered_sessions, filtered_continuous = historical_backtest._filter_backtest_window(
        session_table=dominant_table,
        continuous=continuous,
        start_session_date=config.resolved_start_session_date(),
    )
    minute_bars = _build_intrabar_minute_bars_from_continuous(filtered_continuous)
    hourly_bars = resample_intraday_to_hourly(minute_bars)
    hourly_bars_forward_filled = forward_fill_intraday_hourly_gaps(hourly_bars, filtered_sessions)
    notes = [
        "Minute data comes from the local Databento-style CSV, not yfinance.",
        "Rows with accepted minute-bar rtypes are normalized onto the historical cleaning pipeline before spread removal and dominant-contract selection.",
        "Intrabar replay runs on adjusted minute OHLC values from the continuous series. Because positions are flattened by session end, the per-trade PnL impact of the constant adjustment offset cancels within a session.",
        "The forward-filled hourly table is a diagnostic resample only; the execution engine remains minute-based.",
    ]
    metadata.update(
        {
            "rows_clean": int(clean.shape[0]),
            "rows_continuous": int(filtered_continuous.shape[0]),
            "rows_minute_bars": int(minute_bars.shape[0]),
            "session_count": int(filtered_sessions.shape[0]),
            "dominant_contracts": int(filtered_sessions["dominant_contract"].nunique()) if not filtered_sessions.empty else 0,
        }
    )
    return {
        "config": config,
        "metadata": metadata,
        "data_paths": recent_intrabar_data_paths(config),
        "raw_data": raw,
        "clean_data": clean,
        "audit": audit,
        "notes": notes,
        "minute_bars": minute_bars,
        "session_table": filtered_sessions.reset_index(drop=True),
        "dominant_table": dominant_table.reset_index(drop=True),
        "continuous_data": filtered_continuous.reset_index(drop=True),
        "hourly_bars": hourly_bars,
        "hourly_bars_forward_filled": hourly_bars_forward_filled,
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


def _build_intrabar_concurrency_log(frame_state: pd.DataFrame) -> pd.DataFrame:
    if frame_state.empty:
        return pd.DataFrame(
            columns=[
                "session_id",
                "ts_event",
                "open_positions",
                "simultaneous_long_short",
            ]
        )
    concurrency = (
        frame_state[
            [
                "session_id",
                "ts_event",
                "long_position_open",
                "short_position_open",
            ]
        ]
        .copy()
        .sort_values(["session_id", "ts_event"])
        .reset_index(drop=True)
    )
    concurrency["open_positions"] = (
        concurrency["long_position_open"].astype(int) + concurrency["short_position_open"].astype(int)
    )
    concurrency["simultaneous_long_short"] = (
        concurrency["long_position_open"].astype(bool) & concurrency["short_position_open"].astype(bool)
    )
    return concurrency


def _build_intrabar_equity_curve(
    minute_bars: pd.DataFrame,
    trade_log: pd.DataFrame,
    starting_capital: float,
) -> pd.DataFrame:
    if minute_bars.empty:
        return pd.DataFrame(
            columns=[
                "ts_event",
                "session_id",
                "session_date",
                "session_open_bar",
                "session_close_bar",
                "gross_delta",
                "net_delta",
                "gross_equity",
                "net_equity",
                "gross_running_peak",
                "net_running_peak",
                "gross_drawdown_pct",
                "net_drawdown_pct",
            ]
        )

    curve = (
        minute_bars[
            [
                "ts_event",
                "session_id",
                "session_date",
                "session_open_bar",
                "session_close_bar",
            ]
        ]
        .copy()
        .sort_values("ts_event")
        .reset_index(drop=True)
    )
    curve["gross_delta"] = 0.0
    curve["net_delta"] = 0.0
    if not trade_log.empty:
        realized = (
            trade_log.groupby("exit_bar_ts", as_index=False)
            .agg(
                gross_delta=("gross_pnl", "sum"),
                net_delta=("net_pnl", "sum"),
            )
            .rename(columns={"exit_bar_ts": "ts_event"})
        )
        curve = curve.merge(realized, on="ts_event", how="left", suffixes=("", "_realized"))
        for column in ["gross_delta_realized", "net_delta_realized"]:
            if column in curve.columns:
                curve[column] = curve[column].fillna(0.0)
        curve["gross_delta"] = curve.pop("gross_delta_realized")
        curve["net_delta"] = curve.pop("net_delta_realized")
    curve["gross_equity"] = starting_capital + curve["gross_delta"].cumsum()
    curve["net_equity"] = starting_capital + curve["net_delta"].cumsum()
    curve["gross_running_peak"] = curve["gross_equity"].cummax()
    curve["net_running_peak"] = curve["net_equity"].cummax()
    curve["gross_drawdown_pct"] = (
        curve["gross_equity"] / curve["gross_running_peak"] - 1.0
    ) * 100.0
    curve["net_drawdown_pct"] = (
        curve["net_equity"] / curve["net_running_peak"] - 1.0
    ) * 100.0
    return curve


def _build_intrabar_gc_price_benchmark(
    minute_bars: pd.DataFrame,
    starting_capital: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if minute_bars.empty:
        empty = pd.DataFrame(
            columns=[
                "ts_event",
                "gc_price_index_equity",
                "gc_price_index_running_peak",
                "gc_price_index_drawdown_pct",
                "session_close_bar",
            ]
        )
        return empty, empty
    benchmark = (
        minute_bars[["ts_event", "adj_close", "session_close_bar"]]
        .copy()
        .sort_values("ts_event")
        .reset_index(drop=True)
    )
    starting_price = float(benchmark["adj_close"].iloc[0])
    benchmark["gc_price_index_equity"] = starting_capital * (benchmark["adj_close"] / starting_price)
    benchmark["gc_price_index_running_peak"] = benchmark["gc_price_index_equity"].cummax()
    benchmark["gc_price_index_drawdown_pct"] = (
        benchmark["gc_price_index_equity"] / benchmark["gc_price_index_running_peak"] - 1.0
    ) * 100.0
    session_equity = benchmark.loc[benchmark["session_close_bar"]].copy().reset_index(drop=True)
    return benchmark, session_equity


def _build_recent_benchmark_summary(
    strategy_session_equity: pd.DataFrame,
    gc_price_session_equity: pd.DataFrame,
    config: RecentIntrabarReplayConfig,
) -> pd.DataFrame:
    rows = [
        historical_backtest._summarize_equity_series(
            label="Strategy Net Equity",
            session_equity=strategy_session_equity,
            equity_column="net_equity",
            starting_capital=config.starting_capital,
            risk_free_rate=config.risk_free_rate,
            strategy_session_equity=strategy_session_equity,
        ),
        historical_backtest._summarize_equity_series(
            label="GC Price Index",
            session_equity=gc_price_session_equity,
            equity_column="gc_price_index_equity",
            starting_capital=config.starting_capital,
            risk_free_rate=config.risk_free_rate,
            strategy_session_equity=strategy_session_equity,
        ),
    ]
    return pd.DataFrame(rows)


def _build_recent_validation_results(
    result: RecentIntrabarResult,
) -> pd.DataFrame:
    validations: list[dict[str, Any]] = []
    trades = result.trade_log.copy()
    frame_state = result.frame_state.copy()

    ordering_ok = True
    if not trades.empty:
        ordering_ok = bool(
            pd.to_datetime(trades["entry_bar_ts"], utc=True).le(
                pd.to_datetime(trades["exit_bar_ts"], utc=True)
            ).all()
        )
    validations.append(
        {
            "test": "Trade timestamps are ordered",
            "status": "pass" if ordering_ok else "fail",
            "detail": "Each trade exits at or after its entry timestamp.",
        }
    )

    equity_reconciles_ok = True
    if not result.session_equity.empty:
        expected_terminal = result.config.starting_capital + result.trade_log["net_pnl"].sum()
        actual_terminal = float(result.session_equity["net_equity"].iloc[-1])
        equity_reconciles_ok = bool(np.isclose(actual_terminal, expected_terminal, atol=1e-9))
    validations.append(
        {
            "test": "Net equity reconciles to trade PnL",
            "status": "pass" if equity_reconciles_ok else "fail",
            "detail": "Terminal net equity equals starting capital plus cumulative trade net PnL.",
        }
    )

    session_close_ok = True
    if result.config.close_positions_at_session_end and not frame_state.empty:
        session_end_frames = frame_state.groupby("session_id", as_index=False).tail(1)
        session_close_ok = bool(
            (~session_end_frames["long_position_open"].astype(bool)).all()
            and (~session_end_frames["short_position_open"].astype(bool)).all()
        )
    validations.append(
        {
            "test": "Session-close flattening",
            "status": "pass" if session_close_ok else "fail",
            "detail": "No positions remain open at the final bar of a session when session-close exits are enabled.",
        }
    )

    exit_count_ok = bool(
        result.trade_log.shape[0]
        == result.event_log.loc[result.event_log["event_type"].eq("exit")].shape[0]
    )
    validations.append(
        {
            "test": "Exit events reconcile to closed trades",
            "status": "pass" if exit_count_ok else "fail",
            "detail": "The event log contains one exit event for each closed trade.",
        }
    )
    return pd.DataFrame(validations)


def run_recent_intrabar_backtest(
    context: dict[str, Any],
    replay_config: RecentIntrabarReplayConfig,
) -> RecentIntrabarResult:
    _validate_replay_config(replay_config)
    diagnostic = simulate_intrabar_diagnostic(context, replay_config)
    minute_bars = diagnostic["minute_bars"].copy()
    session_table = context["session_table"].copy()
    trade_log = diagnostic["trade_log"].copy()
    event_log = diagnostic["event_log"].copy()
    if event_log.empty:
        event_log = pd.DataFrame(
            columns=[
                "session_id",
                "session_date",
                "ts_event",
                "trade_id",
                "direction",
                "event_type",
                "event_reason",
                "price",
            ]
        )
    else:
        event_log = event_log.sort_values("ts_event").reset_index(drop=True)
    frame_state = diagnostic["frame_state"].copy()
    if frame_state.empty:
        frame_state = pd.DataFrame(
            columns=[
                "session_id",
                "session_date",
                "ts_event",
                "bar_index",
                "open",
                "high",
                "low",
                "close",
                "buy_stop",
                "sell_stop",
                "long_order_active",
                "short_order_active",
                "long_position_open",
                "short_position_open",
                "long_active_trail_stop",
                "short_active_trail_stop",
                "long_tp",
                "short_tp",
                "event_text",
            ]
        )
    else:
        frame_state = frame_state.sort_values(["session_id", "ts_event"]).reset_index(drop=True)
    session_summary = diagnostic["session_summary"].copy()
    if session_summary.empty:
        session_summary = pd.DataFrame(
            columns=[
                "session_id",
                "session_date",
                "prev_session_high",
                "prev_session_low",
                "sl_distance_price",
                "allow_independent_long_short",
                "trade_count",
                "entry_count",
                "exit_count",
                "net_pnl_usd",
            ]
        )
    else:
        session_summary = session_summary.sort_values("session_id").reset_index(drop=True)

    trade_columns = [
        "trade_id",
        "session_id",
        "session_date",
        "direction",
        "entry_ts",
        "entry_bar_ts",
        "entry_price",
        "tp_price",
        "initial_sl_price",
        "exit_ts",
        "exit_bar_ts",
        "exit_price",
        "exit_reason",
        "exit_fill_basis",
        "bars_held",
        "pnl_usd",
        "gross_pnl",
        "trade_cost",
        "net_pnl",
        "mfe_price",
        "mae_price",
        "mfe_usd",
        "mae_usd",
    ]
    if trade_log.empty:
        trade_log = pd.DataFrame(columns=trade_columns)
    else:
        trade_log["entry_bar_ts"] = pd.to_datetime(trade_log["entry_ts"], utc=True, errors="coerce")
        trade_log["exit_bar_ts"] = pd.to_datetime(trade_log["exit_ts"], utc=True, errors="coerce")
        trade_log["gross_pnl"] = pd.to_numeric(trade_log["pnl_usd"], errors="coerce")
        trade_log["trade_cost"] = float(replay_config.round_turn_cost)
        trade_log["net_pnl"] = trade_log["gross_pnl"] - trade_log["trade_cost"]
        trade_log["mfe_usd"] = pd.to_numeric(trade_log["mfe_price"], errors="coerce") * CONTRACT_MULTIPLIER
        trade_log["mae_usd"] = pd.to_numeric(trade_log["mae_price"], errors="coerce") * CONTRACT_MULTIPLIER
        trade_log = trade_log.sort_values("exit_bar_ts").reset_index(drop=True)
        trade_log = trade_log[trade_columns]

    skipped_sessions = session_table.loc[
        ~session_table["session_id"].isin(session_summary["session_id"])
    ].copy()
    if not skipped_sessions.empty:
        skipped_sessions["ts_event"] = skipped_sessions["session_open_ts"]
        skipped_sessions["reason"] = "Missing prior-session breakout inputs or stop-distance."
        skipped_sessions = skipped_sessions[
            [
                "session_id",
                "session_date",
                "ts_event",
                "reason",
            ]
        ].reset_index(drop=True)
    else:
        skipped_sessions = pd.DataFrame(columns=["session_id", "session_date", "ts_event", "reason"])

    concurrency_log = _build_intrabar_concurrency_log(frame_state)
    equity_curve = _build_intrabar_equity_curve(
        minute_bars=minute_bars,
        trade_log=trade_log,
        starting_capital=replay_config.starting_capital,
    )
    session_equity = (
        equity_curve.loc[equity_curve["session_close_bar"]].copy().reset_index(drop=True)
        if not equity_curve.empty
        else pd.DataFrame()
    )

    gc_price_curve, gc_price_session_equity = _build_intrabar_gc_price_benchmark(
        minute_bars=minute_bars,
        starting_capital=replay_config.starting_capital,
    )
    benchmark_curves = equity_curve.copy()
    if not gc_price_curve.empty:
        benchmark_curves = benchmark_curves.merge(
            gc_price_curve[
                [
                    "ts_event",
                    "gc_price_index_equity",
                    "gc_price_index_drawdown_pct",
                ]
            ],
            on="ts_event",
            how="left",
            validate="one_to_one",
        )

    empty_df = pd.DataFrame()
    performance_summary = historical_backtest._build_performance_summary(
        trade_log=trade_log,
        session_equity=session_equity,
        concurrency_df=concurrency_log,
        floor_events_df=empty_df,
        order_cancellations_df=empty_df,
        margin_flags_df=empty_df,
        session_sizing_df=empty_df,
        skipped_sessions_df=skipped_sessions,
        config=replay_config,
    )
    direction_summary = historical_backtest._build_direction_summary(trade_log)
    annual_summary = historical_backtest._build_annual_summary(session_equity, trade_log)
    exit_breakdown = historical_backtest._build_exit_breakdown(trade_log)
    benchmark_summary = _build_recent_benchmark_summary(
        strategy_session_equity=session_equity,
        gc_price_session_equity=gc_price_session_equity,
        config=replay_config,
    )

    result = RecentIntrabarResult(
        config=replay_config,
        data=minute_bars,
        session_table=session_table,
        trade_log=trade_log,
        roll_events=pd.DataFrame(columns=["roll_ts"]),
        order_cancellations=empty_df.copy(),
        floor_events=empty_df.copy(),
        margin_flags=empty_df.copy(),
        skipped_sessions=skipped_sessions,
        event_log=event_log,
        equity_curve=equity_curve,
        session_equity=session_equity,
        performance_summary=performance_summary,
        direction_summary=direction_summary,
        annual_summary=annual_summary,
        exit_breakdown=exit_breakdown,
        benchmark_curves=benchmark_curves,
        benchmark_summary=benchmark_summary,
        validation_results=pd.DataFrame(),
        diagnostics={
            "concurrency_log": concurrency_log,
            "hourly_bars": context.get("hourly_bars", pd.DataFrame()).copy(),
            "hourly_bars_forward_filled": context.get("hourly_bars_forward_filled", pd.DataFrame()).copy(),
            "metadata": dict(context.get("metadata", {})),
            "audit": context.get("audit", {}),
            "notes": list(context.get("notes", [])),
            "data_paths": dict(context.get("data_paths", {})),
        },
        frame_state=frame_state,
        session_summary=session_summary,
    )
    result.validation_results = _build_recent_validation_results(result)
    return result


def run_recent_intrabar_sensitivity_analysis(
    context: dict[str, Any],
    replay_config: RecentIntrabarReplayConfig,
) -> dict[str, pd.DataFrame]:
    _validate_replay_config(replay_config)
    rows: list[dict[str, Any]] = []
    for atr_period in replay_config.sensitivity_atr_periods:
        for tp_sl_mode in replay_config.sensitivity_tp_sl_modes:
            k_values = replay_config.sensitivity_ks if tp_sl_mode == "prior_day_range" else (replay_config.k,)
            for atr_multiplier in replay_config.sensitivity_atr_multipliers:
                for k_value in k_values:
                    candidate = replace(
                        replay_config,
                        atr_period=atr_period,
                        tp_sl_mode=tp_sl_mode,
                        atr_multiplier=atr_multiplier,
                        k=k_value,
                    )
                    candidate_result = run_recent_intrabar_backtest(context, candidate)
                    metrics = candidate_result.performance_summary
                    rows.append(
                        {
                            "atr_period": atr_period,
                            "tp_sl_mode": tp_sl_mode,
                            "atr_multiplier": atr_multiplier,
                            "k": k_value,
                            "win_rate": metrics.get("win_rate_pct", np.nan),
                            "profit_factor": metrics.get("profit_factor", np.nan),
                            "sharpe_ratio": metrics.get("sharpe_ratio", np.nan),
                            "max_drawdown_pct": metrics.get("max_drawdown_pct", np.nan),
                            "terminal_net_equity": metrics.get("terminal_net_equity", np.nan),
                            "total_return_pct": metrics.get("total_return_pct", np.nan),
                            "total_trades": metrics.get("total_trades", np.nan),
                        }
                    )
    primary_grid = pd.DataFrame(rows)
    return {"primary_grid": primary_grid}


def _subset_context_by_session_date_range(
    context: dict[str, Any],
    start_session_date: pd.Timestamp,
    end_session_date: pd.Timestamp,
) -> dict[str, Any]:
    session_table = context["session_table"].copy()
    session_mask = session_table["session_date"].ge(start_session_date) & session_table["session_date"].lt(end_session_date)
    subset_sessions = session_table.loc[session_mask].copy().reset_index(drop=True)
    subset_ids = subset_sessions["session_id"].tolist()
    minute_bars = context["minute_bars"].loc[context["minute_bars"]["session_id"].isin(subset_ids)].copy().reset_index(drop=True)
    hourly_bars = context.get("hourly_bars", pd.DataFrame())
    hourly_forward_filled = context.get("hourly_bars_forward_filled", pd.DataFrame())
    subset_hourly = hourly_bars.loc[hourly_bars["session_id"].isin(subset_ids)].copy().reset_index(drop=True)
    subset_hourly_ff = hourly_forward_filled.loc[
        hourly_forward_filled["session_id"].isin(subset_ids)
    ].copy().reset_index(drop=True)
    subset_continuous = context.get("continuous_data", pd.DataFrame())
    subset_clean = context.get("clean_data", pd.DataFrame())
    subset_dominant = context.get("dominant_table", pd.DataFrame())
    if not subset_continuous.empty:
        subset_continuous = subset_continuous.loc[subset_continuous["session_id"].isin(subset_ids)].copy().reset_index(drop=True)
    if not subset_clean.empty and "session_id" in subset_clean.columns:
        subset_clean = subset_clean.loc[subset_clean["session_id"].isin(subset_ids)].copy().reset_index(drop=True)
    if not subset_dominant.empty:
        subset_dominant = subset_dominant.loc[subset_dominant["session_id"].isin(subset_ids)].copy().reset_index(drop=True)
    metadata = dict(context.get("metadata", {}))
    metadata.update(
        {
            "subset_train_start": start_session_date.isoformat(),
            "subset_end_exclusive": end_session_date.isoformat(),
            "subset_session_count": int(subset_sessions.shape[0]),
            "subset_minute_bar_count": int(minute_bars.shape[0]),
        }
    )
    return {
        "config": context["config"],
        "metadata": metadata,
        "data_paths": dict(context.get("data_paths", {})),
        "raw_data": context.get("raw_data", pd.DataFrame()),
        "clean_data": subset_clean,
        "audit": context.get("audit", {}),
        "notes": list(context.get("notes", [])),
        "minute_bars": minute_bars,
        "session_table": subset_sessions,
        "dominant_table": subset_dominant,
        "continuous_data": subset_continuous,
        "hourly_bars": subset_hourly,
        "hourly_bars_forward_filled": subset_hourly_ff,
    }


def _build_recent_walk_forward_windows(
    replay_config: RecentIntrabarReplayConfig,
    session_table: pd.DataFrame,
) -> pd.DataFrame:
    if session_table.empty:
        return pd.DataFrame()
    unique_dates = session_table["session_date"].drop_duplicates().sort_values().reset_index(drop=True)
    first_session_date = pd.Timestamp(unique_dates.iloc[0])
    last_session_date_exclusive = pd.Timestamp(unique_dates.iloc[-1]) + pd.Timedelta(days=1)
    windows: list[dict[str, Any]] = []
    fold_id = 1
    train_start = first_session_date
    while True:
        train_end = train_start + pd.DateOffset(months=replay_config.walk_forward_train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=replay_config.walk_forward_test_months)
        if test_end > last_session_date_exclusive:
            break
        train_mask = session_table["session_date"].ge(train_start) & session_table["session_date"].lt(train_end)
        test_mask = session_table["session_date"].ge(test_start) & session_table["session_date"].lt(test_end)
        if not train_mask.any() or not test_mask.any():
            break
        windows.append(
            {
                "fold_id": fold_id,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "train_sessions": int(train_mask.sum()),
                "test_sessions": int(test_mask.sum()),
            }
        )
        fold_id += 1
        train_start = train_start + pd.DateOffset(months=replay_config.walk_forward_step_months)
    return pd.DataFrame(windows)


def _recent_walk_forward_candidate_configs(
    replay_config: RecentIntrabarReplayConfig,
) -> list[RecentIntrabarReplayConfig]:
    candidates: list[RecentIntrabarReplayConfig] = []
    for atr_period in replay_config.walk_forward_atr_periods:
        for tp_sl_mode in replay_config.walk_forward_tp_sl_modes:
            k_values = replay_config.walk_forward_ks if tp_sl_mode == "prior_day_range" else (replay_config.k,)
            for atr_multiplier in replay_config.walk_forward_atr_multipliers:
                for k_value in k_values:
                    candidates.append(
                        replace(
                            replay_config,
                            atr_period=atr_period,
                            tp_sl_mode=tp_sl_mode,
                            atr_multiplier=atr_multiplier,
                            k=k_value,
                        )
                    )
    return candidates


def run_recent_intrabar_walk_forward_analysis(
    context: dict[str, Any],
    replay_config: RecentIntrabarReplayConfig,
) -> RecentIntrabarWalkForwardResult:
    _validate_replay_config(replay_config)
    fold_windows = _build_recent_walk_forward_windows(replay_config, context["session_table"])
    if fold_windows.empty:
        raise ValueError("No complete walk-forward folds are available for the configured month-based windows.")

    candidate_configs = _recent_walk_forward_candidate_configs(replay_config)
    optimization_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    oos_equity_frames: list[pd.DataFrame] = []
    oos_trade_frames: list[pd.DataFrame] = []
    subset_cache: dict[tuple[int, str], dict[str, Any]] = {}

    for fold in fold_windows.to_dict("records"):
        fold_id = int(fold["fold_id"])
        best_candidate: RecentIntrabarReplayConfig | None = None
        best_train_result: RecentIntrabarResult | None = None
        best_selection_key: tuple[float, float, float, float, float] | None = None

        train_cache_key = (fold_id, "train")
        if train_cache_key not in subset_cache:
            subset_cache[train_cache_key] = _subset_context_by_session_date_range(
                context,
                pd.Timestamp(fold["train_start"]),
                pd.Timestamp(fold["train_end"]),
            )
        train_context = subset_cache[train_cache_key]

        for candidate in candidate_configs:
            train_result = run_recent_intrabar_backtest(train_context, candidate)
            train_metrics = train_result.performance_summary
            objective_value = historical_backtest._walk_forward_objective_value(
                train_metrics,
                replay_config.walk_forward_objective,
            )
            optimization_rows.append(
                {
                    "fold_id": fold_id,
                    "train_start": fold["train_start"],
                    "train_end": fold["train_end"],
                    "test_start": fold["test_start"],
                    "test_end": fold["test_end"],
                    "atr_period": candidate.atr_period,
                    "tp_sl_mode": candidate.tp_sl_mode,
                    "atr_multiplier": candidate.atr_multiplier,
                    "k": candidate.k,
                    "objective": replay_config.walk_forward_objective,
                    "train_objective_value": objective_value,
                    "train_sharpe_ratio": train_metrics.get("sharpe_ratio", np.nan),
                    "train_total_return_pct": train_metrics.get("total_return_pct", np.nan),
                    "train_max_drawdown_pct": train_metrics.get("max_drawdown_pct", np.nan),
                    "train_total_trades": train_metrics.get("total_trades", np.nan),
                }
            )
            selection_key = historical_backtest._walk_forward_selection_key(
                train_metrics,
                replay_config.walk_forward_objective,
            )
            if best_selection_key is None or selection_key > best_selection_key:
                best_selection_key = selection_key
                best_candidate = candidate
                best_train_result = train_result

        if best_candidate is None or best_train_result is None:
            raise ValueError(f"Walk-forward fold {fold_id} produced no candidate results.")

        test_cache_key = (fold_id, "test")
        if test_cache_key not in subset_cache:
            subset_cache[test_cache_key] = _subset_context_by_session_date_range(
                context,
                pd.Timestamp(fold["test_start"]),
                pd.Timestamp(fold["test_end"]),
            )
        test_context = subset_cache[test_cache_key]
        oos_result = run_recent_intrabar_backtest(test_context, best_candidate)
        oos_metrics = oos_result.performance_summary
        train_metrics = best_train_result.performance_summary

        fold_rows.append(
            {
                "fold_id": fold_id,
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "train_sessions": fold["train_sessions"],
                "test_sessions": fold["test_sessions"],
                "objective": replay_config.walk_forward_objective,
                "selected_atr_period": best_candidate.atr_period,
                "selected_tp_sl_mode": best_candidate.tp_sl_mode,
                "selected_atr_multiplier": best_candidate.atr_multiplier,
                "selected_k": best_candidate.k,
                "train_objective_value": historical_backtest._walk_forward_objective_value(
                    train_metrics,
                    replay_config.walk_forward_objective,
                ),
                "train_sharpe_ratio": train_metrics.get("sharpe_ratio", np.nan),
                "train_total_return_pct": train_metrics.get("total_return_pct", np.nan),
                "train_max_drawdown_pct": train_metrics.get("max_drawdown_pct", np.nan),
                "train_total_trades": train_metrics.get("total_trades", np.nan),
                "oos_sharpe_ratio": oos_metrics.get("sharpe_ratio", np.nan),
                "oos_total_return_pct": oos_metrics.get("total_return_pct", np.nan),
                "oos_max_drawdown_pct": oos_metrics.get("max_drawdown_pct", np.nan),
                "oos_total_trades": oos_metrics.get("total_trades", np.nan),
                "oos_terminal_net_equity": oos_metrics.get("terminal_net_equity", np.nan),
            }
        )

        if not oos_result.equity_curve.empty:
            oos_equity = oos_result.equity_curve.copy()
            oos_equity["fold_id"] = fold_id
            oos_equity["selected_atr_period"] = best_candidate.atr_period
            oos_equity["selected_tp_sl_mode"] = best_candidate.tp_sl_mode
            oos_equity["selected_atr_multiplier"] = best_candidate.atr_multiplier
            oos_equity["selected_k"] = best_candidate.k
            oos_equity_frames.append(oos_equity)
        if not oos_result.trade_log.empty:
            oos_trade_log = oos_result.trade_log.copy()
            oos_trade_log["fold_id"] = fold_id
            oos_trade_log["selected_atr_period"] = best_candidate.atr_period
            oos_trade_log["selected_tp_sl_mode"] = best_candidate.tp_sl_mode
            oos_trade_log["selected_atr_multiplier"] = best_candidate.atr_multiplier
            oos_trade_log["selected_k"] = best_candidate.k
            oos_trade_frames.append(oos_trade_log)

    fold_summary = pd.DataFrame(fold_rows)
    optimization_results = pd.DataFrame(optimization_rows)
    parameter_stability = historical_backtest._build_walk_forward_parameter_stability(
        fold_summary.rename(
            columns={
                "selected_atr_period": "atr_period",
                "selected_tp_sl_mode": "tp_sl_mode",
                "selected_atr_multiplier": "atr_multiplier",
                "selected_k": "k",
            }
        )
    )
    oos_equity_curve, oos_session_equity = historical_backtest._stitch_walk_forward_oos_equity(
        oos_equity_frames=oos_equity_frames,
        starting_capital=replay_config.starting_capital,
    )
    oos_trade_log = pd.concat(oos_trade_frames, ignore_index=True) if oos_trade_frames else pd.DataFrame()
    diagnostics = {
        "fold_windows": fold_windows,
        "candidate_count": len(candidate_configs),
        "objective": replay_config.walk_forward_objective,
    }
    return RecentIntrabarWalkForwardResult(
        config=replay_config,
        fold_summary=fold_summary,
        optimization_results=optimization_results,
        parameter_stability=parameter_stability,
        oos_equity_curve=oos_equity_curve,
        oos_session_equity=oos_session_equity,
        oos_trade_log=oos_trade_log,
        diagnostics=diagnostics,
    )


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


def animate_intrabar_session_replay(
    diagnostic: dict[str, Any],
    session_id: int,
    figsize: tuple[float, float] = (15.0, 7.5),
) -> FuncAnimation:
    figure, ax = plt.subplots(figsize=figsize)
    session_frames = (
        diagnostic["frame_state"]
        .loc[diagnostic["frame_state"]["session_id"].eq(session_id)]
        .sort_values("bar_index")
        .reset_index(drop=True)
    )
    if session_frames.empty:
        raise ValueError(f"No intrabar replay data is available for session_id={session_id}.")
    frame_step = max(1, int(diagnostic["replay_config"].frame_step))
    frame_indices = session_frames["bar_index"].iloc[::frame_step].astype(int).tolist()
    final_index = int(session_frames["bar_index"].iloc[-1])
    if not frame_indices or frame_indices[-1] != final_index:
        frame_indices.append(final_index)

    def _update(end_bar_index: int) -> list[Any]:
        _render_intrabar_session_frame(ax, diagnostic, session_id, end_bar_index=end_bar_index)
        artists: list[Any] = []
        artists.extend(ax.lines)
        artists.extend(ax.collections)
        artists.extend(ax.patches)
        artists.extend(ax.texts)
        return artists

    animation = FuncAnimation(
        figure,
        _update,
        frames=frame_indices,
        interval=int(diagnostic["replay_config"].animation_interval_ms),
        blit=False,
        repeat=False,
    )
    plt.close(figure)
    return animation


def _empty_figure(message: str, figsize: tuple[float, float]) -> plt.Figure:
    figure, ax = plt.subplots(figsize=figsize)
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_axis_off()
    figure.tight_layout()
    return figure


def plot_equity_curve(result: RecentIntrabarResult) -> plt.Figure:
    if result.equity_curve.empty:
        return _empty_figure("No equity curve is available for the current recent sample.", (12.0, 5.0))
    return historical_backtest.plot_equity_curve(result)


def plot_drawdown_curve(result: RecentIntrabarResult) -> plt.Figure:
    if result.equity_curve.empty:
        return _empty_figure("No drawdown curve is available for the current recent sample.", (12.0, 4.0))
    return historical_backtest.plot_drawdown_curve(result)


def plot_directional_equity_curves(result: RecentIntrabarResult) -> plt.Figure:
    return historical_backtest.plot_directional_equity_curves(result)


def plot_directional_drawdown_curves(result: RecentIntrabarResult) -> plt.Figure:
    return historical_backtest.plot_directional_drawdown_curves(result)


def plot_monthly_returns_heatmap(result: RecentIntrabarResult) -> plt.Figure:
    if result.session_equity.empty or result.session_equity.shape[0] < 2:
        return _empty_figure("At least two session closes are required for monthly return heatmaps.", (12.0, 5.0))
    monthly = (
        result.session_equity.set_index("ts_event")["net_equity"].resample("M").last().pct_change().dropna() * 100.0
    )
    if monthly.empty:
        return _empty_figure("Monthly returns are not available for the current recent sample.", (12.0, 5.0))
    monthly = monthly.to_frame("return_pct")
    monthly["year"] = monthly.index.year
    monthly["month"] = monthly.index.month
    heatmap = monthly.pivot(index="year", columns="month", values="return_pct").sort_index()
    figure, ax = plt.subplots(figsize=(12, 5))
    image = ax.imshow(heatmap.values, aspect="auto", cmap="RdYlGn", interpolation="nearest")
    ax.set_title("Monthly Net Returns Heatmap (%)")
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
    ax.set_yticks(np.arange(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index.tolist())
    figure.colorbar(image, ax=ax, shrink=0.8)
    figure.tight_layout()
    return figure


def plot_trade_pnl_distribution(result: RecentIntrabarResult) -> plt.Figure:
    if result.trade_log.empty:
        return _empty_figure("No closed trades are available for a PnL distribution chart.", (12.0, 8.0))
    return historical_backtest.plot_trade_pnl_distribution(result)


def plot_exit_reason_breakdown(result: RecentIntrabarResult) -> plt.Figure:
    if result.trade_log.empty:
        return _empty_figure("No closed trades are available for an exit breakdown chart.", (8.0, 4.0))
    return historical_backtest.plot_exit_reason_breakdown(result)


def plot_heatmap_grid(
    primary_grid: pd.DataFrame,
    metric: str,
    atr_period: int,
    tp_sl_modes: tuple[str, ...] = VALID_TP_SL_MODES,
) -> plt.Figure:
    subset = primary_grid.loc[primary_grid["atr_period"] == atr_period].copy()
    if subset.empty or subset[metric].dropna().empty:
        return _empty_figure("No sensitivity data is available for the selected ATR period.", (16.0, 4.0))
    figure, axes = plt.subplots(1, len(tp_sl_modes), figsize=(16, 4), sharey=True)
    if len(tp_sl_modes) == 1:
        axes = [axes]
    for axis, mode in zip(axes, tp_sl_modes):
        mode_grid = subset.loc[subset["tp_sl_mode"] == mode]
        if mode_grid.empty:
            axis.text(0.5, 0.5, "No data", ha="center", va="center")
            axis.set_axis_off()
            continue
        pivot = mode_grid.pivot(index="atr_multiplier", columns="k", values=metric).sort_index()
        image = axis.imshow(pivot.values, aspect="auto", cmap="viridis", interpolation="nearest")
        axis.set_title(f"{mode}\n{metric}")
        axis.set_xticks(np.arange(len(pivot.columns)))
        axis.set_xticklabels([f"{value:.2f}" for value in pivot.columns])
        axis.set_yticks(np.arange(len(pivot.index)))
        axis.set_yticklabels([f"{value:.2f}" for value in pivot.index])
        axis.set_xlabel("k")
        axis.set_ylabel("ATR multiplier")
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    return figure


def plot_trail_sensitivity_curve(
    primary_grid: pd.DataFrame,
    replay_config: RecentIntrabarReplayConfig,
) -> plt.Figure:
    subset = primary_grid.loc[
        (primary_grid["atr_period"] == replay_config.atr_period)
        & (primary_grid["tp_sl_mode"] == replay_config.tp_sl_mode)
        & (primary_grid["k"] == replay_config.k)
    ].sort_values("atr_multiplier")
    if subset.empty:
        return _empty_figure("No trail-sensitivity data is available for the selected configuration.", (10.0, 4.0))
    figure, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()
    ax1.plot(subset["atr_multiplier"], subset["win_rate"], marker="o", label="Win rate %")
    ax2.plot(subset["atr_multiplier"], subset["sharpe_ratio"], marker="s", color="darkorange", label="Sharpe")
    ax1.set_title("Trail Sensitivity")
    ax1.set_xlabel("ATR multiplier")
    ax1.set_ylabel("Win rate %")
    ax2.set_ylabel("Sharpe ratio")
    ax1.grid(True, alpha=0.2)
    figure.tight_layout()
    return figure


def plot_session_replay(result: RecentIntrabarResult, session_id: int) -> plt.Figure:
    diagnostic = {
        "replay_config": result.config,
        "minute_bars": result.data,
        "event_log": result.event_log,
        "trade_log": result.trade_log,
        "frame_state": result.frame_state,
    }
    return plot_intrabar_session_replay(diagnostic, session_id)
