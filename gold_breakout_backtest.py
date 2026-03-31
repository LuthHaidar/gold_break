from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_COLUMNS = [
    "ts_event",
    "rtype",
    "publisher_id",
    "instrument_id",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
]
VALID_TP_SL_MODES = ("symmetric_fixed", "atr_based", "prior_day_range")
VALID_WALK_FORWARD_OBJECTIVES = ("sharpe_ratio", "total_return_pct", "calmar_ratio")
TICK_SIZE = 0.10
TICK_VALUE = 10.00
CONTRACT_MULTIPLIER = 100.0
ROUND_TURN_COST = 24.74
ROLL_ROUND_TURN_COST = ROUND_TURN_COST
DEFAULT_FIXED_TICKS = 100
DEFAULT_ATR_MULTIPLIER_TPSL = 2.0


@dataclass(frozen=True)
class BacktestConfig:
    csv_path: Path = Path("glbx-mdp3-20100606-20260325.ohlcv-1h.csv")
    backtest_start_session_date: str = "2018-08-08"
    starting_capital: float = 100_000.0
    risk_free_rate: float = 0.0
    max_ohlc_violations: int = 0
    min_session_volume: int = 0
    tp_sl_mode: str = "prior_day_range"
    fixed_ticks: int = DEFAULT_FIXED_TICKS
    atr_multiplier_tpsl: float = DEFAULT_ATR_MULTIPLIER_TPSL
    k: float = 0.50
    atr_period: int = 14
    atr_multiplier: float = 1.5
    risk_fraction: float = 0.01
    initial_margin_per_contract: float = 15_000.0
    max_margin_fraction: float = 0.50
    max_contracts: int = 3
    sensitivity_atr_periods: tuple[int, ...] = (7, 14, 21)
    sensitivity_atr_multipliers: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)
    sensitivity_ks: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
    sensitivity_tp_sl_modes: tuple[str, ...] = VALID_TP_SL_MODES
    sensitivity_risk_fractions: tuple[float, ...] = (0.005, 0.01, 0.02)
    sensitivity_max_contracts: tuple[int, ...] = (1, 2, 3, 5, 10)
    walk_forward_train_years: int = 3
    walk_forward_test_years: int = 1
    walk_forward_step_months: int = 12
    walk_forward_objective: str = "sharpe_ratio"
    walk_forward_atr_periods: tuple[int, ...] = (14,)
    walk_forward_tp_sl_modes: tuple[str, ...] = ("prior_day_range",)
    walk_forward_atr_multipliers: tuple[float, ...] = (1.0, 1.5, 2.0)
    walk_forward_ks: tuple[float, ...] = (0.25, 0.50, 0.75)

    def resolved_start_session_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.backtest_start_session_date, tz="UTC").normalize()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["csv_path"] = str(self.csv_path)
        return payload


@dataclass
class PositionState:
    trade_id: int
    direction: str
    entry_bar_ts: pd.Timestamp
    entry_bar_seq: int
    entry_session_id: int
    entry_session_date: pd.Timestamp
    entry_price: float
    current_leg_entry_price: float
    entry_adj_price: float
    contracts: int
    tp_adj: float
    initial_sl_adj: float
    current_trail_stop_adj: float
    highest_favorable_price_adj: float | None
    lowest_favorable_price_adj: float | None
    max_price_seen_adj: float
    min_price_seen_adj: float
    current_contract: str
    sizing_equity: float
    floor_applied: bool
    risk_per_trade: float
    sl_distance_price: float
    sl_distance_usd: float
    entry_fill_basis: str
    status: str = "open"
    exit_price: float | None = None
    exit_bar_ts: pd.Timestamp | None = None
    exit_reason: str | None = None
    exit_fill_basis: str | None = None
    realized_roll_pnl: float = 0.0
    realized_roll_cost: float = 0.0
    roll_count: int = 0
    bars_held: int = 0


@dataclass
class PreparedData:
    clean_data: pd.DataFrame
    session_table: pd.DataFrame
    dominant_table: pd.DataFrame
    continuous_data: pd.DataFrame
    atr_source_data: pd.DataFrame
    audit: dict[str, Any]
    notes: list[str]


@dataclass
class BacktestResult:
    config: BacktestConfig
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
    validation_results: pd.DataFrame
    diagnostics: dict[str, Any]


@dataclass
class WalkForwardResult:
    config: BacktestConfig
    fold_summary: pd.DataFrame
    optimization_results: pd.DataFrame
    parameter_stability: pd.DataFrame
    oos_equity_curve: pd.DataFrame
    oos_session_equity: pd.DataFrame
    oos_trade_log: pd.DataFrame
    diagnostics: dict[str, Any]


def make_default_config(**overrides: Any) -> BacktestConfig:
    base = BacktestConfig()
    return replace(base, **overrides)


def config_frame(config: BacktestConfig) -> pd.DataFrame:
    return pd.DataFrame(
        {"parameter": list(config.to_dict().keys()), "value": list(config.to_dict().values())}
    )


def summarize_audit(audit: dict[str, Any]) -> pd.DataFrame:
    rows = [{"metric": key, "value": value} for key, value in audit["counts"].items()]
    return pd.DataFrame(rows)


def _validate_config(config: BacktestConfig) -> None:
    if config.tp_sl_mode not in VALID_TP_SL_MODES:
        raise ValueError(f"Unsupported tp_sl_mode: {config.tp_sl_mode}")
    if config.walk_forward_objective not in VALID_WALK_FORWARD_OBJECTIVES:
        raise ValueError(f"Unsupported walk_forward_objective: {config.walk_forward_objective}")
    if config.fixed_ticks <= 0:
        raise ValueError("fixed_ticks must be positive.")
    if config.atr_period <= 0:
        raise ValueError("atr_period must be positive.")
    if config.atr_multiplier <= 0:
        raise ValueError("atr_multiplier must be positive.")
    if config.atr_multiplier_tpsl <= 0:
        raise ValueError("atr_multiplier_tpsl must be positive.")
    if config.k <= 0:
        raise ValueError("k must be positive.")
    if config.risk_fraction <= 0:
        raise ValueError("risk_fraction must be positive.")
    if config.initial_margin_per_contract <= 0:
        raise ValueError("initial_margin_per_contract must be positive.")
    if config.max_margin_fraction <= 0 or config.max_margin_fraction > 1:
        raise ValueError("max_margin_fraction must be in the interval (0, 1].")
    if config.max_contracts <= 0:
        raise ValueError("max_contracts must be positive.")
    if any(limit <= 0 for limit in config.sensitivity_max_contracts):
        raise ValueError("All sensitivity_max_contracts values must be positive.")
    if config.walk_forward_train_years <= 0:
        raise ValueError("walk_forward_train_years must be positive.")
    if config.walk_forward_test_years <= 0:
        raise ValueError("walk_forward_test_years must be positive.")
    if config.walk_forward_step_months <= 0:
        raise ValueError("walk_forward_step_months must be positive.")
    if config.walk_forward_step_months < config.walk_forward_test_years * 12:
        raise ValueError("walk_forward_step_months must be at least the walk-forward test horizon in months.")
    if any(period <= 0 for period in config.walk_forward_atr_periods):
        raise ValueError("All walk_forward_atr_periods values must be positive.")
    if any(multiplier <= 0 for multiplier in config.walk_forward_atr_multipliers):
        raise ValueError("All walk_forward_atr_multipliers values must be positive.")
    if any(k <= 0 for k in config.walk_forward_ks):
        raise ValueError("All walk_forward_ks values must be positive.")
    if any(mode not in VALID_TP_SL_MODES for mode in config.walk_forward_tp_sl_modes):
        raise ValueError("walk_forward_tp_sl_modes contains an unsupported mode.")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if pd.isna(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


def _directional_pnl(direction: str, entry_price: float, exit_price: float, contracts: int) -> float:
    if direction == "long":
        return (exit_price - entry_price) * contracts * CONTRACT_MULTIPLIER
    return (entry_price - exit_price) * contracts * CONTRACT_MULTIPLIER


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


def _position_size(
    current_equity: float,
    sl_distance_price: float,
    risk_fraction: float,
    initial_margin_per_contract: float,
    max_margin_fraction: float,
    max_contracts: int,
) -> dict[str, Any]:
    risk_per_trade = current_equity * risk_fraction
    sl_distance_ticks = sl_distance_price / TICK_SIZE
    sl_distance_usd = sl_distance_ticks * TICK_VALUE
    raw_contracts = int(np.floor(risk_per_trade / sl_distance_usd)) if sl_distance_usd > 0 else 0
    floor_contracts = max(raw_contracts, 1)
    allowed_initial_margin = current_equity * max_margin_fraction
    margin_allowed_contracts = (
        int(np.floor(allowed_initial_margin / initial_margin_per_contract))
        if initial_margin_per_contract > 0
        else 0
    )
    pre_contract_cap_contracts = min(floor_contracts, margin_allowed_contracts) if margin_allowed_contracts >= 1 else 0
    contracts = min(pre_contract_cap_contracts, max_contracts) if pre_contract_cap_contracts >= 1 else 0
    margin_cap_applied = margin_allowed_contracts >= 1 and pre_contract_cap_contracts < floor_contracts
    margin_cap_blocked = margin_allowed_contracts < 1
    contract_cap_applied = pre_contract_cap_contracts >= 1 and contracts < pre_contract_cap_contracts
    return {
        "risk_per_trade": risk_per_trade,
        "sl_distance_usd": sl_distance_usd,
        "raw_contracts": raw_contracts,
        "floor_contracts": floor_contracts,
        "contracts": contracts,
        "floor_applied": raw_contracts < 1 and contracts == 1,
        "allowed_initial_margin": allowed_initial_margin,
        "initial_margin_per_contract": initial_margin_per_contract,
        "margin_allowed_contracts": margin_allowed_contracts,
        "required_initial_margin": floor_contracts * initial_margin_per_contract,
        "selected_initial_margin": contracts * initial_margin_per_contract,
        "margin_cap_applied": margin_cap_applied,
        "margin_cap_blocked": margin_cap_blocked,
        "max_contracts": max_contracts,
        "contract_cap_applied": contract_cap_applied,
    }


def _wilder_atr(true_range: pd.Series, period: int) -> pd.Series:
    values = true_range.astype(float).to_numpy()
    atr = np.full(values.shape, np.nan, dtype=float)
    if len(values) < period:
        return pd.Series(atr, index=true_range.index)
    atr[period - 1] = np.nanmean(values[:period])
    for index in range(period, len(values)):
        atr[index] = ((atr[index - 1] * (period - 1)) + values[index]) / period
    return pd.Series(atr, index=true_range.index)


def _streaks(net_pnl: pd.Series) -> tuple[int, int]:
    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0
    for value in net_pnl.fillna(0.0):
        if value > 0:
            current_wins += 1
            current_losses = 0
        elif value < 0:
            current_losses += 1
            current_wins = 0
        else:
            current_wins = 0
            current_losses = 0
        max_wins = max(max_wins, current_wins)
        max_losses = max(max_losses, current_losses)
    return max_wins, max_losses


def _max_drawdown_duration(drawdown_pct: pd.Series) -> int:
    duration = 0
    max_duration = 0
    for value in drawdown_pct.fillna(0.0):
        if value < 0:
            duration += 1
            max_duration = max(max_duration, duration)
        else:
            duration = 0
    return max_duration


def prepare_research_data(config: BacktestConfig) -> PreparedData:
    _validate_config(config)
    audit: dict[str, Any] = {"counts": {}, "frames": {}, "lists": {}}
    notes = [
        (
            "Session boundaries are derived from the observed one-hour maintenance-gap pattern "
            "inside the approved post-2018-08-08 sample, because the UTC open shifts seasonally."
        ),
        (
            "The plan's roll-gap sign was internally inconsistent with the continuity test. "
            "The implementation uses the specified roll_gap definition but applies the negative "
            "cumulative adjustment required to satisfy Panama continuity."
        ),
        (
            "The plan's backward-adjustment live-only leakage assertion is mathematically "
            "incompatible with a standard backward Panama series. Validation therefore focuses on "
            "roll continuity and on keeping all fills and P&L in unadjusted price space."
        ),
        (
            f"Configurable defaults chosen for missing plan parameters: fixed_ticks={config.fixed_ticks}, "
            f"atr_multiplier_tpsl={config.atr_multiplier_tpsl}."
        ),
        (
            "ATR is always warmed up on the full pre-sample feature history before the active "
            "backtest window is filtered, and all ATR-period recomputations reuse that same source."
        ),
    ]
    raw = _load_csv(config)
    clean = _clean_outright_data(raw, config, audit)
    session_table, clean = _assign_sessions(clean)
    audit["counts"]["session_count_all_outrights"] = int(session_table.shape[0])
    audit["frames"]["low_volume_sessions_all_outrights"] = _flag_low_volume_sessions(
        session_table, config.min_session_volume
    )
    dominant_table, continuous = _build_continuous_series(clean, session_table, audit)
    continuous = _attach_session_features(continuous)
    continuous = _attach_true_range(continuous)
    atr_source_data = continuous.reset_index(drop=True)
    continuous = _attach_atr(atr_source_data, config.atr_period)
    filtered_sessions, filtered_continuous = _filter_backtest_window(
        session_table=dominant_table,
        continuous=continuous,
        start_session_date=config.resolved_start_session_date(),
    )
    return PreparedData(
        clean_data=clean,
        session_table=filtered_sessions.reset_index(drop=True),
        dominant_table=dominant_table.reset_index(drop=True),
        continuous_data=filtered_continuous.reset_index(drop=True),
        atr_source_data=atr_source_data,
        audit=audit,
        notes=notes,
    )


def with_atr_period(prepared: PreparedData, atr_period: int) -> PreparedData:
    source_with_atr = _attach_atr(prepared.atr_source_data, atr_period)
    session_ids = prepared.session_table["session_id"].tolist()
    data = source_with_atr.loc[source_with_atr["session_id"].isin(session_ids)].copy()
    session_table = prepared.session_table[prepared.session_table["session_id"].isin(session_ids)].copy()
    return PreparedData(
        clean_data=prepared.clean_data,
        session_table=session_table.reset_index(drop=True),
        dominant_table=prepared.dominant_table,
        continuous_data=data.reset_index(drop=True),
        atr_source_data=prepared.atr_source_data,
        audit=prepared.audit,
        notes=prepared.notes,
    )


def preview_signal_inputs(prepared: PreparedData, config: BacktestConfig, rows: int = 10) -> pd.DataFrame:
    open_bars = prepared.continuous_data.loc[prepared.continuous_data["session_open_bar"]].copy()
    open_bars["sl_distance_price"] = open_bars.apply(
        lambda row: _compute_sl_distance(
            tp_sl_mode=config.tp_sl_mode,
            fixed_ticks=config.fixed_ticks,
            atr_multiplier_tpsl=config.atr_multiplier_tpsl,
            k=config.k,
            atr_value=row["atr"],
            prev_session_range=row["prev_session_range"],
        ),
        axis=1,
    )
    preview = open_bars[
        [
            "ts_event",
            "session_date",
            "dominant_contract",
            "prev_session_high",
            "prev_session_low",
            "prev_session_range",
            "atr",
            "sl_distance_price",
            "adj_factor",
        ]
    ].head(rows)
    return preview.reset_index(drop=True)


def _load_csv(config: BacktestConfig) -> pd.DataFrame:
    header = pd.read_csv(config.csv_path, nrows=0).columns.tolist()
    if header != EXPECTED_COLUMNS:
        raise AssertionError(f"Schema mismatch. Expected {EXPECTED_COLUMNS}, found {header}.")
    dtype_map = {
        "rtype": "int64",
        "publisher_id": "int64",
        "instrument_id": "int64",
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "int64",
        "symbol": "string",
    }
    df = pd.read_csv(config.csv_path, dtype=dtype_map)
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="raise")
    if df["ts_event"].dt.tz is None:
        raise AssertionError("ts_event must be timezone aware in UTC.")
    return df


def _clean_outright_data(
    raw: pd.DataFrame,
    config: BacktestConfig,
    audit: dict[str, Any],
) -> pd.DataFrame:
    df = raw.copy()
    audit["counts"]["raw_rows"] = int(df.shape[0])
    rtype_mask = df["rtype"] == 34
    audit["counts"]["excluded_non_rtype_34"] = int((~rtype_mask).sum())
    df = df.loc[rtype_mask].copy()

    df["is_spread"] = df["symbol"].str.contains("-", regex=False)
    spread_rows = df.loc[df["is_spread"]].copy()
    audit["counts"]["excluded_spread_rows"] = int(spread_rows.shape[0])
    audit["frames"]["excluded_spreads"] = spread_rows[["symbol"]].drop_duplicates().sort_values("symbol")
    df = df.loc[~df["is_spread"]].copy()

    ohlc_ok = (
        df["low"].le(df["open"])
        & df["low"].le(df["close"])
        & df["open"].le(df["high"])
        & df["close"].le(df["high"])
    )
    ohlc_violations = df.loc[~ohlc_ok].copy()
    audit["frames"]["ohlc_violations"] = ohlc_violations
    audit["counts"]["ohlc_violation_count"] = int(ohlc_violations.shape[0])
    if ohlc_violations.shape[0] > config.max_ohlc_violations:
        raise AssertionError(
            f"OHLC violation count {ohlc_violations.shape[0]} exceeds threshold {config.max_ohlc_violations}."
        )
    positive_prices = (df[["open", "high", "low", "close"]] > 0).all(axis=1)
    if not positive_prices.all():
        invalid = df.loc[~positive_prices]
        raise AssertionError(f"Found non-positive outright OHLC values after spread exclusion: {invalid.shape[0]} rows.")

    multi_publisher = (
        df.groupby(["ts_event", "symbol"], as_index=False)["publisher_id"].nunique().query("publisher_id > 1")
    )
    audit["frames"]["multi_publisher_symbol_bars"] = multi_publisher

    duplicate_mask = df.duplicated(subset=["ts_event", "instrument_id"], keep=False)
    duplicate_rows = df.loc[duplicate_mask].copy()
    audit["counts"]["duplicate_ts_instrument_rows"] = int(duplicate_rows.shape[0])
    if not duplicate_rows.empty:
        duplicate_rows = duplicate_rows.sort_values(
            ["ts_event", "instrument_id", "volume", "publisher_id"],
            ascending=[True, True, False, True],
        )
        duplicate_rows["selected"] = False
        selected_indices = (
            duplicate_rows.groupby(["ts_event", "instrument_id"], sort=False).head(1).index.to_list()
        )
        duplicate_rows.loc[selected_indices, "selected"] = True
        audit["frames"]["duplicate_resolution"] = duplicate_rows
        df = (
            df.sort_values(["ts_event", "instrument_id", "volume", "publisher_id"], ascending=[True, True, False, True])
            .drop_duplicates(subset=["ts_event", "instrument_id"], keep="first")
            .copy()
        )
    else:
        audit["frames"]["duplicate_resolution"] = pd.DataFrame(columns=df.columns.tolist() + ["selected"])

    if (df["volume"] < 0).any():
        raise AssertionError("Negative volume bars detected.")
    zero_volume = df.loc[df["volume"] == 0].copy()
    audit["frames"]["zero_volume_bars"] = zero_volume
    audit["counts"]["zero_volume_bar_count"] = int(zero_volume.shape[0])
    low_liquidity = df.loc[df["volume"].isin([1, 2])].copy()
    audit["frames"]["low_liquidity_bars"] = low_liquidity
    audit["counts"]["low_liquidity_bar_count"] = int(low_liquidity.shape[0])
    return df.sort_values(["ts_event", "symbol", "instrument_id"]).reset_index(drop=True)


def _assign_sessions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_ts = pd.Series(df["ts_event"].drop_duplicates().sort_values().reset_index(drop=True), name="ts_event")
    gap = unique_ts.diff()
    session_starts = gap.isna() | gap.gt(pd.Timedelta(hours=1))
    session_id = session_starts.cumsum() - 1
    ts_map = pd.DataFrame({"ts_event": unique_ts, "session_id": session_id.astype(int)})
    session_table = (
        ts_map.groupby("session_id", as_index=False)
        .agg(
            session_open_ts=("ts_event", "min"),
            session_close_ts=("ts_event", "max"),
            session_bars=("ts_event", "count"),
        )
        .sort_values("session_id")
        .reset_index(drop=True)
    )
    session_table["session_date"] = session_table["session_open_ts"].dt.normalize()
    session_table["session_open_hour_utc"] = session_table["session_open_ts"].dt.hour
    session_table["session_close_hour_utc"] = session_table["session_close_ts"].dt.hour
    session_table["previous_session_id"] = session_table["session_id"].shift()
    df = df.merge(ts_map, on="ts_event", how="left", validate="many_to_one")
    df = df.merge(session_table, on="session_id", how="left", validate="many_to_one")
    df["session_open_bar"] = df["ts_event"].eq(df["session_open_ts"])
    df["session_close_bar"] = df["ts_event"].eq(df["session_close_ts"])
    total_volume = (
        df.groupby("session_id", as_index=False)["volume"]
        .sum()
        .rename(columns={"volume": "session_total_volume"})
    )
    session_table = session_table.merge(total_volume, on="session_id", how="left")
    return session_table, df


def _flag_low_volume_sessions(session_table: pd.DataFrame, threshold: int) -> pd.DataFrame:
    if threshold <= 0:
        return session_table.iloc[0:0].copy()
    return session_table.loc[session_table["session_total_volume"] < threshold].copy()


def _build_continuous_series(
    clean_df: pd.DataFrame,
    session_table: pd.DataFrame,
    audit: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    session_symbol_volume = (
        clean_df.groupby(["session_id", "session_date", "symbol"], as_index=False)["volume"].sum()
    )
    dominant = (
        session_symbol_volume.sort_values(["session_id", "volume", "symbol"], ascending=[True, False, True])
        .drop_duplicates(subset=["session_id"], keep="first")
        .rename(columns={"symbol": "dominant_contract", "volume": "dominant_volume"})
        .sort_values("session_id")
        .reset_index(drop=True)
    )
    dominant["previous_dominant_contract"] = dominant["dominant_contract"].shift()
    dominant["previous_dominant_volume"] = dominant["dominant_volume"].shift()
    dominant["previous_session_id"] = dominant["session_id"].shift()
    dominant["roll_flag"] = dominant["dominant_contract"].ne(dominant["previous_dominant_contract"]) & dominant[
        "previous_dominant_contract"
    ].notna()

    continuous = clean_df.merge(
        dominant[["session_id", "dominant_contract"]],
        left_on=["session_id", "symbol"],
        right_on=["session_id", "dominant_contract"],
        how="inner",
        validate="many_to_one",
    )
    continuous = continuous.drop(columns=["dominant_contract"])
    continuous = continuous.sort_values(["ts_event", "instrument_id"]).reset_index(drop=True)
    continuous = continuous.merge(
        dominant[
            [
                "session_id",
                "dominant_contract",
                "dominant_volume",
                "previous_dominant_contract",
                "previous_dominant_volume",
                "roll_flag",
            ]
        ],
        on="session_id",
        how="left",
        validate="many_to_one",
    )
    session_bar_counts = (
        continuous.groupby("session_id", as_index=False)["ts_event"]
        .nunique()
        .rename(columns={"ts_event": "continuous_session_bars"})
    )
    dominant = dominant.merge(session_table, on=["session_id", "session_date"], how="left", validate="one_to_one")
    dominant = dominant.rename(columns={"previous_session_id_x": "previous_session_id"})
    if "previous_session_id_y" in dominant.columns:
        dominant = dominant.drop(columns=["previous_session_id_y"])
    dominant = dominant.merge(session_bar_counts, on="session_id", how="left", validate="one_to_one")
    dominant["complete_session"] = dominant["continuous_session_bars"] == 23
    audit["frames"]["incomplete_or_irregular_sessions"] = dominant.loc[
        dominant["continuous_session_bars"] != 23,
        ["session_id", "session_date", "continuous_session_bars", "dominant_contract", "session_total_volume"],
    ].copy()

    session_edge_prices = (
        continuous.groupby("session_id", as_index=False)
        .agg(
            first_bar_ts=("ts_event", "first"),
            last_bar_ts=("ts_event", "last"),
            first_close=("close", "first"),
            last_close=("close", "last"),
            first_open=("open", "first"),
        )
        .sort_values("session_id")
    )
    dominant = dominant.merge(session_edge_prices, on="session_id", how="left", validate="one_to_one")
    dominant["outgoing_last_close"] = dominant["last_close"].shift()
    dominant["outgoing_last_bar_ts"] = dominant["last_bar_ts"].shift()
    dominant["roll_gap"] = np.where(dominant["roll_flag"], dominant["outgoing_last_close"] - dominant["first_close"], 0.0)
    dominant["roll_adjustment"] = np.where(dominant["roll_flag"], -dominant["roll_gap"], 0.0)
    dominant["adj_factor"] = dominant["roll_adjustment"][::-1].cumsum()[::-1] - dominant["roll_adjustment"]
    dominant["previous_adj_factor"] = dominant["adj_factor"].shift()
    dominant["roll_continuity_error"] = np.where(
        dominant["roll_flag"],
        (dominant["outgoing_last_close"] + dominant["previous_adj_factor"])
        - (dominant["first_close"] + dominant["adj_factor"]),
        0.0,
    )
    if not dominant.loc[dominant["roll_flag"], "roll_continuity_error"].abs().lt(TICK_SIZE).all():
        raise AssertionError("Adjusted roll continuity exceeded one tick.")

    continuous = continuous.merge(
        dominant[
            [
                "session_id",
                "adj_factor",
                "roll_gap",
            ]
        ],
        on="session_id",
        how="left",
        validate="many_to_one",
    )
    for column in ["open", "high", "low", "close"]:
        continuous[f"adj_{column}"] = continuous[column] + continuous["adj_factor"]
    continuous["bar_seq"] = np.arange(len(continuous), dtype=int)
    audit["frames"]["roll_log"] = dominant.loc[
        dominant["roll_flag"],
        [
            "session_id",
            "session_date",
            "previous_dominant_contract",
            "dominant_contract",
            "previous_dominant_volume",
            "dominant_volume",
            "roll_gap",
            "roll_continuity_error",
        ],
    ].copy()
    return dominant.reset_index(drop=True), continuous.reset_index(drop=True)


def _attach_session_features(continuous: pd.DataFrame) -> pd.DataFrame:
    df = continuous.copy()
    session_stats = (
        df.groupby(["session_id", "session_date"], as_index=False)
        .agg(
            session_high=("adj_high", "max"),
            session_low=("adj_low", "min"),
            session_open_ts=("session_open_ts", "first"),
        )
        .sort_values("session_id")
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
    return df


def _attach_true_range(continuous: pd.DataFrame) -> pd.DataFrame:
    df = continuous.copy()
    previous_close = df["adj_close"].shift()
    components = pd.concat(
        [
            df["adj_high"] - df["adj_low"],
            (df["adj_high"] - previous_close).abs(),
            (df["adj_low"] - previous_close).abs(),
        ],
        axis=1,
    )
    df["true_range"] = components.max(axis=1)
    return df


def _attach_atr(continuous: pd.DataFrame, atr_period: int) -> pd.DataFrame:
    df = continuous.copy()
    atr_unshifted = _wilder_atr(df["true_range"], atr_period)
    df["atr_period"] = atr_period
    df["atr_unshifted"] = atr_unshifted
    df["atr"] = atr_unshifted.shift(1)
    return df


def _filter_backtest_window(
    session_table: pd.DataFrame,
    continuous: pd.DataFrame,
    start_session_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered_sessions = session_table.loc[session_table["session_date"] >= start_session_date].copy()
    filtered_ids = filtered_sessions["session_id"].tolist()
    filtered_continuous = continuous.loc[continuous["session_id"].isin(filtered_ids)].copy()
    return filtered_sessions, filtered_continuous


def _prior_session_quality_skip_reason(
    previous_session_row: pd.Series | None,
    config: BacktestConfig,
) -> str | None:
    if previous_session_row is None:
        return None
    quality_failures: list[str] = []
    if not bool(previous_session_row["complete_session"]):
        quality_failures.append(
            f"incomplete previous session ({int(previous_session_row['continuous_session_bars'])} bars)"
        )
    if (
        config.min_session_volume > 0
        and float(previous_session_row["session_total_volume"]) < float(config.min_session_volume)
    ):
        quality_failures.append(
            "previous session volume below configured threshold"
        )
    if not quality_failures:
        return None
    return f"Previous session failed quality filter: {', '.join(quality_failures)}."


def run_backtest(prepared: PreparedData, config: BacktestConfig) -> BacktestResult:
    data = prepared.continuous_data.copy().reset_index(drop=True)
    sessions = prepared.session_table.copy().reset_index(drop=True)
    session_lookup = sessions.set_index("session_id")
    full_session_lookup = prepared.dominant_table.set_index("session_id")
    open_positions: list[PositionState] = []
    trade_records: list[dict[str, Any]] = []
    roll_events: list[dict[str, Any]] = []
    order_cancellations: list[dict[str, Any]] = []
    floor_events: list[dict[str, Any]] = []
    margin_flags: list[dict[str, Any]] = []
    skipped_sessions: list[dict[str, Any]] = []
    event_log: list[dict[str, Any]] = []
    concurrency_log: list[dict[str, Any]] = []
    session_sizing_log: list[dict[str, Any]] = []
    session_concurrent_long_short: set[int] = set()
    current_equity = float(config.starting_capital)
    trade_id_counter = 1
    data_by_session = {sid: frame.copy() for sid, frame in data.groupby("session_id", sort=True)}

    for session_id, session_bars in data_by_session.items():
        session_row = session_lookup.loc[session_id]
        session_bars = session_bars.reset_index(drop=True)
        open_bar = session_bars.iloc[0]

        if bool(session_row["roll_flag"]) and open_positions:
            previous_session_id = int(session_row["previous_session_id"])
            previous_close_bar = data_by_session[previous_session_id].iloc[-1]
            for position in list(open_positions):
                leg_gross = _directional_pnl(
                    direction=position.direction,
                    entry_price=position.current_leg_entry_price,
                    exit_price=float(previous_close_bar["close"]),
                    contracts=position.contracts,
                )
                roll_cost = position.contracts * ROLL_ROUND_TURN_COST
                position.realized_roll_pnl += leg_gross
                position.realized_roll_cost += roll_cost
                position.roll_count += 1
                position.current_leg_entry_price = float(open_bar["open"])
                position.current_contract = str(session_row["dominant_contract"])
                current_equity += leg_gross - roll_cost
                roll_events.append(
                    {
                        "trade_id": position.trade_id,
                        "roll_session_id": session_id,
                        "roll_session_date": pd.Timestamp(session_row["session_date"]),
                        "roll_ts": open_bar["ts_event"],
                        "outgoing_contract": str(session_row["previous_dominant_contract"]),
                        "incoming_contract": str(session_row["dominant_contract"]),
                        "outgoing_close": float(previous_close_bar["close"]),
                        "incoming_open": float(open_bar["open"]),
                        "contracts": position.contracts,
                        "gross_roll_pnl": leg_gross,
                        "roll_cost": roll_cost,
                        "roll_continuity_error": float(session_row["roll_continuity_error"]),
                    }
                )
                event_log.append(
                    {
                        "ts_event": open_bar["ts_event"],
                        "session_id": session_id,
                        "trade_id": position.trade_id,
                        "event_type": "roll",
                        "gross_delta": leg_gross,
                        "net_delta": leg_gross - roll_cost,
                        "equity_after_event": current_equity,
                    }
                )

        open_positions, closed_trades, current_equity = _process_session_open_gap_checks(
            open_positions=open_positions,
            open_bar=open_bar,
            current_equity=current_equity,
        )
        trade_records.extend(closed_trades)
        for closed_trade in closed_trades:
            event_log.append(
                {
                    "ts_event": closed_trade["exit_bar_ts"],
                    "session_id": session_id,
                    "trade_id": closed_trade["trade_id"],
                    "event_type": "gap_exit",
                    "gross_delta": closed_trade["incremental_gross_pnl"],
                    "net_delta": closed_trade["incremental_net_pnl"],
                    "equity_after_event": current_equity,
                }
            )

        session_open_equity = current_equity
        if open_positions:
            pending_orders = {
                "active": False,
                "skip_reason": "Open position carried into session; new session orders suppressed.",
                "skip_reason_code": "carry_position",
            }
        else:
            previous_session_row = None
            if pd.notna(session_row["previous_session_id"]):
                previous_session_id = int(session_row["previous_session_id"])
                if previous_session_id in full_session_lookup.index:
                    previous_session_row = full_session_lookup.loc[previous_session_id]
            prior_session_quality_skip_reason = _prior_session_quality_skip_reason(previous_session_row, config)
            if prior_session_quality_skip_reason is not None:
                pending_orders = {
                    "active": False,
                    "skip_reason": prior_session_quality_skip_reason,
                    "skip_reason_code": "prior_session_quality",
                }
            else:
                pending_orders = _initialize_pending_orders(open_bar, config, session_open_equity)
        if pending_orders["active"]:
            session_sizing_log.append(
                {
                    "session_id": session_id,
                    "session_date": session_row["session_date"],
                    "session_open_equity": session_open_equity,
                    "risk_per_trade": pending_orders["risk_per_trade"],
                    "contracts": pending_orders["contracts"],
                    "raw_contracts": pending_orders["raw_contracts"],
                    "floor_contracts": pending_orders["floor_contracts"],
                    "margin_allowed_contracts": pending_orders["margin_allowed_contracts"],
                    "floor_applied": pending_orders["floor_applied"],
                    "margin_cap_applied": pending_orders["margin_cap_applied"],
                    "max_contracts": pending_orders["max_contracts"],
                    "contract_cap_applied": pending_orders["contract_cap_applied"],
                    "sl_distance_price": pending_orders["sl_distance_price"],
                    "sl_distance_usd": pending_orders["sl_distance_usd"],
                    "allowed_initial_margin": pending_orders["allowed_initial_margin"],
                    "required_initial_margin": pending_orders["required_initial_margin"],
                    "selected_initial_margin": pending_orders["selected_initial_margin"],
                    "initial_margin_per_contract": pending_orders["initial_margin_per_contract"],
                }
            )
            if pending_orders["margin_cap_applied"]:
                margin_flags.append(
                    {
                        "session_id": session_id,
                        "session_date": session_row["session_date"],
                        "ts_event": open_bar["ts_event"],
                        "flag_type": "margin_cap_applied",
                        "requested_contracts": pending_orders["floor_contracts"],
                        "allowed_contracts": pending_orders["margin_allowed_contracts"],
                        "selected_contracts": pending_orders["contracts"],
                        "allowed_initial_margin": pending_orders["allowed_initial_margin"],
                        "required_initial_margin": pending_orders["required_initial_margin"],
                        "selected_initial_margin": pending_orders["selected_initial_margin"],
                    }
                )
        else:
            skipped_sessions.append(
                {
                    "session_id": session_id,
                    "session_date": session_row["session_date"],
                    "ts_event": open_bar["ts_event"],
                    "reason": pending_orders["skip_reason"],
                }
            )
            if pending_orders.get("skip_reason_code") == "margin_cap_blocked":
                margin_flags.append(
                    {
                        "session_id": session_id,
                        "session_date": session_row["session_date"],
                        "ts_event": open_bar["ts_event"],
                        "flag_type": "margin_cap_blocked",
                        "requested_contracts": pending_orders["floor_contracts"],
                        "allowed_contracts": pending_orders["margin_allowed_contracts"],
                        "selected_contracts": 0,
                        "allowed_initial_margin": pending_orders["allowed_initial_margin"],
                        "required_initial_margin": pending_orders["required_initial_margin"],
                        "selected_initial_margin": 0.0,
                    }
                )

        for _, bar in session_bars.iterrows():
            existing_positions = [pos for pos in open_positions if pos.entry_bar_seq < int(bar["bar_seq"])]
            to_close: list[dict[str, Any]] = []
            for position in existing_positions:
                position.bars_held += 1
                exit_payload = _evaluate_position_on_bar(position, bar, config)
                if exit_payload is not None:
                    to_close.append(exit_payload)

            closed_ids: set[int] = set()
            for exit_payload in to_close:
                trade_record, current_equity = _close_position(
                    position=exit_payload["position"],
                    exit_bar=bar,
                    exit_price=exit_payload["exit_price"],
                    exit_reason=exit_payload["exit_reason"],
                    exit_fill_basis=exit_payload["exit_fill_basis"],
                    current_equity=current_equity,
                )
                trade_records.append(trade_record)
                event_log.append(
                    {
                        "ts_event": trade_record["exit_bar_ts"],
                        "session_id": session_id,
                        "trade_id": trade_record["trade_id"],
                        "event_type": "bar_exit",
                        "gross_delta": trade_record["incremental_gross_pnl"],
                        "net_delta": trade_record["incremental_net_pnl"],
                        "equity_after_event": current_equity,
                    }
                )
                closed_ids.add(trade_record["trade_id"])
            if closed_ids:
                open_positions = [pos for pos in open_positions if pos.trade_id not in closed_ids]

            if pending_orders["active"]:
                new_positions, floor_rows, cancellation_rows, skip_reason, trade_id_counter = _check_entry_triggers(
                    bar=bar,
                    pending_orders=pending_orders,
                    trade_id_counter=trade_id_counter,
                )
                open_positions.extend(new_positions)
                floor_events.extend(floor_rows)
                order_cancellations.extend(cancellation_rows)
                if skip_reason is not None:
                    skipped_sessions.append(
                        {
                            "session_id": session_id,
                            "session_date": session_row["session_date"],
                            "ts_event": bar["ts_event"],
                            "reason": skip_reason,
                        }
                    )

            concurrency_snapshot = _capture_concurrency_snapshot(
                ts_event=bar["ts_event"],
                session_id=session_id,
                open_positions=open_positions,
            )
            concurrency_log.append(concurrency_snapshot)
            if concurrency_snapshot["simultaneous_long_short"]:
                session_concurrent_long_short.add(session_id)

            if bool(bar["session_close_bar"]) and pending_orders["active"]:
                if not pending_orders["long_triggered"]:
                    order_cancellations.append(
                        {
                            "session_id": session_id,
                            "session_date": session_row["session_date"],
                            "direction": "long",
                            "cancel_ts": bar["ts_event"],
                            "stop_level_adj": pending_orders["buy_stop_adj"],
                        }
                    )
                if not pending_orders["short_triggered"]:
                    order_cancellations.append(
                        {
                            "session_id": session_id,
                            "session_date": session_row["session_date"],
                            "direction": "short",
                            "cancel_ts": bar["ts_event"],
                            "stop_level_adj": pending_orders["sell_stop_adj"],
                        }
                    )
                pending_orders["active"] = False

    return _finalize_backtest_results(
        prepared=prepared,
        config=config,
        data=data,
        sessions=sessions,
        trade_records=trade_records,
        roll_events=roll_events,
        order_cancellations=order_cancellations,
        floor_events=floor_events,
        margin_flags=margin_flags,
        skipped_sessions=skipped_sessions,
        event_log=event_log,
        concurrency_log=concurrency_log,
        session_sizing_log=session_sizing_log,
        session_concurrent_long_short=session_concurrent_long_short,
    )


def _process_session_open_gap_checks(
    open_positions: list[PositionState],
    open_bar: pd.Series,
    current_equity: float,
) -> tuple[list[PositionState], list[dict[str, Any]], float]:
    remaining: list[PositionState] = []
    closed_trades: list[dict[str, Any]] = []
    for position in open_positions:
        tp_unadjusted = position.tp_adj - float(open_bar["adj_factor"])
        stop_unadjusted = position.current_trail_stop_adj - float(open_bar["adj_factor"])
        initial_sl_unadjusted = position.initial_sl_adj - float(open_bar["adj_factor"])
        open_price = float(open_bar["open"])
        open_price_adj = open_price + float(open_bar["adj_factor"])
        position.max_price_seen_adj = max(position.max_price_seen_adj, open_price_adj)
        position.min_price_seen_adj = min(position.min_price_seen_adj, open_price_adj)
        should_close = False
        exit_reason = None
        if position.direction == "long":
            if open_price >= tp_unadjusted:
                should_close = True
                exit_reason = "tp"
            elif open_price <= stop_unadjusted or open_price <= initial_sl_unadjusted:
                should_close = True
                exit_reason = "gap_stop"
        else:
            if open_price <= tp_unadjusted:
                should_close = True
                exit_reason = "tp"
            elif open_price >= stop_unadjusted or open_price >= initial_sl_unadjusted:
                should_close = True
                exit_reason = "gap_stop"
        if should_close:
            trade_record, current_equity = _close_position(
                position=position,
                exit_bar=open_bar,
                exit_price=open_price,
                exit_reason=str(exit_reason),
                exit_fill_basis="unadjusted_open_gap_exit",
                current_equity=current_equity,
            )
            closed_trades.append(trade_record)
        else:
            remaining.append(position)
    return remaining, closed_trades, current_equity


def _initialize_pending_orders(open_bar: pd.Series, config: BacktestConfig, session_open_equity: float) -> dict[str, Any]:
    sl_distance = _compute_sl_distance(
        tp_sl_mode=config.tp_sl_mode,
        fixed_ticks=config.fixed_ticks,
        atr_multiplier_tpsl=config.atr_multiplier_tpsl,
        k=config.k,
        atr_value=open_bar["atr"],
        prev_session_range=open_bar["prev_session_range"],
    )
    missing_inputs = [
        pd.isna(open_bar["prev_session_high"]),
        pd.isna(open_bar["prev_session_low"]),
        pd.isna(sl_distance),
        sl_distance <= 0 if not pd.isna(sl_distance) else True,
    ]
    if any(missing_inputs):
        return {
            "active": False,
            "skip_reason": "Insufficient confirmed session inputs for order placement.",
            "skip_reason_code": "insufficient_inputs",
        }
    sizing = _position_size(
        session_open_equity,
        float(sl_distance),
        config.risk_fraction,
        config.initial_margin_per_contract,
        config.max_margin_fraction,
        config.max_contracts,
    )
    if sizing["margin_cap_blocked"]:
        return {
            "active": False,
            "skip_reason": "Initial margin cap prevents minimum one-contract entry.",
            "skip_reason_code": "margin_cap_blocked",
            "session_id": int(open_bar["session_id"]),
            "session_date": open_bar["session_date"],
            "floor_contracts": sizing["floor_contracts"],
            "margin_allowed_contracts": sizing["margin_allowed_contracts"],
            "allowed_initial_margin": sizing["allowed_initial_margin"],
            "required_initial_margin": sizing["required_initial_margin"],
            "max_contracts": sizing["max_contracts"],
            "selected_initial_margin": sizing["selected_initial_margin"],
            "contract_cap_applied": sizing["contract_cap_applied"],
        }
    return {
        "active": True,
        "skip_reason": None,
        "skip_reason_code": None,
        "session_id": int(open_bar["session_id"]),
        "session_date": open_bar["session_date"],
        "buy_stop_adj": float(open_bar["prev_session_high"]),
        "sell_stop_adj": float(open_bar["prev_session_low"]),
        "sl_distance_price": float(sl_distance),
        "risk_per_trade": sizing["risk_per_trade"],
        "sl_distance_usd": sizing["sl_distance_usd"],
        "contracts": sizing["contracts"],
        "raw_contracts": sizing["raw_contracts"],
        "floor_contracts": sizing["floor_contracts"],
        "floor_applied": sizing["floor_applied"],
        "allowed_initial_margin": sizing["allowed_initial_margin"],
        "initial_margin_per_contract": sizing["initial_margin_per_contract"],
        "margin_allowed_contracts": sizing["margin_allowed_contracts"],
        "required_initial_margin": sizing["required_initial_margin"],
        "selected_initial_margin": sizing["selected_initial_margin"],
        "margin_cap_applied": sizing["margin_cap_applied"],
        "max_contracts": sizing["max_contracts"],
        "contract_cap_applied": sizing["contract_cap_applied"],
        "sizing_equity": session_open_equity,
        "long_triggered": False,
        "short_triggered": False,
    }


def _check_entry_triggers(
    bar: pd.Series,
    pending_orders: dict[str, Any],
    trade_id_counter: int,
) -> tuple[list[PositionState], list[dict[str, Any]], list[dict[str, Any]], str | None, int]:
    new_positions: list[PositionState] = []
    floor_rows: list[dict[str, Any]] = []
    order_cancellations: list[dict[str, Any]] = []
    skip_reason: str | None = None
    adj_factor = float(bar["adj_factor"])
    long_trade: PositionState | None = None
    short_trade: PositionState | None = None

    buy_stop_unadjusted = pending_orders["buy_stop_adj"] - adj_factor
    sell_stop_unadjusted = pending_orders["sell_stop_adj"] - adj_factor
    long_gap_trigger = float(bar["open"]) > buy_stop_unadjusted
    short_gap_trigger = float(bar["open"]) < sell_stop_unadjusted
    long_bar_trigger = float(bar["adj_high"]) >= pending_orders["buy_stop_adj"]
    short_bar_trigger = float(bar["adj_low"]) <= pending_orders["sell_stop_adj"]

    def _cancellation_row(direction: str, reason: str) -> dict[str, Any]:
        stop_level = pending_orders["buy_stop_adj"] if direction == "long" else pending_orders["sell_stop_adj"]
        return {
            "session_id": pending_orders["session_id"],
            "session_date": pending_orders["session_date"],
            "direction": direction,
            "cancel_ts": bar["ts_event"],
            "stop_level_adj": stop_level,
            "cancel_reason": reason,
        }

    if long_gap_trigger:
        long_trade = _open_position(
            trade_id=trade_id_counter,
            direction="long",
            bar=bar,
            entry_price=float(bar["open"]),
            entry_fill_basis="unadjusted_open_gap_entry",
            pending_orders=pending_orders,
        )
        trade_id_counter += 1
        pending_orders["long_triggered"] = True
        pending_orders["active"] = False
        new_positions.append(long_trade)
        order_cancellations.append(_cancellation_row("short", "oco_after_long_fill"))
    elif short_gap_trigger:
        short_trade = _open_position(
            trade_id=trade_id_counter,
            direction="short",
            bar=bar,
            entry_price=float(bar["open"]),
            entry_fill_basis="unadjusted_open_gap_entry",
            pending_orders=pending_orders,
        )
        trade_id_counter += 1
        pending_orders["short_triggered"] = True
        pending_orders["active"] = False
        new_positions.append(short_trade)
        order_cancellations.append(_cancellation_row("long", "oco_after_short_fill"))
    elif long_bar_trigger and short_bar_trigger:
        pending_orders["active"] = False
        skip_reason = "Ambiguous hourly OCO trigger: both entry stops touched in the same bar."
        order_cancellations.append(_cancellation_row("long", "ambiguous_oco_skip"))
        order_cancellations.append(_cancellation_row("short", "ambiguous_oco_skip"))
    elif long_bar_trigger:
        long_trade = _open_position(
            trade_id=trade_id_counter,
            direction="long",
            bar=bar,
            entry_price=buy_stop_unadjusted + TICK_SIZE,
            entry_fill_basis="unadjusted_stop_plus_tick_entry",
            pending_orders=pending_orders,
        )
        trade_id_counter += 1
        pending_orders["long_triggered"] = True
        pending_orders["active"] = False
        new_positions.append(long_trade)
        order_cancellations.append(_cancellation_row("short", "oco_after_long_fill"))
    elif short_bar_trigger:
        short_trade = _open_position(
            trade_id=trade_id_counter,
            direction="short",
            bar=bar,
            entry_price=sell_stop_unadjusted - TICK_SIZE,
            entry_fill_basis="unadjusted_stop_minus_tick_entry",
            pending_orders=pending_orders,
        )
        trade_id_counter += 1
        pending_orders["short_triggered"] = True
        pending_orders["active"] = False
        new_positions.append(short_trade)
        order_cancellations.append(_cancellation_row("long", "oco_after_short_fill"))

    for trade in [long_trade, short_trade]:
        if trade is not None and trade.floor_applied:
            floor_rows.append(
                {
                    "trade_id": trade.trade_id,
                    "direction": trade.direction,
                    "entry_bar_ts": trade.entry_bar_ts,
                    "contracts": trade.contracts,
                    "sizing_equity": trade.sizing_equity,
                    "risk_per_trade": trade.risk_per_trade,
                    "sl_distance_usd": trade.sl_distance_usd,
                }
            )
    return new_positions, floor_rows, order_cancellations, skip_reason, trade_id_counter


def _open_position(
    trade_id: int,
    direction: str,
    bar: pd.Series,
    entry_price: float,
    entry_fill_basis: str,
    pending_orders: dict[str, Any],
) -> PositionState:
    adj_entry = entry_price + float(bar["adj_factor"])
    distance = float(pending_orders["sl_distance_price"])
    if direction == "long":
        tp_adj = adj_entry + distance
        initial_sl_adj = adj_entry - distance
        high_favorable = adj_entry
        low_favorable = None
    else:
        tp_adj = adj_entry - distance
        initial_sl_adj = adj_entry + distance
        high_favorable = None
        low_favorable = adj_entry
    return PositionState(
        trade_id=trade_id,
        direction=direction,
        entry_bar_ts=bar["ts_event"],
        entry_bar_seq=int(bar["bar_seq"]),
        entry_session_id=int(bar["session_id"]),
        entry_session_date=pd.Timestamp(bar["session_date"]),
        entry_price=entry_price,
        current_leg_entry_price=entry_price,
        entry_adj_price=adj_entry,
        contracts=int(pending_orders["contracts"]),
        tp_adj=tp_adj,
        initial_sl_adj=initial_sl_adj,
        current_trail_stop_adj=initial_sl_adj,
        highest_favorable_price_adj=high_favorable,
        lowest_favorable_price_adj=low_favorable,
        max_price_seen_adj=adj_entry,
        min_price_seen_adj=adj_entry,
        current_contract=str(bar["dominant_contract"]),
        sizing_equity=float(pending_orders["sizing_equity"]),
        floor_applied=bool(pending_orders["floor_applied"]),
        risk_per_trade=float(pending_orders["risk_per_trade"]),
        sl_distance_price=distance,
        sl_distance_usd=float(pending_orders["sl_distance_usd"]),
        entry_fill_basis=entry_fill_basis,
    )


def _evaluate_position_on_bar(
    position: PositionState,
    bar: pd.Series,
    config: BacktestConfig,
) -> dict[str, Any] | None:
    active_trail_stop_adj = float(position.current_trail_stop_adj)
    active_initial_sl_adj = float(position.initial_sl_adj)
    current_tp_unadjusted = position.tp_adj - float(bar["adj_factor"])
    current_stop_unadjusted = active_trail_stop_adj - float(bar["adj_factor"])
    initial_sl_unadjusted = active_initial_sl_adj - float(bar["adj_factor"])
    trail_is_active = not np.isclose(active_trail_stop_adj, active_initial_sl_adj)

    if position.direction == "long":
        tp_hit = float(bar["adj_high"]) >= position.tp_adj
        stop_hit = float(bar["adj_low"]) <= active_trail_stop_adj
        if tp_hit and stop_hit:
            exit_reason = "trail_stop" if trail_is_active else "initial_sl"
            return {
                "position": position,
                "exit_price": current_stop_unadjusted - TICK_SIZE,
                "exit_reason": exit_reason,
                "exit_fill_basis": "unadjusted_stop_minus_tick_exit",
            }
        if stop_hit:
            exit_reason = "trail_stop" if trail_is_active else "initial_sl"
            return {
                "position": position,
                "exit_price": current_stop_unadjusted - TICK_SIZE,
                "exit_reason": exit_reason,
                "exit_fill_basis": "unadjusted_stop_minus_tick_exit",
            }
        if tp_hit:
            return {
                "position": position,
                "exit_price": current_tp_unadjusted,
                "exit_reason": "tp",
                "exit_fill_basis": "unadjusted_tp_limit_exit",
            }
        if not trail_is_active and float(bar["adj_low"]) <= active_initial_sl_adj:
            return {
                "position": position,
                "exit_price": initial_sl_unadjusted - TICK_SIZE,
                "exit_reason": "initial_sl",
                "exit_fill_basis": "unadjusted_stop_minus_tick_exit",
            }
    else:
        tp_hit = float(bar["adj_low"]) <= position.tp_adj
        stop_hit = float(bar["adj_high"]) >= active_trail_stop_adj
        if tp_hit and stop_hit:
            exit_reason = "trail_stop" if trail_is_active else "initial_sl"
            return {
                "position": position,
                "exit_price": current_stop_unadjusted + TICK_SIZE,
                "exit_reason": exit_reason,
                "exit_fill_basis": "unadjusted_stop_plus_tick_exit",
            }
        if stop_hit:
            exit_reason = "trail_stop" if trail_is_active else "initial_sl"
            return {
                "position": position,
                "exit_price": current_stop_unadjusted + TICK_SIZE,
                "exit_reason": exit_reason,
                "exit_fill_basis": "unadjusted_stop_plus_tick_exit",
            }
        if tp_hit:
            return {
                "position": position,
                "exit_price": current_tp_unadjusted,
                "exit_reason": "tp",
                "exit_fill_basis": "unadjusted_tp_limit_exit",
            }
        if not trail_is_active and float(bar["adj_high"]) >= active_initial_sl_adj:
            return {
                "position": position,
                "exit_price": initial_sl_unadjusted + TICK_SIZE,
                "exit_reason": "initial_sl",
                "exit_fill_basis": "unadjusted_stop_plus_tick_exit",
            }

    position.max_price_seen_adj = max(position.max_price_seen_adj, float(bar["adj_high"]))
    position.min_price_seen_adj = min(position.min_price_seen_adj, float(bar["adj_low"]))

    atr_value = bar["atr"]
    if pd.notna(atr_value):
        if position.direction == "long":
            position.highest_favorable_price_adj = max(
                float(position.highest_favorable_price_adj),
                float(bar["adj_high"]),
            )
            new_trail = position.highest_favorable_price_adj - (config.atr_multiplier * float(atr_value))
            position.current_trail_stop_adj = max(position.current_trail_stop_adj, new_trail)
        else:
            position.lowest_favorable_price_adj = min(
                float(position.lowest_favorable_price_adj),
                float(bar["adj_low"]),
            )
            new_trail = position.lowest_favorable_price_adj + (config.atr_multiplier * float(atr_value))
            position.current_trail_stop_adj = min(position.current_trail_stop_adj, new_trail)
    return None


def _compute_trade_excursions(
    position: PositionState,
    exit_bar: pd.Series,
    exit_price: float,
) -> dict[str, float]:
    exit_adj_price = exit_price + float(exit_bar["adj_factor"])
    max_price_seen_adj = max(position.max_price_seen_adj, exit_adj_price)
    min_price_seen_adj = min(position.min_price_seen_adj, exit_adj_price)
    if position.direction == "long":
        mfe_price = max_price_seen_adj - position.entry_adj_price
        mae_price = position.entry_adj_price - min_price_seen_adj
    else:
        mfe_price = position.entry_adj_price - min_price_seen_adj
        mae_price = max_price_seen_adj - position.entry_adj_price
    mfe_price = max(float(mfe_price), 0.0)
    mae_price = max(float(mae_price), 0.0)
    return {
        "mfe_price": mfe_price,
        "mae_price": mae_price,
        "mfe_usd": mfe_price * position.contracts * CONTRACT_MULTIPLIER,
        "mae_usd": mae_price * position.contracts * CONTRACT_MULTIPLIER,
    }


def _close_position(
    position: PositionState,
    exit_bar: pd.Series,
    exit_price: float,
    exit_reason: str,
    exit_fill_basis: str,
    current_equity: float,
) -> tuple[dict[str, Any], float]:
    incremental_gross = _directional_pnl(
        direction=position.direction,
        entry_price=position.current_leg_entry_price,
        exit_price=exit_price,
        contracts=position.contracts,
    )
    excursions = _compute_trade_excursions(position, exit_bar, exit_price)
    incremental_net = incremental_gross - (position.contracts * ROUND_TURN_COST)
    current_equity += incremental_net
    gross_pnl = position.realized_roll_pnl + incremental_gross
    transaction_cost = position.realized_roll_cost + (position.contracts * ROUND_TURN_COST)
    net_pnl = gross_pnl - transaction_cost
    tp_price_unadjusted = position.entry_price + position.sl_distance_price if position.direction == "long" else position.entry_price - position.sl_distance_price
    initial_sl_unadjusted = position.entry_price - position.sl_distance_price if position.direction == "long" else position.entry_price + position.sl_distance_price
    current_trail_unadjusted = position.current_trail_stop_adj - float(exit_bar["adj_factor"])
    trade_record = {
        "trade_id": position.trade_id,
        "direction": position.direction,
        "entry_bar_ts": position.entry_bar_ts,
        "entry_session_id": position.entry_session_id,
        "entry_session_date": position.entry_session_date,
        "entry_price": position.entry_price,
        "contracts": position.contracts,
        "tp_price": tp_price_unadjusted,
        "initial_sl_price": initial_sl_unadjusted,
        "current_trail_stop": current_trail_unadjusted,
        "tp_price_adj": position.tp_adj,
        "initial_sl_price_adj": position.initial_sl_adj,
        "current_trail_stop_adj": position.current_trail_stop_adj,
        "status": "closed",
        "exit_price": exit_price,
        "exit_bar_ts": exit_bar["ts_event"],
        "exit_reason": exit_reason,
        "gross_pnl": gross_pnl,
        "transaction_cost": transaction_cost,
        "net_pnl": net_pnl,
        "entry_fill_basis": position.entry_fill_basis,
        "exit_fill_basis": exit_fill_basis,
        "bars_held": position.bars_held,
        "roll_count": position.roll_count,
        "roll_cost": position.realized_roll_cost,
        "sizing_equity": position.sizing_equity,
        "risk_per_trade": position.risk_per_trade,
        "floor_applied": position.floor_applied,
        "sl_distance_price": position.sl_distance_price,
        "sl_distance_usd": position.sl_distance_usd,
        "current_contract_at_exit": position.current_contract,
        "incremental_gross_pnl": incremental_gross,
        "incremental_net_pnl": incremental_net,
        "mfe_price": excursions["mfe_price"],
        "mae_price": excursions["mae_price"],
        "mfe_usd": excursions["mfe_usd"],
        "mae_usd": excursions["mae_usd"],
    }
    return trade_record, current_equity


def _capture_concurrency_snapshot(ts_event: pd.Timestamp, session_id: int, open_positions: list[PositionState]) -> dict[str, Any]:
    directions = {position.direction for position in open_positions}
    open_contracts = int(sum(position.contracts for position in open_positions))
    return {
        "ts_event": ts_event,
        "session_id": session_id,
        "open_positions": int(len(open_positions)),
        "open_contracts": open_contracts,
        "simultaneous_long_short": directions == {"long", "short"},
    }


def _build_equity_curve(data: pd.DataFrame, event_log: pd.DataFrame, starting_capital: float) -> pd.DataFrame:
    curve = data[
        ["ts_event", "session_id", "session_date", "session_open_bar", "session_close_bar"]
    ].drop_duplicates().sort_values("ts_event")
    if event_log.empty:
        curve["gross_delta"] = 0.0
        curve["net_delta"] = 0.0
    else:
        event_deltas = event_log.groupby("ts_event", as_index=False)[["gross_delta", "net_delta"]].sum()
        curve = curve.merge(event_deltas, on="ts_event", how="left")
        curve[["gross_delta", "net_delta"]] = curve[["gross_delta", "net_delta"]].fillna(0.0)
    curve["gross_equity"] = starting_capital + curve["gross_delta"].cumsum()
    curve["net_equity"] = starting_capital + curve["net_delta"].cumsum()
    curve["gross_running_peak"] = curve["gross_equity"].cummax()
    curve["net_running_peak"] = curve["net_equity"].cummax()
    curve["gross_drawdown_pct"] = (curve["gross_equity"] / curve["gross_running_peak"] - 1.0) * 100.0
    curve["net_drawdown_pct"] = (curve["net_equity"] / curve["net_running_peak"] - 1.0) * 100.0
    return curve.reset_index(drop=True)


def _build_performance_summary(
    trade_log: pd.DataFrame,
    session_equity: pd.DataFrame,
    concurrency_df: pd.DataFrame,
    floor_events_df: pd.DataFrame,
    order_cancellations_df: pd.DataFrame,
    margin_flags_df: pd.DataFrame,
    session_sizing_df: pd.DataFrame,
    skipped_sessions_df: pd.DataFrame,
    config: BacktestConfig,
) -> pd.Series:
    if session_equity.empty:
        return pd.Series(dtype=float)
    net_equity = session_equity["net_equity"]
    gross_equity = session_equity["gross_equity"]
    total_return = (net_equity.iloc[-1] / config.starting_capital) - 1.0
    years = max((session_equity["ts_event"].iloc[-1] - session_equity["ts_event"].iloc[0]).days / 365.25, 1 / 365.25)
    annualized_return = np.nan
    if net_equity.iloc[-1] > 0 and config.starting_capital > 0:
        annualized_return = (net_equity.iloc[-1] / config.starting_capital) ** (1 / years) - 1.0
    session_returns = net_equity.pct_change().dropna()
    rf_session = (1.0 + config.risk_free_rate) ** (1.0 / 252.0) - 1.0
    sharpe = _safe_ratio((session_returns - rf_session).mean(), session_returns.std(ddof=0)) * np.sqrt(252.0)
    downside = session_returns[session_returns < rf_session]
    sortino = _safe_ratio((session_returns - rf_session).mean(), downside.std(ddof=0)) * np.sqrt(252.0)
    max_drawdown_pct = abs(session_equity["net_drawdown_pct"].min())
    calmar = _safe_ratio(annualized_return * 100.0, max_drawdown_pct)
    max_drawdown_duration = _max_drawdown_duration(session_equity["net_drawdown_pct"])

    total_trades = int(trade_log.shape[0])
    wins = trade_log.loc[trade_log["net_pnl"] > 0]
    losses = trade_log.loc[trade_log["net_pnl"] < 0]
    win_rate = 100.0 * wins.shape[0] / total_trades if total_trades else np.nan
    avg_win = wins["net_pnl"].mean() if not wins.empty else np.nan
    avg_loss = losses["net_pnl"].mean() if not losses.empty else np.nan
    profit_factor = _safe_ratio(wins["gross_pnl"].sum(), abs(losses["gross_pnl"].sum()))
    max_wins, max_losses = _streaks(trade_log["net_pnl"])
    avg_bars_long = trade_log.loc[trade_log["direction"] == "long", "bars_held"].mean()
    avg_bars_short = trade_log.loc[trade_log["direction"] == "short", "bars_held"].mean()
    exit_breakdown = trade_log["exit_reason"].value_counts()
    peak_concurrent_positions = int(concurrency_df["open_positions"].max()) if not concurrency_df.empty else 0
    concurrent_long_short_sessions = int(concurrency_df.groupby("session_id")["simultaneous_long_short"].max().sum()) if not concurrency_df.empty else 0
    gross_total_pnl = trade_log["gross_pnl"].sum() if not trade_log.empty else 0.0
    net_total_pnl = trade_log["net_pnl"].sum() if not trade_log.empty else 0.0
    cost_drag = gross_total_pnl - net_total_pnl
    cost_drag_pct = _safe_ratio(cost_drag, gross_total_pnl) * 100.0 if gross_total_pnl != 0 else np.nan
    ambiguous_entry_skips = 0
    carry_position_skips = 0
    prior_session_quality_skips = 0
    margin_cap_applied_sessions = 0
    margin_cap_blocked_sessions = 0
    max_contract_cap_applied_sessions = 0
    if not skipped_sessions_df.empty and "reason" in skipped_sessions_df.columns:
        ambiguous_entry_skips = int(
            skipped_sessions_df["reason"].fillna("").str.startswith("Ambiguous hourly OCO trigger").sum()
        )
        carry_position_skips = int(
            skipped_sessions_df["reason"].fillna("").eq("Open position carried into session; new session orders suppressed.").sum()
        )
        prior_session_quality_skips = int(
            skipped_sessions_df["reason"].fillna("").str.startswith("Previous session failed quality filter:").sum()
        )
    if not margin_flags_df.empty and "flag_type" in margin_flags_df.columns:
        margin_cap_applied_sessions = int(
            margin_flags_df["flag_type"].eq("margin_cap_applied").sum()
        )
        margin_cap_blocked_sessions = int(
            margin_flags_df["flag_type"].eq("margin_cap_blocked").sum()
        )
    if not session_sizing_df.empty and "contract_cap_applied" in session_sizing_df.columns:
        max_contract_cap_applied_sessions = int(session_sizing_df["contract_cap_applied"].sum())
    return pd.Series(
        {
            "terminal_gross_equity": gross_equity.iloc[-1],
            "terminal_net_equity": net_equity.iloc[-1],
            "total_return_pct": total_return * 100.0,
            "annualized_return_pct": annualized_return * 100.0,
            "max_drawdown_pct": max_drawdown_pct,
            "max_drawdown_duration_sessions": max_drawdown_duration,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "total_trades": total_trades,
            "win_rate_pct": win_rate,
            "avg_win_usd": avg_win,
            "avg_loss_usd": avg_loss,
            "profit_factor": profit_factor,
            "max_consecutive_wins": max_wins,
            "max_consecutive_losses": max_losses,
            "avg_bars_held_long": avg_bars_long,
            "avg_bars_held_short": avg_bars_short,
            "exit_tp_count": int(exit_breakdown.get("tp", 0)),
            "exit_trail_stop_count": int(exit_breakdown.get("trail_stop", 0)),
            "exit_initial_sl_count": int(exit_breakdown.get("initial_sl", 0)),
            "exit_gap_stop_count": int(exit_breakdown.get("gap_stop", 0)),
            "peak_concurrent_positions": peak_concurrent_positions,
            "sessions_with_concurrent_long_short": concurrent_long_short_sessions,
            "sizing_floor_events": int(floor_events_df.shape[0]),
            "gross_total_pnl": gross_total_pnl,
            "net_total_pnl": net_total_pnl,
            "total_cost_drag": cost_drag,
            "cost_drag_pct_of_gross": cost_drag_pct,
            "cancelled_orders": int(order_cancellations_df.shape[0]),
            "margin_flag_sessions": int(margin_flags_df.shape[0]),
            "margin_cap_applied_sessions": margin_cap_applied_sessions,
            "margin_cap_blocked_sessions": margin_cap_blocked_sessions,
            "max_contract_cap_applied_sessions": max_contract_cap_applied_sessions,
            "ambiguous_entry_skip_sessions": ambiguous_entry_skips,
            "carry_position_skip_sessions": carry_position_skips,
            "prior_session_quality_skip_sessions": prior_session_quality_skips,
        }
    )


def _build_direction_summary(trade_log: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for direction in ["long", "short"]:
        subset = trade_log.loc[trade_log["direction"] == direction].copy()
        if subset.empty:
            rows.append({"direction": direction})
            continue
        wins = subset.loc[subset["net_pnl"] > 0]
        losses = subset.loc[subset["net_pnl"] < 0]
        rows.append(
            {
                "direction": direction,
                "trades": int(subset.shape[0]),
                "win_rate_pct": 100.0 * wins.shape[0] / subset.shape[0],
                "gross_pnl": subset["gross_pnl"].sum(),
                "net_pnl": subset["net_pnl"].sum(),
                "avg_win_usd": wins["net_pnl"].mean() if not wins.empty else np.nan,
                "avg_loss_usd": losses["net_pnl"].mean() if not losses.empty else np.nan,
                "profit_factor": _safe_ratio(wins["gross_pnl"].sum(), abs(losses["gross_pnl"].sum())),
                "avg_bars_held": subset["bars_held"].mean(),
                "avg_mfe_price": subset["mfe_price"].mean(),
                "avg_mae_price": subset["mae_price"].mean(),
                "avg_mfe_usd": subset["mfe_usd"].mean(),
                "avg_mae_usd": subset["mae_usd"].mean(),
            }
        )
    return pd.DataFrame(rows)


def _build_annual_summary(session_equity: pd.DataFrame, trade_log: pd.DataFrame) -> pd.DataFrame:
    yearly_equity = session_equity.copy()
    yearly_equity["year"] = yearly_equity["ts_event"].dt.year
    equity_summary = yearly_equity.groupby("year").agg(
        start_equity=("net_equity", "first"),
        end_equity=("net_equity", "last"),
    )
    equity_summary["annual_return_pct"] = (equity_summary["end_equity"] / equity_summary["start_equity"] - 1.0) * 100.0
    if trade_log.empty:
        equity_summary["trade_count"] = 0
        equity_summary["win_rate_pct"] = np.nan
        return equity_summary.reset_index()
    trade_counts = trade_log.copy()
    trade_counts["year"] = pd.to_datetime(trade_counts["exit_bar_ts"], utc=True).dt.year
    trade_stats = trade_counts.groupby("year").agg(
        trade_count=("trade_id", "count"),
        win_rate_pct=("net_pnl", lambda values: 100.0 * (values > 0).mean()),
    )
    return equity_summary.merge(trade_stats, on="year", how="left").reset_index()


def _build_exit_breakdown(trade_log: pd.DataFrame) -> pd.DataFrame:
    if trade_log.empty:
        return pd.DataFrame(columns=["exit_reason", "count", "pct"])
    counts = trade_log["exit_reason"].value_counts().rename_axis("exit_reason").reset_index(name="count")
    counts["pct"] = 100.0 * counts["count"] / counts["count"].sum()
    return counts


def _finalize_backtest_results(
    prepared: PreparedData,
    config: BacktestConfig,
    data: pd.DataFrame,
    sessions: pd.DataFrame,
    trade_records: list[dict[str, Any]],
    roll_events: list[dict[str, Any]],
    order_cancellations: list[dict[str, Any]],
    floor_events: list[dict[str, Any]],
    margin_flags: list[dict[str, Any]],
    skipped_sessions: list[dict[str, Any]],
    event_log: list[dict[str, Any]],
    concurrency_log: list[dict[str, Any]],
    session_sizing_log: list[dict[str, Any]],
    session_concurrent_long_short: set[int],
) -> BacktestResult:
    trade_log = pd.DataFrame(trade_records)
    roll_events_df = pd.DataFrame(roll_events)
    order_cancellations_df = pd.DataFrame(order_cancellations)
    floor_events_df = pd.DataFrame(floor_events)
    margin_flags_df = pd.DataFrame(margin_flags)
    skipped_sessions_df = pd.DataFrame(skipped_sessions)
    event_log_df = pd.DataFrame(event_log)
    concurrency_df = pd.DataFrame(concurrency_log)
    session_sizing_df = pd.DataFrame(session_sizing_log)
    equity_curve = _build_equity_curve(data, event_log_df, config.starting_capital)
    session_equity = equity_curve.loc[equity_curve["session_close_bar"]].copy().reset_index(drop=True)
    performance_summary = _build_performance_summary(
        trade_log=trade_log,
        session_equity=session_equity,
        concurrency_df=concurrency_df,
        floor_events_df=floor_events_df,
        order_cancellations_df=order_cancellations_df,
        margin_flags_df=margin_flags_df,
        session_sizing_df=session_sizing_df,
        skipped_sessions_df=skipped_sessions_df,
        config=config,
    )
    direction_summary = _build_direction_summary(trade_log)
    annual_summary = _build_annual_summary(session_equity, trade_log)
    exit_breakdown = _build_exit_breakdown(trade_log)
    diagnostics = {
        "concurrency_log": concurrency_df,
        "session_sizing": session_sizing_df,
        "sessions_with_concurrent_long_short": sorted(session_concurrent_long_short),
        "peak_concurrent_positions": int(concurrency_df["open_positions"].max()) if not concurrency_df.empty else 0,
        "peak_concurrent_contracts": int(concurrency_df["open_contracts"].max()) if not concurrency_df.empty else 0,
    }
    validation_results = run_validations(
        prepared=prepared,
        result_frames={
            "trade_log": trade_log,
            "roll_events": roll_events_df,
            "floor_events": floor_events_df,
            "event_log": event_log_df,
            "session_sizing": session_sizing_df,
            "skipped_sessions": skipped_sessions_df,
            "equity_curve": equity_curve,
        },
        config=config,
    )
    return BacktestResult(
        config=config,
        data=data,
        session_table=sessions,
        trade_log=trade_log,
        roll_events=roll_events_df,
        order_cancellations=order_cancellations_df,
        floor_events=floor_events_df,
        margin_flags=margin_flags_df,
        skipped_sessions=skipped_sessions_df,
        event_log=event_log_df,
        equity_curve=equity_curve,
        session_equity=session_equity,
        performance_summary=performance_summary,
        direction_summary=direction_summary,
        annual_summary=annual_summary,
        exit_breakdown=exit_breakdown,
        validation_results=validation_results,
        diagnostics=diagnostics,
    )


def run_sensitivity_analysis(prepared: PreparedData, config: BacktestConfig) -> dict[str, pd.DataFrame]:
    atr_cache: dict[int, PreparedData] = {}
    grid_rows: list[dict[str, Any]] = []
    for atr_period in config.sensitivity_atr_periods:
        atr_prepared = atr_cache.setdefault(atr_period, with_atr_period(prepared, atr_period))
        for tp_sl_mode in config.sensitivity_tp_sl_modes:
            k_values = config.sensitivity_ks if tp_sl_mode == "prior_day_range" else (config.k,)
            for atr_multiplier in config.sensitivity_atr_multipliers:
                mode_rows: list[dict[str, Any]] = []
                for k in k_values:
                    run_config = replace(
                        config,
                        atr_period=atr_period,
                        tp_sl_mode=tp_sl_mode,
                        atr_multiplier=atr_multiplier,
                        k=k,
                    )
                    result = run_backtest(atr_prepared, run_config)
                    metrics = result.performance_summary
                    mode_rows.append(
                        {
                            "atr_period": atr_period,
                            "tp_sl_mode": tp_sl_mode,
                            "atr_multiplier": atr_multiplier,
                            "k": k,
                            "win_rate": metrics["win_rate_pct"],
                            "profit_factor": metrics["profit_factor"],
                            "sharpe_ratio": metrics["sharpe_ratio"],
                            "max_drawdown_pct": metrics["max_drawdown_pct"],
                            "terminal_net_equity": metrics["terminal_net_equity"],
                            "total_return_pct": metrics["total_return_pct"],
                            "total_trades": metrics["total_trades"],
                        }
                    )
                if tp_sl_mode == "prior_day_range":
                    grid_rows.extend(mode_rows)
                else:
                    replicated_row = mode_rows[0]
                    for k in config.sensitivity_ks:
                        grid_rows.append({**replicated_row, "k": k})
    sizing_rows: list[dict[str, Any]] = []
    default_atr_prepared = atr_cache.setdefault(config.atr_period, with_atr_period(prepared, config.atr_period))
    for risk_fraction in config.sensitivity_risk_fractions:
        run_config = replace(config, risk_fraction=risk_fraction)
        result = run_backtest(default_atr_prepared, run_config)
        metrics = result.performance_summary
        sizing_rows.append(
            {
                "risk_fraction": risk_fraction,
                "terminal_net_equity": metrics["terminal_net_equity"],
                "max_drawdown_pct": metrics["max_drawdown_pct"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "total_return_pct": metrics["total_return_pct"],
                "total_trades": metrics["total_trades"],
            }
        )
    contract_cap_rows: list[dict[str, Any]] = []
    for max_contracts in config.sensitivity_max_contracts:
        run_config = replace(config, max_contracts=max_contracts)
        result = run_backtest(default_atr_prepared, run_config)
        metrics = result.performance_summary
        contract_cap_rows.append(
            {
                "max_contracts": max_contracts,
                "terminal_net_equity": metrics["terminal_net_equity"],
                "max_drawdown_pct": metrics["max_drawdown_pct"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "total_return_pct": metrics["total_return_pct"],
                "total_trades": metrics["total_trades"],
                "max_contract_cap_applied_sessions": metrics["max_contract_cap_applied_sessions"],
            }
        )
    return {
        "primary_grid": pd.DataFrame(grid_rows),
        "sizing_grid": pd.DataFrame(sizing_rows),
        "contract_cap_grid": pd.DataFrame(contract_cap_rows),
    }


def _subset_prepared_by_session_date_range(
    prepared: PreparedData,
    start_session_date: pd.Timestamp,
    end_session_date: pd.Timestamp,
) -> PreparedData:
    session_mask = prepared.session_table["session_date"].ge(start_session_date) & prepared.session_table[
        "session_date"
    ].lt(end_session_date)
    subset_sessions = prepared.session_table.loc[session_mask].copy()
    subset_ids = subset_sessions["session_id"].tolist()
    subset_data = prepared.continuous_data.loc[prepared.continuous_data["session_id"].isin(subset_ids)].copy()
    return PreparedData(
        clean_data=prepared.clean_data,
        session_table=subset_sessions.reset_index(drop=True),
        dominant_table=prepared.dominant_table,
        continuous_data=subset_data.reset_index(drop=True),
        atr_source_data=prepared.atr_source_data,
        audit=prepared.audit,
        notes=prepared.notes,
    )


def _build_walk_forward_windows(config: BacktestConfig, session_table: pd.DataFrame) -> pd.DataFrame:
    if session_table.empty:
        return pd.DataFrame()
    unique_dates = (
        session_table["session_date"].drop_duplicates().sort_values().reset_index(drop=True)
    )
    first_session_date = pd.Timestamp(unique_dates.iloc[0])
    last_session_date_exclusive = pd.Timestamp(unique_dates.iloc[-1]) + pd.Timedelta(days=1)
    windows: list[dict[str, Any]] = []
    fold_id = 1
    train_start = first_session_date
    while True:
        train_end = train_start + pd.DateOffset(years=config.walk_forward_train_years)
        test_start = train_end
        test_end = test_start + pd.DateOffset(years=config.walk_forward_test_years)
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
        train_start = train_start + pd.DateOffset(months=config.walk_forward_step_months)
    return pd.DataFrame(windows)


def _walk_forward_candidate_configs(config: BacktestConfig) -> list[BacktestConfig]:
    candidates: list[BacktestConfig] = []
    for atr_period in config.walk_forward_atr_periods:
        for tp_sl_mode in config.walk_forward_tp_sl_modes:
            k_values = config.walk_forward_ks if tp_sl_mode == "prior_day_range" else (config.k,)
            for atr_multiplier in config.walk_forward_atr_multipliers:
                for k in k_values:
                    candidates.append(
                        replace(
                            config,
                            atr_period=atr_period,
                            tp_sl_mode=tp_sl_mode,
                            atr_multiplier=atr_multiplier,
                            k=k,
                        )
                    )
    return candidates


def _walk_forward_objective_value(metrics: pd.Series, objective: str) -> float:
    value = metrics.get(objective, np.nan)
    return float(value) if pd.notna(value) else float("-inf")


def _walk_forward_selection_key(metrics: pd.Series, objective: str) -> tuple[float, float, float, float]:
    objective_value = _walk_forward_objective_value(metrics, objective)
    total_return = metrics.get("total_return_pct", np.nan)
    sharpe_ratio = metrics.get("sharpe_ratio", np.nan)
    max_drawdown = metrics.get("max_drawdown_pct", np.nan)
    total_trades = metrics.get("total_trades", np.nan)
    return (
        objective_value,
        float(total_return) if pd.notna(total_return) else float("-inf"),
        float(sharpe_ratio) if pd.notna(sharpe_ratio) else float("-inf"),
        -float(max_drawdown) if pd.notna(max_drawdown) else float("-inf"),
        float(total_trades) if pd.notna(total_trades) else float("-inf"),
    )


def _build_walk_forward_parameter_stability(fold_summary: pd.DataFrame) -> pd.DataFrame:
    if fold_summary.empty:
        return pd.DataFrame(columns=["parameter", "value", "selected_folds", "selected_pct"])
    rows: list[dict[str, Any]] = []
    total_folds = int(fold_summary["fold_id"].nunique())
    for parameter in ["atr_period", "tp_sl_mode", "atr_multiplier", "k"]:
        counts = fold_summary[parameter].value_counts(dropna=False).sort_index()
        for value, count in counts.items():
            rows.append(
                {
                    "parameter": parameter,
                    "value": value,
                    "selected_folds": int(count),
                    "selected_pct": (100.0 * int(count) / total_folds) if total_folds else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _stitch_walk_forward_oos_equity(
    oos_equity_frames: list[pd.DataFrame],
    starting_capital: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not oos_equity_frames:
        empty_curve = pd.DataFrame(
            columns=[
                "ts_event",
                "session_id",
                "session_date",
                "session_open_bar",
                "session_close_bar",
                "gross_delta",
                "net_delta",
                "fold_id",
                "selected_atr_period",
                "selected_tp_sl_mode",
                "selected_atr_multiplier",
                "selected_k",
                "gross_equity",
                "net_equity",
                "gross_running_peak",
                "net_running_peak",
                "gross_drawdown_pct",
                "net_drawdown_pct",
            ]
        )
        return empty_curve, empty_curve.loc[empty_curve.get("session_close_bar", pd.Series(dtype=bool))].copy()
    stitched = pd.concat(oos_equity_frames, ignore_index=True).sort_values(["ts_event", "fold_id"]).reset_index(drop=True)
    if stitched["ts_event"].duplicated().any():
        raise ValueError("Walk-forward OOS windows overlap; stitched OOS equity cannot contain duplicate timestamps.")
    stitched["gross_equity"] = starting_capital + stitched["gross_delta"].cumsum()
    stitched["net_equity"] = starting_capital + stitched["net_delta"].cumsum()
    stitched["gross_running_peak"] = stitched["gross_equity"].cummax()
    stitched["net_running_peak"] = stitched["net_equity"].cummax()
    stitched["gross_drawdown_pct"] = (stitched["gross_equity"] / stitched["gross_running_peak"] - 1.0) * 100.0
    stitched["net_drawdown_pct"] = (stitched["net_equity"] / stitched["net_running_peak"] - 1.0) * 100.0
    stitched_session_equity = stitched.loc[stitched["session_close_bar"]].copy().reset_index(drop=True)
    return stitched.reset_index(drop=True), stitched_session_equity


def run_walk_forward_analysis(prepared: PreparedData, config: BacktestConfig) -> WalkForwardResult:
    _validate_config(config)
    fold_windows = _build_walk_forward_windows(config, prepared.session_table)
    if fold_windows.empty:
        raise ValueError("No complete walk-forward folds are available for the configured train/test windows.")

    candidate_configs = _walk_forward_candidate_configs(config)
    atr_prepared_cache = {
        atr_period: with_atr_period(prepared, atr_period)
        for atr_period in sorted({candidate.atr_period for candidate in candidate_configs})
    }
    subset_cache: dict[tuple[int, str, int], PreparedData] = {}
    optimization_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    oos_equity_frames: list[pd.DataFrame] = []
    oos_trade_frames: list[pd.DataFrame] = []

    for fold in fold_windows.to_dict("records"):
        fold_id = int(fold["fold_id"])
        best_candidate: BacktestConfig | None = None
        best_train_result: BacktestResult | None = None
        best_selection_key: tuple[float, float, float, float] | None = None

        for candidate in candidate_configs:
            train_cache_key = (fold_id, "train", candidate.atr_period)
            if train_cache_key not in subset_cache:
                subset_cache[train_cache_key] = _subset_prepared_by_session_date_range(
                    atr_prepared_cache[candidate.atr_period],
                    fold["train_start"],
                    fold["train_end"],
                )
            train_prepared = subset_cache[train_cache_key]
            train_result = run_backtest(train_prepared, candidate)
            train_metrics = train_result.performance_summary
            train_objective_value = _walk_forward_objective_value(train_metrics, config.walk_forward_objective)
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
                    "objective": config.walk_forward_objective,
                    "train_objective_value": train_objective_value,
                    "train_sharpe_ratio": train_metrics.get("sharpe_ratio", np.nan),
                    "train_total_return_pct": train_metrics.get("total_return_pct", np.nan),
                    "train_max_drawdown_pct": train_metrics.get("max_drawdown_pct", np.nan),
                    "train_total_trades": train_metrics.get("total_trades", np.nan),
                }
            )
            selection_key = _walk_forward_selection_key(train_metrics, config.walk_forward_objective)
            if best_selection_key is None or selection_key > best_selection_key:
                best_selection_key = selection_key
                best_candidate = candidate
                best_train_result = train_result

        if best_candidate is None or best_train_result is None:
            raise ValueError(f"Walk-forward fold {fold_id} produced no candidate results.")

        test_cache_key = (fold_id, "test", best_candidate.atr_period)
        if test_cache_key not in subset_cache:
            subset_cache[test_cache_key] = _subset_prepared_by_session_date_range(
                atr_prepared_cache[best_candidate.atr_period],
                fold["test_start"],
                fold["test_end"],
            )
        test_prepared = subset_cache[test_cache_key]
        oos_result = run_backtest(test_prepared, best_candidate)
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
                "objective": config.walk_forward_objective,
                "selected_atr_period": best_candidate.atr_period,
                "selected_tp_sl_mode": best_candidate.tp_sl_mode,
                "selected_atr_multiplier": best_candidate.atr_multiplier,
                "selected_k": best_candidate.k,
                "train_objective_value": _walk_forward_objective_value(train_metrics, config.walk_forward_objective),
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
    parameter_stability = _build_walk_forward_parameter_stability(
        fold_summary.rename(
            columns={
                "selected_atr_period": "atr_period",
                "selected_tp_sl_mode": "tp_sl_mode",
                "selected_atr_multiplier": "atr_multiplier",
                "selected_k": "k",
            }
        )
    )
    oos_equity_curve, oos_session_equity = _stitch_walk_forward_oos_equity(
        oos_equity_frames=oos_equity_frames,
        starting_capital=config.starting_capital,
    )
    oos_trade_log = pd.concat(oos_trade_frames, ignore_index=True) if oos_trade_frames else pd.DataFrame()
    diagnostics = {
        "fold_windows": fold_windows,
        "candidate_count": len(candidate_configs),
        "objective": config.walk_forward_objective,
    }
    return WalkForwardResult(
        config=config,
        fold_summary=fold_summary,
        optimization_results=optimization_results,
        parameter_stability=parameter_stability,
        oos_equity_curve=oos_equity_curve,
        oos_session_equity=oos_session_equity,
        oos_trade_log=oos_trade_log,
        diagnostics=diagnostics,
    )


def run_validations(
    prepared: PreparedData,
    result_frames: dict[str, pd.DataFrame],
    config: BacktestConfig,
) -> pd.DataFrame:
    data = prepared.continuous_data
    trade_log = result_frames["trade_log"]
    roll_events = result_frames["roll_events"]
    floor_events = result_frames["floor_events"]
    event_log = result_frames["event_log"]
    session_sizing = result_frames["session_sizing"]
    skipped_sessions = result_frames["skipped_sessions"]
    validations: list[dict[str, Any]] = []

    open_bars = data.loc[data["session_open_bar"], ["session_id", "prev_session_high", "prev_session_low"]].copy()
    open_bars = open_bars.iloc[1:].copy()
    session_stats = (
        data.groupby("session_id", as_index=False)
        .agg(session_high=("adj_high", "max"), session_low=("adj_low", "min"))
        .sort_values("session_id")
    )
    expected_prev = session_stats.assign(
        expected_prev_session_high=session_stats["session_high"].shift(),
        expected_prev_session_low=session_stats["session_low"].shift(),
    )[["session_id", "expected_prev_session_high", "expected_prev_session_low"]]
    temporal_check = open_bars.merge(expected_prev, on="session_id", how="left")
    signal_temporal_ok = bool(
        np.allclose(
            temporal_check["prev_session_high"].to_numpy(dtype=float),
            temporal_check["expected_prev_session_high"].to_numpy(dtype=float),
            equal_nan=True,
        )
        and np.allclose(
            temporal_check["prev_session_low"].to_numpy(dtype=float),
            temporal_check["expected_prev_session_low"].to_numpy(dtype=float),
            equal_nan=True,
        )
    )
    validations.append(
        {
            "test": "Signal temporal integrity",
            "status": "pass" if signal_temporal_ok else "fail",
            "detail": "Session-open breakout levels equal the previous session adjusted high/low.",
        }
    )

    atr_expected = data["atr_unshifted"].shift(1)
    atr_temporal_ok = bool(
        np.allclose(
            data["atr"].iloc[1:].to_numpy(dtype=float),
            atr_expected.iloc[1:].to_numpy(dtype=float),
            equal_nan=True,
        )
    )
    validations.append(
        {
            "test": "ATR temporal integrity",
            "status": "pass" if atr_temporal_ok else "fail",
            "detail": "Shifted ATR equals the one-bar-lagged Wilder ATR series.",
        }
    )

    atr_warmup_ok = True
    if not data.empty:
        recomputed = with_atr_period(prepared, config.atr_period).continuous_data
        atr_warmup_ok = bool(
            data["ts_event"].equals(recomputed["ts_event"])
            and np.allclose(
                data["atr"].to_numpy(dtype=float),
                recomputed["atr"].to_numpy(dtype=float),
                equal_nan=True,
            )
        )
    validations.append(
        {
            "test": "ATR warm-up consistency",
            "status": "pass" if atr_warmup_ok else "fail",
            "detail": "ATR recomputation for the configured period reuses the full pre-sample source history.",
        }
    )

    fill_basis_ok = True
    if not trade_log.empty:
        fill_basis_ok = bool(
            trade_log["entry_fill_basis"].str.startswith("unadjusted").all()
            and trade_log["exit_fill_basis"].str.startswith("unadjusted").all()
        )
    validations.append(
        {
            "test": "Fill price series provenance",
            "status": "pass" if fill_basis_ok else "fail",
            "detail": "All fills are explicitly tagged as sourced from the unadjusted price series.",
        }
    )

    roll_continuity_ok = True
    if not roll_events.empty:
        roll_continuity_ok = bool((roll_events["roll_continuity_error"].abs() < TICK_SIZE).all())
    validations.append(
        {
            "test": "Roll continuity",
            "status": "pass" if roll_continuity_ok else "fail",
            "detail": "Adjusted close continuity at each roll is within one tick.",
        }
    )

    cost_accounting_ok = True
    if not trade_log.empty:
        expected_cost = ((1 + trade_log["roll_count"]) * trade_log["contracts"] * ROUND_TURN_COST).sum()
        cost_accounting_ok = np.isclose(trade_log["transaction_cost"].sum(), expected_cost)
    validations.append(
        {
            "test": "Cost accounting",
            "status": "pass" if cost_accounting_ok else "fail",
            "detail": "Trade transaction costs equal base round-turn cost plus approved rollover costs.",
        }
    )

    equity_entry_ok = True
    if not event_log.empty:
        entry_events = event_log.loc[event_log["event_type"] == "entry"]
        equity_entry_ok = bool(entry_events.empty or (entry_events["net_delta"] == 0).all())
    sizing_consistency_ok = True
    if not trade_log.empty and not session_sizing.empty:
        sizing_reference = session_sizing[["session_id", "session_open_equity"]].drop_duplicates()
        merged = trade_log.merge(sizing_reference, left_on="entry_session_id", right_on="session_id", how="left")
        sizing_consistency_ok = bool(np.allclose(merged["sizing_equity"], merged["session_open_equity"]))
    validations.append(
        {
            "test": "Equity monotonicity check",
            "status": "pass" if (equity_entry_ok and sizing_consistency_ok) else "fail",
            "detail": "Entries do not change equity, and each trade uses the session-open sizing equity.",
        }
    )

    margin_cap_ok = True
    if not trade_log.empty:
        margin_budget = trade_log["sizing_equity"] * config.max_margin_fraction
        required_margin = trade_log["contracts"] * config.initial_margin_per_contract
        margin_cap_ok = bool((required_margin <= margin_budget + 1e-9).all())
    validations.append(
        {
            "test": "Margin cap enforcement",
            "status": "pass" if margin_cap_ok else "fail",
            "detail": "Each trade's initial margin fits within the configured margin-fraction cap.",
        }
    )

    max_contract_cap_ok = True
    if not trade_log.empty:
        max_contract_cap_ok = bool((trade_log["contracts"] <= config.max_contracts).all())
    validations.append(
        {
            "test": "Max contract cap enforcement",
            "status": "pass" if max_contract_cap_ok else "fail",
            "detail": "Each trade's contracts do not exceed the configured hard contract cap.",
        }
    )

    mae_mfe_ok = True
    if not trade_log.empty:
        mae_mfe_columns = ["mfe_price", "mae_price", "mfe_usd", "mae_usd"]
        mae_mfe_ok = bool(
            set(mae_mfe_columns).issubset(trade_log.columns)
            and trade_log[mae_mfe_columns].notna().all().all()
            and (trade_log[mae_mfe_columns] >= 0.0).all().all()
        )
    validations.append(
        {
            "test": "MAE/MFE diagnostics",
            "status": "pass" if mae_mfe_ok else "fail",
            "detail": "Trade-level MAE/MFE diagnostics are populated and non-negative.",
        }
    )

    floor_logging_ok = True
    if not trade_log.empty:
        expected_floor_ids = set(trade_log.loc[trade_log["floor_applied"], "trade_id"].tolist())
        logged_floor_ids = set(floor_events["trade_id"].tolist()) if not floor_events.empty else set()
        floor_logging_ok = expected_floor_ids == logged_floor_ids
    validations.append(
        {
            "test": "Minimum floor logging",
            "status": "pass" if floor_logging_ok else "fail",
            "detail": "Every trade that used the one-contract floor appears in the floor-event log.",
        }
    )

    oco_same_bar_ok = True
    if not trade_log.empty:
        same_bar_direction_counts = trade_log.groupby(["entry_session_id", "entry_bar_ts"])["direction"].nunique()
        oco_same_bar_ok = bool((same_bar_direction_counts <= 1).all())
    validations.append(
        {
            "test": "OCO same-bar enforcement",
            "status": "pass" if oco_same_bar_ok else "fail",
            "detail": "No session bar creates simultaneous long and short entries under the hourly OCO rule.",
        }
    )

    carry_position_skip_ok = True
    if not skipped_sessions.empty and not trade_log.empty and "reason" in skipped_sessions.columns:
        carry_skip_sessions = set(
            skipped_sessions.loc[
                skipped_sessions["reason"].eq("Open position carried into session; new session orders suppressed."),
                "session_id",
            ].tolist()
        )
        carry_position_skip_ok = bool(
            trade_log.loc[trade_log["entry_session_id"].isin(carry_skip_sessions)].empty
        )
    validations.append(
        {
            "test": "Carry-position session suppression",
            "status": "pass" if carry_position_skip_ok else "fail",
            "detail": "Sessions skipped because a position was already open do not create new entries.",
        }
    )

    prior_session_quality_skip_ok = True
    if not skipped_sessions.empty and not trade_log.empty and "reason" in skipped_sessions.columns:
        prior_quality_skip_sessions = set(
            skipped_sessions.loc[
                skipped_sessions["reason"].fillna("").str.startswith("Previous session failed quality filter:"),
                "session_id",
            ].tolist()
        )
        prior_session_quality_skip_ok = bool(
            trade_log.loc[trade_log["entry_session_id"].isin(prior_quality_skip_sessions)].empty
        )
    validations.append(
        {
            "test": "Prior-session quality suppression",
            "status": "pass" if prior_session_quality_skip_ok else "fail",
            "detail": "Sessions skipped because the previous session was incomplete or abnormal do not create new entries.",
        }
    )
    return pd.DataFrame(validations)


def plot_equity_curve(result: BacktestResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(result.equity_curve["ts_event"], result.equity_curve["gross_equity"], label="Gross equity")
    ax.plot(result.equity_curve["ts_event"], result.equity_curve["net_equity"], label="Net equity")
    if not result.roll_events.empty:
        for roll_ts in pd.to_datetime(result.roll_events["roll_ts"], utc=True):
            ax.axvline(roll_ts, color="grey", alpha=0.15, linewidth=0.8)
    ax.set_yscale("log")
    ax.set_title("Equity Curve")
    ax.set_ylabel("Equity")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def plot_drawdown_curve(result: BacktestResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(result.equity_curve["ts_event"], result.equity_curve["net_drawdown_pct"], color="firebrick")
    ax.set_title("Net Drawdown")
    ax.set_ylabel("Drawdown %")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def plot_monthly_returns_heatmap(result: BacktestResult) -> plt.Figure:
    session_equity = result.session_equity.copy()
    monthly = (
        session_equity.set_index("ts_event")["net_equity"].resample("M").last().pct_change().dropna() * 100.0
    )
    monthly = monthly.to_frame("return_pct")
    monthly["year"] = monthly.index.year
    monthly["month"] = monthly.index.month
    heatmap = monthly.pivot(index="year", columns="month", values="return_pct").sort_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(heatmap.values, aspect="auto", cmap="RdYlGn", interpolation="nearest")
    ax.set_title("Monthly Net Returns Heatmap (%)")
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
    ax.set_yticks(np.arange(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index.tolist())
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def plot_trade_pnl_distribution(result: BacktestResult) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)
    trade_log = result.trade_log.copy()
    series_to_plot = [
        ("long", "gross_pnl", "Long Gross P&L"),
        ("short", "gross_pnl", "Short Gross P&L"),
        ("long", "net_pnl", "Long Net P&L"),
        ("short", "net_pnl", "Short Net P&L"),
    ]
    for axis, (direction, column, title) in zip(axes.flat, series_to_plot):
        values = trade_log.loc[trade_log["direction"] == direction, column]
        axis.hist(values, bins=30, color="steelblue", alpha=0.8)
        axis.set_title(title)
        axis.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def plot_exit_reason_breakdown(result: BacktestResult) -> plt.Figure:
    exit_counts = result.trade_log["exit_reason"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    exit_counts.plot(kind="bar", ax=ax, color="slateblue")
    ax.set_title("Exit Reason Breakdown")
    ax.set_ylabel("Trades")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def plot_heatmap_grid(
    primary_grid: pd.DataFrame,
    metric: str,
    atr_period: int,
    tp_sl_modes: tuple[str, ...] = VALID_TP_SL_MODES,
) -> plt.Figure:
    subset = primary_grid.loc[primary_grid["atr_period"] == atr_period].copy()
    fig, axes = plt.subplots(1, len(tp_sl_modes), figsize=(16, 4), sharey=True)
    if len(tp_sl_modes) == 1:
        axes = [axes]
    for axis, mode in zip(axes, tp_sl_modes):
        mode_grid = subset.loc[subset["tp_sl_mode"] == mode]
        pivot = mode_grid.pivot(index="atr_multiplier", columns="k", values=metric).sort_index()
        im = axis.imshow(pivot.values, aspect="auto", cmap="viridis", interpolation="nearest")
        axis.set_title(f"{mode}\n{metric}")
        axis.set_xticks(np.arange(len(pivot.columns)))
        axis.set_xticklabels([f"{value:.2f}" for value in pivot.columns])
        axis.set_yticks(np.arange(len(pivot.index)))
        axis.set_yticklabels([f"{value:.2f}" for value in pivot.index])
        axis.set_xlabel("k")
        axis.set_ylabel("ATR multiplier")
        fig.colorbar(im, ax=axis, shrink=0.8)
    fig.tight_layout()
    return fig


def plot_trail_sensitivity_curve(primary_grid: pd.DataFrame, config: BacktestConfig) -> plt.Figure:
    subset = primary_grid.loc[
        (primary_grid["atr_period"] == config.atr_period)
        & (primary_grid["tp_sl_mode"] == config.tp_sl_mode)
        & (primary_grid["k"] == config.k)
    ].sort_values("atr_multiplier")
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()
    ax1.plot(subset["atr_multiplier"], subset["win_rate"], marker="o", label="Win rate %")
    ax2.plot(subset["atr_multiplier"], subset["sharpe_ratio"], marker="s", color="darkorange", label="Sharpe")
    ax1.set_title("Trail Sensitivity")
    ax1.set_xlabel("ATR multiplier")
    ax1.set_ylabel("Win rate %")
    ax2.set_ylabel("Sharpe ratio")
    ax1.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def plot_sizing_sensitivity_curve(sizing_grid: pd.DataFrame) -> plt.Figure:
    subset = sizing_grid.sort_values("risk_fraction")
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()
    ax1.plot(subset["risk_fraction"], subset["terminal_net_equity"], marker="o", label="Terminal net equity")
    ax2.plot(subset["risk_fraction"], subset["max_drawdown_pct"], marker="s", color="firebrick", label="Max DD %")
    ax1.set_title("Sizing Sensitivity")
    ax1.set_xlabel("Risk fraction")
    ax1.set_ylabel("Terminal net equity")
    ax2.set_ylabel("Max drawdown %")
    ax1.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def plot_max_contracts_sensitivity_curve(contract_cap_grid: pd.DataFrame) -> plt.Figure:
    subset = contract_cap_grid.sort_values("max_contracts")
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()
    ax1.plot(subset["max_contracts"], subset["terminal_net_equity"], marker="o", label="Terminal net equity")
    ax2.plot(subset["max_contracts"], subset["max_drawdown_pct"], marker="s", color="firebrick", label="Max DD %")
    ax1.set_title("Max Contracts Sensitivity")
    ax1.set_xlabel("Max contracts")
    ax1.set_ylabel("Terminal net equity")
    ax2.set_ylabel("Max drawdown %")
    ax1.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig
