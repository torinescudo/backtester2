from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from bt_config import (
    COOLDOWN_HOURS, INITIAL_EQUITY, MAX_ALLOCATION, MAX_NEW_TRADES_PER_DAY,
    MAX_OPEN_POSITIONS, RISK_PER_TRADE, TEST_END, TEST_START, CandidateTrade,
)


def select_sequential(trades: list[CandidateTrade]) -> list[CandidateTrade]:
    selected: list[CandidateTrade] = []
    last_entry: pd.Timestamp | None = None
    last_exit: pd.Timestamp | None = None
    for trade in sorted(trades, key=lambda item: (item.entry_time, -item.score)):
        if last_exit is not None and trade.entry_time < last_exit:
            continue
        if last_entry is not None and trade.entry_time < last_entry + pd.Timedelta(hours=COOLDOWN_HOURS):
            continue
        selected.append(trade)
        last_entry, last_exit = trade.entry_time, trade.exit_time
    return selected


def daily_sharpe(events: list[dict[str, Any]]) -> float:
    if not events:
        return float("nan")
    series = pd.Series(
        [event["equity"] for event in events],
        index=pd.DatetimeIndex([event["time"] for event in events]), dtype=float,
    )
    series = series[~series.index.duplicated(keep="last")].sort_index()
    days = pd.date_range(TEST_START.normalize(), TEST_END.normalize(), freq="1D", tz="UTC")
    daily = series.reindex(series.index.union(days)).sort_index().ffill().reindex(days).fillna(INITIAL_EQUITY)
    returns = daily.pct_change().dropna()
    std = returns.std(ddof=1)
    return float(math.sqrt(365.0) * returns.mean() / std) if len(returns) > 1 and std > 0 else float("nan")


def max_drawdown(equities: list[float]) -> float:
    if not equities:
        return 0.0
    values = np.asarray(equities, dtype=float)
    peaks = np.maximum.accumulate(values)
    return float((values / peaks - 1.0).min())


def simulate_single_asset(trades: list[CandidateTrade]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    equity = INITIAL_EQUITY
    executed: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = [{"time": TEST_START, "equity": equity}]
    for trade in select_sequential(trades):
        size = min(
            equity * RISK_PER_TRADE / trade.price_risk_per_unit,
            equity * MAX_ALLOCATION / trade.entry_price,
        )
        if size <= 0:
            continue
        pnl = size * trade.net_pnl_per_unit
        record = trade.to_record()
        record.update(
            size=size,
            notional=size * trade.entry_price,
            pnl_net=pnl,
            fees=size * trade.fees_per_unit,
            equity_before=equity,
            equity_after=equity + pnl,
        )
        equity += pnl
        executed.append(record)
        events.append({"time": trade.exit_time, "equity": equity})
    return executed, events


def summarize(executed: list[dict[str, Any]], events: list[dict[str, Any]], buy_hold: float | None = None) -> dict[str, Any]:
    if not executed:
        return {
            "trades": 0, "wins": 0, "win_rate": float("nan"),
            "profit_factor": float("nan"), "expectancy_r": float("nan"),
            "median_r": float("nan"), "gross_r_sum": 0.0, "net_r_sum": 0.0,
            "return_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe_daily": float("nan"),
            "avg_duration_hours": float("nan"), "fees_total": 0.0,
            "buy_hold_return_pct": buy_hold * 100.0 if buy_hold is not None else float("nan"),
        }
    pnls = np.asarray([float(row["pnl_net"]) for row in executed])
    net_r = np.asarray([float(row["net_r"]) for row in executed])
    gross_r = np.asarray([float(row["gross_r"]) for row in executed])
    fees = np.asarray([float(row["fees"]) for row in executed])
    durations = np.asarray([float(row["duration_hours"]) for row in executed])
    gains = pnls[pnls > 0].sum()
    losses = -pnls[pnls < 0].sum()
    equities = [float(event["equity"]) for event in events]
    final_equity = equities[-1] if equities else INITIAL_EQUITY
    return {
        "trades": int(len(executed)),
        "wins": int((pnls > 0).sum()),
        "win_rate": float((pnls > 0).mean()),
        "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
        "expectancy_r": float(net_r.mean()),
        "median_r": float(np.median(net_r)),
        "gross_r_sum": float(gross_r.sum()),
        "net_r_sum": float(net_r.sum()),
        "return_pct": float((final_equity / INITIAL_EQUITY - 1.0) * 100.0),
        "max_drawdown_pct": float(max_drawdown(equities) * 100.0),
        "sharpe_daily": daily_sharpe(events),
        "avg_duration_hours": float(durations.mean()),
        "fees_total": float(fees.sum()),
        "buy_hold_return_pct": buy_hold * 100.0 if buy_hold is not None else float("nan"),
    }


def simulate_portfolio(trades: list[CandidateTrade]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    equity = INITIAL_EQUITY
    open_positions: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = [{"time": TEST_START, "equity": equity}]
    last_entry_by_symbol: dict[str, pd.Timestamp] = {}
    entries_by_day: dict[str, int] = {}
    skipped = {"open_limit": 0, "same_symbol_open": 0, "daily_limit": 0, "cooldown": 0}

    def close_due(as_of: pd.Timestamp) -> None:
        nonlocal equity
        due = sorted(
            [position for position in open_positions if position["trade"].exit_time <= as_of],
            key=lambda position: position["trade"].exit_time,
        )
        for position in due:
            trade: CandidateTrade = position["trade"]
            position["record"]["equity_at_exit_before"] = equity
            equity += position["pnl"]
            position["record"]["equity_after"] = equity
            accepted.append(position["record"])
            events.append({"time": trade.exit_time, "equity": equity})
            open_positions.remove(position)

    for trade in sorted(trades, key=lambda item: (item.entry_time, -item.score, item.symbol)):
        close_due(trade.entry_time)
        if any(position["trade"].symbol == trade.symbol for position in open_positions):
            skipped["same_symbol_open"] += 1
            continue
        if len(open_positions) >= MAX_OPEN_POSITIONS:
            skipped["open_limit"] += 1
            continue
        day = trade.entry_time.strftime("%Y-%m-%d")
        if entries_by_day.get(day, 0) >= MAX_NEW_TRADES_PER_DAY:
            skipped["daily_limit"] += 1
            continue
        last_entry = last_entry_by_symbol.get(trade.symbol)
        if last_entry is not None and trade.entry_time < last_entry + pd.Timedelta(hours=COOLDOWN_HOURS):
            skipped["cooldown"] += 1
            continue

        size = min(
            equity * RISK_PER_TRADE / trade.price_risk_per_unit,
            equity * MAX_ALLOCATION / trade.entry_price,
        )
        if size <= 0:
            continue
        pnl = size * trade.net_pnl_per_unit
        record = trade.to_record()
        record.update(
            size=size,
            notional=size * trade.entry_price,
            pnl_net=pnl,
            fees=size * trade.fees_per_unit,
            equity_at_entry=equity,
            concurrent_positions_before=len(open_positions),
        )
        open_positions.append({"trade": trade, "pnl": pnl, "record": record})
        last_entry_by_symbol[trade.symbol] = trade.entry_time
        entries_by_day[day] = entries_by_day.get(day, 0) + 1

    close_due(TEST_END + pd.Timedelta(days=2))
    accepted.sort(key=lambda row: row["entry_time"])
    events.sort(key=lambda row: row["time"])
    return accepted, events, skipped
