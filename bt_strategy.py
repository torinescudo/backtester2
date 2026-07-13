from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from bt_config import (
    ATR_PERIOD, MIN_ATR_MULTIPLE, MIN_BARS_AFTER_SPIKE, MIN_DROP_PCT,
    MIN_NET_DROP_PCT, MIN_REWARD_RISK, MIN_SWEEP_PCT,
    MAX_BARS_AFTER_SPIKE, ORIGIN_LOOKBACK, SPIKE_BARS_MAX,
    STOP_BUFFER_PCT, SWEEP_LOOKBACK, TARGET_BUFFER_PCT,
    TEST_END, TEST_START, CandidateTrade,
)
from bt_data import ExitEvaluator, TrendRegime


def add_atr(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    previous_close = result["close"].shift(1)
    true_range = pd.concat(
        [
            result["high"] - result["low"],
            (result["high"] - previous_close).abs(),
            (result["low"] - previous_close).abs(),
        ], axis=1,
    ).max(axis=1)
    result["atr"] = true_range.rolling(ATR_PERIOD).mean()
    result["prior_low"] = result["low"].shift(1).rolling(SWEEP_LOOKBACK).min()
    result["pre_origin_high"] = result["high"].shift(1).rolling(ORIGIN_LOOKBACK).max()
    result["pre_volume_median"] = result["volume"].shift(1).rolling(20).median()
    return result


def detect_candidates(
    symbol: str,
    timeframe: str,
    timeframe_minutes: int,
    frame: pd.DataFrame,
    regime: TrendRegime,
    exit_evaluator: ExitEvaluator,
) -> list[CandidateTrade]:
    """Find spikes and enter only after the first confirmed bullish BOS."""
    df = add_atr(frame).reset_index(drop=True)
    n = len(df)
    if n < ATR_PERIOD + SWEEP_LOOKBACK + MAX_BARS_AFTER_SPIKE + 5:
        return []

    open_arr = df["open"].to_numpy(dtype=float)
    high_arr = df["high"].to_numpy(dtype=float)
    low_arr = df["low"].to_numpy(dtype=float)
    close_arr = df["close"].to_numpy(dtype=float)
    volume_arr = df["volume"].to_numpy(dtype=float)
    atr_arr = df["atr"].to_numpy(dtype=float)
    prior_low_arr = df["prior_low"].to_numpy(dtype=float)
    pre_origin_arr = df["pre_origin_high"].to_numpy(dtype=float)
    pre_vol_arr = df["pre_volume_median"].to_numpy(dtype=float)
    times = df["time"].tolist()

    candidates: list[CandidateTrade] = []
    first_start = max(ATR_PERIOD + 1, SWEEP_LOOKBACK, ORIGIN_LOOKBACK)
    last_start = n - MAX_BARS_AFTER_SPIKE - SPIKE_BARS_MAX - 2

    for event_start in range(first_start, last_start + 1):
        for spike_bars in range(1, SPIKE_BARS_MAX + 1):
            event_end = event_start + spike_bars - 1
            if event_end + MIN_BARS_AFTER_SPIKE + 1 >= n:
                continue

            event_open = open_arr[event_start]
            event_close = close_arr[event_end]
            event_high = float(np.max(high_arr[event_start:event_end + 1]))
            extreme = float(np.min(low_arr[event_start:event_end + 1]))
            origin = max(float(pre_origin_arr[event_start]), event_high)
            atr_ref = atr_arr[event_start - 1]
            prior_low = prior_low_arr[event_start]

            if not np.isfinite(origin) or not np.isfinite(extreme) or origin <= 0 or extreme <= 0:
                continue
            if not np.isfinite(atr_ref) or atr_ref <= 0 or not np.isfinite(prior_low):
                continue

            drop_pct = (origin - extreme) / origin
            net_drop_pct = (event_open - event_close) / event_open if event_open > 0 else 0.0
            drop_atr = (origin - extreme) / atr_ref
            if not (
                extreme < prior_low * (1.0 - MIN_SWEEP_PCT)
                and event_close < event_open
                and drop_pct >= MIN_DROP_PCT
                and net_drop_pct >= MIN_NET_DROP_PCT
                and drop_atr >= MIN_ATR_MULTIPLE
            ):
                continue

            normal_volume = pre_vol_arr[event_start]
            event_volume = float(np.mean(volume_arr[event_start:event_end + 1]))
            volume_ratio = event_volume / normal_volume if np.isfinite(normal_volume) and normal_volume > 0 else 1.0

            for bars_after in range(MIN_BARS_AFTER_SPIKE, MAX_BARS_AFTER_SPIKE + 1):
                breakout_idx = event_end + bars_after
                if breakout_idx + 1 >= n:
                    break
                prior_rebound = high_arr[event_end + 1:breakout_idx]
                if prior_rebound.size == 0:
                    continue
                bos_level = float(np.max(prior_rebound))
                previous_close = close_arr[breakout_idx - 1]
                signal_close = close_arr[breakout_idx]
                if not (signal_close > bos_level and previous_close <= bos_level):
                    continue

                signal_time = times[breakout_idx] + pd.Timedelta(minutes=timeframe_minutes)
                if signal_time < TEST_START or signal_time >= TEST_END or not regime.bullish(signal_time):
                    break

                stop = extreme * (1.0 - STOP_BUFFER_PCT)
                target = origin * (1.0 - TARGET_BUFFER_PCT)
                if not (stop < signal_close < target):
                    break
                risk_signal = signal_close - stop
                reward_signal = target - signal_close
                reward_risk = reward_signal / risk_signal if risk_signal > 0 else 0.0
                if reward_risk < MIN_REWARD_RISK:
                    break

                entry_idx = breakout_idx + 1
                entry_time = times[entry_idx]
                if entry_time != signal_time:
                    break
                result = exit_evaluator.evaluate(entry_time, float(open_arr[entry_idx]), stop, target)
                if result is None:
                    break

                candidates.append(CandidateTrade(
                    symbol=symbol,
                    timeframe=timeframe,
                    signal_time=signal_time,
                    entry_time=entry_time,
                    exit_time=result["exit_time"],
                    entry_price=result["entry_price"],
                    stop_price=stop,
                    target_price=target,
                    exit_price=result["exit_price"],
                    outcome=result["outcome"],
                    duration_hours=result["duration_hours"],
                    score=drop_atr + reward_risk + min(volume_ratio, 3.0) * 0.20,
                    drop_pct=drop_pct,
                    drop_atr=drop_atr,
                    reward_risk_signal=reward_risk,
                    gross_r=result["gross_r"],
                    net_r=result["net_r"],
                    net_return_on_notional=result["net_return_on_notional"],
                    price_risk_per_unit=result["price_risk"],
                    net_pnl_per_unit=result["net_pnl_per_unit"],
                    fees_per_unit=result["fees_per_unit"],
                    origin=origin,
                    extreme=extreme,
                    bos_level=bos_level,
                ))
                break

    return candidates


def deduplicate_candidates(candidates: Iterable[CandidateTrade]) -> list[CandidateTrade]:
    best: dict[tuple[str, pd.Timestamp], CandidateTrade] = {}
    for trade in candidates:
        key = (trade.symbol, trade.signal_time)
        current = best.get(key)
        if current is None or trade.score > current.score:
            best[key] = trade
    return sorted(best.values(), key=lambda trade: (trade.entry_time, -trade.score, trade.symbol))
