from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from bt_config import (
    CACHE_DIR, DAILY_EMA_FAST, DAILY_EMA_SLOW, FEE_RATE, HF_BASE,
    MONTHLY_EMA, SLIPPAGE_RATE, TEST_END, WARMUP_START,
    WEEKLY_EMA_FAST, WEEKLY_EMA_SLOW,
)


class TrendRegime:
    """Higher-timeframe filter using only completed daily/weekly/monthly candles."""

    def __init__(self, daily: pd.DataFrame):
        d = daily.copy().sort_values("time").reset_index(drop=True)
        d["ema_fast"] = d["close"].ewm(span=DAILY_EMA_FAST, adjust=False).mean()
        d["ema_slow"] = d["close"].ewm(span=DAILY_EMA_SLOW, adjust=False).mean()
        d["bull"] = (
            (d["close"] > d["ema_fast"])
            & (d["ema_fast"] > d["ema_slow"])
            & (d["ema_fast"] > d["ema_fast"].shift(3))
        )
        d["available_at"] = d["time"] + pd.Timedelta(days=1)

        periods = d.copy()
        naive = periods["time"].dt.tz_convert(None)
        periods["week_period"] = naive.dt.to_period("W-SUN")
        weekly = periods.groupby("week_period", observed=True).agg(
            open=("open", "first"), high=("high", "max"), low=("low", "min"),
            close=("close", "last"), volume=("volume", "sum"),
        ).reset_index()
        weekly["ema_fast"] = weekly["close"].ewm(span=WEEKLY_EMA_FAST, adjust=False).mean()
        weekly["ema_slow"] = weekly["close"].ewm(span=WEEKLY_EMA_SLOW, adjust=False).mean()
        weekly["bull"] = (
            (weekly["close"] > weekly["ema_fast"])
            & (weekly["ema_fast"] > weekly["ema_slow"])
            & (weekly["ema_fast"] > weekly["ema_fast"].shift(2))
        )
        weekly["available_at"] = pd.to_datetime(
            [p.end_time + pd.Timedelta(nanoseconds=1) for p in weekly["week_period"]], utc=True
        )

        periods["month_period"] = naive.dt.to_period("M")
        monthly = periods.groupby("month_period", observed=True).agg(
            open=("open", "first"), high=("high", "max"), low=("low", "min"),
            close=("close", "last"), volume=("volume", "sum"),
        ).reset_index()
        monthly["ema"] = monthly["close"].ewm(span=MONTHLY_EMA, adjust=False).mean()
        monthly["bull"] = (
            (monthly["close"] > monthly["ema"])
            & (monthly["ema"] > monthly["ema"].shift(1))
        )
        monthly["available_at"] = pd.to_datetime(
            [p.end_time + pd.Timedelta(nanoseconds=1) for p in monthly["month_period"]], utc=True
        )

        self.daily_times, self.daily_values = self._arrays(d[["available_at", "bull"]])
        self.weekly_times, self.weekly_values = self._arrays(weekly[["available_at", "bull"]])
        self.monthly_times, self.monthly_values = self._arrays(monthly[["available_at", "bull"]])

    @staticmethod
    def _arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        frame = frame.dropna(subset=["available_at"]).sort_values("available_at")
        return (
            frame["available_at"].astype("int64").to_numpy(),
            frame["bull"].fillna(False).astype(bool).to_numpy(),
        )

    @staticmethod
    def _lookup(times: np.ndarray, values: np.ndarray, timestamp: pd.Timestamp) -> bool:
        idx = int(np.searchsorted(times, timestamp.value, side="right") - 1)
        return bool(values[idx]) if idx >= 0 else False

    def bullish(self, timestamp: pd.Timestamp) -> bool:
        return (
            self._lookup(self.daily_times, self.daily_values, timestamp)
            and self._lookup(self.weekly_times, self.weekly_values, timestamp)
            and self._lookup(self.monthly_times, self.monthly_values, timestamp)
        )


class ExitEvaluator:
    """Evaluate exits on the underlying 15-minute bars; stop wins ambiguous bars."""

    def __init__(self, base15: pd.DataFrame):
        self.times_ns = base15["time"].astype("int64").to_numpy()
        self.times = base15["time"].tolist()
        self.lows = base15["low"].to_numpy(dtype=float)
        self.highs = base15["high"].to_numpy(dtype=float)
        self.closes = base15["close"].to_numpy(dtype=float)

    def evaluate(self, entry_time: pd.Timestamp, entry_raw: float, stop: float, target: float) -> dict[str, Any] | None:
        start_idx = int(np.searchsorted(self.times_ns, entry_time.value, side="left"))
        if start_idx >= len(self.times):
            return None
        entry_price = float(entry_raw) * (1.0 + SLIPPAGE_RATE)
        if not (stop < entry_price < target):
            return None

        outcome = "END_OF_TEST"
        exit_idx = len(self.times) - 1
        exit_price = self.closes[exit_idx] * (1.0 - SLIPPAGE_RATE)
        for idx in range(start_idx, len(self.times)):
            if self.lows[idx] <= stop:
                outcome = "STOP"
                exit_idx = idx
                exit_price = stop * (1.0 - SLIPPAGE_RATE)
                break
            if self.highs[idx] >= target:
                outcome = "TARGET"
                exit_idx = idx
                exit_price = target * (1.0 - SLIPPAGE_RATE)
                break

        price_risk = entry_price - stop
        gross_unit = exit_price - entry_price
        fees_unit = FEE_RATE * entry_price + FEE_RATE * exit_price
        net_unit = gross_unit - fees_unit
        return {
            "entry_price": entry_price,
            "exit_time": self.times[exit_idx],
            "exit_price": exit_price,
            "outcome": outcome,
            "duration_hours": max(0.0, (self.times[exit_idx] - entry_time).total_seconds() / 3600.0),
            "price_risk": price_risk,
            "gross_r": gross_unit / price_risk,
            "net_r": net_unit / price_risk,
            "net_return_on_notional": net_unit / entry_price,
            "net_pnl_per_unit": net_unit,
            "fees_per_unit": fees_unit,
        }


def make_session() -> requests.Session:
    retry = Retry(
        total=5, connect=5, read=5, backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4))
    session.headers.update({"User-Agent": "Mozilla/5.0 spike-bos-backtest/1.0"})
    return session


def download_symbol(session: requests.Session, symbol: str) -> Path:
    destination = CACHE_DIR / f"{symbol}USDT_15m.csv"
    if destination.exists() and destination.stat().st_size > 1_000_000:
        return destination
    url = f"{HF_BASE}/{symbol}USDT_15m.csv?download=true"
    temporary = destination.with_suffix(".part")
    print(f"Downloading {symbol}: {url}", flush=True)
    with session.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        with temporary.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    temporary.replace(destination)
    if destination.stat().st_size < 1_000_000:
        raise RuntimeError(f"Downloaded file for {symbol} is unexpectedly small")
    return destination


def read_symbol_csv(path: Path) -> pd.DataFrame:
    columns = ["Open time", "open", "high", "low", "close", "volume"]
    df = pd.read_csv(path, usecols=columns, low_memory=False).rename(columns={"Open time": "time"})
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = (
        df.dropna(subset=["time", "open", "high", "low", "close"])
        .drop_duplicates(subset=["time"], keep="last")
        .sort_values("time")
        .reset_index(drop=True)
    )
    return df[(df["time"] >= WARMUP_START) & (df["time"] < TEST_END)].reset_index(drop=True)


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        df.set_index("time")
        .resample(rule, label="left", closed="left")
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
             close=("close", "last"), volume=("volume", "sum"))
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
