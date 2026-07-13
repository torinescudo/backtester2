from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import pandas as pd

SYMBOLS = [
    "BTC", "ETH", "BNB", "XRP", "SOL", "TRX", "DOGE", "ADA", "BCH", "LINK",
    "XLM", "LTC", "AVAX", "SHIB", "DOT", "UNI", "AAVE", "ETC", "FIL", "ALGO",
]
TEST_START = pd.Timestamp("2024-01-01T00:00:00Z")
TEST_END = pd.Timestamp("2026-07-01T00:00:00Z")
WARMUP_START = pd.Timestamp("2020-01-01T00:00:00Z")
INITIAL_EQUITY = 10_000.0
FEE_RATE = 0.006
SLIPPAGE_RATE = 0.0005
RISK_PER_TRADE = 0.005
MAX_ALLOCATION = 0.20
MAX_OPEN_POSITIONS = 2
MAX_NEW_TRADES_PER_DAY = 3
COOLDOWN_HOURS = 12
ATR_PERIOD = 14
SWEEP_LOOKBACK = 20
ORIGIN_LOOKBACK = 3
SPIKE_BARS_MAX = 2
MIN_BARS_AFTER_SPIKE = 2
MAX_BARS_AFTER_SPIKE = 8
MIN_DROP_PCT = 0.018
MIN_NET_DROP_PCT = 0.007
MIN_ATR_MULTIPLE = 2.0
MIN_SWEEP_PCT = 0.0005
STOP_BUFFER_PCT = 0.003
TARGET_BUFFER_PCT = 0.001
MIN_REWARD_RISK = 1.5
DAILY_EMA_FAST = 20
DAILY_EMA_SLOW = 50
WEEKLY_EMA_FAST = 8
WEEKLY_EMA_SLOW = 20
MONTHLY_EMA = 6
OUT_DIR = Path("results")
CACHE_DIR = Path(".cache")
HF_BASE = "https://huggingface.co/datasets/linxy/CryptoCoin/resolve/main"
OUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

@dataclass
class CandidateTrade:
    symbol: str
    timeframe: str
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    stop_price: float
    target_price: float
    exit_price: float
    outcome: str
    duration_hours: float
    score: float
    drop_pct: float
    drop_atr: float
    reward_risk_signal: float
    gross_r: float
    net_r: float
    net_return_on_notional: float
    price_risk_per_unit: float
    net_pnl_per_unit: float
    fees_per_unit: float
    origin: float
    extreme: float
    bos_level: float

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        for key in ("signal_time", "entry_time", "exit_time"):
            record[key] = record[key].isoformat()
        return record
