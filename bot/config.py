"""Configuration loading for the trading bot.

Loads ``strategies.yaml`` (or a user-supplied config file) into typed
dataclasses used by the backtester, paper engine and reports.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(BASE_DIR, "strategies.yaml")

# Coinbase granularity enum -> candle length in seconds.
GRANULARITY_SECONDS: Dict[str, int] = {
    "ONE_MINUTE": 60,
    "FIVE_MINUTE": 300,
    "FIFTEEN_MINUTE": 900,
    "THIRTY_MINUTE": 1800,
    "ONE_HOUR": 3600,
    "TWO_HOUR": 7200,
    "FOUR_HOUR": 14400,
    "SIX_HOUR": 21600,
    "ONE_DAY": 86400,
}


@dataclass
class BotConfig:
    pairs: list[str] = field(default_factory=lambda: ["BTC-USDC", "ETH-USDC", "SOL-USDC"])
    granularity: str = "ONE_HOUR"
    history_days: int = 365
    paper_capital: float = 10_000.0
    taker_fee: float = 0.006      # Coinbase Advanced Trade retail taker fee (~0.6%)
    slippage: float = 0.001       # pessimistic 0.1% per fill
    position_fraction: float = 0.25  # fraction of equity deployed per trade
    max_positions: int = 3
    poll_seconds: int = 30        # REST polling interval for the paper engine
    db_path: str = os.path.join(BASE_DIR, "data", "trading.db")
    out_dir: str = os.path.join(BASE_DIR, "reports")
    strategies: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def candle_seconds(self) -> int:
        return GRANULARITY_SECONDS[self.granularity]

    @classmethod
    def from_yaml(cls, path: str | None = None) -> "BotConfig":
        path = path or DEFAULT_CONFIG
        with open(path, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}
        known = {f for f in cls.__dataclass_fields__}
        cfg = cls()
        for key, value in raw.items():
            if key in known and value is not None:
                setattr(cfg, key, value)
        return cfg
