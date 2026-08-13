"""Strategy registry."""
from __future__ import annotations

from typing import Dict, Optional, Type

from bot.strategies.base import Strategy
from bot.strategies.community import (BBandsBreakout, DCABot, DonchianBreakout,
                                      GoldenCross, GridTrader, MACDCross,
                                      RSI2, StochasticReversion)
from bot.strategies.mean_reversion import MeanReversionStrategy
from bot.strategies.ml import MLOptimizerStrategy
from bot.strategies.ml_trend import MLTrendStrategy
from bot.strategies.momentum import MomentumStrategy

REGISTRY: Dict[str, Type[Strategy]] = {
    cls.name: cls for cls in (
        MomentumStrategy, MeanReversionStrategy, MLOptimizerStrategy,
        MLTrendStrategy,
        MACDCross, GoldenCross, DonchianBreakout, RSI2,
        StochasticReversion, BBandsBreakout, GridTrader, DCABot,
    )
}


def build_strategy(name: str, params: Optional[dict] = None) -> Strategy:
    if name not in REGISTRY:
        raise KeyError(f"unknown strategy '{name}'; available: {sorted(REGISTRY)}")
    return REGISTRY[name](params)


def build_strategies(config: dict) -> list[Strategy]:
    """Build strategies from a config section like {'momentum': {...}, 'ml': {...}}."""
    out = []
    for name, params in (config or {}).items():
        out.append(build_strategy(name, params or None))
    return out