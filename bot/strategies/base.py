"""Strategy interface.

Signal convention (0/1/-1):
  * ``1``  -> enter long (buy) on the *next* candle's open
  * ``-1`` -> exit long (sell) on the *next* candle's open
  * ``0``  -> no action

Causality contract: ``compute_signals`` must only use data up to and
including bar ``i`` when producing the signal at index ``i``. Execution
on the next bar's open is enforced by the backtest engine, which makes
look-ahead bias structurally impossible for TA strategies.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd


class Strategy(ABC):
    name: str = "base"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params: Dict[str, Any] = params or {}

    # ---- optional interface for ML strategies ---------------------------
    def fit(self, df: pd.DataFrame) -> None:
        """Train on historical data up to the last closed candle."""
        return None

    @abstractmethod
    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        """Return a pd.Series (aligned to df.index) of 1 / -1 / 0 signals.

        ``live=True`` is used by the paper engine: stateful strategies
        (ML) may reuse/refresh a trained model instead of re-walking the
        whole history.
        """

    def warmup_bars(self) -> int:
        """Bars at the start of df that can never produce a signal."""
        return 0


def entry_or_exit(signals: pd.Series, i: int, in_position: bool) -> int:
    """Resolve a raw signal series to an action, respecting position state."""
    sig = signals.iloc[i]
    if in_position:
        return -1 if sig == -1 else 0
    return 1 if sig == 1 else 0