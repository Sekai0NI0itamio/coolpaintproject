"""DonchianSage: breakout entries confirmed by Sage's evidence panel.

Lab-born winner (sim-lab R&D, Aug 2026): the classic 20/30-bar Donchian
breakout only fires when the Sage evidence panel (trend/dip/panic/
value/band/momentum witnesses) already agrees — killing the false
breakouts that make raw turtle trading bleed fees (Costa 2026: >75% of
breakout attempts are liquidity sweeps).

Walk-forward record (2y real 4H, 5 folds, capped grid, purge gaps):
  mean OOS excess +10.2%, won 4/5 folds, 23 trades across folds
  (the only lab idea that both trades AND beats buy&hold OOS —
  everything else earned its excess by standing aside)
Live-path replay (last 600 bars, past-as-live, all 6 pairs):
  +15.8% .. +37.3% excess, 1-3 trades per pair

Final params {min_score: 2.0, entry_period: 30} frozen from the last
walk-forward fold.
"""
from __future__ import annotations

import pandas as pd

from bot.strategies.base import Strategy
from bot.strategies.sage import SageStrategy


class DonchianSage(Strategy):
    name = "donchian_sage"
    DEFAULTS = {"entry_period": 30, "exit_period": 10, "min_score": 2.0}

    def warmup_bars(self) -> int:
        return 220

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        p = {**self.DEFAULTS, **self.params}
        close = df["close"]
        hi = df["high"].rolling(int(p["entry_period"])).max().shift(1)
        lo = df["low"].rolling(int(p["exit_period"])).min().shift(1)
        panel = SageStrategy({}).score_series(df)
        buy = (close > hi) & (panel >= float(p["min_score"]))
        sell = close < lo
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[buy.fillna(False)] = 1
        sig[sell.fillna(False)] = -1
        return sig
