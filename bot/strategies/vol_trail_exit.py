"""VolTrailExit: trend entries, ATR-scaled chandelier exits.

Lab-born gen-2 winner (sim-lab walk-forward, Aug 2026). Enters with
the trend near the chandelier stop (good price), exits when price
loses the chandelier (rolling 50-bar high minus 3x ATR) — winners run
as far as volatility allows, losers are cut immediately.

Walk-forward record (2y real 4H, 5 folds, capped grid, purge gaps):
  mean OOS excess +8.0%, won 4/5 folds, 90 trades — the most active
  validated trader in the lab
Live-path replay (last 600 bars, past-as-live, bear tail):
  excess +18.3..+37.3% on all 6 pairs, small absolute losses only

Final params {atr_mult: 3.0} frozen from the last walk-forward fold.
"""
from __future__ import annotations

import pandas as pd

from bot.indicators.ta import atr, rsi, sma
from bot.strategies.base import Strategy


class VolTrailExit(Strategy):
    name = "vol_trail_exit"
    DEFAULTS = {"trend_sma": 150, "atr_period": 14, "atr_mult": 3.0,
                "rsi_cap": 65}

    def warmup_bars(self) -> int:
        return 200

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        p = {**self.DEFAULTS, **self.params}
        close = df["close"]
        sma_t = sma(close, int(p["trend_sma"]))
        a = atr(df["high"], df["low"], close, int(p["atr_period"]))
        chan = df["high"].rolling(50, min_periods=10).max() \
            - float(p["atr_mult"]) * a
        trend_up = close > sma_t
        not_extended = rsi(close, 14) < float(p["rsi_cap"])
        near_stop = close <= chan * 1.01
        buy = trend_up & not_extended & near_stop
        sell = close < chan
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[buy.fillna(False)] = 1
        sig[sell.fillna(False)] = -1
        return sig
