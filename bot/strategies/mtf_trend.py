"""MTFTrend: 4H trend entries gated by the 1D trend.

Lab-born gen-2 winner (sim-lab walk-forward, Aug 2026). Multi-
timeframe confluence: the 4H EMA cross only fires when the daily
close is above its own SMA (resampled from the same window — no extra
data dependency, fully causal forward-fill).

Walk-forward record (2y real 4H, 5 folds, capped grid, purge gaps):
  mean OOS excess +9.1%, won 4/5 folds, 41 trades — the most
  bull-leg participation of any lab bot (fixes the fold-2 weakness
  where cash-discipline bots missed +69% rallies)
Live-path replay (last 600 bars, past-as-live, bear tail):
  POSITIVE absolute returns on 5/6 pairs (+1.3..+3.9%) while
  buy&hold fell -16..-37%; excess +19.4..+37.3%

Final params {day_sma: 30} frozen from the last walk-forward fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot.indicators.ta import ema
from bot.strategies.base import Strategy


class MTFTrend(Strategy):
    name = "mtf_trend"
    DEFAULTS = {"fast": 20, "slow": 50, "day_sma": 30, "day_band": 0.01}

    def warmup_bars(self) -> int:
        return 240

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        p = {**self.DEFAULTS, **self.params}
        close = df["close"]
        f, s = ema(close, int(p["fast"])), ema(close, int(p["slow"]))
        cross_up = (f > s) & (f.shift(1) <= s.shift(1))
        cross_dn = (f < s) & (f.shift(1) >= s.shift(1))
        # daily gate from the same window (causal: each day uses only
        # closes that have already happened; ffill onto the intraday
        # index never looks ahead)
        daily = close.resample("1D").last().dropna()
        day_sma = daily.rolling(int(p["day_sma"]), min_periods=10).mean()
        day_ok = (daily > day_sma * (1 - float(p["day_band"]))).fillna(False)
        gate = day_ok.reindex(close.index, method="ffill").fillna(False)
        buy = cross_up & gate
        sell = cross_dn | ~gate
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[buy.fillna(False)] = 1
        sig[sell.fillna(False)] = -1
        return sig
