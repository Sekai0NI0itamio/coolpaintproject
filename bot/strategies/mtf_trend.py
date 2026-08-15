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

from bot.indicators.ta import atr, ema
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


class SwingRider(Strategy):
    """Momentum-ignition entries + chandelier trail.

    Born from the sim-lab miss analysis (Aug 2026): 84% of missed >=8%
    swings were rallies starting inside downtrends (signal-blind), and
    winners were cashed at ~+2% with +19.8% left behind. SwingRider
    enters on surge ignition (>=5% over 12 bars, vol regime ok) —
    regime-agnostic by design, risk bounded by the chassis stop — and
    trails with a 4.5xATR chandelier so winners ride the swing.

    Walk-forward (2y 4H, 5 folds): +6.9% mean OOS, 119 trades (most
    active validated bot). Capture rate 40% of all >=8% swings vs 15%
    for the next best (the miss-analysis evidence that motivated its
    promotion despite missing the +8% excess bar)."""
    name = "swing_rider"
    DEFAULTS = {"surge_pct": 0.05, "surge_bars": 12,
                "atr_period": 14, "atr_mult": 4.5, "cooldown": 6,
                "vol_ok_pctile": 0.30}

    def warmup_bars(self) -> int:
        return 200

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        p = {**self.DEFAULTS, **self.params}
        close = df["close"]
        surge = close / close.shift(int(p["surge_bars"])) - 1.0
        a = atr(df["high"], df["low"], close, int(p["atr_period"]))
        a_pctile = (a / close).rolling(540, min_periods=120).apply(
            lambda v: (v <= v[-1]).mean(), raw=True).fillna(0.5)
        ignite = (surge >= float(p["surge_pct"])) & \
                 (a_pctile >= float(p["vol_ok_pctile"]))
        buy = ignite & ~ignite.shift(1, fill_value=False)
        buy = buy & ~buy.rolling(int(p["cooldown"])).sum().shift(1) \
            .fillna(0).gt(0)
        chan = df["high"].rolling(50, min_periods=10).max() \
            - float(p["atr_mult"]) * a
        sell = close < chan
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[buy.fillna(False)] = 1
        sig[sell.fillna(False)] = -1
        return sig
