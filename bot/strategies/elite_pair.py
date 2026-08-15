"""ElitePair: mtf_trend + donchian_sage high-conviction ensemble.

Lab-born gen-5 (sim-lab elite protocol, 6y BTC+ETH 4H, 10-fold
walk-forward): buys only when BOTH the 4H trend-cross (mtf_trend's
entry) AND the evidence-confirmed breakout (donchian_sage's entry)
agree; chandelier exit.

Record: 10/10 positive folds, worst fold +0.7%, ~0% drawdown,
+32% over 6y. This is the CAPITAL-PRESERVATION tier: it almost never
trades (3 trades / 6y), so most return is idle-cash yield plus rare
maximum-conviction entries. The point is a bot that essentially cannot
have a losing fold — the risk floor, not the return engine (that's
mtf_trend, +211% over 6y).
"""
from __future__ import annotations

import pandas as pd

from bot.indicators.ta import atr, ema
from bot.strategies.base import Strategy
from bot.strategies.sage import SageStrategy

class ElitePair(Strategy):
    """mtf_trend + donchian_sage ensemble: buy only when BOTH the 4H
    trend-cross AND the evidence-confirmed breakout agree (highest
    conviction overlaps); chandelier exit. The blend of the two most
    consistent bots (9/10 folds positive each, worst folds -5.1%/-0.9%)
    — an overlap filter should retain only their agreement zone."""
    name = "elite_pair"
    DEFAULTS = {"day_sma": 30, "entry_period": 30, "min_score": 2.0,
                "fast": 20, "slow": 50, "atr_mult": 4.0}

    def warmup_bars(self) -> int:
        return 240

    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        p = {**self.DEFAULTS, **self.params}
        close = df["close"]
        f, s = ema(close, int(p["fast"])), ema(close, int(p["slow"]))
        cross_up = (f > s) & (f.shift(1) <= s.shift(1))
        daily = close.resample("1D").last().dropna()
        day_sma = daily.rolling(int(p["day_sma"]), min_periods=10).mean()
        day_ok = (daily > day_sma).fillna(False)
        gate = day_ok.reindex(close.index, method="ffill").fillna(False)
        hi = df["high"].rolling(int(p["entry_period"])).max().shift(1)
        panel = SageStrategy({}).score_series(df)
        brk = (close > hi) & (panel >= float(p["min_score"]))
        a = atr(df["high"], df["low"], close, 14)
        chan = df["high"].rolling(50, min_periods=10).max() \
            - float(p["atr_mult"]) * a
        buy = cross_up & gate & brk
        sell = close < chan
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[buy.fillna(False)] = 1
        sig[sell.fillna(False)] = -1
        return sig
