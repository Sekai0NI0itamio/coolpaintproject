"""Technical indicators (pandas/numpy only, no heavy dependencies)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI computed on closed candles only."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)


def bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = sma(close, period)
    std = close.rolling(period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    position = (close - lower) / (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower,
                         "bb_position": position.fillna(0.5)})


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def volume_zscore(volume: pd.Series, window: int = 48) -> pd.Series:
    mean = volume.rolling(window).mean()
    std = volume.rolling(window).std(ddof=0).replace(0.0, np.nan)
    return ((volume - mean) / std).fillna(0.0)


def volume_to_volatility(volume: pd.Series, atr_pct: pd.Series, window: int = 24) -> pd.Series:
    """Volume-to-volatility ratio: rolling mean volume normalised by ATR%."""
    return volume.rolling(window).mean() / atr_pct.replace(0.0, np.nan)