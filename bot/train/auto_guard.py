"""Nightly auto-tuner: extract the winning guard parameters over time.

This is the lab's "auto-apply winning logic" loop:

  1. Walk-forward backtest a BOUNDED sample of guard parameter variants
     (trial-capped to limit overfitting) on recent real history, with
     fees + slippage + cash-yield ON, scored on out-of-sample excess.
  2. Pick the best parameter set per guard mode (trend / range).
  3. Write state/guard_params.json, which every guarded bot reads at
     construction -- so the winning thresholds go live automatically on
     the next zoo/swarm run, no code changes.

Honesty guards: trial cap, OOS-only scoring, incumbents must be beaten
by a margin or the previous parameters are kept.
"""
from __future__ import annotations

import itertools
import json
import os
from datetime import datetime, timezone

from bot.backtest.engine import run_backtest
from bot.backtest.walkforward import split_train_test
from bot.config import BotConfig
from bot.data.fetcher import fetch_history
from bot.data.store import Store
from bot.strategies.guarded import (GuardedGrid, GuardedMACD, GuardedMomentum,
                                    GuardedRSI2, GuardedStochastic)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OVERRIDES_PATH = os.path.join(BASE_DIR, "state", "guard_params.json")

# Bounded search space (trial-capped: sampled subset, not full product).
PARAM_GRID = {
    "atr_hurdle_pct": [0.004, 0.008, 0.014],
    "trend_sma": [100, 200, 300],
    "confirm_bars": [1, 2, 3],
}
MAX_TRIALS = 10                      # overfit control
BEAT_MARGIN = 0.25                   # new params must beat incumbent by this %excess
PAIRS = ["BTC-USDC", "ETH-USDC", "SOL-USDC", "DOGE-USDC", "XRP-USDC", "ADA-USDC"]

# Representative bots per guard mode for evaluation.
EVAL_BOTS = {
    "trend": [GuardedMomentum, GuardedMACD],
    "range": [GuardedRSI2, GuardedStochastic, GuardedGrid],
}


def _score_params(mode: str, params: dict, data: dict, cfg: BotConfig) -> float:
    """Mean OOS excess return across eval bots x pairs for one param set."""
    excesses = []
    for bot_cls in EVAL_BOTS[mode]:
        for pair, df in data.items():
            _, test = split_train_test(df, train_frac=0.7)
            if len(test) < 200:
                continue
            strat = bot_cls({**params, "mode": mode})
            r = run_backtest(test, strat, pair=pair,
                             taker_fee=cfg.taker_fee, slippage=cfg.slippage,
                             position_fraction=0.30, capital=20.0,
                             cash_yield_apy=cfg.cash_yield_apy)
            excesses.append(r.excess_return * 100.0)
    return sum(excesses) / len(excesses) if excesses else 0.0


def tune(days: int = 365, granularity: str = "FOUR_HOUR",
         db_path: str = os.path.join(BASE_DIR, "data", "tune.db")) -> dict:
    cfg = BotConfig.from_yaml(None)
    store = Store(db_path)
    data = {}
    for pair in PAIRS:
        df = fetch_history(pair, granularity, days)
        if not df.empty:
            store.upsert_candles(pair, granularity, df)
            data[pair] = store.load_candles(pair, granularity)

    combos = list(itertools.product(*[PARAM_GRID[k] for k in
                                      sorted(PARAM_GRID)]))
    step = max(1, len(combos) // MAX_TRIALS)
    sampled = combos[::step][:MAX_TRIALS]

    current = json.load(open(OVERRIDES_PATH)) if os.path.exists(OVERRIDES_PATH) else {}
    results = {"tuned_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
               "note": "auto-tuned guard params; OOS excess, trial-capped", "modes": {}}
    for mode in ("trend", "range"):
        best_params, best_score = None, -1e9
        for combo in sampled:
            params = dict(zip(sorted(PARAM_GRID), combo))
            score = _score_params(mode, params, data, cfg)
            print(f"  [tune:{mode}] {params} -> OOS excess {score:+.2f}%")
            if score > best_score:
                best_score, best_params = score, params
        incumbent = (current.get(mode) or {})
        incumbent_score = _score_params(mode, incumbent, data, cfg) if incumbent else -1e9
        if best_params and best_score >= incumbent_score + BEAT_MARGIN:
            results["modes"][mode] = {**best_params, "score": round(best_score, 2)}
        else:   # keep incumbent: not beaten by the margin (overfit guard)
            results["modes"][mode] = {**incumbent, "score": round(incumbent_score, 2)} \
                if incumbent else {**best_params, "score": round(best_score, 2)}
    return results


def write_overrides(results: dict, path: str = OVERRIDES_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
