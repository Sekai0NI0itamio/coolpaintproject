#!/usr/bin/env python3
"""Train seed tunings for the swarm on historical data (runs on GitHub).

Downloads real Coinbase history, backtests a bounded grid of strategy
tunings (trial count capped to fight overfitting), ranks them by
out-of-sample excess return vs buy & hold AFTER fees, and writes the
top 5 configs to state/seeds.json for the swarm to clone from.

Usage:
    python run_train.py --days 365
    python run_train.py --days 730 --granularity ONE_HOUR
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot.backtest.engine import run_backtest  # noqa: E402
from bot.backtest.walkforward import split_train_test  # noqa: E402
from bot.config import BotConfig  # noqa: E402
from bot.data.fetcher import fetch_history  # noqa: E402
from bot.data.store import Store  # noqa: E402
from bot.strategies import build_strategy  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEEDS_PATH = os.path.join(BASE_DIR, "state", "seeds.json")
HISTORY_DIR = os.path.join(BASE_DIR, "state", "history")

GRID = {
    "mean_reversion": {
        "rsi_period": [10, 14, 20],
        "bb_period": [14, 20, 30],
        "bb_std": [1.8, 2.0, 2.5],
        "oversold": [25, 30, 35],
        "exit_rsi": [50, 55, 60],
    },
    "momentum": {
        "ema_fast": [8, 12, 16],
        "ema_slow": [21, 26, 40],
        "trend_ema": [100, 200, 300],
    },
}
MAX_TRIALS = {"mean_reversion": 24, "momentum": 12}   # overfit control
MIN_TOTAL_TRADES = 8


def _combos(strategy: str, rng: random.Random) -> list[dict]:
    keys = sorted(GRID[strategy])
    all_c = [dict(zip(keys, values))
             for values in itertools.product(*(GRID[strategy][k] for k in keys))]
    rng.shuffle(all_c)
    return all_c[:MAX_TRIALS[strategy]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Train swarm seed tunings on history")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--granularity", default="FIFTEEN_MINUTE",
                    help="must match the swarm timeframe (FIFTEEN_MINUTE)")
    ap.add_argument("--pairs", default="BTC-USDC,ETH-USDC,SOL-USDC")
    ap.add_argument("--db", default=os.path.join(BASE_DIR, "data", "train.db"))
    args = ap.parse_args()

    cfg = BotConfig.from_yaml(None)
    pairs = args.pairs.split(",")
    rng = random.Random(42)
    store = Store(args.db)

    data = {}
    for pair in pairs:
        print(f"[train] fetching {args.days}d {args.granularity} for {pair}...")
        df = fetch_history(pair, args.granularity, args.days)
        store.upsert_candles(pair, args.granularity, df)
        df = store.load_candles(pair, args.granularity)
        data[pair] = df
        print(f"[train]   {len(df)} candles {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")

    results = []
    for strategy_name in ("mean_reversion", "momentum"):
        for params in _combos(strategy_name, rng):
            per_pair, total_trades = {}, 0
            for pair, df in data.items():
                strat = build_strategy(strategy_name, params)
                _, test = split_train_test(df, train_frac=0.7)
                if len(test) < 200:
                    continue
                r = run_backtest(test, strat, pair=pair,
                                 taker_fee=cfg.taker_fee, slippage=cfg.slippage,
                                 position_fraction=0.30, capital=20.0)
                per_pair[pair] = {"excess%": round(r.excess_return * 100, 2),
                                  "trades": r.n_trades,
                                  "win%": round(r.win_rate * 100, 1)}
                total_trades += r.n_trades
            if not per_pair or total_trades < MIN_TOTAL_TRADES:
                continue
            score = sum(v["excess%"] for v in per_pair.values()) / len(per_pair)
            results.append({"strategy": strategy_name, "params": params,
                            "score": round(score, 3), "total_trades": total_trades,
                            "per_pair": per_pair})

    results.sort(key=lambda r: r["score"], reverse=True)
    top = results[:5]

    os.makedirs(os.path.dirname(SEEDS_PATH), exist_ok=True)
    payload = {
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "days": args.days, "granularity": args.granularity, "pairs": pairs,
        "note": "Ranked by mean OOS excess return vs buy&hold after 0.6% taker "
                "fees + slippage. Trial count capped to limit overfitting.",
        "seeds": top,
    }
    with open(SEEDS_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)

    lines = [f"# Swarm seed training - {payload['trained_at']}",
             f"Data: {args.days} days {args.granularity} on {', '.join(pairs)}",
             f"Evaluated: {len(results)} viable tunings (fees ON, OOS last 30%)", "",
             "## Top 5 seeds", ""]
    for i, r in enumerate(top, 1):
        lines.append(f"### {i}. {r['strategy']} (score {r['score']:+.2f}% avg excess, "
                     f"{r['total_trades']} trades)")
        lines.append(f"params: `{json.dumps(r['params'])}`")
        for pair, v in r["per_pair"].items():
            lines.append(f"- {pair}: excess {v['excess%']:+.2f}%, "
                         f"{v['trades']} trades, win {v['win%']}%")
        lines.append("")
    os.makedirs(HISTORY_DIR, exist_ok=True)
    report_path = os.path.join(
        HISTORY_DIR, f"train-{datetime.now(timezone.utc):%Y-%m-%d}.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"\n[train] wrote {len(top)} seeds -> {SEEDS_PATH}")
    print(f"[train] report -> {report_path}")
    for i, r in enumerate(top, 1):
        print(f"  {i}. {r['strategy']} score={r['score']:+.2f}% {r['params']}")


if __name__ == "__main__":
    main()