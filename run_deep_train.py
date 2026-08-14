#!/usr/bin/env python3
"""Deep Time trainer: evolve bots for hundreds of simulated years on real
multi-year history, then promote the validated champions to the live bots.

The world replays real Coinbase candles at a coarse granularity (default
4H -- fewer round trips, so the ~1.2% fee wall shrinks ~4x). Bots live
through years of history in seconds; selection runs at segment
boundaries; each epoch's top candidates face an untouched VALIDATION
GAUNTLET (the most recent slice of history). Only gauntlet survivors
that beat buy & hold after fees, with enough closed trades, are written
to state/champions.json -- which the live zoo applies automatically.

Usage:
    python run_deep_train.py --minutes 300            # one CI budget window
    python run_deep_train.py --years 3 --granularity FOUR_HOUR
    python run_deep_train.py --report                 # print the deep board
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot.config import BotConfig  # noqa: E402
from bot.data.fetcher import fetch_history  # noqa: E402
from bot.data.store import Store  # noqa: E402
from bot.train.deep_time import (DEFAULT_STATE_DIR, MIN_CHAMPION_EXCESS,  # noqa: E402
                                 MIN_VALIDATION_TRADES, DeepWorld)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = DEFAULT_STATE_DIR
WORLD_PATH = os.path.join(STATE_DIR, "world.json")
CHAMPIONS_PATH = os.path.join(BASE_DIR, "state", "champions.json")
BOARD_PATH = os.path.join(STATE_DIR, "DEEP_BOARD.md")
DEFAULT_PAIRS = "BTC-USDC,ETH-USDC,SOL-USDC,DOGE-USDC,XRP-USDC,ADA-USDC"


def load_data(pairs: list[str], granularity: str, days: int,
              db_path: str, refresh: bool = False) -> dict:
    """Multi-year candles per pair, cached in SQLite between runs."""
    store = Store(db_path)
    data = {}
    for pair in pairs:
        df = store.load_candles(pair, granularity)
        bar_sec = None
        if not refresh and len(df) > 500:
            from bot.config import GRANULARITY_SECONDS
            bar_sec = GRANULARITY_SECONDS[granularity]
            age = time.time() - df.index[-1].timestamp()
            if age < 3 * bar_sec:      # cache is fresh enough
                data[pair] = df
                print(f"[deep] {pair}: {len(df)} cached candles "
                      f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
                continue
        print(f"[deep] fetching {days}d {granularity} for {pair}...")
        fresh = fetch_history(pair, granularity, days)
        if len(fresh):
            store.upsert_candles(pair, granularity, fresh)
            df = store.load_candles(pair, granularity)
        data[pair] = df
        print(f"[deep]   {pair}: {len(df)} candles "
              f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
    return {p: df for p, df in data.items() if len(df) > 500}


def write_champions(world: DeepWorld, granularity: str) -> list[dict]:
    """Promote gauntlet survivors with a real, fee-adjusted edge."""
    if not world.best:
        return []
    b = world.best
    promotable = b["excess_pct"] >= MIN_CHAMPION_EXCESS and b["eligible"]
    payload = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "granularity": granularity,
        "promotable": promotable,
        "threshold_excess_pct": MIN_CHAMPION_EXCESS,
        "champion": b,
        "recent_epochs": world.champion_history[-10:],
    }
    os.makedirs(os.path.dirname(CHAMPIONS_PATH), exist_ok=True)
    tmp = CHAMPIONS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    os.replace(tmp, CHAMPIONS_PATH)
    return [b] if promotable else []


def render_board(world: DeepWorld, granularity: str) -> str:
    lines = [
        "# Deep Time board",
        f"Epoch {world.epoch} | {granularity} | updated "
        f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
        f"Evolution zone: bars 0..{world.train_idx[1]} | "
        f"validation gauntlet: bars {world.valid_idx[0]}..{world.valid_idx[1]}",
        "",
    ]
    if world.best:
        b = world.best
        lines.append(
            f"**Champion (epoch {b.get('epoch', '?')}): {b['strategy']}** | "
            f"validation excess **{b['excess_pct']:+.2f}%** vs buy&hold "
            f"(return {b['return_pct']:+.2f}%, bench {b['benchmark_pct']:+.2f}%, "
            f"sharpe {b['sharpe']}, {b['trades']} trades, ${b['fees']:.2f} fees)")
        lines.append(f"params: `{json.dumps(b['params'])}`")
        lines.append("")
    else:
        lines.append("_No eligible champion yet (needs "
                     f"{MIN_VALIDATION_TRADES}+ validation trades)._")
    lines.append(f"Converged: {world.converged} | epochs without improvement: "
                 f"{world.no_improve}")
    hist = world.champion_history[-8:][::-1]
    if hist:
        lines += ["", "| epoch | improved | best excess% | top strategy |",
                  "|---|---|---|---|"]
        for h in hist:
            top = h.get("top") or {}
            lines.append(f"| {h['epoch']} | {'YES' if h['improved'] else 'no'} "
                         f"| {h.get('best_excess_pct') if h.get('best_excess_pct') is not None else '-'} "
                         f"| {top.get('strategy', '-')} "
                         f"({top.get('excess_pct', '-')}%) |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Deep Time evolutionary trainer")
    ap.add_argument("--minutes", type=float, default=300.0,
                    help="wall-clock training budget for this invocation")
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--granularity", default="FOUR_HOUR",
                    choices=["FOUR_HOUR", "SIX_HOUR", "ONE_DAY"])
    ap.add_argument("--pairs", default=DEFAULT_PAIRS)
    ap.add_argument("--agents", type=int, default=40)
    ap.add_argument("--segment-bars", type=int, default=2190,
                    help="bars per selection segment (~1y at 4H)")
    ap.add_argument("--max-epochs", type=int, default=1000)
    ap.add_argument("--db", default=os.path.join(BASE_DIR, "data", "deep.db"))
    ap.add_argument("--refresh", action="store_true",
                    help="force refetch even if the cache is fresh")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    cfg = BotConfig.from_yaml(None)
    fee_cfg = {"taker_fee": cfg.taker_fee, "slippage": cfg.slippage,
               "position_fraction": 0.50, "max_positions": 3,
               "cash_yield_apy": cfg.cash_yield_apy}
    pairs = args.pairs.split(",")
    days = int(args.years * 365)

    if args.report:
        with open(WORLD_PATH, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        data = load_data(pairs, args.granularity, days, args.db)
        world = DeepWorld.load(data, WORLD_PATH, fee_cfg)
        world.bind_data(data)
        board = render_board(world, args.granularity)
        os.makedirs(STATE_DIR, exist_ok=True)
        with open(BOARD_PATH, "w", encoding="utf-8") as fh:
            fh.write(board)
        print(board)
        return

    data = load_data(pairs, args.granularity, days, args.db, args.refresh)
    if os.path.exists(WORLD_PATH):
        world = DeepWorld.load(data, WORLD_PATH, fee_cfg)
        world.bind_data(data)
        print(f"[deep] resumed world at epoch {world.epoch}")
        if world.converged:
            # Diversity restart: keep the global champion genome in the
            # gene pool (it re-mutates), fill the rest with fresh
            # immigrants, and keep exploring -- the champion file already
            # preserves the best-ever tuning, so a restart risks nothing.
            from bot.swarm.genome import Genome
            elites = []
            if world.best and world.best.get("params"):
                elites.append(Genome(id="champion", strategy=world.best["strategy"],
                                     params=dict(world.best["params"]),
                                     lineage=["champion"]))
            world._reseed_from(elites, world.rng_seed + 7919 * world.epoch)
            world.converged = False
            world.no_improve = 0
            print("[deep] converged -> diversity restart with champion elite")
    else:
        world = DeepWorld(data, pairs=pairs, granularity=args.granularity,
                          capital=20.0, fee_cfg=fee_cfg, n_agents=args.agents,
                          segment_bars=args.segment_bars)
        print(f"[deep] new world: {len(world.timeline)} timeline bars, "
              f"train 0..{world.train_idx[1]}, valid "
              f"{world.valid_idx[0]}..{world.valid_idx[1]}")

    deadline = time.monotonic() + args.minutes * 60.0
    epochs_run = 0
    while epochs_run < args.max_epochs and not world.converged:
        if time.monotonic() > deadline - 5.0:
            break
        report = world.run_epoch()
        epochs_run += 1
        top = report.candidates[0] if report.candidates else {}
        print(f"[deep] epoch {report.epoch}: {report.elapsed_sec:.1f}s | "
              f"best candidate {top.get('strategy', '-')} "
              f"excess {top.get('excess_pct', '-')}% "
              f"({top.get('trades', 0)} trades) | "
              f"champion excess "
              f"{world.best['excess_pct'] if world.best else '-'}% | "
              f"no-improve {world.no_improve}/{report.converged and 'CONVERGED'}")
        world.save(WORLD_PATH)   # checkpoint after every epoch
        write_champions(world, args.granularity)

    board = render_board(world, args.granularity)
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(BOARD_PATH, "w", encoding="utf-8") as fh:
        fh.write(board)
    world.save(WORLD_PATH)
    promoted = write_champions(world, args.granularity)
    print(f"[deep] trained {epochs_run} epoch(s); board -> {BOARD_PATH}")
    if promoted:
        print(f"[deep] CHAMPION PROMOTED: {promoted[0]['strategy']} "
              f"{promoted[0]['excess_pct']:+.2f}% excess -> {CHAMPIONS_PATH}")


if __name__ == "__main__":
    main()
