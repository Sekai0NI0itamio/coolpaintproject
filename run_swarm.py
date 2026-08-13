#!/usr/bin/env python3
"""Evolutionary swarm paper-trading on live Coinbase data.

40 bots, each with $20 of pretend USDC, all trading the same strategy
type with different tunings. Every UTC day the bottom 35 are killed,
the top 5 are cloned 8x with slight mutations, and the new generation
starts again with $20 each.

Designed to run inside GitHub Actions in ~5.75h windows (4 per day);
state persists in a small JSON file committed to the repo, and any
missed time is automatically replayed ("gap-fill") on the next run.

Usage:
    python run_swarm.py --hours 5.75            # one trading window (CI)
    python run_swarm.py --hours 0.05 --poll 5   # quick local smoke test
    python run_swarm.py --select                # force daily selection now
    python run_swarm.py --leaderboard           # print standings and exit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot.config import BotConfig  # noqa: E402
from bot.swarm.population import (CLONES_PER_SURVIVOR, DEFAULT_CAPITAL,  # noqa: E402
                                  POP_SIZE, TOP_K, Population)
from bot.swarm.runner import SwarmRunner  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STATE = os.path.join(BASE_DIR, "state", "population.json")
SEEDS_PATH = os.path.join(BASE_DIR, "state", "seeds.json")
HISTORY_DIR = os.path.join(BASE_DIR, "state", "history")
LEADERBOARD_PATH = os.path.join(BASE_DIR, "state", "LEADERBOARD.md")


def load_seeds() -> list[dict] | None:
    if not os.path.exists(SEEDS_PATH):
        return None
    try:
        with open(SEEDS_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        seeds = data.get("seeds") or []
        return seeds or None
    except Exception:  # noqa: BLE001
        return None


def render_daily_report(summary: dict, capital: float) -> str:
    lines = [
        f"# Swarm day report - generation {summary['generation_finished']} finished",
        f"Selection at {summary['date']} | starting capital ${capital:.2f}/bot",
        "",
        "## Final standings (top 10)",
        "| rank | bot | strategy | equity | pnl | trades |",
        "|---|---|---|---|---|---|",
    ]
    for i, row in enumerate(summary["final_standings"][:10], 1):
        lines.append(f"| {i} | {row['id']} | {row['strategy']} | "
                     f"${row['equity']:.2f} | ${row['pnl']:+.2f} | {row['trades']} |")
    lines += ["", f"**Survivors (cloned 8x into generation "
              f"{summary['generation_finished'] + 1}):** "
              + ", ".join(summary["survivors"]), ""]
    return "\n".join(lines)


def write_leaderboard(pop: Population) -> None:
    os.makedirs(os.path.dirname(LEADERBOARD_PATH), exist_ok=True)
    lines = [
        f"# Swarm leaderboard",
        f"Generation {pop.generation} | day {pop.day} | "
        f"updated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC | "
        f"capital ${pop.capital:.2f}/bot",
        "",
        "| rank | bot | strategy | equity | pnl | trades | tuning |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, agent in enumerate(pop.leaderboard(), 1):
        pnl = agent.equity - pop.capital
        params = ", ".join(f"{k}={v}" for k, v in sorted(agent.genome.params.items()))
        lines.append(f"| {i} | {agent.genome.id} | {agent.genome.strategy} | "
                     f"${agent.equity:.2f} | ${pnl:+.2f} | {agent.account.n_trades} | "
                     f"{params} |")
    with open(LEADERBOARD_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evolutionary swarm paper trading")
    ap.add_argument("--hours", type=float, default=None, help="window length in hours")
    ap.add_argument("--state", default=DEFAULT_STATE)
    ap.add_argument("--poll", type=int, default=60, help="seconds between data polls")
    ap.add_argument("--bots", type=int, default=POP_SIZE)
    ap.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    ap.add_argument("--strategy", default="mean_reversion",
                    help="seed strategy type (mean_reversion|momentum)")
    ap.add_argument("--pairs", default="BTC-USDC,ETH-USDC,SOL-USDC")
    ap.add_argument("--granularity", default="FIFTEEN_MINUTE")
    ap.add_argument("--select", action="store_true", help="force daily selection now")
    ap.add_argument("--leaderboard", action="store_true", help="print standings and exit")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = BotConfig.from_yaml(None)
    fee_cfg = {"taker_fee": cfg.taker_fee, "slippage": cfg.slippage,
               "position_fraction": 0.30, "max_positions": 3}
    pairs = args.pairs.split(",")

    seeds = load_seeds()
    pop = Population.load_or_seed(args.state, pairs=pairs, granularity=args.granularity,
                                  capital=args.capital, fee_cfg=fee_cfg,
                                  n=args.bots, strategy=args.strategy, seeds=seeds)
    if seeds and pop.generation == 0 and not os.path.exists(args.state + ".seeded"):
        print(f"[swarm] seeded generation 0 from {len(seeds)} trained seeds "
              f"({args.bots} bots x ${args.capital:.0f})")

    if args.select:
        summary = pop.select_and_repopulate(TOP_K, CLONES_PER_SURVIVOR)
        pop.day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        os.makedirs(HISTORY_DIR, exist_ok=True)
        path = os.path.join(HISTORY_DIR, f"day-{summary['date'][:10]}.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(render_daily_report(summary, args.capital))
        pop.save(args.state)
        print(f"[swarm] selection done -> generation {pop.generation}, report: {path}")
        return

    if args.leaderboard:
        runner = SwarmRunner(pairs, args.granularity, pop, verbose=False)
        pop.mark_equity(runner.latest_prices())
        write_leaderboard(pop)
        for i, agent in enumerate(pop.leaderboard()[:15], 1):
            print(f"{i:2d}. {agent.genome.id}  ${agent.equity:8.2f}  "
                  f"({agent.equity - pop.capital:+.2f})  trades={agent.account.n_trades}")
        return

    # daily rollover (selection) if the UTC day changed since the last run
    summary = pop.maybe_rollover(TOP_K, CLONES_PER_SURVIVOR)
    if summary:
        os.makedirs(HISTORY_DIR, exist_ok=True)
        path = os.path.join(HISTORY_DIR, f"day-{summary['date'][:10]}.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(render_daily_report(summary, args.capital))
        pop.save(args.state)
        print(f"[swarm] new UTC day -> selected top {TOP_K}, generation "
              f"{pop.generation} started ({len(pop.agents)} bots)")

    if args.hours is None:
        ap.error("--hours is required unless using --select/--leaderboard")

    runner = SwarmRunner(pairs, args.granularity, pop,
                         poll_seconds=args.poll, verbose=not args.quiet)
    print(f"[swarm] generation {pop.generation} | {len(pop.agents)} bots x "
          f"${pop.capital:.0f} | {pairs} | {args.granularity} | window {args.hours}h")
    runner.run(args.hours, state_path=args.state)

    pop.mark_equity(runner.latest_prices())
    pop.save(args.state)
    write_leaderboard(pop)
    print("\n[swarm] window complete. Current standings (top 10):")
    for i, agent in enumerate(pop.leaderboard()[:10], 1):
        print(f"{i:2d}. {agent.genome.id}  ${agent.equity:8.2f}  "
              f"({agent.equity - pop.capital:+.2f})  trades={agent.account.n_trades}")


if __name__ == "__main__":
    main()