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
                                  IMMIGRANTS, MIN_TRADES, POP_SIZE, SHARPE_WINDOW,
                                  TOP_K, Population)
from bot.swarm.runner import SwarmRunner  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STATE = os.path.join(BASE_DIR, "state", "population.json")
SEEDS_PATH = os.path.join(BASE_DIR, "state", "seeds.json")
DEPLOYED_PATH = os.path.join(BASE_DIR, "state", "deployed_model.json")
HISTORY_DIR = os.path.join(BASE_DIR, "state", "history")
LEADERBOARD_PATH = os.path.join(BASE_DIR, "state", "LEADERBOARD.md")


def load_seeds() -> list[dict] | None:
    seeds: list[dict] = []
    # 1) ML-regime-gate deployment (the primary candidate we trust).
    if os.path.exists(DEPLOYED_PATH):
        try:
            with open(DEPLOYED_PATH, "r", encoding="utf-8") as fh:
                dep = json.load(fh)
            if dep.get("deployment_source") == "ml_regime_gate" and dep.get("hyper"):
                seeds.append({"strategy": "ml_trend", "params": dict(dep["hyper"])})
                print(f"[swarm] seeding ml_trend from deployed model "
                      f"({dep.get('trained_at', '?')})")
        except Exception:  # noqa: BLE001
            pass
    # 2) Classic seed grid (diagnostic diversity).
    if os.path.exists(SEEDS_PATH):
        try:
            with open(SEEDS_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            seeds.extend(data.get("seeds") or [])
        except Exception:  # noqa: BLE001
            pass
    return seeds or None


def render_daily_report(summary: dict, capital: float) -> str:
    lines = [
        f"# Swarm day report - generation {summary['generation_finished']} finished",
        f"Selection at {summary['date']} | starting capital ${capital:.2f}/bot | "
        f"min-trades gate: {summary.get('min_trades_gate', MIN_TRADES)} | "
        f"fitness = realized fee-paid per-trade Sharpe over window of {SHARPE_WINDOW}",
        f"Disqualified (too few trades): {summary.get('disqualified_no_trades', 0)}"
        + (f" | fallback survivors used: {summary['fallback_survivors']}"
           if summary.get("fallback_survivors") else ""),
        "",
        "## Final standings by fitness (top 10)",
        "| rank | bot | strategy | fitness | sharpe_20 | net worth | realized | trades | eligible |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, row in enumerate(summary["final_standings"][:10], 1):
        holdings = ", ".join(f"{q:.6f} {p.split('-')[0]}"
                             for p, q in row.get("holdings", {}).items()) or "-"
        ls = (f"| {i} | {row['id']} | {row['strategy']} | "
              f"{row['fitness']:.4f} | {row['sharpe_20']:.4f} | "
              f"${row['net_worth']:.2f} | ${row['realized_pnl']:+.2f} | "
              f"{row['trades']} | {'yes' if row['eligible'] else 'NO'} |")
        lines.append(ls)
    survivors = summary.get("survivors", [])
    lines += ["", f"**Survivors (cloned x{CLONES_PER_SURVIVOR + 1}, including self, plus immigrants from all strategy families):** "
              + ", ".join(survivors), ""]
    return "\n".join(lines)


def write_leaderboard(pop: Population) -> None:
    os.makedirs(os.path.dirname(LEADERBOARD_PATH), exist_ok=True)
    lines = [
        f"# Swarm leaderboard",
        f"Generation {pop.generation} | day {pop.day} | "
        f"updated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC | "
        f"capital ${pop.capital:.2f}/bot | ranked by REALIZED SHARPE "
        f"(realized fee-paid per-trade return consistency)",
        "",
        "| rank | bot | fitness | sharpe_20 | net worth | realized | trades | holdings |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, agent in enumerate(pop.leaderboard(), 1):
        holdings = ", ".join(f"{pos.qty:.6f} {pair.split('-')[0]}"
                             for pair, pos in agent.account.positions.items()) or "-"
        nw = agent.account.equity({})  # mark-to-market with no prices -> just cash
        lines.append(f"| {i} | {agent.genome.id} | "
                     f"{agent.fitness:.4f} | "
                     f"{agent.account.realized_sharpe(20):.4f} | "
                     f"${agent.account.cash:.2f} | "
                     f"${agent.account.realized_pnl:+.2f} | "
                     f"{agent.account.n_trades} | {holdings} |")
    with open(LEADERBOARD_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evolutionary swarm paper trading")
    ap.add_argument("--hours", type=float, default=None, help="window length in hours")
    ap.add_argument("--state", default=DEFAULT_STATE)
    ap.add_argument("--poll", type=int, default=60, help="seconds between data polls")
    ap.add_argument("--bots", type=int, default=POP_SIZE)
    ap.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    ap.add_argument("--strategy", default="ml_trend",
                    help="seed strategy type (ml_trend|mean_reversion|momentum)")
    ap.add_argument("--pairs", default="BTC-USDC,ETH-USDC,SOL-USDC,DOGE-USDC,XRP-USDC,ADA-USDC")
    ap.add_argument("--granularity", default="ONE_HOUR")
    ap.add_argument("--select", action="store_true", help="force daily selection now")
    ap.add_argument("--leaderboard", action="store_true", help="print standings and exit")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = BotConfig.from_yaml(None)
    from bot.trade_gate import set_fee_model
    set_fee_model(cfg.taker_fee, cfg.slippage)
    fee_cfg = {"taker_fee": cfg.taker_fee, "slippage": cfg.slippage,
               "position_fraction": 0.30, "max_positions": 3,
               "cash_yield_apy": cfg.cash_yield_apy}
    pairs = args.pairs.split(",")

    seeds = load_seeds()
    pop = Population.load_or_seed(args.state, pairs=pairs, granularity=args.granularity,
                                  capital=args.capital, fee_cfg=fee_cfg,
                                  n=args.bots, strategy=args.strategy, seeds=seeds)
    if seeds and pop.generation == 0 and not os.path.exists(args.state + ".seeded"):
        print(f"[swarm] seeded generation 0 from {len(seeds)} trained seeds "
              f"({args.bots} bots x ${args.capital:.0f})")

    if args.select:
        runner = SwarmRunner(pairs, args.granularity, pop, verbose=False)
        runner.sync()                              # finish any pending candles first
        pop.mark_equity(runner.latest_prices())    # rank by fresh net worth
        summary = pop.select_and_repopulate(TOP_K, CLONES_PER_SURVIVOR,
                                            IMMIGRANTS, MIN_TRADES)
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

    runner = SwarmRunner(pairs, args.granularity, pop,
                         poll_seconds=args.poll, verbose=not args.quiet)

    # Catch up on any candles since the last run (gap-fill) so the OLD
    # generation finishes its trading, then mark everyone at LIVE prices
    # and only then perform the daily selection on fresh net worth.
    runner.sync()
    pop.mark_equity(runner.latest_prices())

    summary = pop.maybe_rollover(TOP_K, CLONES_PER_SURVIVOR, IMMIGRANTS, MIN_TRADES)
    if summary:
        os.makedirs(HISTORY_DIR, exist_ok=True)
        path = os.path.join(HISTORY_DIR, f"day-{summary['date'][:10]}.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(render_daily_report(summary, args.capital))
        pop.save(args.state)
        print(f"[swarm] new UTC day -> selected top {TOP_K} (min {MIN_TRADES} trades), "
              f"generation {pop.generation} started ({len(pop.agents)} bots, "
              f"{IMMIGRANTS} immigrants)")

    if args.hours is None:
        ap.error("--hours is required unless using --select/--leaderboard")

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