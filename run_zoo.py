#!/usr/bin/env python3
"""Strategy zoo: community-classic models raced head-to-head on live data.

10 published strategies (momentum, mean reversion, MACD, golden cross,
Donchian/Turtle breakout, Connors RSI2, stochastic, Bollinger squeeze
breakout, percentage grid, DCA-with-exit) each get their own $20 paper
account and trade the same live Coinbase candles. Every ISO week a
ranking report is written so we can see which published solution
actually works.

Usage:
    python run_zoo.py --hours 5.75     # one trading window (CI)
    python run_zoo.py --report         # print current standings and exit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot.config import BotConfig  # noqa: E402
from bot.swarm.runner import SwarmRunner  # noqa: E402
from bot.zoo.roster import ROSTER, build_zoo_population  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(BASE_DIR, "state", "zoo.json")
META_PATH = os.path.join(BASE_DIR, "state", "zoo.meta.json")
BOARD_PATH = os.path.join(BASE_DIR, "state", "ZOO_BOARD.md")
WEEKLY_DIR = os.path.join(BASE_DIR, "state", "zoo")


def iso_week() -> str:
    d = datetime.now(timezone.utc).isocalendar()
    return f"{d[0]}-W{d[1]:02d}"


def load_meta() -> dict:
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:  # noqa: BLE001
            pass
    return {}


def save_meta(meta: dict) -> None:
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)


def standings_rows(pop) -> list[dict]:
    rows = []
    for agent in pop.leaderboard():
        rows.append({
            "model": agent.genome.id,
            "net_worth": agent.equity,
            "revenue": agent.equity - pop.capital,
            "realized": agent.account.realized_pnl,
            "trades": agent.account.n_trades,
            "fees": agent.account.fee_take,
            "holdings": ", ".join(f"{pos.qty:.6f} {pair.split('-')[0]}"
                                  for pair, pos in agent.account.positions.items()) or "-",
        })
    return rows


def render_board(pop, week: str) -> str:
    lines = [
        "# Zoo leaderboard - community models head-to-head",
        f"Week {week} | updated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC | "
        f"capital ${pop.capital:.2f}/model | ranked by NET WORTH "
        f"(cash + holdings at live prices)",
        "",
        "| rank | model | net worth | revenue | realized | trades | fees | holdings |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(standings_rows(pop), 1):
        lines.append(f"| {i} | {r['model']} | ${r['net_worth']:.2f} | "
                     f"${r['revenue']:+.2f} | ${r['realized']:+.2f} | "
                     f"{r['trades']} | ${r['fees']:.2f} | {r['holdings']} |")
    return "\n".join(lines) + "\n"


def write_weekly_report(pop, week: str) -> str:
    os.makedirs(WEEKLY_DIR, exist_ok=True)
    path = os.path.join(WEEKLY_DIR, f"week-{week}.md")
    rows = standings_rows(pop)
    lines = [
        f"# Zoo weekly report - {week}",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC | "
        f"capital ${pop.capital:.2f}/model",
        "",
        f"**Week's best model: {rows[0]['model']} "
        f"(${rows[0]['net_worth']:.2f}, {rows[0]['revenue']:+.2f} revenue)**",
        "",
        "| rank | model | net worth | revenue | realized | trades | fees |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(f"| {i} | {r['model']} | ${r['net_worth']:.2f} | "
                     f"${r['revenue']:+.2f} | ${r['realized']:+.2f} | "
                     f"{r['trades']} | ${r['fees']:.2f} |")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Strategy zoo paper trading")
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--state", default=STATE_PATH)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--capital", type=float, default=20.0)
    ap.add_argument("--pairs", default="BTC-USDC,ETH-USDC,SOL-USDC,DOGE-USDC,XRP-USDC,ADA-USDC")
    ap.add_argument("--granularity", default="ONE_HOUR")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = BotConfig.from_yaml(None)
    fee_cfg = {"taker_fee": cfg.taker_fee, "slippage": cfg.slippage,
               "position_fraction": 0.50, "max_positions": 3,
               "cash_yield_apy": cfg.cash_yield_apy}
    pairs = args.pairs.split(",")

    if os.path.exists(args.state):
        from bot.swarm.population import Population
        pop = Population.load(args.state, fee_cfg)
    else:
        pop = build_zoo_population(pairs, args.granularity, args.capital, fee_cfg)
        pop.save(args.state)
        print(f"[zoo] seeded {len(ROSTER)} community models x ${args.capital:.0f}")

    runner = SwarmRunner(pairs, args.granularity, pop,
                         poll_seconds=args.poll, verbose=not args.quiet)

    def _render_board_cb() -> None:
        """Re-render ZOO_BOARD.md on every checkpoint so the committed
        board always matches the committed zoo.json (no stale boards)."""
        os.makedirs(os.path.dirname(BOARD_PATH), exist_ok=True)
        with open(BOARD_PATH, "w", encoding="utf-8") as fh:
            fh.write(render_board(pop, week))
    runner.on_save = _render_board_cb

    if args.report:
        runner.sync()
        pop.mark_equity(runner.latest_prices())
        board = render_board(pop, iso_week())
        os.makedirs(os.path.dirname(BOARD_PATH), exist_ok=True)
        with open(BOARD_PATH, "w", encoding="utf-8") as fh:
            fh.write(board)
        print(board)
        return

    # catch up on missed candles, mark at live prices
    runner.sync()
    pop.mark_equity(runner.latest_prices())

    # weekly rollover: write the ranking for the week that just ended
    meta = load_meta()
    week = iso_week()
    if meta.get("week") and meta["week"] != week:
        path = write_weekly_report(pop, meta["week"])
        print(f"[zoo] weekly report for {meta['week']} -> {path}")
    meta["week"] = week
    save_meta(meta)

    if args.hours is None:
        ap.error("--hours is required unless using --report")

    print(f"[zoo] week {week} | {len(pop.agents)} models x ${pop.capital:.0f} | "
          f"{pairs} | {args.granularity} | window {args.hours}h")
    runner.run(args.hours, state_path=args.state, save_every_loops=1)

    pop.mark_equity(runner.latest_prices())
    pop.save(args.state)
    board = render_board(pop, week)
    os.makedirs(os.path.dirname(BOARD_PATH), exist_ok=True)
    with open(BOARD_PATH, "w", encoding="utf-8") as fh:
        fh.write(board)
    print("\n[zoo] window complete. Standings:")
    for i, r in enumerate(standings_rows(pop), 1):
        print(f"{i:2d}. {r['model']:<22} ${r['net_worth']:7.2f} "
              f"({r['revenue']:+.2f})  trades={r['trades']}")


if __name__ == "__main__":
    main()