#!/usr/bin/env python3
"""Live paper trading on real Coinbase data with pretend USDC.

Usage:
    python run_paper.py                          # until Ctrl-C (default)
    python run_paper.py --hours 12 --capital 10000
    python run_paper.py --strategies momentum,mean_reversion --pairs BTC-USDC
    python run_paper.py --report                 # print stored P&L, no trading

The bot polls the public Coinbase candles API (free, no keys), runs the
configured strategies on each new closed candle, and records all fills,
trades and equity into SQLite. Run it for a few hours/days, then compare
strategy P&L against the buy-hold baseline with --report.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot.config import BotConfig  # noqa: E402
from bot.data.store import Store  # noqa: E402
from bot.paper.engine import PaperEngine  # noqa: E402
from bot.paper.ledger import load_account  # noqa: E402
from bot.report.report import (paper_report, plot_paper_equity,  # noqa: E402
                               save_paper_report)
from bot.strategies import build_strategies  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper trade with pretend USDC on live Coinbase data")
    ap.add_argument("--config", default=None)
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--granularity", default=None)
    ap.add_argument("--strategies", default=None, help="comma-separated names (default: all)")
    ap.add_argument("--capital", type=float, default=None, help="pretend USDC balance")
    ap.add_argument("--hours", type=float, default=None, help="run duration in hours")
    ap.add_argument("--poll", type=int, default=None, help="REST poll interval in seconds")
    ap.add_argument("--report", action="store_true", help="print paper report and exit")
    args = ap.parse_args()

    config = BotConfig.from_yaml(args.config)
    from bot.trade_gate import set_fee_model
    set_fee_model(config.taker_fee, config.slippage)
    if args.pairs:
        config.pairs = args.pairs.split(",")
    if args.granularity:
        config.granularity = args.granularity
    if args.capital:
        config.paper_capital = args.capital
    if args.poll:
        config.poll_seconds = args.poll
    if args.strategies:
        config.strategies = {name: config.strategies.get(name, {})
                             for name in args.strategies.split(",")}

    store = Store(config.db_path)

    if args.report:
        names = list(config.strategies) if config.strategies else ["momentum",
                                                                   "mean_reversion", "ml"]
        text = paper_report(store, names, config.pairs)
        print(text)
        path = save_paper_report(text, config.out_dir)
        png = plot_paper_equity(store, names, config.out_dir)
        print(f"\nReport saved: {path}")
        if png:
            print(f"Equity chart: {png}")
        return

    strategies = build_strategies(config.strategies)
    engine = PaperEngine(config, strategies, store)
    try:
        asyncio.run(engine.run(duration_hours=args.hours))
    except KeyboardInterrupt:
        print("\n[paper] stopped by user")
    finally:
        for strategy in engine.strategies:
            acc = load_account(store, strategy.name)
            if acc is not None:
                print(f"[paper] {strategy.name}: cash=${acc.cash:,.2f} "
                      f"realized P&L=${acc.realized_pnl:,.2f} "
                      f"({acc.n_trades} closed, {acc.n_positions} open)")


if __name__ == "__main__":
    main()