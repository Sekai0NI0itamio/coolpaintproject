#!/usr/bin/env python3
"""Backtest every configured strategy on real Coinbase USDC history.

Usage:
    python run_backtest.py                 # uses strategies.yaml defaults
    python run_backtest.py --days 730 --granularity FOUR_HOUR --strategies momentum,ml
    python run_backtest.py --pairs BTC-USDC,ETH-USDC

Flow:
  1. Download (paginated) public Coinbase candles into SQLite.
  2. For TA strategies: backtest the full window, report OOS = last 30%.
  3. For ML strategies: fit on the first 70% (train), backtest the last
     30% (held-out test) with a purge gap -- honest out-of-sample.
  4. Print comparison table + save equity PNG/JSON.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot.backtest.engine import BacktestResult, run_backtest  # noqa: E402
from bot.backtest.walkforward import split_train_test  # noqa: E402
from bot.config import BotConfig  # noqa: E402
from bot.data.fetcher import fetch_history  # noqa: E402
from bot.data.store import Store  # noqa: E402
from bot.report.report import (backtest_table, plot_backtest_equity,  # noqa: E402
                               save_backtest_results)
from bot.strategies import build_strategies  # noqa: E402
from bot.strategies.base import Strategy  # noqa: E402


def _load_or_fetch(config: BotConfig, store: Store,
                   pairs: List[str]) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        df = store.load_candles(pair, config.granularity)
        if len(df) < config.history_days * (24 * 3600 // config.candle_seconds) // 2:
            print(f"[backtest] downloading {config.history_days}d of {pair} "
                  f"({config.granularity})...")
            df = fetch_history(pair, config.granularity, config.history_days)
            store.upsert_candles(pair, config.granularity, df)
            df = store.load_candles(pair, config.granularity)
        data[pair] = df
        print(f"[backtest] {pair}: {len(df)} candles "
              f"({df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d})")
    return data


def _run_ta_backtest(df: pd.DataFrame, strategy: Strategy, config: BotConfig) -> List[BacktestResult]:
    full = run_backtest(df, strategy, pair=df.attrs.get("pair"),
                        taker_fee=config.taker_fee, slippage=config.slippage,
                        position_fraction=config.position_fraction,
                        max_positions=config.max_positions,
                        capital=config.paper_capital,
                        cash_yield_apy=config.cash_yield_apy)
    _, test = split_train_test(df, train_frac=0.7)
    if len(test) > 200:
        test = test.copy()
        test.attrs["pair"] = df.attrs.get("pair")
            oos = run_backtest(test, strategy, pair=df.attrs.get("pair"),
                               taker_fee=config.taker_fee, slippage=config.slippage,
                               position_fraction=config.position_fraction,
                               max_positions=config.max_positions,
                               capital=config.paper_capital,
                               cash_yield_apy=config.cash_yield_apy)
        oos.strategy = f"{strategy.name}[OOS]"
        return [full, oos]
    return [full]


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest strategies on Coinbase USDC history")
    ap.add_argument("--config", default=None, help="path to strategies.yaml")
    ap.add_argument("--pairs", default=None, help="comma-separated product ids, e.g. BTC-USDC")
    ap.add_argument("--granularity", default=None,
                    help="ONE_MINUTE..ONE_DAY (default from config)")
    ap.add_argument("--days", type=int, default=None, help="days of history to fetch")
    ap.add_argument("--strategies", default=None, help="comma-separated strategy names")
    args = ap.parse_args()

    config = BotConfig.from_yaml(args.config)
    if args.pairs:
        config.pairs = args.pairs.split(",")
    if args.granularity:
        config.granularity = args.granularity
    if args.days:
        config.history_days = args.days
    if args.strategies:
        config.strategies = {name: config.strategies.get(name, {})
                             for name in args.strategies.split(",")}

    store = Store(config.db_path)
    data = _load_or_fetch(config, store, config.pairs)

    results: List[BacktestResult] = []
    for pair, df in data.items():
        df = df.copy()
        df.attrs["pair"] = pair
        for strategy in build_strategies(config.strategies):
            if strategy.name == "ml":
                train, test = split_train_test(df, train_frac=0.7)
                strategy.fit(train)
                if len(test) > 200:
                    test = test.copy()
                    test.attrs["pair"] = pair
                        r = run_backtest(test, strategy, pair=pair,
                                         taker_fee=config.taker_fee,
                                         slippage=config.slippage,
                                         position_fraction=config.position_fraction,
                                         max_positions=config.max_positions,
                                         capital=config.paper_capital,
                                         cash_yield_apy=config.cash_yield_apy)
                    r.strategy = "ml[OOS]"
                    results.append(r)
                    print(f"[backtest] ml {pair}: fitted on {len(train)} bars, "
                          f"OOS on {len(test)} bars")
            else:
                results.extend(_run_ta_backtest(df, strategy, config))

    print()
    print(backtest_table(results))
    json_path = save_backtest_results(results, config.out_dir)
    png_path = plot_backtest_equity(results, config.out_dir)
    print(f"\nResults saved to:\n  {json_path}\n  {png_path}")


if __name__ == "__main__":
    main()