#!/usr/bin/env python3
"""Offline test suite (no network). Run: python tests/run_tests.py"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bot.backtest.engine import run_backtest  # noqa: E402
from bot.paper.account import PaperAccount  # noqa: E402
from bot.strategies.base import Strategy  # noqa: E402
from bot.swarm.genome import PARAM_BOUNDS, Genome, make_genome, mutate  # noqa: E402
from bot.swarm.population import Agent, Population  # noqa: E402
from bot.swarm.runner import SwarmRunner  # noqa: E402
import bot.swarm.runner as R  # noqa: E402
from bot.data.store import Store  # noqa: E402

import random  # noqa: E402

FEE_CFG = {"taker_fee": 0.006, "slippage": 0.001,
           "position_fraction": 0.30, "max_positions": 3}
PASSED = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED
    if not cond:
        print(f"FAIL: {name} {detail}")
        sys.exit(1)
    PASSED += 1
    print(f"PASS: {name}")


def test_account_math() -> None:
    acc = PaperAccount(capital=10000, taker_fee=0.006, slippage=0.001,
                       position_fraction=0.25)
    pos = acc.open_position("BTC-USDC", 100.0, ts=1)
    check("account: qty uses slippage fill", abs(pos.qty - 2500 / 100.1) < 1e-9)
    check("account: cash debited incl fee",
          abs(acc.cash - (10000 - 2500 - 15)) < 1e-6)
    closed = acc.close_position("BTC-USDC", 110.0, ts=2)
    proceeds = pos.qty * 109.89 - pos.qty * 109.89 * 0.006
    check("account: pnl matches hand calc",
          abs(closed["pnl"] - (proceeds - (pos.entry_cost + pos.entry_fee))) < 1e-6)


def test_next_bar_fills() -> None:
    class Forced(Strategy):
        name = "forced"

        def compute_signals(self, df, live=False):
            s = pd.Series(0, index=df.index)
            s.iloc[5] = 1
            s.iloc[10] = -1
            return s

    idx = pd.date_range("2026-01-01", periods=20, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "open": [100 + i for i in range(20)],
        "high": [101 + i for i in range(20)],
        "low": [99 + i for i in range(20)],
        "close": [100.5 + i for i in range(20)],
        "volume": [1000.0] * 20,
    }, index=idx)
    r = run_backtest(df, Forced(), pair="TEST")
    check("backtest: exactly one round trip", r.n_trades == 1)
    t = r.trades[0]
    check("backtest: entry at NEXT bar open",
          t.entry_ts == idx[6] and abs(t.entry_price - 106 * 1.001) < 1e-9)
    check("backtest: exit at NEXT bar open",
          t.exit_ts == idx[11] and abs(t.exit_price - 111 * 0.999) < 1e-9)


def test_mutation_bounds() -> None:
    rng = random.Random(3)
    for strategy in PARAM_BOUNDS:
        g = make_genome(strategy, "root", rng)
        cur = g
        for i in range(200):
            cur = mutate(cur, f"c{i}", rng, strength=0.2)
            for name, (lo, hi, typ) in PARAM_BOUNDS[strategy].items():
                v = cur.params[name]
                check(f"mutation bounds {strategy}.{name}", lo <= v <= hi,
                      f"got {v}")
            if strategy == "momentum":
                check("mutation: ema_fast < ema_slow",
                      cur.params["ema_fast"] < cur.params["ema_slow"])
            if strategy == "mean_reversion":
                check("mutation: oversold < exit_rsi < overbought",
                      cur.params["oversold"] < cur.params["exit_rsi"]
                      < cur.params["overbought"])
    check("mutation: lineage tracked", len(cur.lineage) >= 1)


def test_selection_cycle() -> None:
    pop = Population(pairs=["BTC-USDC"], granularity="FIFTEEN_MINUTE",
                     capital=20.0, fee_cfg=FEE_CFG)
    pop.seed(n=40, strategy="mean_reversion")
    check("seed: 40 agents", len(pop.agents) == 40)
    check("seed: capital 20", all(a.account.cash == 20.0 for a in pop.agents))
    for i, agent in enumerate(pop.agents):
        agent.equity = 20.0 + (40 - i) * 0.1  # deterministic ranking
        agent.account.n_trades = 5            # all pass the min-trades gate
    summary = pop.select_and_repopulate(top_k=5, clones=7, immigrants=5)
    check("selection: back to 40 agents", len(pop.agents) == 40)
    check("selection: generation incremented", pop.generation == 1)
    check("selection: capital reset to 20",
          all(a.account.cash == 20.0 for a in pop.agents))
    check("selection: survivors recorded", len(summary["survivors"]) == 5)
    check("selection: children carry lineage",
          all(len(a.genome.lineage) >= 1 for a in pop.agents))
    immigrants = [a for a in pop.agents if a.genome.lineage == ["immigrant"]]
    check("selection: 5 immigrants present", len(immigrants) == 5)
    ids = [a.genome.id for a in pop.agents]
    check("selection: unique ids", len(set(ids)) == 40)


def test_min_trades_gate() -> None:
    """A bot that never trades ('never sell' trick) cannot survive on
    unrealized equity alone."""
    pop = Population(pairs=["BTC-USDC"], granularity="FIFTEEN_MINUTE",
                     capital=20.0, fee_cfg=FEE_CFG)
    pop.seed(n=40, strategy="mean_reversion")
    # agent 0: richest by far but never traded (hoarder)
    pop.agents[0].equity = 100.0
    pop.agents[0].account.n_trades = 0
    # agents 1-5: modest gains, actually traded
    for i in range(1, 6):
        pop.agents[i].equity = 21.0 + i * 0.1
        pop.agents[i].account.n_trades = 4
    summary = pop.select_and_repopulate(top_k=5, clones=7, immigrants=5,
                                        min_trades=3)
    hoarder_standing = next(r for r in summary["final_standings"]
                            if r["id"] == "g0-00")
    check("gate: hoarder marked ineligible", hoarder_standing["eligible"] is False)
    check("gate: hoarder not a survivor",
          "g0-00" not in summary["survivors"])
    check("gate: traded bots survived", len(summary["survivors"]) == 5)
    check("gate: disqualified count reported",
          summary["disqualified_no_trades"] >= 35)


def test_state_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "pop.json")
        pop = Population(pairs=["BTC-USDC", "ETH-USDC"],
                         granularity="FIFTEEN_MINUTE", capital=20.0,
                         fee_cfg=FEE_CFG)
        pop.seed(n=40)
        pop.agents[0].account.open_position("BTC-USDC", 50000.0, ts=123)
        pop.last_ts = 999
        pop.generation = 2
        pop.save(path)
        pop2 = Population.load(path, fee_cfg=FEE_CFG)
        check("roundtrip: agents", len(pop2.agents) == 40)
        check("roundtrip: generation", pop2.generation == 2)
        check("roundtrip: last_ts", pop2.last_ts == 999)
        check("roundtrip: open position restored",
              "BTC-USDC" in pop2.agents[0].account.positions)
        check("roundtrip: fee cfg restored",
              pop2.agents[0].account.taker_fee == 0.006)


def test_runner_replay_offline() -> None:
    """Inject synthetic candles directly into the store and verify the
    swarm executes a full buy-low/sell-high round trip through the
    virtual account tool."""
    with tempfile.TemporaryDirectory() as tmp:
        idx = pd.date_range("2026-01-01", periods=60, freq="15min", tz="UTC")
        closes = np.array([100.0] * 40 + [97, 94, 91, 88, 86]
                          + list(np.linspace(88, 101, 15)))
        df = pd.DataFrame({
            "open": np.concatenate([[100.0], closes[:-1]]),
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": [1000.0] * 60,
        }, index=idx)
        pop = Population(pairs=["TEST-USDC"], granularity="FIFTEEN_MINUTE",
                         capital=20.0, fee_cfg=FEE_CFG)
        genome = Genome(id="t-00", strategy="mean_reversion", params={
            "rsi_period": 14, "bb_period": 20, "bb_std": 2.0,
            "oversold": 30, "overbought": 70, "exit_rsi": 55})
        pop.agents = [Agent(genome=genome,
                            account=PaperAccount(capital=20.0, **FEE_CFG),
                            equity=20.0)]
        runner = SwarmRunner(["TEST-USDC"], "FIFTEEN_MINUTE", pop,
                             db_path=os.path.join(tmp, "t.db"), verbose=False)
        runner.store.upsert_candles("TEST-USDC", "FIFTEEN_MINUTE", df)
        for ts in (int(t.timestamp()) for t in idx):
            runner._step_candle("TEST-USDC", ts)
        acc = pop.agents[0].account
        check("runner: bot bought the dip and sold the recovery",
              acc.n_trades >= 1, f"trades={acc.n_trades}")
        check("runner: profitable round trip after fees",
              acc.realized_pnl > 0, f"pnl={acc.realized_pnl}")
        check("runner: fees were charged", acc.fee_take > 0)


def test_sync_window_fill() -> None:
    """Even with a recent last_ts (resuming run on a fresh CI runner),
    sync must fetch a full indicator window, not just 1 bar."""
    from bot.config import GRANULARITY_SECONDS
    bar_sec = GRANULARITY_SECONDS["FIFTEEN_MINUTE"]
    pop = Population(pairs=["BTC-USDC"], granularity="FIFTEEN_MINUTE",
                     capital=20.0, fee_cfg=FEE_CFG)
    pop.seed(n=2)
    now = 1_800_000_000 - (1_800_000_000 % bar_sec)
    pop.last_ts = now - 3600  # resumed 1h ago
    runner = SwarmRunner(["BTC-USDC"], "FIFTEEN_MINUTE", pop,
                         db_path=":memory:", verbose=False)
    start = runner._sync_start_ts(now)
    check("sync: fetches full window on resume",
          now - start >= (R.WINDOW_BARS - 1) * bar_sec,
          f"got {(now - start) // bar_sec} bars")
    pop.last_ts = 0
    start = runner._sync_start_ts(now)
    check("sync: fresh seed fetches full backfill",
          now - start >= (R.FETCH_BACK_BARS - 1) * bar_sec)


def main() -> None:
    test_account_math()
    test_next_bar_fills()
    test_mutation_bounds()
    test_selection_cycle()
    test_min_trades_gate()
    test_state_roundtrip()
    test_runner_replay_offline()
    test_sync_window_fill()
    print(f"\nAll {PASSED} checks passed.")


if __name__ == "__main__":
    main()