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

from bot.strategies.ml_trend import MLTrendStrategy  # noqa: E402
from bot.train.checkpoint import Checkpoint, load_checkpoint  # noqa: E402
from bot.train.features import FEATURE_COLS, build_features  # noqa: E402
from bot.train.models import (ModelBundle, ModelBundleMeta, RegimeClassifier,  # noqa: E402
                              TimingModel, build_labels)
from bot.train.pipeline import DEFAULT_GRID, TrainConfig, TrainingRun  # noqa: E402

FEE_CFG = {"taker_fee": 0.006, "slippage": 0.001,
           "position_fraction": 0.30, "max_positions": 3}
PASSED = 0


def _synth(n: int = 1300, seed: int = 11) -> pd.DataFrame:
    """Trend + oscillation + noise OHLCV (causal, closed candles)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    closes = (110 + 0.02 * t + 8 * np.sin(t / 40.0)
              + rng.normal(0, 0.5, n).cumsum() * 0.2)
    closes = np.maximum(closes, 10.0)
    idx = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": np.concatenate([[closes[0]], closes[:-1]]),
        "high": closes * 1.004,
        "low": closes * 0.996,
        "close": closes,
        "volume": 1000 + rng.normal(0, 100, n).clip(-500, 500),
    }, index=idx)


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
        idx = pd.date_range("2026-01-01", periods=53, freq="15min", tz="UTC")
        closes = np.array(
            [100.0] * 40                       # calm base
            + [98, 96, 94, 92, 90, 88, 86]     # steady dip to the low
            + [89, 92, 96, 101, 106, 112]      # sharp recovery
        )
        df = pd.DataFrame({
            "open": np.concatenate([[100.0], closes[:-1]]),
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": [1000.0] * 53,
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


def test_zoo_strategies_signals() -> None:
    """Every community model must produce clean causal signals on real
    shaped data (trend + oscillation + noise) without errors."""
    from bot.strategies import REGISTRY
    rng = np.random.default_rng(11)
    n = 400
    t = np.arange(n)
    closes = 100 + 0.05 * t + 6 * np.sin(t / 9.0) + rng.normal(0, 0.6, n).cumsum() * 0.3
    closes = np.maximum(closes, 5.0)
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "open": np.concatenate([[closes[0]], closes[:-1]]),
        "high": closes * 1.004,
        "low": closes * 0.996,
        "close": closes,
        "volume": 1000 + rng.normal(0, 100, n).clip(-500, 500),
    }, index=idx)
    for name, cls in REGISTRY.items():
        strat = cls({})
        sig = strat.compute_signals(df)
        check(f"zoo signals valid: {name}",
              len(sig) == n and set(np.unique(sig.dropna())) <= {-1, 0, 1},
              f"got {set(np.unique(sig))}")


def test_account_overwrite_guard() -> None:
    acc = PaperAccount(capital=1000, taker_fee=0.006, slippage=0.001,
                       position_fraction=0.3, max_positions=3)
    p1 = acc.open_position("BTC-USDC", 100.0, ts=1)
    cash_after_first = acc.cash
    p2 = acc.open_position("BTC-USDC", 105.0, ts=2)
    check("guard: second buy on same pair refused", p2 is None)
    check("guard: cash untouched by refused buy", acc.cash == cash_after_first)
    check("guard: original position intact",
          acc.positions["BTC-USDC"].qty == p1.qty)


def test_dca_bot_lifecycle() -> None:
    """DCA: first tranche, average down on dips, take profit on recovery."""
    from bot.strategies.community import DCABot
    acc = PaperAccount(capital=100.0, taker_fee=0.006, slippage=0.001,
                       position_fraction=0.3, max_positions=3,
                       allow_averaging=True)
    bot = DCABot({})
    idx = pd.date_range("2026-01-01", periods=5, freq="15min", tz="UTC")
    df = pd.DataFrame({"close": [100.0] * 5}, index=idx)
    r1 = bot.execute(acc, "BTC-USDC", df, 100.0, ts=1)
    check("dca: first tranche bought", r1 is not None and r1["action"] == "buy")
    qty1 = acc.positions["BTC-USDC"].qty
    r2 = bot.execute(acc, "BTC-USDC", df, 98.0, ts=2)   # 2% below avg -> average down
    check("dca: averaged down on dip",
          r2 is not None and acc.positions["BTC-USDC"].qty > qty1)
    avg = acc.positions["BTC-USDC"].entry_cost / acc.positions["BTC-USDC"].qty
    r3 = bot.execute(acc, "BTC-USDC", df, avg * 1.025, ts=3)  # above target
    check("dca: took profit", r3 is not None and r3["action"] == "sell")
    check("dca: position closed", "BTC-USDC" not in acc.positions)
    check("dca: profitable round trip", acc.realized_pnl > 0,
          f"pnl={acc.realized_pnl}")
    # stop loss path
    acc2 = PaperAccount(capital=100.0, taker_fee=0.006, slippage=0.001,
                        position_fraction=0.3, allow_averaging=True)
    bot.execute(acc2, "BTC-USDC", df, 100.0, ts=1)
    r4 = bot.execute(acc2, "BTC-USDC", df, 91.0, ts=2)   # > 8% below avg
    check("dca: stop loss fired", r4 is not None and r4["action"] == "sell")


def test_grid_trader_signals() -> None:
    from bot.strategies.community import GridTrader
    n = 200
    closes = np.array([100.0] * 100 + [97.0] * 50 + [103.5] * 50)
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "open": closes, "high": closes + 0.2, "low": closes - 0.2,
        "close": closes, "volume": [1000.0] * n,
    }, index=idx)
    sig = GridTrader({}).compute_signals(df)
    check("grid: buys below median band", (sig.iloc[100:150] == 1).any())
    check("grid: sells above median band", (sig.iloc[150:] == -1).any())


def test_zoo_population_seed() -> None:
    from bot.zoo.roster import ROSTER, build_zoo_population
    pop = build_zoo_population(["BTC-USDC"], "FIFTEEN_MINUTE", 20.0, FEE_CFG)
    check("zoo: one bot per model", len(pop.agents) == len(ROSTER))
    ids = [a.genome.id for a in pop.agents]
    check("zoo: unique model ids", len(set(ids)) == len(ROSTER))
    dca = next(a for a in pop.agents if a.genome.id == "dca_bot")
    check("zoo: dca account allows averaging", dca.account.allow_averaging)
    check("zoo: every model starts with $20",
          all(a.account.cash == 20.0 for a in pop.agents))


def test_features_build() -> None:
    feats = build_features(_synth())
    missing = set(FEATURE_COLS) - set(feats.columns)
    check("features: all documented columns present", not missing, f"got {missing}")
    check("features: index aligned", len(feats) == 1300)
    check("features: calendar cols present",
          {"hour_sin", "hour_cos", "dow_sin", "dow_cos"} <= set(feats.columns))


def test_build_labels() -> None:
    feats = build_features(_synth())
    df = _synth()
    feats = build_features(df)
    lab = build_labels(feats, horizon=6, min_gain=0.013,
                       regime_horizon=24, regime_tol=0.004,
                       close=df["close"])
    check("labels: timing in {0,1,nan}",
          set(lab["timing"].dropna().unique()) <= {0.0, 1.0})
    check("labels: regime in {0,1,2}",
          set(lab["regime"].unique()) <= {0, 1, 2})
    check("labels: timing tail NaN (no future leak)",
          lab["timing"].iloc[-6:].isna().all())


def test_checkpoint_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "chk.json")
        cp = Checkpoint("abc123", {"pairs": ["X-USDC"]})
        cp.mark_pair_fetched("X-USDC", 100, 123)
        cp.mark_cv_done(0, "X-USDC", 0, {"excess%": 1.2})
        cp.save(p)
        cp2 = load_checkpoint(p, "abc123")
        check("checkpoint: roundtrip", cp2 is not None)
        check("checkpoint: cv markers preserved", len(cp2.cv_done) == 1)
        cp3 = load_checkpoint(p, "different")
        check("checkpoint: config-hash mismatch invalidates", cp3 is None)


def test_ml_trend_signals() -> None:
    df = _synth()
    strat = MLTrendStrategy({})
    strat.fit(df)
    check("ml_trend: bundle fitted", strat.bundle is not None)
    sig = strat.compute_signals(df)
    check("ml_trend: signals aligned", len(sig) == len(df))
    vals = set(sig.dropna().unique())
    check("ml_trend: values in {-1,0,1}", vals <= {-1, 0, 1}, f"got {vals}")


def test_model_bundle_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        df = _synth(n=800)
        strat = MLTrendStrategy({})
        strat.fit(df)
        b = strat.bundle
        d = os.path.join(tmp, "bundle")
        b.save(d)
        b2 = ModelBundle.load(d)
        feats = build_features(df)
        cols = [c for c in feats.columns]
        p1 = b.timing.predict(feats[cols])
        p2 = b2.timing.predict(feats[cols])
        check("bundle: load roundtrip", abs(p1.mean() - p2.mean()) < 1e-4)


def test_cv_resume_skips_done() -> None:
    """First run completes CV folds; a second run (same config) skips
    them instead of recomputing, proving resumability."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "t.db")
        cp_path = os.path.join(tmp, "chk.json")
        df = _synth()
        cfg = TrainConfig(pairs=["T-USDC"], granularity="ONE_HOUR", days=400,
                          grid=[dict(DEFAULT_GRID[0])], n_folds=3,
                          db_path=db, checkpoint_path=cp_path,
                          deployed_path=os.path.join(tmp, "dep.json"))
        def _run() -> TrainingRun:
            r = TrainingRun(cfg, budget_sec=1e6)
            r.store.upsert_candles("T-USDC", "ONE_HOUR", df)
            r._cv()
            return r
        r1 = _run()
        done1 = len(r1.cp.cv_done)
        check("cv: folds computed on first run", done1 > 0, f"got {done1}")
        r2 = _run()
        done2 = len(r2.cp.cv_done)
        check("cv: resume skips done folds (no recompute)", done2 == done1,
              f"{done1} -> {done2}")


def main() -> None:
    test_account_math()
    test_next_bar_fills()
    test_mutation_bounds()
    test_selection_cycle()
    test_min_trades_gate()
    test_state_roundtrip()
    test_runner_replay_offline()
    test_sync_window_fill()
    test_zoo_strategies_signals()
    test_account_overwrite_guard()
    test_dca_bot_lifecycle()
    test_grid_trader_signals()
    test_zoo_population_seed()
    test_features_build()
    test_build_labels()
    test_checkpoint_roundtrip()
    test_ml_trend_signals()
    test_model_bundle_roundtrip()
    test_cv_resume_skips_done()
    print(f"\nAll {PASSED} checks passed.")


if __name__ == "__main__":
    main()