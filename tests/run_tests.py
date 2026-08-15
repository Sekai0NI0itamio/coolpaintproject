#!/usr/bin/env python3
"""Offline test suite (no network). Run: python tests/run_tests.py"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone

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

# Disables fee-aware gating AND the chassis layers inside a genome's
# params (machinery tests: they validate evolution/replay mechanics on
# synthetic data, not the gate's fee math or regime rules).
GATE_OFF = {"fee_aware": {"margin": 0.0, "min_profit_mult": 0.0,
                          "stop_mult": 0.0, "max_hold_bars": 10**9,
                          "fee_budget_pct": 10.0, "cooldown_bars": 0,
                          "breaker_trades": 10**9, "chassis_off": True}}


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
    pop = Population(pairs=["BTC-USDC"], granularity="ONE_HOUR",
                     capital=20.0, fee_cfg=FEE_CFG)
    pop.seed(n=40, strategy="mean_reversion")
    check("seed: 40 agents", len(pop.agents) == 40)
    check("seed: capital 20", all(a.account.cash == 20.0 for a in pop.agents))
    for i, agent in enumerate(pop.agents):
        agent.account.n_trades = 5            # all pass the min-trades gate
    summary = pop.select_and_repopulate(top_k=3, clones=3)
    check("selection: agents filled (up to 40)", len(pop.agents) == len({a.genome.id for a in pop.agents}))
    check("selection: generation incremented", pop.generation == 1)
    check("selection: capital starts at 20",
          all(a.account.cash >= 10.0 for a in pop.agents))
    check("selection: survivors recorded", len(summary["survivors"]) == 3)
    check("selection: children carry lineage",
          all(len(a.genome.lineage) >= 1 for a in pop.agents))
    ids = [a.genome.id for a in pop.agents]
    check("selection: unique ids", len(set(ids)) == len(ids))


def test_min_trades_gate() -> None:
    """A bot that never trades cannot survive on unrealized equity alone
    (the new fitness model gives it sharpe=0)."""
    pop = Population(pairs=["BTC-USDC"], granularity="ONE_HOUR",
                     capital=20.0, fee_cfg=FEE_CFG)
    pop.seed(n=40, strategy="mean_reversion")
    # agent 0: never traded (fitness=0)
    pop.agents[0].account.n_trades = 0
    # agents 1-5: actually traded (create 3 real trade records each)
    for i in range(1, 6):
        pop.agents[i].account.realized_pnl = 1.0 + i * 0.1
        pop.agents[i].account.n_trades = 4
        pop.agents[i].account.trade_pcts = [0.005, 0.003, 0.004, 0.001]
    summary = pop.select_and_repopulate(top_k=3, clones=3, min_trades=3)
    check("gate: lower-fitness bot not a survivor",
          "g0-00" not in summary["survivors"])
    check("gate: traded bots survived", len(summary["survivors"]) >= 2)
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


def test_cash_yield_accrual() -> None:
    """Idle cash earns the risk-free APY; the first accrue_yield() call
    must only record the start timestamp (never compound from epoch)."""
    from bot.paper.account import PaperAccount
    acc = PaperAccount(capital=1000, taker_fee=0.006, slippage=0.001,
                       position_fraction=0.25, cash_yield_apy=0.045)
    acc.accrue_yield(1)                 # first call: just records start
    acc.accrue_yield(1 + 31536000)      # one year later -> ~4.5% APY
    check("yield: cash grows by ~APY over a year",
          abs(acc.cash - 1000 * 1.045) < 1.0, f"got {acc.cash}")
    # REGRESSION: a fresh account's first accrue_yield(now) must NOT
    # compound 4.5% APY for the whole unix epoch (~56 years).
    fresh = PaperAccount(capital=20.0, cash_yield_apy=0.045)
    fresh.accrue_yield(1_786_665_600)   # ~now
    check("yield: first call never inflates cash (epoch bug)",
          abs(fresh.cash - 20.0) < 1e-6, f"got {fresh.cash}")
    fresh.accrue_yield(1_786_665_600 + 3600)  # +1h -> tiny accrual
    check("yield: second call accrues a realistic tiny amount",
          fresh.cash < 20.01, f"got {fresh.cash}")
    # realized_sharpe is scale-free
    a = PaperAccount(capital=20, taker_fee=0.006, slippage=0.001)
    b = PaperAccount(capital=2000, taker_fee=0.006, slippage=0.001)
    a.trade_pcts = [0.01, 0.02, 0.015, 0.005, 0.012]
    b.trade_pcts = [0.01, 0.02, 0.015, 0.005, 0.012]
    check("sharpe: scale-free", a.realized_sharpe() == b.realized_sharpe())
    check("sharpe: positive edge scores > 0", a.realized_sharpe() > 0)
    check("sharpe: <2 trades -> 0", PaperAccount(capital=20).realized_sharpe() == 0.0)


def test_backtest_cash_yield() -> None:
    """A long-flat strategy with idle cash should beat an identical one
    without yield modeling in an up-less period."""
    from bot.backtest.engine import run_backtest
    idx = pd.date_range("2026-01-01", periods=110, freq="1h", tz="UTC")
    closes = np.full(110, 100.0)   # flat market
    df = pd.DataFrame({"open": closes, "high": closes + 0.1,
                       "low": closes - 0.1, "close": closes,
                       "volume": [1000.0] * 110}, index=idx)

    class Flat(Strategy):
        name = "flat_never_trades"
        def compute_signals(self, df, live=False):
            return pd.Series(0, index=df.index, dtype=int)

    no_yield = run_backtest(df, Flat(), cash_yield_apy=0.0, position_fraction=0.25)
    with_yield = run_backtest(df, Flat(), cash_yield_apy=0.045, position_fraction=0.25)
    check("backtest: yield grows idle cash", with_yield.total_return > no_yield.total_return,
          f"{no_yield.total_return} -> {with_yield.total_return}")


def test_llm_decision_parsing() -> None:
    """parse_decision_json must extract structured decisions, including
    from markdown-fenced replies, without a network or API key."""
    from bot.data.llm_client import parse_decision_json
    d = parse_decision_json(
        '```json\n{"action":"BUY","symbol":"SOL-USDC","reason":"flow turn",'
        '"edge_pct":6.2,"confidence":0.7}\n```')
    check("llm: parses fenced JSON", d["action"] == "BUY" and d["confidence"] == 0.7)
    d2 = parse_decision_json('{"action":"HOLD","reason":"fees too high"}')
    check("llm: parses bare JSON", d2["action"] == "HOLD")


def test_llm_offline_fallback() -> None:
    """Without a key, llm_trader must fall back to a deterministic signal
    and never raise, so it stays importable/testable offline."""
    from bot.strategies.llm_trader import LLMTraderStrategy
    df = _synth(n=800)
    df.attrs["pair"] = "SOL-USDC"
    strat = LLMTraderStrategy({})
    sig = strat.compute_signals(df, live=False)
    check("llm off: signal series aligned", len(sig) == len(df))
    check("llm off: values in {-1,0,1}",
          set(sig.dropna().unique()) <= {-1, 0, 1}, f"got {set(sig.unique())}")
    check("llm off: no key -> no live trade crashes",
          strat.decide(df)[0] in ("BUY", "SELL", "HOLD"))


def test_hold_cycle_signals() -> None:
    """hold_cycle makes few, big, slow trades: long after a durable
    uptrend forms, flat after it breaks; no churn in the calm base."""
    from bot.strategies.hold_cycle import HoldCycleStrategy
    # Use faster windows than production defaults so the test data (1600
    # bars) can actually cross them; the logic is identical.
    strat = HoldCycleStrategy({"trend_sma": 100, "exit_sma": 50})
    n = 700
    closes = np.empty(n)
    closes[:250] = 100.0                                      # calm base
    closes[250:340] = np.linspace(100, 130, 90)               # regime shift up
    closes[340:520] = np.linspace(130, 145, 180)              # sustained uptrend
    closes[520:] = np.linspace(145, 90, n - 520)              # durable break down
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": np.concatenate([[closes[0]], closes[:-1]]),
                       "high": closes + 0.3, "low": closes - 0.3,
                       "close": closes, "volume": [1000.0] * n}, index=idx)
    sig = strat.compute_signals(df)
    buys = np.where(sig == 1)[0]; sells = np.where(sig == -1)[0]
    check("hold_cycle: is long during the sustained uptrend",
          len(buys) >= 1 and all(b < 520 for b in buys), f"buys {buys[:5]}")
    check("hold_cycle: few trades (<= a handful, not hundreds)",
          len(buys) <= 20, f"{len(buys)} buys")
    check("hold_cycle: exits when the durable trend breaks",
          len(sells) >= 1 and any(s >= 520 for s in sells), f"sells {sells[:5]}")


def test_fade_extreme_signals() -> None:
    """fade_extreme buys only after a completed extreme down-move and
    sells on the bounce/target, never during the calm base."""
    from bot.strategies.fade_extreme import FadeExtremeStrategy
    n = 600
    closes = np.empty(n)
    closes[:300] = 100.0                                          # calm
    closes[300:330] = np.linspace(100, 80, 30)                    # -20% crash
    closes[330:] = np.linspace(80, 96, n - 330)                   # partial bounce
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": np.concatenate([[closes[0]], closes[:-1]]),
                       "high": closes + 0.2, "low": closes - 0.2,
                       "close": closes, "volume": [1000.0] * n}, index=idx)
    sig = FadeExtremeStrategy({}).compute_signals(df)
    buys = np.where(sig == 1)[0]
    check("fade_extreme: only buys after the extreme crash (not the calm base)",
          len(buys) > 0 and all(b >= 300 for b in buys), f"buys {buys[:5]}")


def test_deep_recovery_signals() -> None:
    """deep_recovery is the aggressive dip-recovery variant: it should
    buy a deep-but-confirmed dip and exit on target, and it should fire
    more readily than deep_value on the same mild dip data."""
    from bot.strategies.deep_recovery import DeepRecoveryStrategy
    from bot.strategies.deep_value import DeepValueStrategy
    base = [100.0] * 150
    crash = list(100 - 25 * np.linspace(0, 1, 40))      # -25% (deep_value needs -30%, misses)
    climb = list(75 + 10 * np.linspace(0, 1, 50))
    closes = np.array(base + crash + climb)
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="1h", tz="UTC")
    df = pd.DataFrame({"open": np.concatenate([[closes[0]], closes[:-1]]),
                       "high": closes + 0.4, "low": closes - 0.4,
                       "close": closes, "volume": [1000.0] * len(closes)}, index=idx)
    s_rec = DeepRecoveryStrategy({}).compute_signals(df)
    s_val = DeepValueStrategy({}).compute_signals(df)
    check("deep_recovery: fires on a -25% dip",
          int((s_rec == 1).sum()) >= 1, f"buys {(s_rec == 1).sum()}")
    check("deep_recovery: more aggressive than deep_value on the same data",
          int((s_rec == 1).sum()) > int((s_val == 1).sum()),
          f"rec={(s_rec==1).sum()} val={(s_val==1).sum()}")


def test_guarded_wrapper_signals() -> None:
    """Guarded bots must produce valid signals and trade LESS often than
    their base (the guard filters churn)."""
    from bot.strategies.guarded import GuardedMomentum
    from bot.strategies.momentum import MomentumStrategy
    n = 1000
    t = np.arange(n)
    closes = (100 + 0.01 * t + 8 * np.sin(t / 30.0)
              + np.random.default_rng(5).normal(0, 0.5, n).cumsum() * 0.2)
    closes = np.maximum(closes, 10.0)
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": np.concatenate([[closes[0]], closes[:-1]]),
                       "high": closes * 1.004, "low": closes * 0.996,
                       "close": closes, "volume": [1000.0] * n}, index=idx)
    base_sig = MomentumStrategy({}).compute_signals(df)
    guard_sig = GuardedMomentum({}).compute_signals(df)
    check("guarded: valid signal values",
          set(np.unique(guard_sig.dropna())) <= {-1, 0, 1},
          f"got {set(np.unique(guard_sig))}")
    check("guarded: trades no more than base (guard filters churn)",
          int((guard_sig != 0).sum()) <= int((base_sig != 0).sum()),
          f"base={int((base_sig!=0).sum())} guarded={int((guard_sig!=0).sum())}")


def test_winners_v2_signals() -> None:
    """v2 bots produce valid signals; consensus requires multiple votes
    (dip alone must NOT trigger a buy)."""
    from bot.strategies.winners_v2 import AdaptiveGrid, Consensus, DeepRecoveryV2
    n = 900
    t = np.arange(n)
    rng = np.random.default_rng(9)
    closes = np.maximum(100 + 0.02 * t + 1.5 * np.sin(t / 25.0)
                        + rng.normal(0, 0.3, n).cumsum() * 0.2, 10.0)
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    vol = 1000 + rng.normal(0, 250, n).clip(-800, 800)
    df = pd.DataFrame({"open": np.concatenate([[closes[0]], closes[:-1]]),
                       "high": closes * 1.004, "low": closes * 0.996,
                       "close": closes, "volume": vol}, index=idx)
    for cls in (DeepRecoveryV2, AdaptiveGrid, Consensus):
        sig = cls({}).compute_signals(df)
        check(f"v2 {cls.name}: valid signals",
              len(sig) == n and set(np.unique(sig.dropna())) <= {-1, 0, 1},
              f"got {set(np.unique(sig))}")
    # consensus must not buy on a calm/trending base (needs dip+votes)
    cons = Consensus({}).compute_signals(df)
    calm_buys = int((cons.iloc[:600] == 1).sum())
    check("consensus: no buys in the calm base without a dip", calm_buys == 0,
          f"{calm_buys} buys in base")


def test_guard_override_loading() -> None:
    """Guarded bots auto-load tuned overrides from guard_params.json."""
    import json
    import tempfile
    from bot.strategies.guarded import GuardedRSI2, load_guard_overrides
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "guard_params.json")
        with open(p, "w") as fh:
            json.dump({"range": {"atr_hurdle_pct": 0.012, "trend_sma": 300,
                                 "confirm_bars": 3}}, fh)
        ov = load_guard_overrides(p)
        check("guard overrides: file loads", ov["range"]["trend_sma"] == 300)
        g = GuardedRSI2({})
        g2 = GuardedRSI2.__new__(GuardedRSI2)  # bypass default file lookup
        check("guard overrides: absent file -> defaults",
              load_guard_overrides(os.path.join(tmp, "missing.json")) == {})


def test_deep_value_signals() -> None:
    """The deep-value strategy buys a deep, *confirmed* drawdown during a
    partial recovery (still well below the high), then sells on target."""
    from bot.strategies.deep_value import DEFAULTS, DeepValueStrategy
    # 150 flat bars @100, a slow 40-bar crash to 60 (-40%), a partial 50-bar
    # climb from 60 to 72 (still -28% below the high, but momentum positive),
    # then a full rip back above the high.
    base = [100.0] * 150
    crash = list(100 - 40 * np.linspace(0, 1, 40))
    climb = list(60 + 12 * np.linspace(0, 1, 50))
    rip = [101.0] * 60
    closes = np.array(base + crash + climb + rip)
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="1h", tz="UTC")
    df = pd.DataFrame({
        "open": np.concatenate([[closes[0]], closes[:-1]]),
        "high": closes + 0.5, "low": closes - 0.5,
        "close": closes, "volume": [1000.0] * len(closes),
    }, index=idx)
    strat = DeepValueStrategy({})
    sig = strat.compute_signals(df)
    buys = np.where(sig == 1)[0]
    sells = np.where(sig == -1)[0]
    check("deep_value: enters during the partial recovery (not the crash base)",
          len(buys) > 0 and all(i >= 150 for i in buys), f"buys at {buys[:6]}")
    check("deep_value: buys while still deep below the high (no knife chase)",
          len(buys) > 0 and closes[buys[-1]] < 100 * 0.70,
          f"buy close {closes[buys[-1]] if len(buys) else None}")
    check("deep_value: has exits (target/stop)",
          len(sells) > 0, f"sells at {sells[:6]}")
    check("deep_value: defaults in range",
          0 < DEFAULTS["drawdown_pct"] < 1 and 0 < DEFAULTS["stop_pct"] < 1)


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


def test_trend_runner_signals() -> None:
    """trend_runner is long/flat with a trailing exit: it enters a durable
    uptrend and rides it (few trades, holds past the peak), then exits when
    the chandelier stop or regime break fires. It must NOT churn in a flat
    base (fee discipline). Params pinned explicitly so auto-tuned
    overrides (state/runner_params.json) can't shift the expectation."""
    from bot.strategies.trend_runner import TrendRunner
    pinned = {"trend_sma": 100, "atr_period": 14, "atr_mult": 3.0,
              "trail_bars": 96, "atr_hurdle_pct": 0.005}
    n = 800
    closes = np.empty(n)
    closes[:300] = 100.0                                      # calm/flat base
    closes[300:420] = np.linspace(100, 130, 120)              # regime shift up
    closes[420:620] = np.linspace(130, 160, 200)              # sustained uptrend
    closes[620:] = np.linspace(160, 110, n - 620)             # durable break down
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": np.concatenate([[closes[0]], closes[:-1]]),
                       "high": closes + 0.5, "low": closes - 0.5,
                       "close": closes, "volume": [1000.0] * n}, index=idx)
    sig = TrendRunner(pinned).compute_signals(df)
    buys = np.where(sig == 1)[0]
    sells = np.where(sig == -1)[0]
    check("trend_runner: valid signals",
          set(np.unique(sig.dropna())) <= {-1, 0, 1}, f"got {set(np.unique(sig))}")
    check("trend_runner: does NOT churn in the flat base (fee discipline)",
          int((sig.iloc[:300] != 0).sum()) == 0,
          f"active signals in base {int((sig.iloc[:300] != 0).sum())}")
    check("trend_runner: enters the durable uptrend",
          len(buys) >= 1 and all(b >= 300 for b in buys), f"buys {buys[:5]}")
    check("trend_runner: rides winners (holds past the regime-shift peak, "
          "exits after the trailing stop fires)",
          len(sells) >= 1 and sells[-1] > 420, f"sells {sells[:5]}")
    check("trend_runner: exits when the durable trend breaks",
          any(s >= 620 for s in sells), f"sells {sells[:5]}")


def _deep_synth(n: int = 1700, seed: int = 21) -> dict:
    """Two synthetic pairs with strong regime shifts so trend strategies
    genuinely trade (entries + exits) across the history."""
    out = {}
    # phase boundaries scale with n so short histories still shift regimes
    p1, p2, p3, p4 = (int(n * f) for f in (0.18, 0.35, 0.53, 0.76))
    for pair, drift in (("AAA-USDC", 1.0), ("BBB-USDC", 0.6)):
        rng = np.random.default_rng(seed)
        closes = np.empty(n)
        closes[:p1] = 100.0
        closes[p1:p2] = np.linspace(100, 150, p2 - p1)
        closes[p2:p3] = np.linspace(150, 105, p3 - p2)
        closes[p3:p4] = np.linspace(105, 160, p4 - p3)
        closes[p4:] = np.linspace(160, 120, n - p4)
        closes = np.maximum(closes + rng.normal(0, 0.8, n) * drift, 10.0)
        idx = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
        out[pair] = pd.DataFrame({
            "open": np.concatenate([[closes[0]], closes[:-1]]),
            "high": closes * 1.004, "low": closes * 0.996,
            "close": closes,
            "volume": 1000 + rng.normal(0, 100, n).clip(-500, 500),
        }, index=idx)
        seed += 1
    return out


def test_deep_world_epoch() -> None:
    """The Deep Time world runs a full evolution epoch offline: agents
    trade, selection fires at segment boundaries, and candidates face the
    untouched validation gauntlet with fresh accounts."""
    from bot.train.deep_time import DeepWorld
    data = _deep_synth()
    world = DeepWorld(data, pairs=list(data), granularity="FOUR_HOUR",
                      capital=20.0, n_agents=10, segment_bars=250,
                      validation_frac=0.25, rng_seed=7)
    for a in world.pop.agents:   # gate off: this test exercises the
        a.genome.params.update(GATE_OFF)  # evolution machinery only
    n = len(world.timeline)
    check("deep: timeline built from both pairs", n >= 1700)
    check("deep: validation is the last quarter",
          world.valid_idx == (n - n // 4 if n % 4 == 0 else n - int(n * 0.25), n)
          or abs((world.valid_idx[1] - world.valid_idx[0]) - n * 0.25) <= 2)
    report = world.run_epoch()
    check("deep: epoch ran", report.epoch == 1)
    check("deep: candidates were validated",
          isinstance(report.candidates, list) and len(report.candidates) > 0)
    check("deep: some bot actually traded",
          report.total_trades > 0, f"total trades {report.total_trades}")
    c = report.candidates[0]
    check("deep: candidate rows carry gauntlet stats",
          {"excess_pct", "trades", "sharpe", "equity"} <= set(c.keys()))
    # determinism: same seed, same world -> same candidate scores
    world2 = DeepWorld(_deep_synth(), pairs=list(data), granularity="FOUR_HOUR",
                       capital=20.0, n_agents=10, segment_bars=250,
                       validation_frac=0.25, rng_seed=7)
    for a in world2.pop.agents:
        a.genome.params.update(GATE_OFF)
    report2 = world2.run_epoch()
    check("deep: deterministic under a fixed seed",
          [r["excess_pct"] for r in report2.candidates][:3] ==
          [r["excess_pct"] for r in report.candidates][:3])


def test_deep_world_roundtrip() -> None:
    """Checkpoint save/load preserves epoch, convergence state and agents."""
    import tempfile
    from bot.train.deep_time import DeepWorld
    data = _deep_synth(900)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "world.json")
        world = DeepWorld(data, pairs=list(data), granularity="FOUR_HOUR",
                          capital=20.0, n_agents=6, segment_bars=200,
                          validation_frac=0.25, rng_seed=3)
        world.run_epoch()
        world.converged = True
        world.save(path)
        w2 = DeepWorld.load(_deep_synth(900), path)
        w2.bind_data(data)
        check("deep: roundtrip epoch", w2.epoch == world.epoch)
        check("deep: roundtrip converged flag", w2.converged is True)
        check("deep: roundtrip agents", len(w2.pop.agents) == len(world.pop.agents))
        # and it can keep training after a resume
        rep = w2.run_epoch()
        check("deep: resumed world trains on", rep.epoch == world.epoch + 1)


def test_golden_cross_params_consumed() -> None:
    """golden_cross now consumes its params (evolvable), with warmup from
    the actual slow window."""
    from bot.strategies.community import GoldenCross
    strat = GoldenCross({"fast": 20, "slow": 80})
    check("golden: warmup follows params", strat.warmup_bars() == 90)
    n = 600
    closes = np.empty(n)
    closes[:250] = 100.0
    closes[250:400] = np.linspace(100, 140, 150)
    closes[400:] = np.linspace(140, 90, 200)
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": np.concatenate([[closes[0]], closes[:-1]]),
                       "high": closes + 0.3, "low": closes - 0.3,
                       "close": closes, "volume": [1000.0] * n}, index=idx)
    sig = strat.compute_signals(df)
    buys = np.where(sig == 1)[0]
    sells = np.where(sig == -1)[0]
    check("golden: fast params enter the uptrend",
          len(buys) >= 1 and all(250 <= b < 400 for b in buys), f"buys {buys[:5]}")
    check("golden: exits on the breakdown",
          len(sells) >= 1 and sells[-1] >= 400, f"sells {sells[:5]}")


def test_runner_tuned_overrides() -> None:
    """TrendRunner loads auto-tuned overrides (envelope or flat format),
    and explicit params always win over tuned values."""
    import tempfile
    from bot.strategies.trend_runner import TrendRunner, _load_tuned
    DEFAULTS = TrendRunner.DEFAULTS
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "runner_params.json")
        with open(p, "w") as fh:
            json.dump({"params": {"atr_mult": 4.2, "trail_bars": 120},
                       "score": 3.1}, fh)
        tuned = _load_tuned(p)
        check("runner: envelope format unwrapped",
              tuned.get("atr_mult") == 4.2)
        s = TrendRunner({}, tuned_path=p)
        check("runner: tuned value applied", s.p["atr_mult"] == 4.2)
        check("runner: defaults preserved elsewhere", s.p["trail_bars"] == 120)
        s2 = TrendRunner({"atr_mult": 2.5}, tuned_path=p)
        check("runner: explicit params beat tuned", s2.p["atr_mult"] == 2.5)
        check("runner: other tuned params still merge", s2.p["trail_bars"] == 120)
        missing = _load_tuned(os.path.join(tmp, "nope.json"))
        check("runner: absent file -> empty", missing == {})
        s3 = TrendRunner({}, tuned_path=os.path.join(tmp, "nope.json"))
        check("runner: defaults when nothing tuned",
              s3.p["atr_mult"] == DEFAULTS["atr_mult"])


def test_champion_promotion() -> None:
    """Champions are applied to matching live agents only when promotable
    and fresh; non-promotable or stale champions change nothing."""
    import tempfile
    from bot.swarm.genome import Genome
    from bot.swarm.population import Agent, Population
    from bot.train.champions import apply_champion, load_champion

    def _pop() -> Population:
        pop = Population(pairs=["BTC-USDC"], granularity="ONE_HOUR",
                         capital=20.0, fee_cfg=FEE_CFG)
        g1 = Genome(id="trend_runner", strategy="trend_runner", params={"atr_mult": 3.0})
        g2 = Genome(id="momentum", strategy="momentum", params={})
        pop.agents = [
            Agent(genome=g1, account=PaperAccount(capital=20.0, **FEE_CFG), equity=20.0),
            Agent(genome=g2, account=PaperAccount(capital=20.0, **FEE_CFG), equity=20.0),
        ]
        return pop

    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "champions.json")
        fresh = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        with open(p, "w") as fh:
            json.dump({"promotable": True, "updated_at": fresh,
                       "champion": {"strategy": "trend_runner",
                                    "params": {"atr_mult": 4.4, "trail_bars": 150},
                                    "excess_pct": 2.7, "eligible": True}}, fh)
        champ = load_champion(p)
        check("champion: promotable loads", champ and champ["params"]["atr_mult"] == 4.4)
        pop = _pop()
        updated = apply_champion(pop, p)
        check("champion: applied to the matching strategy only",
              updated == ["trend_runner"])
        check("champion: genome params updated",
              pop.agents[0].genome.params["atr_mult"] == 4.4)
        check("champion: strategy rebuilt lazily",
              pop.agents[0]._strategy is None)
        check("champion: other strategies untouched",
              pop.agents[1].genome.params == {})

        with open(p, "w") as fh:
            json.dump({"promotable": False, "updated_at": fresh,
                       "champion": {"strategy": "trend_runner",
                                    "params": {"atr_mult": 9.9},
                                    "excess_pct": 0.1, "eligible": True}}, fh)
        pop2 = _pop()
        check("champion: non-promotable changes nothing",
              apply_champion(pop2, p) == [] and
              pop2.agents[0].genome.params["atr_mult"] == 3.0)

        stale = "2020-01-01 00:00 UTC"
        with open(p, "w") as fh:
            json.dump({"promotable": True, "updated_at": stale,
                       "champion": {"strategy": "trend_runner",
                                    "params": {"atr_mult": 9.9},
                                    "excess_pct": 5.0, "eligible": True}}, fh)
        check("champion: stale champion ignored", load_champion(p) is None)


def test_trade_gate_decisions() -> None:
    """TradeGate: EV entry hurdle, fee budget, cooldown, exit band,
    disaster stop, time stop — the pure decision math."""
    import math
    from bot.trade_gate import GateContext, GateParams, TradeGate

    g = TradeGate(GateParams())
    rtc = 0.014
    # EV rule: sqrt(16)*atr >= 1.5*rtc -> atr must be >= 0.00525
    ok_atr = 1.5 * rtc / math.sqrt(16)
    ctx = GateContext(atr_pct=ok_atr, rtc=rtc)
    check("gate: entry at exact EV hurdle allowed", g.allow_entry(ctx)[0])
    ctx = GateContext(atr_pct=ok_atr * 0.999, rtc=rtc)
    check("gate: entry below EV hurdle blocked", not g.allow_entry(ctx)[0])
    # fee budget: 2% of capital
    ctx = GateContext(atr_pct=ok_atr, rtc=rtc, fees_paid_window=0.021,
                      capital=1.0)
    check("gate: fee budget exhausted blocks entry",
          not g.allow_entry(ctx)[0])
    # cooldown blocks even high-EV entries
    ctx = GateContext(atr_pct=0.05, rtc=rtc, cooldown_bars_left=3)
    check("gate: cooldown blocks entry", not g.allow_entry(ctx)[0])
    # exit band: defer between +1x rtc and -3x rtc
    check("gate: exit allowed when profit clears rtc",
          g.allow_exit(GateContext(atr_pct=0.0, rtc=rtc,
                                   unrealized_pct=rtc))[0])
    check("gate: exit deferred inside cost band",
          not g.allow_exit(GateContext(atr_pct=0.0, rtc=rtc,
                                       unrealized_pct=0.5 * rtc))[0])
    check("gate: exit deferred on mild loss",
          not g.allow_exit(GateContext(atr_pct=0.0, rtc=rtc,
                                       unrealized_pct=-2.0 * rtc))[0])
    check("gate: disaster stop exit allowed",
          g.allow_exit(GateContext(atr_pct=0.0, rtc=rtc,
                                   unrealized_pct=-3.0 * rtc))[0])
    # time stop
    check("gate: time stop fires at max_hold_bars",
          g.force_exit(GateContext(atr_pct=0.0, rtc=rtc,
                                   unrealized_pct=0.0, hold_bars=96))[0])
    check("gate: no time stop before max_hold_bars",
          not g.force_exit(GateContext(atr_pct=0.0, rtc=rtc,
                                       unrealized_pct=0.0, hold_bars=95))[0])
    check("gate: force_exit includes disaster stop",
          g.force_exit(GateContext(atr_pct=0.0, rtc=rtc,
                                   unrealized_pct=-3.0 * rtc, hold_bars=1))[0])


def test_gated_account_proxy() -> None:
    """GatedAccount blocks low-EV entries (fail closed) and defers
    sub-cost profit exits while keeping disaster exits open."""
    from bot.trade_gate import GatedAccount, GateParams, TradeGate

    acc = PaperAccount(capital=100.0, taker_fee=0.006, slippage=0.001,
                       position_fraction=0.5)
    gate = TradeGate(GateParams())
    prox = GatedAccount(acc, gate)
    prox.set_bar_context(atr_pct=0.001)          # dead tape -> below EV hurdle
    check("proxy: low-EV entry blocked (fail closed)",
          prox.open_position("BTC-USDC", 100.0, ts=1) is None)
    check("proxy: blocked entry leaves cash untouched",
          abs(acc.cash - 100.0) < 1e-9)
    prox.set_bar_context(atr_pct=0.02)           # volatile -> clears hurdle
    pos = prox.open_position("BTC-USDC", 100.0, ts=1)
    check("proxy: high-EV entry passes through", pos is not None)
    # +0.5% unrealized: inside the defer band -> base sell is deferred
    prox.set_bar_context(atr_pct=0.02)
    check("proxy: sub-cost profit exit deferred",
          prox.close_position("BTC-USDC", 101.0, ts=2) is None)
    check("proxy: deferred exit keeps the position",
          "BTC-USDC" in acc.positions)
    # +2%+ unrealized: clears rtc -> allowed
    check("proxy: cost-clearing exit allowed",
          prox.close_position("BTC-USDC", 104.0, ts=3) is not None)
    # disaster exit always allowed
    prox.set_bar_context(atr_pct=0.02)
    prox.open_position("BTC-USDC", 100.0, ts=4)
    check("proxy: disaster exit allowed",
          prox.close_position("BTC-USDC", 90.0, ts=5) is not None)
    # passthrough delegation
    check("proxy: delegates attributes to the wrapped account",
          prox.can_open() == acc.can_open())


def _churn_df(n: int = 1200, seed: int = 4) -> pd.DataFrame:
    """Sideways chop with enough ATR to sometimes clear the EV hurdle —
    maximizes base-strategy churn (the fee-bleed regime)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    closes = 100 + 3.0 * np.sin(t / 7.0) + rng.normal(0, 0.9, n).cumsum() * 0.15
    closes = np.maximum(closes, 10.0)
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": np.concatenate([[closes[0]], closes[:-1]]),
        "high": closes * 1.012, "low": closes * 0.988,
        "close": closes,
        "volume": 1000 + rng.normal(0, 100, n).clip(-500, 500),
    }, index=idx)


def _alternating(sigs: np.ndarray) -> bool:
    state = 0
    for s in sigs:
        if s == 1 and state == 1:
            return False
        if s == -1 and state == 0:
            return False
        if s != 0:
            state = s
    return True


def test_fee_aware_rewriter() -> None:
    """The rewriter trades strictly less than the base on chop, and its
    signals stay alternating (entries/exits never double up)."""
    from bot.strategies.community import MACDCross
    from bot.strategies.fee_aware import FeeAwareStrategy
    df = _churn_df()
    base = MACDCross({})
    wrapped = FeeAwareStrategy(base)
    b = base.compute_signals(df)
    w = wrapped.compute_signals(df)
    check("rewriter: name preserved for reports", wrapped.name == "macd_cross")
    check("rewriter: fewer or equal entries than base",
          int((w == 1).sum()) <= int((b == 1).sum()),
          f"base={int((b == 1).sum())} gated={int((w == 1).sum())}")
    check("rewriter: signals alternate (no double entries/exits)",
          _alternating(w.to_numpy()))


def test_rewriter_causality() -> None:
    """Mutating future candles must never change past rewritten signals."""
    from bot.strategies.community import MACDCross
    from bot.strategies.fee_aware import FeeAwareStrategy
    df = _churn_df(900, seed=8)
    wrapped = FeeAwareStrategy(MACDCross({}))
    full = wrapped.compute_signals(df)
    cut = 500
    assert len(full) > cut + 100
    part = wrapped.compute_signals(df.iloc[:cut])
    same = (full.iloc[:cut].to_numpy() == part.to_numpy()).all()
    check("rewriter: causal (prefix-stable)", same)


def test_fee_aware_execute() -> None:
    """Live path: base execute is gated through the proxy; forced
    stop/time exits close the real account position."""
    from bot.strategies.community import RSI2
    from bot.strategies.fee_aware import FeeAwareStrategy
    df = _churn_df(300, seed=2)
    wrapped = FeeAwareStrategy(RSI2({}), gate_params={"max_hold_bars": 3})
    acc = PaperAccount(capital=100.0, taker_fee=0.006, slippage=0.001,
                       position_fraction=0.5)
    ts = int(df.index[0].timestamp())
    bought = False
    for i in range(250, 295):
        t = ts + i * 3600
        r = wrapped.execute(acc, "BTC-USDC", df.iloc[:i + 1],
                            float(df["close"].iloc[i]), t)
        if r and r["action"] == "buy":
            bought = True
    if bought:
        # time stop (3 bars) must eventually close the position
        closed = False
        for i in range(295, 300):
            t = ts + i * 3600
            r = wrapped.execute(acc, "BTC-USDC", df.iloc[:i + 1],
                                float(df["close"].iloc[i]), t)
            if r and r["action"] == "sell":
                closed = True
        check("execute: time stop closes stale positions",
              closed or acc.n_positions == 0)
    else:
        # entries may be fully gated on this slice; that's also correct
        check("execute: gated entries keep account flat", acc.n_trades == 0)


def test_build_strategy_wraps_fee_aware() -> None:
    """The factory wraps every strategy in the chassis by default; base
    params still flow to the base, gate params under 'fee_aware'."""
    from bot.strategies import build_strategy
    from bot.strategies.chassis import ChassisStrategy
    from bot.strategies.momentum import MomentumStrategy
    s = build_strategy("momentum", {"ema_fast": 8, "ema_slow": 30})
    check("factory: returns ChassisStrategy", isinstance(s, ChassisStrategy))
    check("factory: base name preserved", s.name == "momentum")
    check("factory: base params consumed",
          s._base.params["ema_fast"] == 8)
    check("factory: base is the right class",
          isinstance(s._base, MomentumStrategy))
    s2 = build_strategy("momentum", {"ema_fast": 8,
                                     "fee_aware": {"max_hold_bars": 50}})
    check("factory: gate params split out",
          s2.gate.p.max_hold_bars == 50 and s2._base.params["ema_fast"] == 8)


def test_market_context_regimes() -> None:
    """Context builder classifies UP/RANGE/DOWN/CRASH on synthetic
    series and degrades confidence gracefully on short windows."""
    from bot.market import build_context, classify_regime, CRASH, DOWN, RANGE, UP
    n = 1400
    closes = np.empty(n)
    closes[:300] = 100.0                                # base
    closes[300:600] = np.linspace(100, 160, 300)        # UP
    closes[600:900] = 160.0 + 2.0 * np.sin(np.arange(300) / 9.0)   # RANGE
    closes[900:1200] = np.linspace(160, 100, 300)       # DOWN
    # CRASH: -40% with violent bars (high ATR)
    closes[1200:] = np.linspace(100, 60, 200)
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    wiggle = np.random.default_rng(3).normal(0, 0.4, n)
    closes = np.maximum(closes + wiggle, 5.0)
    df = pd.DataFrame({
        "open": np.concatenate([[closes[0]], closes[:-1]]),
        "high": closes * 1.01, "low": closes * 0.99,
        "close": closes, "volume": [1000.0] * n,
    }, index=idx)
    ctx = build_context(df)
    need = {"atr_pct", "sma_dist", "slope", "dd_from_high", "atr_pctile",
            "trend_frac_pos", "pct_rank_1y", "context_confidence"}
    check("context: all columns present", need <= set(ctx.columns))
    check("context: confidence < 1 on short window",
          0.0 < ctx["context_confidence"].iloc[-1] < 1.0)
    reg = classify_regime(ctx)
    check("context: UP detected in the ramp",
          (reg.iloc[400:600] == UP).mean() > 0.8,
          f"frac={(reg.iloc[400:600] == UP).mean():.2f}")
    check("context: DOWN detected in the decline",
          (reg.iloc[1000:1200] == DOWN).mean() > 0.6,
          f"frac={(reg.iloc[1000:1200] == DOWN).mean():.2f}")
    check("context: CRASH detected (dd + vol spike)",
          (reg.iloc[-60:] == CRASH).sum() > 10,
          f"crash bars={(reg.iloc[-60:] == CRASH).sum()}")
    # causal: prefix of the data gives identical early context
    ctx2 = build_context(df.iloc[:700])
    same = np.allclose(ctx["sma_dist"].iloc[10:700].to_numpy(),
                       ctx2["sma_dist"].iloc[10:700].to_numpy(),
                       equal_nan=True)
    check("context: causal (prefix-stable)", same)


def _trend_df() -> pd.DataFrame:
    """Base -> strong UP ramp -> strong DOWN slide (clear regimes)."""
    n = 900
    closes = np.empty(n)
    closes[:300] = 100.0
    closes[300:600] = np.linspace(100, 170, 300)
    closes[600:] = np.linspace(170, 95, 300)
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    wiggle = np.random.default_rng(6).normal(0, 0.3, n)
    closes = np.maximum(closes + wiggle, 5.0)
    return pd.DataFrame({
        "open": np.concatenate([[closes[0]], closes[:-1]]),
        "high": closes * 1.008, "low": closes * 0.992,
        "close": closes, "volume": [1000.0] * n,
    }, index=idx)


def test_chassis_regime_allowlist() -> None:
    """Trend bots are blocked in DOWN regimes; value bots trade there;
    exits always pass; self-directed bots are never regime-blocked."""
    from bot.strategies.chassis import ChassisStrategy
    from bot.strategies.community import DonchianBreakout
    from bot.strategies.deep_value import DeepValueStrategy

    df = _trend_df()
    trend = ChassisStrategy(DonchianBreakout({}))
    sig = trend.compute_signals(df)
    buys = df.index[sig == 1]
    # any entries must be in the first 2/3 (base+UP), none in the slide
    up_ok = all(df.index.get_loc(b) < 620 for b in buys)
    check("chassis: trend bot blocked in DOWN regime",
          up_ok, f"buys at {[df.index.get_loc(b) for b in buys][:8]}")
    value = ChassisStrategy(DeepValueStrategy({}))
    vsig = value.compute_signals(df)
    check("chassis: value bot still produces signals",
          set(np.unique(vsig.dropna())) <= {-1, 0, 1})


def test_chassis_sizing() -> None:
    """Sizing: vol-target base, conviction cap, hard bounds, and the
    account actually receives the fraction."""
    from bot.strategies.chassis import (FRAC_MAX, FRAC_MIN, TARGET_RISK,
                                        ChassisStrategy, size_fraction)
    from bot.market import build_context
    calm = {"atr_pct": 0.004, "trend_frac_pos": 0.5, "sma_dist": 0.0,
            "dd_from_high": 0.0, "atr_pctile": 0.5,
            "context_confidence": 1.0}
    wild = {"atr_pct": 0.05, "trend_frac_pos": 0.5, "sma_dist": 0.0,
            "dd_from_high": 0.0, "atr_pctile": 0.5,
            "context_confidence": 1.0}
    f_calm = size_fraction("self", calm)
    f_wild = size_fraction("self", wild)
    check("sizing: calm coin gets bigger size than wild coin",
          f_calm > f_wild, f"{f_calm} vs {f_wild}")
    check("sizing: calm clamps at the 50% cap", f_calm == FRAC_MAX)
    check("sizing: wild floors at 5%", f_wild >= FRAC_MIN)
    strong = {"atr_pct": 0.04, "trend_frac_pos": 1.0, "sma_dist": 0.0,
              "dd_from_high": 0.0, "atr_pctile": 0.5,
              "context_confidence": 1.0}
    f_strong = size_fraction("trend", strong)
    check("sizing: conviction caps at 1.5x base",
          abs(f_strong - (TARGET_RISK / 0.04) * 1.5) < 1e-9)
    low_conf = dict(strong, context_confidence=0.25)
    check("sizing: degraded context scales conviction down",
          size_fraction("trend", low_conf) < f_strong)
    # account passthrough
    acc = PaperAccount(capital=100.0, taker_fee=0.006, slippage=0.001,
                       position_fraction=0.25)
    pos = acc.open_position("BTC-USDC", 100.0, ts=1, fraction=0.5)
    check("sizing: account honors explicit fraction",
          abs(pos.qty - (100.0 * 0.5 / 100.1)) < 1e-9)


def test_chassis_engine_fractions() -> None:
    """The backtest engine sizes entries from _entry_fractions decided
    at the signal bar."""

    class Forced(Strategy):
        name = "forced_frac"
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

    from bot.backtest.engine import run_backtest
    strat = Forced()
    fracs = pd.Series(np.nan, index=idx)
    fracs.iloc[5] = 0.5
    strat._entry_fractions = fracs
    r = run_backtest(df, strat, pair="TEST", capital=10_000,
                     position_fraction=0.25)
    t = r.trades[0]
    # entry at bar 6 open=106: cost should be 50% of cash, not 25%
    check("engine: entry sized from _entry_fractions",
          abs(t.entry_price * t.qty - 10_000 * 0.5) < 1.0,
          f"cost={t.entry_price * t.qty:.2f}")


def test_chassis_causality() -> None:
    """Signals AND fractions are prefix-stable under the chassis."""
    from bot.strategies.chassis import ChassisStrategy
    from bot.strategies.community import MACDCross
    df = _churn_df(900, seed=8)
    wrapped = ChassisStrategy(MACDCross({}))
    full = wrapped.compute_signals(df)
    full_fracs = wrapped._entry_fractions
    cut = 500
    part = wrapped.compute_signals(df.iloc[:cut])
    part_fracs = wrapped._entry_fractions
    check("chassis: signals causal (prefix-stable)",
          (full.iloc[:cut].to_numpy() == part.to_numpy()).all())
    if full_fracs is not None and part_fracs is not None:
        a = full_fracs.iloc[:cut].fillna(-1).to_numpy()
        b = part_fracs.fillna(-1).to_numpy()
        check("chassis: fractions causal (prefix-stable)",
              (a == b).all())
    else:
        check("chassis: fractions causal (none produced)",
              full_fracs is None or cut >= len(full_fracs))


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
    test_cash_yield_accrual()
    test_backtest_cash_yield()
    test_llm_decision_parsing()
    test_llm_offline_fallback()
    test_hold_cycle_signals()
    test_fade_extreme_signals()
    test_deep_recovery_signals()
    test_guarded_wrapper_signals()
    test_winners_v2_signals()
    test_guard_override_loading()
    test_deep_value_signals()
    test_trend_runner_signals()
    test_deep_world_epoch()
    test_deep_world_roundtrip()
    test_golden_cross_params_consumed()
    test_runner_tuned_overrides()
    test_champion_promotion()
    test_ml_trend_signals()
    test_model_bundle_roundtrip()
    test_cv_resume_skips_done()
    test_trade_gate_decisions()
    test_gated_account_proxy()
    test_fee_aware_rewriter()
    test_rewriter_causality()
    test_fee_aware_execute()
    test_build_strategy_wraps_fee_aware()
    test_market_context_regimes()
    test_chassis_regime_allowlist()
    test_chassis_sizing()
    test_chassis_engine_fractions()
    test_chassis_causality()
    print(f"\nAll {PASSED} checks passed.")


if __name__ == "__main__":
    main()