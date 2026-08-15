# Fee-Aware Trading System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every bot (zoo, swarm, backtest, paper) refuses trades whose expected move can't clear the ~1.4% round-trip cost, never realizes sub-cost "wins", and auto-pauses when it starts bleeding fees.

**Architecture:** One `TradeGate` policy module (`bot/trade_gate.py`) holds all fee math. Two adapters consume it: a causal signal rewriter inside `FeeAwareStrategy` (backtest path) and a `GatedAccount` proxy (live zoo/paper path). Default-on wiring happens in the single strategy factory `build_strategy`, which every entry point already uses.

**Tech Stack:** Python 3 (stdlib + pandas/numpy already in requirements), house test runner `python tests/run_tests.py` (check()-style, no pytest).

**Spec:** `docs/superpowers/specs/2026-08-15-fee-aware-trading-design.md`

---

### Task 1: FeeAwareConfig in config.py + strategies.yaml section

**Files:**
- Modify: `bot/config.py`
- Modify: `strategies.yaml`

- [ ] **Step 1: Add FeeAwareConfig dataclass to bot/config.py** (after `BotConfig`)

```python
@dataclass
class FeeAwareConfig:
    """Gate constants for the fee-aware trading system (see
    docs/superpowers/specs/2026-08-15-fee-aware-trading-design.md).

    Fixed by principle — the auto-tuner must NOT tune these
    (Bailey/Lopez de Prado overfit guard).
    """
    margin: float = 1.5            # expected move must clear rtc * margin
    expected_hold_bars: int = 16   # sqrt-scaled horizon for the EV estimate
    min_profit_mult: float = 1.0   # profit exits need >= rtc * this
    stop_mult: float = 3.0         # disaster stop at -rtc * this
    max_hold_bars: int = 96        # time stop
    breaker_trades: int = 8        # trailing gross-loss window
    cooldown_bars: int = 24        # entry pause after breaker trips
    fee_budget_pct: float = 0.02   # max fees per 24h as fraction of capital
    position_fraction: float = 0.25  # sizing used for the fee-budget ledger

    @classmethod
    def from_yaml(cls, path: str | None = None) -> "FeeAwareConfig":
        path = path or DEFAULT_CONFIG
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw: Dict[str, Any] = (yaml.safe_load(fh) or {}).get("fee_aware") or {}
        except Exception:  # noqa: BLE001 - defaults on any config problem
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        cfg = cls()
        for key, value in raw.items():
            if key in known and value is not None:
                setattr(cfg, key, value)
        return cfg
```

- [ ] **Step 2: Add the yaml section to strategies.yaml** (after the `slippage:` line)

```yaml
# Fee-aware gate (see docs/superpowers/specs/2026-08-15-fee-aware-trading-design.md).
# Constants are principled fixed values — do NOT tune these per strategy.
fee_aware:
  margin: 1.5              # expected move must clear rtc x margin
  expected_hold_bars: 16   # sqrt-scaled horizon for the EV estimate
  min_profit_mult: 1.0     # profit exits need >= rtc x this
  stop_mult: 3.0           # disaster stop at -rtc x this
  max_hold_bars: 96        # time stop (bars)
  breaker_trades: 8        # trailing gross-loss window
  cooldown_bars: 24        # entry pause after breaker trips
  fee_budget_pct: 0.02     # max fees per 24h as fraction of capital
```

- [ ] **Step 3: Verify it loads**

Run: `venv/bin/python -c "from bot.config import FeeAwareConfig; c=FeeAwareConfig.from_yaml(); print(c.margin, c.max_hold_bars)"`
Expected: `1.5 96`

- [ ] **Step 4: Commit**

```bash
git add bot/config.py strategies.yaml
git commit -m "feat: FeeAwareConfig dataclass + yaml section"
```

---

### Task 2: TradeGate core (bot/trade_gate.py)

**Files:**
- Create: `bot/trade_gate.py`
- Test: `tests/run_tests.py` (house check() style, add `test_trade_gate_decisions` and register in `main()`)

- [ ] **Step 1: Write the failing test** (add to tests/run_tests.py, and add `test_trade_gate_decisions()` to `main()`)

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python -c "import tests.run_tests as t; t.test_trade_gate_decisions()"`
Expected: FAIL with ModuleNotFoundError (bot.trade_gate)

- [ ] **Step 3: Create bot/trade_gate.py**

```python
"""Fee-aware trade gate: the single source of truth for cost-aware
trade decisions (see docs/superpowers/specs/2026-08-15-fee-aware-trading-design.md).

Every bot's entries must clear the round-trip cost with margin, exits
never realize a sub-cost "win", and repeated fee-bleeding pauses the
bot. The gate itself is STATELESS — adapters (signal rewriter for
backtests, GatedAccount proxy for live) own all state.

Failure asymmetry (adapters): entries fail CLOSED (broken math never
opens a position), exits fail OPEN (selling is always possible).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

from bot.config import FeeAwareConfig

# Module fee model — entry points sync this from strategies.yaml so the
# gate's math always matches the engine's fills. Defaults = verified
# Coinbase Advanced Trade retail tier (0.6% taker) + 0.1% slippage.
_TAKER_FEE = 0.006
_SLIPPAGE = 0.001


def set_fee_model(taker_fee: float, slippage: float) -> None:
    """Sync the gate's cost model with the run's actual config."""
    global _TAKER_FEE, _SLIPPAGE
    _TAKER_FEE = float(taker_fee)
    _SLIPPAGE = float(slippage)


def round_trip_cost() -> float:
    """Full round-trip cost as a fraction (2 x (taker + slippage))."""
    return 2.0 * (_TAKER_FEE + _SLIPPAGE)


@dataclass
class GateParams:
    margin: float = 1.5
    expected_hold_bars: int = 16
    min_profit_mult: float = 1.0
    stop_mult: float = 3.0
    max_hold_bars: int = 96
    breaker_trades: int = 8
    cooldown_bars: int = 24
    fee_budget_pct: float = 0.02
    position_fraction: float = 0.25

    @classmethod
    def from_config(cls, cfg: Optional[FeeAwareConfig] = None) -> "GateParams":
        c = cfg or FeeAwareConfig.from_yaml()
        return cls(margin=c.margin, expected_hold_bars=c.expected_hold_bars,
                   min_profit_mult=c.min_profit_mult, stop_mult=c.stop_mult,
                   max_hold_bars=c.max_hold_bars, breaker_trades=c.breaker_trades,
                   cooldown_bars=c.cooldown_bars, fee_budget_pct=c.fee_budget_pct,
                   position_fraction=c.position_fraction)


@dataclass
class GateContext:
    atr_pct: float              # ATR(14)/price on the decision bar
    rtc: float                  # round-trip cost fraction
    unrealized_pct: float = 0.0
    hold_bars: int = 0
    recent_gross_pcts: List[float] = field(default_factory=list)
    fees_paid_window: float = 0.0
    capital: float = 0.0
    cooldown_bars_left: int = 0


@dataclass
class TradeGate:
    """Stateless fee-brain. Pure decisions over a GateContext."""

    p: GateParams = field(default_factory=lambda: GateParams.from_config())

    def allow_entry(self, ctx: GateContext) -> Tuple[bool, str]:
        if ctx.cooldown_bars_left > 0:
            return False, f"circuit-breaker cooldown ({ctx.cooldown_bars_left} bars left)"
        if ctx.capital > 0 and ctx.fees_paid_window > self.p.fee_budget_pct * ctx.capital:
            return False, "fee budget exhausted (24h window)"
        hurdle = self.p.margin * ctx.rtc
        expected = math.sqrt(self.p.expected_hold_bars) * ctx.atr_pct
        if expected < hurdle:
            return False, (f"EV: expected move {expected * 100:.2f}% "
                           f"< hurdle {hurdle * 100:.2f}%")
        return True, "entry clears round-trip cost"

    def allow_exit(self, ctx: GateContext) -> Tuple[bool, str]:
        if ctx.unrealized_pct >= self.p.min_profit_mult * ctx.rtc:
            return True, "profit clears round-trip cost"
        if ctx.unrealized_pct <= -self.p.stop_mult * ctx.rtc:
            return True, "disaster stop"
        return False, (f"exit deferred: {ctx.unrealized_pct * 100:+.2f}% "
                       f"inside cost band")

    def force_exit(self, ctx: GateContext) -> Tuple[bool, str]:
        if ctx.unrealized_pct <= -self.p.stop_mult * ctx.rtc:
            return True, "disaster stop"
        if ctx.hold_bars >= self.p.max_hold_bars:
            return True, f"time stop ({ctx.hold_bars} bars)"
        return False, ""
```

- [ ] **Step 4: Run the test, verify PASS**

Run: `venv/bin/python -c "import tests.run_tests as t; t.test_trade_gate_decisions()"`
Expected: series of PASS lines

- [ ] **Step 5: Commit**

```bash
git add bot/trade_gate.py tests/run_tests.py
git commit -m "feat: TradeGate — stateless fee-aware decision core"
```

---

### Task 3: GatedAccount proxy (in bot/trade_gate.py)

**Files:**
- Modify: `bot/trade_gate.py` (append class)
- Test: `tests/run_tests.py` (add `test_gated_account_proxy`, register in `main()`)

- [ ] **Step 1: Write the failing test**

```python
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
    # +2% unrealized: clears rtc -> allowed
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python -c "import tests.run_tests as t; t.test_gated_account_proxy()"`
Expected: FAIL with ImportError (GatedAccount)

- [ ] **Step 3: Append GatedAccount to bot/trade_gate.py**

```python
class GatedAccount:
    """Live/zoo adapter: wraps a PaperAccount so every open/close the
    base strategy requests routes through the TradeGate.

    State (recent gross returns, 24h fee ledger, cooldown) lives here,
    in-process; the recent-returns window is re-seeded from the wrapped
    account's trade history so the circuit breaker survives CI restarts.
    """

    def __init__(self, account, gate: TradeGate, bar_sec: int = 3600):
        self._acc = account
        self.gate = gate
        self.bar_sec = bar_sec
        rtc = round_trip_cost()
        self.recent_gross: Deque[float] = deque(
            [p + rtc for p in account.trade_pcts[-gate.p.breaker_trades:]],
            maxlen=gate.p.breaker_trades)
        self.fee_ledger: List[Tuple[int, float]] = []  # (ts, fee$)
        self.cooldown = 0
        self._atr_pct = 0.0
        self.block_reason: Optional[str] = None

    # ---- context ----------------------------------------------------
    def set_bar_context(self, atr_pct: float, ts: int = 0) -> None:
        """Called once per closed candle with this bar's context."""
        self._atr_pct = float(atr_pct)
        if self.cooldown > 0:
            self.cooldown -= 1
        if ts:
            self._drop_stale_fees(ts)

    def _drop_stale_fees(self, now_ts: int, window_sec: int = 86400) -> None:
        self.fee_ledger = [(t, f) for (t, f) in self.fee_ledger
                           if now_ts - t <= window_sec]

    def _fees_window(self) -> float:
        return sum(f for _, f in self.fee_ledger)

    def _unrealized(self, pos, price: float) -> float:
        """Net-of-cost unrealized return — matches PaperAccount math."""
        fill = price * (1.0 - self._acc.slippage)
        proceeds = fill * pos.qty * (1.0 - self._acc.taker_fee)
        basis = pos.entry_cost + pos.entry_fee
        return proceeds / basis - 1.0 if basis else 0.0

    # ---- gated account API ------------------------------------------
    def open_position(self, pair: str, price: float, ts: int):
        try:
            ctx = GateContext(atr_pct=self._atr_pct, rtc=round_trip_cost(),
                              recent_gross_pcts=list(self.recent_gross),
                              fees_paid_window=self._fees_window(),
                              capital=self._acc.capital,
                              cooldown_bars_left=self.cooldown)
            ok, why = self.gate.allow_entry(ctx)
        except Exception:  # noqa: BLE001 — entries fail CLOSED
            self.block_reason = "gate error -> entry blocked (fail closed)"
            return None
        if not ok:
            self.block_reason = f"gate: {why}"
            return None
        self.block_reason = None
        pos = self._acc.open_position(pair, price, ts)
        if pos is not None:
            self.fee_ledger.append((ts, pos.entry_fee))
        return pos

    def close_position(self, pair: str, price: float, ts: int):
        pos = self._acc.positions.get(pair)
        if pos is None:
            return None
        try:
            ctx = GateContext(atr_pct=self._atr_pct, rtc=round_trip_cost(),
                              unrealized_pct=self._unrealized(pos, price),
                              recent_gross_pcts=list(self.recent_gross),
                              fees_paid_window=self._fees_window(),
                              capital=self._acc.capital,
                              cooldown_bars_left=self.cooldown)
            ok, why = self.gate.allow_exit(ctx)
        except Exception:  # noqa: BLE001 — exits fail OPEN
            ok, why = True, "gate error -> exit allowed (fail open)"
        if not ok:
            self.block_reason = f"gate: {why}"
            return None
        self.block_reason = None
        closed = self._acc.close_position(pair, price, ts)
        if closed is not None:
            self._record_close(closed, ts)
        return closed

    def _record_close(self, closed: dict, ts: int = 0) -> None:
        """Update breaker state after any realized close (also used by
        FeeAwareStrategy for forced stop/time exits)."""
        self.fee_ledger.append((ts, closed.get("exit_fee", 0.0)))
        gross = closed["pnl_pct"] + round_trip_cost()
        self.recent_gross.append(gross)
        if (len(self.recent_gross) >= self.gate.p.breaker_trades
                and sum(self.recent_gross) / len(self.recent_gross) < 0):
            self.cooldown = self.gate.p.cooldown_bars

    # ---- delegation --------------------------------------------------
    def __getattr__(self, name: str):
        acc = self.__dict__.get("_acc")
        if acc is None:
            raise AttributeError(name)
        return getattr(acc, name)
```

- [ ] **Step 4: Run the test, verify PASS**

Run: `venv/bin/python -c "import tests.run_tests as t; t.test_gated_account_proxy()"`
Expected: PASS lines. Note: `List`/`Tuple` must be imported at the top of trade_gate.py — extend the existing typing import to `from typing import Deque, List, Optional, Tuple`.

- [ ] **Step 5: Commit**

```bash
git add bot/trade_gate.py tests/run_tests.py
git commit -m "feat: GatedAccount proxy — live fee-gating for every bot"
```

---

### Task 4: FeeAwareStrategy (bot/strategies/fee_aware.py)

**Files:**
- Create: `bot/strategies/fee_aware.py`
- Test: `tests/run_tests.py` (add `test_fee_aware_rewriter`, `test_rewriter_causality`, `test_fee_aware_execute`, register all in `main()`)

- [ ] **Step 1: Write the failing tests**

```python
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


def test_fee_aware_rewriter() -> None:
    """The rewriter trades strictly less than the base on chop, and every
    realized exit clears the cost band or is a stop/time exit."""
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
    check("rewriter: no two entries without an exit between",
          _alternating(w.to_numpy()))


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
    from bot.trade_gate import GateParams
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `venv/bin/python -c "import tests.run_tests as t; t.test_fee_aware_rewriter()"`
Expected: FAIL with ModuleNotFoundError (bot.strategies.fee_aware)

- [ ] **Step 3: Create bot/strategies/fee_aware.py**

```python
"""FeeAwareStrategy: wraps any strategy so every trade decision clears
the round-trip cost (see docs/superpowers/specs/2026-08-15-fee-aware-trading-design.md).

Two adapters, one policy (bot.trade_gate.TradeGate):
  * compute_signals -> causal signal REWRITE for the backtest engine
    (suppress gated entries, defer sub-cost exits, inject stop/time exits)
  * execute -> GatedAccount proxy around the live/paper account, plus
    forced stop/time exits against the real position

The wrapper preserves the base's ``name`` so reports, boards and state
files are unchanged.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from bot.indicators.ta import atr
from bot.strategies.base import Strategy
from bot.trade_gate import (GatedAccount, GateContext, GateParams,
                            TradeGate, round_trip_cost)


def _bar_sec(df: pd.DataFrame) -> int:
    try:
        return int((df.index[-1] - df.index[-2]).total_seconds())
    except Exception:  # noqa: BLE001
        return 3600


class FeeAwareStrategy(Strategy):
    name = "fee_aware"

    def __init__(self, base: Strategy, gate_params: Optional[dict] = None):
        super().__init__({})
        self._base = base
        self.name = base.name          # reports/boards unchanged
        self.gate = TradeGate(GateParams(**(gate_params or {})))
        self._proxies: Dict[int, GatedAccount] = {}
        self._last_reason: Optional[str] = None

    # ---- forwarding ---------------------------------------------------
    def fit(self, df: pd.DataFrame) -> None:
        return self._base.fit(df)

    def warmup_bars(self) -> int:
        return self._base.warmup_bars()

    def last_reason(self) -> Optional[str]:
        return self._last_reason

    # ---- backtest path: causal rewrite --------------------------------
    def compute_signals(self, df: pd.DataFrame, live: bool = False) -> pd.Series:
        base_sig = self._base.compute_signals(df, live=live)
        try:
            return self._rewrite(base_sig, df)
        except Exception:  # noqa: BLE001 — fail closed for entries,
            # fail open for exits: keep only the base's exit signals
            return base_sig.where(base_sig == -1, 0).astype(int)

    def _rewrite(self, sig: pd.Series, df: pd.DataFrame) -> pd.Series:
        gate, p = self.gate, self.gate.p
        rtc = round_trip_cost()
        close = df["close"]
        atr_pct = (atr(df["high"], df["low"], close, 14)
                   / close.replace(0.0, np.nan)).fillna(0.0)
        out = pd.Series(0, index=df.index, dtype=int)
        in_pos = False
        entry_price = 0.0
        entry_i = 0
        recent_gross: list = []
        fee_ledger: list = []      # (ts, fee as fraction of capital)
        cooldown = 0
        for i in range(len(df)):
            if cooldown > 0:
                cooldown -= 1
            px = float(close.iloc[i])
            ts = df.index[i]
            fee_ledger = [(t, f) for (t, f) in fee_ledger
                          if (ts - t).total_seconds() <= 86400]
            s = int(sig.iloc[i]) if not pd.isna(sig.iloc[i]) else 0
            if in_pos:
                unreal = (px * (1.0 - _TAKER_SLIP[0]) * (1.0 - _TAKER_SLIP[1])
                          / (entry_price * (1.0 + _TAKER_SLIP[1])
                             * (1.0 + _TAKER_SLIP[0])) - 1.0)
                hold = i - entry_i
                forced, _ = gate.force_exit(GateContext(
                    atr_pct=float(atr_pct.iloc[i]), rtc=rtc,
                    unrealized_pct=unreal, hold_bars=hold))
                allowed = forced
                if not allowed and s == -1:
                    allowed, _ = gate.allow_exit(GateContext(
                        atr_pct=float(atr_pct.iloc[i]), rtc=rtc,
                        unrealized_pct=unreal, hold_bars=hold))
                if allowed:
                    out.iloc[i] = -1
                    in_pos = False
                    recent_gross.append(px / entry_price - 1.0)
                    recent_gross = recent_gross[-p.breaker_trades:]
                    fee_ledger.append((ts, rtc * p.position_fraction / 2.0))
                    if (len(recent_gross) >= p.breaker_trades
                            and sum(recent_gross) / len(recent_gross) < 0):
                        cooldown = p.cooldown_bars
            elif s == 1:
                ctx = GateContext(atr_pct=float(atr_pct.iloc[i]), rtc=rtc,
                                  recent_gross_pcts=recent_gross,
                                  fees_paid_window=sum(f for _, f in fee_ledger),
                                  capital=1.0,
                                  cooldown_bars_left=cooldown)
                ok, _ = gate.allow_entry(ctx)
                if ok:
                    out.iloc[i] = 1
                    in_pos = True
                    entry_price = px
                    entry_i = i
                    fee_ledger.append((ts, rtc * p.position_fraction / 2.0))
        return out

    # ---- live/zoo path: proxy + forced exits ---------------------------
    def execute(self, account, pair: str, df: pd.DataFrame,
                price: float, ts: int, live: bool = False) -> Optional[dict]:
        key = id(account)
        proxied = self._proxies.get(key)
        if proxied is None or proxied._acc is not account:
            proxied = GatedAccount(account, self.gate, bar_sec=_bar_sec(df))
            self._proxies[key] = proxied
        try:
            atr_pct = float((atr(df["high"], df["low"], df["close"], 14)
                             / df["close"]).iloc[-1])
        except Exception:  # noqa: BLE001
            atr_pct = 0.0
        proxied.set_bar_context(atr_pct, ts=ts)
        result = self._base.execute(proxied, pair, df, price, ts, live=live)
        # forced stop / time exits against the REAL position
        pos = account.positions.get(pair)
        if pos is not None:
            unreal = proxied._unrealized(pos, price)
            hold = max(0, int((ts - pos.entry_ts) / max(1, proxied.bar_sec)))
            forced, why = self.gate.force_exit(GateContext(
                atr_pct=atr_pct, rtc=round_trip_cost(),
                unrealized_pct=unreal, hold_bars=hold))
            if forced:
                closed = account.close_position(pair, price, ts)
                if closed is not None:
                    proxied._record_close(closed, ts)
                    result = {"action": "sell", "qty": closed["qty"],
                              "fee": closed["exit_fee"], "price": price,
                              "pnl": closed["pnl"],
                              "pnl_pct": closed["pnl_pct"]}
        self._last_reason = proxied.block_reason
        return result
```

Note: `_TAKER_SLIP` — add a module helper in trade_gate.py instead:

```python
# in bot/trade_gate.py, after set_fee_model/round_trip_cost
def fee_pair() -> "Tuple[float, float]":
    """(taker_fee, slippage) from the synced fee model."""
    return (_TAKER_FEE, _SLIPPAGE)
```

and in fee_aware.py `_rewrite`, replace the `_TAKER_SLIP` expression with:

```python
                fee, slip = fee_pair()
                unreal = (px * (1.0 - slip) * (1.0 - fee)
                          / (entry_price * (1.0 + slip) * (1.0 + fee)) - 1.0)
```

(import `fee_pair` from bot.trade_gate).

- [ ] **Step 4: Run the new tests, verify PASS**

Run: `venv/bin/python -c "import tests.run_tests as t; t.test_fee_aware_rewriter(); t.test_rewriter_causality(); t.test_fee_aware_execute()"`
Expected: PASS lines. If `test_fee_aware_execute`'s RSI2 slice produces no signals, the fallback branch (`gated entries keep account flat`) still passes — both outcomes are correct behavior.

- [ ] **Step 5: Commit**

```bash
git add bot/strategies/fee_aware.py bot/trade_gate.py tests/run_tests.py
git commit -m "feat: FeeAwareStrategy — signal rewriter + gated execute"
```

---

### Task 5: Default-on wiring

**Files:**
- Modify: `bot/strategies/__init__.py`
- Modify: `run_backtest.py`, `run_zoo.py`, `run_swarm.py`, `run_paper.py` (one line each)
- Test: `tests/run_tests.py` (add `test_build_strategy_wraps_fee_aware`, register in `main()`)

- [ ] **Step 1: Write the failing test**

```python
def test_build_strategy_wraps_fee_aware() -> None:
    """The factory wraps every strategy fee-aware by default; base
    params still flow to the base, gate params under 'fee_aware'."""
    from bot.strategies import build_strategy
    from bot.strategies.fee_aware import FeeAwareStrategy
    from bot.strategies.momentum import MomentumStrategy
    s = build_strategy("momentum", {"ema_fast": 8, "ema_slow": 30})
    check("factory: returns FeeAwareStrategy", isinstance(s, FeeAwareStrategy))
    check("factory: base name preserved", s.name == "momentum")
    check("factory: base params consumed",
          s._base.params["ema_fast"] == 8)
    check("factory: base is the right class",
          isinstance(s._base, MomentumStrategy))
    s2 = build_strategy("momentum", {"ema_fast": 8,
                                     "fee_aware": {"max_hold_bars": 50}})
    check("factory: gate params split out",
          s2.gate.p.max_hold_bars == 50 and s2._base.params["ema_fast"] == 8)
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python -c "import tests.run_tests as t; t.test_build_strategy_wraps_fee_aware()"`
Expected: FAIL (build_strategy returns raw MomentumStrategy)

- [ ] **Step 3: Modify bot/strategies/__init__.py**

Add import at top (with the other strategy imports):

```python
from bot.strategies.fee_aware import FeeAwareStrategy
from bot.trade_gate import GateParams
```

Replace `build_strategy`:

```python
def build_strategy(name: str, params: Optional[dict] = None) -> Strategy:
    if name not in REGISTRY:
        raise KeyError(f"unknown strategy '{name}'; available: {sorted(REGISTRY)}")
    params = dict(params or {})
    gate_params = params.pop("fee_aware", {}) or {}
    base = REGISTRY[name](params or None)
    return FeeAwareStrategy(base, gate_params)
```

(Keep `build_strategies` unchanged — it calls `build_strategy`.)

- [ ] **Step 4: Sync the fee model in the four entry points**

In each of `run_backtest.py`, `run_zoo.py`, `run_swarm.py`, `run_paper.py`, immediately after the `BotConfig.from_yaml(...)` call in `main()` (or module top for run_paper's `config = ...`), add:

```python
    from bot.trade_gate import set_fee_model
    set_fee_model(config.taker_fee, config.slippage)
```

(For run_zoo.py / run_swarm.py the variable is `cfg`, so use `set_fee_model(cfg.taker_fee, cfg.slippage)`. Place imports at the top of the file with the other bot imports, consistent with house style.)

- [ ] **Step 5: Run the FULL existing test suite — nothing regresses**

Run: `venv/bin/python tests/run_tests.py`
Expected: `All N checks passed.` — existing tests that call `REGISTRY[name]({})` directly still get raw strategies (unchanged behavior); everything routed through `build_strategy` is now gated. If `test_runner_replay_offline` fails because the gate blocks the synthetic dip-buy: that test constructs its agent with `Agent(genome=..., account=...)` whose `.strategy` property uses `build_strategy` — if it now trades 0 times, pass `fee_aware` overrides in that test's genome params: `params={..., "fee_aware": {"margin": 0.0, "min_profit_mult": 0.0, "stop_mult": 0.0, "max_hold_bars": 100000, "fee_budget_pct": 10.0}}` and re-run. The synthetic data's ATR (dip of 14% over 7 bars) should clear the real EV hurdle, so prefer verifying the failure reason first (print `agent.strategy.last_reason()`) before weakening the test.

- [ ] **Step 6: Run the new test, verify PASS**

Run: `venv/bin/python -c "import tests.run_tests as t; t.test_build_strategy_wraps_fee_aware()"`
Expected: PASS lines

- [ ] **Step 7: Commit**

```bash
git add bot/strategies/__init__.py run_backtest.py run_zoo.py run_swarm.py run_paper.py tests/run_tests.py
git commit -m "feat: fee-aware gating default-on for every bot"
```

---

### Task 6: Full-suite verification

**Files:** none new

- [ ] **Step 1: Run the complete test suite**

Run: `venv/bin/python tests/run_tests.py`
Expected: `All N checks passed.` (all pre-existing + 5 new test functions)

- [ ] **Step 2: Smoke the zoo one window offline (no network needed if DB has candles; otherwise skip)**

Run: `venv/bin/python run_zoo.py --report 2>&1 | head -30`
Expected: standings render without error

- [ ] **Step 3: Commit any fixups**

```bash
git add -A && git commit -m "fix: fee-aware integration fixups" || true
```

---

### Task 7: A/B backtest on real 1y data (raw vs fee-aware)

**Files:**
- Create: `run_ab_fee.py`
- Output: `reports/ab_fee_aware.md`

- [ ] **Step 1: Create run_ab_fee.py**

```python
#!/usr/bin/env python3
"""A/B backtest: raw (fee-blind) vs fee-aware versions of the churners.

Success = fees and trade count drop sharply (>= 60%) and net excess
return improves. Uses the same engine, fees, and 1y data as
run_backtest.py; raw strategies bypass the factory wrapper on purpose.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot.backtest.engine import run_backtest  # noqa: E402
from bot.config import BotConfig  # noqa: E402
from bot.data.fetcher import fetch_candles  # noqa: E402
from bot.data.store import Store  # noqa: E402
from bot.strategies import REGISTRY  # noqa: E402
from bot.strategies.fee_aware import FeeAwareStrategy  # noqa: E402
from bot.trade_gate import set_fee_model  # noqa: E402
from bot.backtest.engine import BacktestResult  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402

CHURNERS = ["macd_cross", "momentum", "rsi2",
            "stochastic_reversion", "donchian_breakout"]


def main() -> None:
    cfg = BotConfig.from_yaml()
    set_fee_model(cfg.taker_fee, cfg.slippage)
    store = Store(cfg.db_path)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=cfg.history_days)
    rows = []
    for pair in cfg.pairs:
        try:
            df = store.load_candles(pair, cfg.granularity,
                                    start=int(start.timestamp()))
        except Exception:
            df = None
        if df is None or len(df) < 500:
            try:
                df = fetch_candles(pair, cfg.granularity, start, end)
                store.upsert_candles(pair, cfg.granularity, df)
            except Exception as exc:  # noqa: BLE001
                print(f"[ab] {pair} data unavailable: {exc}")
                continue
        df = df.dropna()
        df.attrs["pair"] = pair
        for name in CHURNERS:
            raw = REGISTRY[name]({})            # bypass the wrapper
            gated = FeeAwareStrategy(REGISTRY[name]({}))
            r_raw = run_backtest(df, raw, pair=pair, taker_fee=cfg.taker_fee,
                                 slippage=cfg.slippage,
                                 position_fraction=cfg.position_fraction,
                                 capital=cfg.paper_capital,
                                 cash_yield_apy=cfg.cash_yield_apy)
            r_gated = run_backtest(df, gated, pair=pair,
                                   taker_fee=cfg.taker_fee,
                                   slippage=cfg.slippage,
                                   position_fraction=cfg.position_fraction,
                                   capital=cfg.paper_capital,
                                   cash_yield_apy=cfg.cash_yield_apy)
            rows.append((name, pair, r_raw, r_gated))
    lines = ["# Fee-aware A/B — 1y real data", "",
             "| strategy | pair | raw trades | gated trades | raw fees | "
             "gated fees | raw excess% | gated excess% |", "|---|---|---|---|---|---|---|---|"]
    for name, pair, r_raw, r_gated in rows:
        lines.append(
            f"| {name} | {pair} | {r_raw.n_trades} | {r_gated.n_trades} | "
            f"${r_raw.fee_take:.0f} | ${r_gated.fee_take:.0f} | "
            f"{r_raw.excess_return * 100:+.1f} | "
            f"{r_gated.excess_return * 100:+.1f} |")
    os.makedirs(cfg.out_dir, exist_ok=True)
    out = os.path.join(cfg.out_dir, "ab_fee_aware.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[ab] written to {out}")


if __name__ == "__main__":
    main()
```

Note: check `Store.load_candles` signature matches (`start=` int kwarg) — it is used the same way in `bot/paper/engine.py` and `bot/swarm/runner.py`; mirror those call sites exactly if the signature differs.

- [ ] **Step 2: Run it (network needed for candle fetch; data/trading.db already holds 1y of 1h candles for BTC/ETH/SOL and DOGE/XRP/ADA)**

Run: `venv/bin/python run_ab_fee.py`
Expected: a table showing gated trades/fees far below raw for the churners, and the report saved to `reports/ab_fee_aware.md`

- [ ] **Step 3: Commit**

```bash
git add run_ab_fee.py
git commit -m "feat: raw vs fee-aware A/B backtest report"
```

---

## Self-Review (completed)

- **Spec coverage:** cost model (Task 2 `round_trip_cost` synced in Task 5), gate decisions (Task 2), adapters (Tasks 3-4), wiring default-on (Task 5), unit + causality tests (Tasks 2-5), A/B validation (Task 7). Out-of-scope items intentionally absent. ✓
- **Placeholder scan:** no TBDs; every step has complete code/commands. ✓
- **Type consistency:** `GateParams(**gate_params)` in Task 4 matches the dataclass fields in Task 2; `GatedAccount._record_close(closed, ts)` defined in Task 3 and used in Task 4; `fee_pair()` helper noted in Task 4 Step 3 and added to trade_gate.py in the same task's commit. ✓
