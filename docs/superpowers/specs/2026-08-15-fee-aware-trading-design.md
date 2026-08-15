# Fee-Aware Trading System — Design

Date: 2026-08-15
Status: Approved by user (brainstorming session)

## Problem

Live zoo data (generation 0, 2026-08-14 window) showed that every active
bot had gross P&L (before fees) of roughly breakeven or positive, yet all
were net losers. Losses mapped 1:1 to fees paid:

- macd_cross: gross −$0.68, fees $4.44 (73 round trips) → net −$5.12
- rsi2: gross +$0.06, fees $2.86 → net −$2.80
- momentum: gross −$1.59, fees $1.91 (0% win rate) → net −$3.50

Root cause: strategies emit indicator-event signals (1/-1/0) with zero
knowledge of the ~1.4% round-trip cost. Bots routinely take trades whose
expected move is smaller than the toll, and take profits smaller than the
round trip. The existing `GuardedStrategy` wrapper only gates entries on
a volatility proxy for six bots and leaves exits fee-blind.

Research basis (Research/notes, verified Aug 2026):
- Coinbase Advanced Trade retail tier: 0.60% taker / 0.40% maker on
  crypto-USDC pairs (external consensus re-verified 2026-08-15). The
  0%-maker "stablepair" pricing does NOT apply to crypto-USDC pairs.
- Round trip with slippage: 2 × (0.6% + 0.1%) = 1.4%.
- Han, Kang & Ryu (2024): momentum edges weaken/vanish under realistic
  costs — cost-aware filtering is mandatory, not optional.
- Coinmonks 500-strategy analysis: only ~3% of strategies beat
  buy-and-hold after fees; churn is the dominant failure mode.
- Bailey & Lopez de Prado (2021) + AlgoXpert protocol (Pham 2026):
  parameter plateaus not optima, low trial counts, kill switches /
  circuit breakers. Consequence: gate constants below are principled
  fixed values, never tuned per strategy.

## Decisions (user-approved)

1. Default-on for ALL bots (zoo + swarm + backtests). No raw controls
   kept in the zoo.
2. Full EV system: entry gate + fee-aware exits + circuit breaker +
   daily fee budget. No adaptive position sizing (declined).
3. Architecture: one `TradeGate` policy module + two thin adapters
   (signal rewriter for backtest, account proxy for live/zoo).

## Design

### 1. Cost model

`round_trip_cost (rtc) = 2 × (taker_fee + slippage)` — always derived at
runtime from the account/config (currently 1.4%). Never hardcoded.

### 2. `bot/trade_gate.py` (new) — the fee brain

Pure decisions over a context dataclass:

```python
@dataclass
class GateContext:
    atr_pct: float          # ATR(14) / price on the current bar
    rtc: float              # round-trip cost as fraction (0.014)
    unrealized_pct: float   # position P&L vs entry cost incl. fees
    hold_bars: int
    recent_gross_pcts: list # last N closed trades' gross returns
    fees_paid_window: float # fees paid in trailing 24h
    capital: float
    cooldown_bars_left: int  # maintained by the adapter (stateful part)
```

`TradeGate` itself is stateless — adapters (rewriter/proxy) own all
state: the simulated/real position, the fee-window ledger, and the
cooldown countdown.

- `allow_entry(ctx) -> (bool, reason)`:
  - EV rule: `sqrt(expected_hold_bars) × atr_pct >= 1.5 × rtc`
    (expected_hold_bars default 16; on 1h candles this requires
    ATR% ≳ 0.53% instead of trading dead tape)
  - Circuit breaker: mean of last 8 closed gross returns < 0 → block
    entries for 24 bars from the last closed trade
  - Fee budget: fees_paid_window > 2% of capital → block entries
- `allow_exit(ctx) -> (bool, reason)` — called when the base bot sells:
  - Allow if `unrealized_pct >= +1.0 × rtc` (profit paid the toll)
  - Allow if `unrealized_pct <= −3.0 × rtc` (disaster stop ≈ −4.2%)
  - Otherwise defer (hold): never realize a sub-cost "win"
- `force_exit(ctx) -> (bool, reason)`: time stop — exit after
  `max_hold_bars` (default 96) regardless; checked every bar by the
  adapters so deferred exits cannot become zombie positions.

Defaults (strategies.yaml `fee_aware:` section, typed in config.py):

| key | default | meaning |
|---|---|---|
| margin | 1.5 | expected move must clear rtc × margin |
| expected_hold_bars | 16 | sqrt-scaled horizon for EV estimate |
| min_profit_mult | 1.0 | profit exits need ≥ rtc × this |
| stop_mult | 3.0 | disaster stop at −rtc × this |
| max_hold_bars | 96 | time stop |
| breaker_trades | 8 | trailing gross-loss window |
| cooldown_bars | 24 | entry pause after breaker trips |
| fee_budget_pct | 0.02 | max fees per 24h as % of capital |

These constants are fixed by principle; the auto-tuner may NOT tune
them (overfit guard).

### 3. Adapters

**Signal rewriter** (backtest path). A causal, bar-sequential walk over
the base signal series inside `FeeAwareStrategy.compute_signals`.
Maintains a simulated position (entry at next-bar open approximated by
current close; documented approximation) and rewrites the series:
suppress gated entries, defer sub-cost profit exits, inject stop and
time exits. Uses only data up to each bar — a causality unit test
asserts that mutating future candles never changes past rewritten
signals. Works with the existing `run_backtest` engine unchanged, so
Deep Time training and walk-forward validation test gated behavior
automatically.

**Account proxy** (zoo/paper path). `GatedAccount` wraps `PaperAccount`:
- `open_position` → `allow_entry`; blocked calls return None (same
  contract as insufficient cash)
- `close_position` → `allow_exit`; deferred calls return None
- tracks per-account trailing stats (recent gross pcts, 24h fee window,
  breaker state) from the account's own trade records
`FeeAwareStrategy.execute` passes the proxy to the base strategy —
including custom-execute bots (grid, DCA, LLM trader), which get uniform
gating with zero changes to their files.

Failure asymmetry: if gate math throws, entries fail CLOSED (no trade on
broken math), exits fail OPEN (selling is always possible).

### 4. Wiring (default-on)

- `bot/zoo/roster.py`: every roster entry wrapped in `FeeAwareStrategy`
- `bot/swarm/population.py`: agents' strategies wrapped at build
- `run_backtest.py`: strategies wrapped, fee params passed so the
  rewriter's rtc matches the engine's fills
- `bot/config.py`: `FeeAwareConfig` dataclass bound to `strategies.yaml`
- Guarded bots stay: their regime filter is complementary; the ATR
  hurdle partially overlaps the EV rule (harmless)

### 5. Testing & validation

1. Unit tests (pytest): gate arithmetic (EV threshold boundary, exit
   defer band, stop, time stop, breaker, budget), proxy contract
   (blocked entry returns None), rewriter causality.
2. A/B backtest on existing 1y data in `data/trading.db` for the five
   churners (macd_cross, momentum, rsi2, stochastic_reversion,
   donchian_breakout): success = round trips and fees drop sharply
   (expect ≥ 60%) and net excess return improves. Report saved to
   `reports/`.
3. Zoo: next CI window runs gated automatically; boards unchanged.

### 6. Explicitly out of scope

- No fitted/tuned gate parameters (Bailey/Lopez de Prado guard)
- No per-strategy gate overrides
- No fee-model changes (0.6% taker verified correct)
- No adaptive position sizing (user declined)
- No maker-order simulation

## Risks

- Deferring exits can convert small winners into losers → bounded by
  disaster stop (−4.2%) and time stop (96 bars).
- EV entry gate trades less → fewer data points for swarm selection →
  acceptable; breaker window (8) chosen to stay above the zoo's
  ≥ 3 closed trades eligibility rule.
- Rewriter's close-as-next-open approximation diverges slightly from
  engine fills → acceptable for gate decisions; documented.
