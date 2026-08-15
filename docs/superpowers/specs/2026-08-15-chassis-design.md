# The Chassis — Mechanical Trading Rule System Design

Date: 2026-08-15
Status: Approved by user (brainstorming session)

## Problem / Thesis

The fee gate proved that fixed mechanical rules beat blind evolution
(macd_cross flipped from −22.9% to +18.9% excess on BTC). This design
generalizes that: instead of letting bots learn everything from
scratch, we hard-code what experienced traders already know — look at
the long picture first, only play your play in its regime, size bets
by conviction and volatility — and leave ONLY the signal parameters
for evolution to tune.

User decisions (approved):
1. Unified chassis — the fee gate becomes a layer inside one engine,
   not a wrapper sandwich.
2. Sizing = volatility targeting × conviction multiplier (not Kelly,
   not a fixed ladder).
3. Tunable surface = strategy signal params ONLY. Chassis layers are
   frozen (overfit guard, Bailey/Lopez de Prado).

## Evidence basis

- ATR-inverse sizing with fixed per-trade risk beat fixed-fraction
  sizing on 2020-25 BTC backtests: Sharpe 1.24 / −19.7% maxDD vs
  1.12 / −22.1% (Stratbase 2026, corroborated by Coinquant/DennTech
  2026 writeups).
- Volatility-normalized momentum was the strongest evidence-backed
  signal family in the repo research (Gbadebo 2026, 31.96%/yr).
- Costa 2026 "Illusion of Breakouts": most breakouts are liquidity
  sweeps — trend plays need regime confirmation, supporting the
  regime gate.
- ml pipeline (bot/train/features.py) already computes trend_frac_pos,
  dd_from_high, atr_pct, sma_dist, linreg_slope — patterns reused by
  the context builder.

## Architecture

```
bot/
├── market/
│   └── context.py        # MarketContext: the 1-year look, per pair/bar
├── trade_gate.py          # unchanged (layer 3)
└── strategies/
    └── fee_aware.py       # FeeAwareStrategy -> ChassisStrategy
                          # (layers 2,4,5,6 wired around the base signal)
```

### Layer 1 — MarketContext (bot/market/context.py)

Computed per (pair, bar), cached per compute pass. Deterministic,
vectorized, causal.

Fields (with graceful degradation when history < 1y):
- `pct_rank_1y`: close percentile in the trailing 1y high-low range
  (falls back to available window; `context_confidence` scales with
  window length)
- `trend_sma_dist`: close / SMA200 − 1
- `trend_slope`: linear-regression slope of close over 50 bars,
  normalized by price
- `atr_pct`: ATR(14)/close (already used by the fee gate)
- `atr_pctile`: ATR percentile over trailing 90 bars (panic detector)
- `dd_from_high`: drawdown from trailing 1y high
- `context_confidence`: min(1, window/8760) — scales conviction

### Layer 2 — Regime gate

Regime (deterministic, evaluated in order):
- CRASH: dd_from_high > 25% AND atr_pctile > 0.90
- UP: trend_sma_dist > +3% AND trend_slope > 0
- DOWN: trend_sma_dist < −3% AND trend_slope < 0
- RANGE: otherwise

Play allowlist by strategy family (static table, chassis-owned):

| family | strategies | allowed regimes |
|---|---|---|
| trend | momentum, macd_cross, golden_cross, donchian_breakout, trend_runner, hold_cycle, ml_trend | UP |
| range | mean_reversion, rsi2, stochastic_reversion, grid_trader, bbands_breakout, adaptive_grid | RANGE, UP |
| value | deep_value, deep_recovery, dca_bot, fade_extreme | DOWN, RANGE |
| capitulation | fade_extreme | CRASH |

(fade_extreme's effective allowlist is the union of its rows:
DOWN, RANGE, CRASH — it is the only family allowed to buy a crash.)
| self-directed | llm_trader, order_flow, consensus | any (sizing still applies) |

Entries blocked outside allowlist; exits ALWAYS pass (mirrors the fee
gate's asymmetry). If context is degraded the regime still computes on
available data — no hard dependency on a full year.

### Layer 3 — Fee gate (unchanged TradeGate)

EV hurdle, breaker, budget — exactly as shipped.

### Layer 4 — Signal (the only evolvable surface)

The base strategy's own compute_signals / execute. Evolution (swarm,
Deep Time, auto-tuner) tunes these params within existing
PARAM_BOUNDS. The chassis never mutates signal logic.

### Layer 5 — Sizing (volatility target × conviction)

```
base_fraction = clamp(TARGET_RISK / atr_pct, 0.05, 0.50)
conviction   = 1.0 + 0.5 * context_score * context_confidence
fraction     = clamp(base_fraction * conviction, 0.05, 0.50)
```

- `TARGET_RISK = 0.01` (1% of equity risked per trade; calm coin →
  bigger size, wild coin → smaller — empirically the better scheme)
- `context_score ∈ [0,1]` measures how strongly the context favors
  THIS play: trend_frac_pos for trend family, range-ness (1 −
  |trend_sma_dist|/0.03 clamped) for range family, discount depth
  (min(dd_from_high/0.40, 1)) for value family, 0.5 default for
  self-directed
- Sizing only applies to entries; exits are all-or-nothing (full close)

### Layer 6 — Order

`PaperAccount.open_position(pair, price, ts, fraction=None)` —
optional fraction overrides `position_fraction` (default keeps legacy
behavior for every existing caller). The backtest engine reads a
per-entry fraction series the chassis publishes
(`strategy._entry_fractions`, aligned to df.index; NaN → engine
default).

## Wiring

- `build_strategy()` constructs `ChassisStrategy(base, gate_params)`
  (FeeAwareStrategy kept as a deprecated alias; name preserved for
  boards/state).
- Live/zoo: chassis `execute()` computes context from the closed
  window, applies layers 2-5, passes fraction to the (gated) account.
- Backtest: the causal rewriter applies regime gate + sizing; the
  engine consumes `_entry_fractions`. Causality test extended: prefix
  stability must hold for both signals and fractions.
- Guarded bots: their own regime filter coexists (harmless overlap,
  both must pass — stricter is safer).

## Constants (frozen — not exposed to any tuner)

| key | default |
|---|---|
| target_risk | 0.01 |
| fraction_min / fraction_max | 0.05 / 0.50 |
| conviction_max_mult | 0.5 |
| sma_window | 200 |
| slope_bars | 50 |
| crash_dd / crash_atr_pctile | 0.25 / 0.90 |
| trend_band | 0.03 |
| context_full_bars | 8760 |

## Validation

1. Unit tests: regime classification on synthetic UP/RANGE/DOWN/CRASH
   series; allowlist gating; sizing math (bounds, conviction caps,
   degraded-context scaling); fraction passthrough into the account;
   engine honoring `_entry_fractions`; causality (signals AND
   fractions prefix-stable).
2. 3-way A/B on the same 1y data (extend run_ab_fee.py): raw vs
   fee-gate-only vs chassis. Success = Sharpe/maxDD improvement over
   fee-gate-only without trade-count explosion; sizing must reduce
   maxDD on the high-vol pairs (DOGE/SOL) specifically.
3. Full suite green; zoo runs the chassis in its next window.

## Risks / guards

- Regime misclassification blocks valid entries → mitigated: RANGE is
  the default bucket (only strict conditions give UP/DOWN/CRASH), and
  range+value families are allowed in 2-3 regimes each.
- Dynamic sizing could over-concentrate → hard cap 50% per position;
  max_positions still bounds total exposure; cash constraint is
  structural.
- Short-history CI runners → context degrades gracefully, conviction
  scales down, nothing crashes or blocks.

## Out of scope

- No Kelly, no per-strategy chassis overrides, no maker-order
  simulation, no portfolio-level correlation caps (future work),
  no changes to PARAM_BOUNDS or evolution mechanics.
