# Strategy Lab Ledger

A living record of every bot idea tested in the StockTradeBot lab. Each
entry states the **hypothesis**, the **mechanism** (why it *should* work
and what would kill it — the discipline the Aug 2026 review demanded),
the **backtest evidence** (honest, fees + slippage + cash-yield ON), and
the **live verdict** to be filled in from `state/ZOO_BOARD.md` over time.

Rules of the lab:
- Every idea is **pre-registered** here before/with deployment.
- Verdicts are judged **net of ~1.4% round-trip fees + slippage + the
  ~4.5% risk-free hurdle**, vs buy & hold — not on gross fantasy.
- Backtests are advisory only. The live race (weeks of real data) is the
  only evidence that counts.
- A strategy that clears the bar is promoted to `state/deployed_model.json`;
  everything else stays a diagnostic.

---

## Idea #1 — `deep_value` (the dip-recovery bot)

**Status:** LIVE in the zoo (deployed 2026-08-13)

**Your idea (as stated):** "When a coin falls a lot one day and I could
have bought it cheap and it went back up, I want the bot to buy lows and
sell highs."

**Hypothesis (testable form):** After a sustained, deep drawdown on a
liquid major, a *confirmed* recovery has positive expected return net of
fees.

**Mechanism:** Buy only after a **confirmed** capitulation — price ≥30%
below its rolling high — AND a recovery confirmation (price back above a
fast EMA with positive momentum). This filters out coins still in
free-fall (the "falling knife"). Target +30% above the recent low,
stop −15% below it. Long/flat only; cash earns the risk-free yield while
waiting.

**What would kill it (and we're watching for):** most deep dips never
recover — the strategy must not average into dead coins, and fee bleed
on frequent small trades would eat the edge. Its job is to be *selective*,
not to catch every knife.

**Backtest evidence** (365d, 4h candles, fees + slippage + yield ON):
- **ADA-USDC: +5.50%** vs buy&hold −79.85% → **excess +85.34%**; OOS
  +2.20% vs −25.80% (win rate 100% on 1 trade)
- **SOL-USDC: +4.88%** vs buy&hold −62.20% (win 100%, 1 trade)
- BTC/ETH/DOGE/XRP: **0 trades** — the default params are too strict
  (a confirmed recovery while still ≥30% below the high is rare). It
  sits out most of the year.

**Honest read:** the concept works *when it fires* (the two majors that
actually crashed recovered — exactly your SOL scenario, now pre-registered
not hindsight), but the default thresholds are too conservative to be
useful alone. Tuning decision pending: more aggressive (fires on more
dips, more trades + more fees) vs moderate (fewer, safer trades).

**Live verdict (pending):** ─ will be filled from the zoo board after
weeks of live racing.

---

## Idea #2 — `order_flow` (the exchange-prediction / "who is buying" bot)

**Status:** LIVE in the zoo (deployed 2026-08-13)

**Your idea (as stated):** "See who is buying what on the exchange; if a
lot of people start buying when a coin decreased in value, it will go up
again. Buy low, sell high."

**Hypothesis (testable form):** Aggressive **buy-side order flow**
turning up after a price dip has positive expected short-term return net
of fees.

**Mechanism:** Reads **real, keyless Coinbase order-flow data** live —
trade-by-trade (price, size, BUY/SELL taker side) + the live order book:
- buy:sell $ ratio over a window (buyers stepping in?)
- cumulative-volume-delta (CVD) trend (persistent accumulation?)
- order-book bid-side share (who can absorb fills)
- combined with a drawdown filter (only act after a dip)

Buys when flow turns decisively bullish after a dip; sells when the dip
recovers or flow turns. **Every action logs a full structured reason**
(ratio, CVD, drawdown, book, target minus fees/safety) so the reasoning
is auditable — you can literally see why it bought/sold.

**What would kill it (and we're watching for):** order-flow signal is
real but *weak and crowded*; fees can eat it. Also the live feed only
covers the recent window — there is **no historical trade-flow data**, so
the backtest uses a conservative candle-volume *proxy* that is advisory
only; **the live path is the real strategy**.

**Backtest evidence (proxy only, advisory):** 365d 4h candles → ~0 buys
after tightening (deliberately conservative; the volume proxy over-trades
if left loose, which would mislead). **Do not judge this strategy on the
backtest.**

**Live reasoning (verified against the real feed, 2026-08-14):**
```
[order_flow BTC-USDC] BUY @ 63,440.62
   reason: buy:sell ratio 9.53 (buyers stepping in),
           CVD +81.02% (accumulating), drawdown -0.7% from high,
           book is 39% bid-side, price sees 100 recent trades.
   plan: target ~66,612.65 (5% up) minus ~1.4% fees+slip = net positive
         if reached; stop 4% below entry.
```

**Live verdict (pending):** ─ will be filled from the zoo board after
weeks of live racing.

---

## Idea #3 (next slot) — *fill me in*

- **Idea:** ...
- **Hypothesis:** ...
- **Mechanism / what would kill it:** ...
- **Backtest evidence:** ...
- **Live verdict:** ...
