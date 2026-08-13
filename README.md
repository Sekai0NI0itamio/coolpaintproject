# StockTradeBot — USDC Coinbase Trading Bot (Paper Trading)

A Python trading bot for USDC-quoted pairs on Coinbase Advanced Trade.
It trains on real historical data, then paper-trades on live data with
pretend money so you can measure P&L before ever risking real funds.

## The Swarm (evolutionary experiment, runs on GitHub Actions)

40 bots, each with **$20 of pretend USDC**, all trading the same strategy
type with different tunings on live Coinbase data — entirely on GitHub's
servers, so your computer can be asleep or off.

- **4 windows/day** (00:00, 06:00, 12:00, 18:00 UTC, ~5.75h each — GitHub's
  per-job limit is 6h, so a single 24h run is impossible; the 4 windows
  cover the full day).
- **Gap-fill**: if a window is late/skipped/crashed, the next run replays
  every missed candle deterministically — no data or trades are lost.
- **Daily selection** (end of each UTC day): the 35 worst bots are killed,
  the **top 5 earners are cloned 8x** with slightly mutated tunings, and
  all 40 start fresh with $20 again. Lineage is tracked in the state file.
- **Training happens on GitHub** (`train-seeds` workflow): backtests a
  bounded grid of tunings on 1y of real 15m candles (fees ON, out-of-sample)
  and writes the top 5 configs to `state/seeds.json`.
- State = one small JSON file (`state/population.json`) committed back to
  the repo after every window. Leaderboard: `state/LEADERBOARD.md`,
  daily reports: `state/history/`.

```bash
python run_swarm.py --hours 5.75      # one trading window (what CI runs)
python run_swarm.py --leaderboard     # current standings
python run_swarm.py --select          # force daily selection (testing)
python run_train.py --days 365        # retrain seed tunings
```

Built from investigative research (Aug 2026, 12 research sub-agents,
all sources verified live). Key findings that shaped the design:

- **freqtrade does NOT support Coinbase** (official) → custom bot on the
  Coinbase public API instead.
- **Coinbase has no paper trading** (their sandbox is a static mock) →
  this repo implements its own paper engine on live market data.
- **Data is free and keyless**: public REST candles + public WebSocket.
- **Fees dominate**: Coinbase taker ~0.6% (~1.2% round trip). Every
  backtest and paper fill models fees + slippage, or results are fiction.
  A documented 2026 postmortem found 7 strategies × 100 hyperopt runs —
  all lost money on Coinbase fees. Expectations should be set accordingly.

## Strategies (all compared against buy & hold)

| Strategy | Evidence basis |
|---|---|
| `momentum` | EMA trend-following; strongest academic evidence 2020–25 (TS momentum ~32%/yr study) |
| `mean_reversion` | RSI + Bollinger fade; USDC-quoted pairs show mean-reverting dynamics (Yan et al. 2026) |
| `ml` | LightGBM direction classifier, fee-aware labels, walk-forward trained; realistic AUC 0.57–0.61 |

## Install

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
# macOS: LightGBM needs OpenMP
brew install libomp
```

## Usage

### 1. Backtest on real history ("train on real data")

```bash
venv/bin/python run_backtest.py                    # 1yr, 1h candles, all strategies
venv/bin/python run_backtest.py --days 730 --granularity FOUR_HOUR
venv/bin/python run_backtest.py --strategies momentum,mean_reversion --pairs BTC-USDC
```

Downloads paginated public candles into `data/trading.db`, backtests each
strategy with 0.6% taker fee + 0.1% slippage, prints a comparison table,
and saves `reports/backtest_results.json` + `reports/backtest_equity.png`.
ML strategies are fitted on the first 70% and evaluated on a held-out
last-30% window (honest out-of-sample). TA strategies report both the
full window and an `[OOS]` last-30% row.

### 2. Paper trade on live data ("pretend it paid the money")

```bash
venv/bin/python run_paper.py                       # run until Ctrl-C
venv/bin/python run_paper.py --hours 12            # run for 12 hours
venv/bin/python run_paper.py --capital 10000 --strategies momentum,mean_reversion
```

Polls live Coinbase candles (no API keys), executes pretend fills at each
closed candle with fees + slippage, and records everything to SQLite.
State persists — stop and resume anytime. A buy & hold baseline runs in
parallel for comparison.

### 3. Measure P&L ("calculate how much it lost and earned")

```bash
venv/bin/python run_paper.py --report
```

Prints per-strategy closed trades, win rate, realized P&L; saves
`reports/paper_report.md` + `reports/paper_equity.png`.

**The honesty metric**: compare paper P&L vs backtest P&L. If paper win
rate diverges from backtest by more than ~10% after 30+ trades, something
is wrong (fees, data, overfitting) → go back to step 1.

## Configuration

Everything lives in `strategies.yaml` (pairs, timeframe, capital, fees,
per-strategy parameters). All keys can be overridden via CLI flags.

## Architecture

```
bot/
├── config.py              # YAML config -> typed dataclasses
├── data/
│   ├── fetcher.py         # Coinbase public REST candles (paginated, keyless)
│   ├── live.py            # REST polling feed + WebSocket feed
│   └── store.py           # SQLite: candles, fills, trades, equity curve
├── indicators/ta.py       # EMA, RSI, Bollinger, ATR, volume features
├── strategies/
│   ├── base.py            # signal interface (1=buy, -1=sell, 0=hold)
│   ├── momentum.py        # EMA cross + trend filter
│   ├── mean_reversion.py  # RSI + Bollinger fade
│   └── ml.py              # LightGBM, walk-forward, fee-aware labels
├── backtest/
│   ├── engine.py          # next-bar-open fills, fees, slippage, metrics
│   └── walkforward.py     # train/test splits with purge gaps
├── paper/
│   ├── account.py         # virtual USDC wallet
│   ├── ledger.py          # trade persistence
│   └── engine.py          # live loop: candles -> signals -> pretend fills
└── report/report.py       # tables, equity PNGs, paper reports
```

Anti-bias guarantees:
- Signals use closed candles only; backtest fills at the **next bar's open**.
- Walk-forward validation with purge gaps; ML evaluated out-of-sample.
- Fees + slippage on every fill in backtest AND paper mode.

## First backtest results (real data, Aug 2026)

365 days of 1h candles — a brutal bear year (BTC −46.8%, ETH −59.4%,
SOL −61.5% buy & hold):

| strategy | pair | total | buy&hold | excess | win rate | trades | fees paid |
|---|---|---|---|---|---|---|---|
| mean_reversion | SOL-USDC | −21.3% | −61.5% | **+40.1%** | 44.1% | 59 | $1,588 |
| momentum | ETH-USDC | −24.5% | −59.4% | +35.0% | 18.3% | 60 | $1,534 |
| mean_reversion | BTC-USDC | −22.1% | −46.8% | +24.7% | 34.4% | 64 | $1,675 |

All strategies beat buy & hold by staying out of the drawdown, but fees
(~$1.5k per strategy on $10k) are the dominant cost — exactly as the
research predicted. ML validation AUC came in at 0.606/0.583/0.472 —
inside the realistic 0.57–0.61 band; anything higher would be overfitting.

## Roadmap

- Longer paper runs (days → weeks) and parameter tuning per regime
- More USDC pairs once the core loop is proven on liquid ones
- Walk-forward hyperparameter search with trial caps (overfit control)
- Telegram/console alerts
- Real-money mode via Coinbase CDP API keys (only after paper results
  consistently beat buy & hold net of fees)

## Disclaimer

Educational/research project. Not financial advice. Most retail bot
strategies lose money after fees; paper trade long before risking real
funds.
