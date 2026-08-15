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

from datetime import datetime, timedelta, timezone  # noqa: E402

from bot.backtest.engine import run_backtest  # noqa: E402
from bot.config import BotConfig  # noqa: E402
from bot.data.fetcher import fetch_candles  # noqa: E402
from bot.data.store import Store  # noqa: E402
from bot.strategies import REGISTRY  # noqa: E402
from bot.strategies.chassis import ChassisStrategy  # noqa: E402
from bot.strategies.fee_aware import FeeAwareStrategy  # noqa: E402
from bot.trade_gate import set_fee_model  # noqa: E402

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
        except Exception:  # noqa: BLE001
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
        print(f"[ab] {pair}: {len(df)} bars")
        for name in CHURNERS:
            variants = {
                "raw": REGISTRY[name]({}),            # bypass wrappers
                "fee": FeeAwareStrategy(REGISTRY[name]({})),
                "chassis": ChassisStrategy(REGISTRY[name]({})),
            }
            results = {}
            for label, strat in variants.items():
                results[label] = run_backtest(
                    df, strat, pair=pair, taker_fee=cfg.taker_fee,
                    slippage=cfg.slippage,
                    position_fraction=cfg.position_fraction,
                    capital=cfg.paper_capital,
                    cash_yield_apy=cfg.cash_yield_apy)
            rows.append((name, pair, results))
    lines = ["# Raw vs Fee-gate vs Chassis — 1y real data", "",
             "| strategy | pair | trades r/f/c | excess% r/f/c | "
             "maxDD% r/f/c | sharpe r/f/c |",
             "|---|---|---|---|---|---|"]

    def _cell(rs, fn, fmt):
        return "/".join(fmt(rs[l]) for l in ("raw", "fee", "chassis"))

    for name, pair, rs in rows:
        lines.append(
            f"| {name} | {pair} | "
            + _cell(rs, lambda r: r, lambda r: str(r.n_trades)) + " | "
            + _cell(rs, lambda r: r, lambda r: f"{r.excess_return*100:+.1f}") + " | "
            + _cell(rs, lambda r: r, lambda r: f"{r.max_drawdown*100:.1f}") + " | "
            + _cell(rs, lambda r: r, lambda r: f"{r.sharpe:.2f}") + " |")
    # aggregate verdict
    agg = {l: {"excess": 0.0, "dd": 0.0, "sh": 0.0, "n": 0} for l in ("raw", "fee", "chassis")}
    for _, _, rs in rows:
        for l, r in rs.items():
            agg[l]["excess"] += r.excess_return
            agg[l]["dd"] += r.max_drawdown
            agg[l]["sh"] += r.sharpe
            agg[l]["n"] += 1
    lines += ["", "## Averages", "",
              "| variant | excess% | maxDD% | sharpe |",
              "|---|---|---|---|"]
    for l in ("raw", "fee", "chassis"):
        n = max(agg[l]["n"], 1)
        lines.append(f"| {l} | {agg[l]['excess']/n*100:+.1f} | "
                     f"{agg[l]['dd']/n*100:.1f} | {agg[l]['sh']/n:.2f} |")
    os.makedirs(cfg.out_dir, exist_ok=True)
    out = os.path.join(cfg.out_dir, "ab_fee_aware.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[-8:]))
    print(f"\n[ab] written to {out}")


if __name__ == "__main__":
    main()
