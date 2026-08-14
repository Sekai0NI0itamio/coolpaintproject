#!/usr/bin/env python3
"""Nightly auto-tuner entrypoint: extract winning guard params, apply live.

Walk-forward backtests a trial-capped sample of guard variants on recent
real history (fees + yield ON, OOS-scored) and writes the winners to
state/guard_params.json. Guarded bots read that file when constructed,
so the zoo/swarm automatically pick up the tuned parameters on their next
run -- the "auto-apply winning logic over time" loop.

Usage:
    python run_auto_tune.py --days 365
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot.train.auto_guard import (OVERRIDES_PATH, RUNNER_PATH,  # noqa: E402
                                  tune, tune_runner, write_overrides)


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-tune guard parameters")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--granularity", default="FOUR_HOUR")
    ap.add_argument("--skip-runner", action="store_true",
                    help="skip the trend_runner tuning stage")
    args = ap.parse_args()

    print(f"[tune] starting auto-tune ({args.days}d {args.granularity})")
    results = tune(days=args.days, granularity=args.granularity)
    write_overrides(results)
    print(f"[tune] wrote {OVERRIDES_PATH}")
    for mode, params in results["modes"].items():
        print(f"  {mode}: {params}")

    if not args.skip_runner:
        print(f"[tune] tuning trend_runner ({args.days}d {args.granularity})...")
        runner = tune_runner(days=args.days, granularity=args.granularity)
        print(f"[tune] wrote {RUNNER_PATH}: {runner['params']} "
              f"(score {runner['score']:+.2f}%)")

    # short human report for the state history
    hist = os.path.join(os.path.dirname(OVERRIDES_PATH), "history")
    os.makedirs(hist, exist_ok=True)
    path = os.path.join(hist, f"auto-tune-{datetime.now(timezone.utc):%Y-%m-%d}.md")
    lines = [f"# Guard auto-tune - {results['tuned_at']}",
             f"Data: {args.days}d {args.granularity}; OOS excess, trial-capped", ""]
    for mode, params in results["modes"].items():
        lines.append(f"- **{mode}**: atr_hurdle {params.get('atr_hurdle_pct')}, "
                     f"trend_sma {params.get('trend_sma')}, confirm {params.get('confirm_bars')} "
                     f"(score {params.get('score')}%)")
    if not args.skip_runner:
        lines.append(f"- **trend_runner**: {json.dumps(runner['params'])} "
                     f"(score {runner['score']:+.2f}%)")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[tune] report -> {path}")


if __name__ == "__main__":
    main()
