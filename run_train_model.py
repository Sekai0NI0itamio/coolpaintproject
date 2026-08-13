#!/usr/bin/env python3
"""Resumable model-training entrypoint for the overnight pipeline.

Runs the two-stage (regime + timing) trend model on full Coinbase
history, with walk-forward CV and an untouched holdout gate. Designed to
run as a chain of time-budgeted GitHub Actions jobs: each job does as
much as it can inside ``--budget``, saves a small progress ledger
(state/training/checkpoint.json), commits, and re-dispatches itself.

Usage:
    python run_train_model.py --budget 3400            # ~57 min (seconds)
    python run_train_model.py --resume                  # continue from checkpoint
    python run_train_model.py --report                  # print standings, no training
    python run_train_model.py --gate                    # force gate on best config
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot.train.checkpoint import load_checkpoint  # noqa: E402
from bot.train.pipeline import TrainConfig, TrainingRun, _hash  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Train trend model (resumable)")
    ap.add_argument("--budget", type=int, default=2000,
                    help="seconds of work before stopping cleanly (default 2000)")
    ap.add_argument("--margin", type=int, default=120,
                    help="seconds of stop margin before hard timeout")
    ap.add_argument("--pairs", default="BTC-USDC,ETH-USDC,SOL-USDC")
    ap.add_argument("--granularity", default="ONE_HOUR")
    ap.add_argument("--days", type=int, default=1460)
    ap.add_argument("--db", default="data/train_cache/data.db")
    ap.add_argument("--checkpoint", default="state/training/checkpoint.json")
    ap.add_argument("--deployed", default="state/deployed_model.json")
    ap.add_argument("--report", action="store_true", help="print standings and exit")
    ap.add_argument("--gate", action="store_true", help="run final gate now")
    ap.add_argument("--resume", action="store_true",
                    help="continue from existing checkpoint (default)")
    args = ap.parse_args()

    cfg = TrainConfig(
        pairs=args.pairs.split(","),
        granularity=args.granularity,
        days=args.days,
        db_path=args.db,
        checkpoint_path=args.checkpoint,
        deployed_path=args.deployed,
    )

    if args.report:
        report(cfg)
        return

    runner = TrainingRun(cfg, budget_sec=args.budget, stop_margin_sec=args.margin)
    code = runner.run()
    # Continuation handshake: write state/training/.continue ONLY when the
    # run exhausted its time budget (i.e. genuinely more resumable work than
    # it could finish). When it finishes within budget -- even if nothing was
    # deployable -- we remove the marker so the workflow stops the
    # self-chaining loop instead of spinning. The nightly cron restarts
    # training on fresh data.
    cont = os.path.join(os.path.dirname(args.checkpoint), ".continue")
    if runner.out_of_time:
        with open(cont, "w", encoding="utf-8") as fh:
            fh.write(utcnow_str())
        print("[train] budget exhausted -> wrote state/training/.continue")
    else:
        if os.path.exists(cont):
            os.remove(cont)
        print("[train] finished within budget -> stopping self-chain")
    sys.exit(code)


def utcnow_str() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def report(cfg: TrainConfig) -> None:
    cp = load_checkpoint(cfg.checkpoint_path, cfg.config_hash)
    print(f"[report] config hash {cfg.config_hash}")
    if cp is None:
        print("[report] no checkpoint yet -- nothing trained.")
        return
    print(f"- CV units done: {len(cp.cv_done)}")
    print(f"- best config:  {cp.best_config}")
    if cp.deployed:
        print("- deployed model:")
        print(f"    hyper:  {cp.deployed['hyper']}")
        print(f"    excess: {cp.deployed['metrics_holdout']['excess%']}% "
              f"({cp.deployed['metrics_holdout']['total_trades']} tr, "
              f"{cp.deployed['metrics_holdout']['win%']}% win)")
        print(f"    trained_at: {cp.deployed['trained_at']}")
    else:
        print("- deployed model: none yet")


if __name__ == "__main__":
    main()