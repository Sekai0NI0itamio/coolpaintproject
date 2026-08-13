"""Swarm population: 40 agents, daily selection, cloning with mutation.

State is a single small JSON file (``state/population.json``) so it can
be committed to the repo between GitHub Actions runs.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bot.paper.account import PaperAccount
from bot.strategies import build_strategy
from bot.strategies.base import Strategy
from bot.swarm.genome import Genome, make_genome, mutate

POP_SIZE = 40
TOP_K = 5
CLONES_PER_SURVIVOR = 7    # 5 survivors x 7 clones + 5 immigrants = 40
IMMIGRANTS = 5             # fresh random tunings each generation (anti-convergence)
MIN_TRADES = 3             # bots must actually trade to qualify for survival
DEFAULT_CAPITAL = 20.0
VERSION = 2


@dataclass
class Agent:
    genome: Genome
    account: PaperAccount
    equity: float = DEFAULT_CAPITAL
    _strategy: Optional[Strategy] = None

    @property
    def strategy(self) -> Strategy:
        if self._strategy is None:
            self._strategy = build_strategy(self.genome.strategy, self.genome.params)
        return self._strategy

    def to_dict(self) -> dict:
        return {"genome": self.genome.to_dict(),
                "account": self.account.state_dict(),
                "equity": self.equity}

    @classmethod
    def from_dict(cls, d: dict, fee_cfg: dict) -> "Agent":
        genome = Genome.from_dict(d["genome"])
        acc = PaperAccount.from_dict(d["account"], **fee_cfg)
        return cls(genome=genome, account=acc, equity=d.get("equity", acc.cash))


class Population:
    def __init__(self, pairs: List[str], granularity: str,
                 capital: float = DEFAULT_CAPITAL,
                 fee_cfg: Optional[dict] = None,
                 strategy: str = "mean_reversion"):
        self.pairs = pairs
        self.granularity = granularity
        self.capital = capital
        self.fee_cfg = fee_cfg or {}
        self.strategy = strategy
        self.agents: List[Agent] = []
        self.generation = 0
        self.day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.last_ts = 0
        self.history: List[dict] = []   # per-generation selection summaries

    # ------------------------------------------------------------- seeding
    def seed(self, n: int = POP_SIZE, strategy: str = "mean_reversion",
             seeds: Optional[List[dict]] = None, rng_seed: int = 7) -> None:
        """Create the initial population.

        If ``seeds`` (from run_train.py) is given, clone each of the top
        seeds into n/len(seeds) mutated children; otherwise sample random
        tunings around the strategy defaults.
        """
        rng = random.Random(rng_seed)
        self.agents = []
        self.strategy = strategy
        if seeds:
            per = max(1, n // len(seeds))
            i = 0
            for seed in seeds:
                base = Genome(id=f"seed-{seed['strategy']}", strategy=seed["strategy"],
                              params=dict(seed["params"]))
                for c in range(per):
                    if len(self.agents) >= n:
                        break
                    gid = f"g0-{i:02d}"
                    child = mutate(base, gid, rng, strength=0.05) if c else \
                        Genome(id=gid, strategy=base.strategy, params=dict(base.params))
                    self.agents.append(self._fresh_agent(child))
                    i += 1
            while len(self.agents) < n:
                gid = f"g0-{len(self.agents):02d}"
                g = make_genome(strategy, gid, rng)
                self.agents.append(self._fresh_agent(g))
        else:
            for i in range(n):
                gid = f"g0-{i:02d}"
                g = make_genome(strategy, gid, rng)
                self.agents.append(self._fresh_agent(g))
        self.generation = 0

    def _fresh_agent(self, genome: Genome) -> Agent:
        acc = PaperAccount(capital=self.capital, **self.fee_cfg)
        return Agent(genome=genome, account=acc, equity=self.capital)

    # ---------------------------------------------------------- selection
    def mark_equity(self, prices: Dict[str, float]) -> None:
        for agent in self.agents:
            agent.equity = agent.account.equity(prices)

    def leaderboard(self) -> List[Agent]:
        return sorted(self.agents,
                      key=lambda a: (a.equity, a.account.realized_pnl, a.genome.id),
                      reverse=True)

    def select_and_repopulate(self, top_k: int = TOP_K,
                              clones: int = CLONES_PER_SURVIVOR,
                              immigrants: int = IMMIGRANTS,
                              min_trades: int = MIN_TRADES,
                              rng_seed: Optional[int] = None) -> dict:
        """Daily evolution step.

        Fitness is NET WORTH: cash + open positions marked at the latest
        market prices (call ``mark_equity`` right before this) — so a bot
        holding a coin that pumped 40% ranks as rich even with $0 cash.

        Anti-gaming rules:
        * A bot needs >= ``min_trades`` closed trades to be ELIGIBLE.
          "Buy and never sell" bots cannot win on unrealized equity alone;
          if too few bots qualify, remaining slots fall back to best equity.
        * ``immigrants`` fresh random genomes join every generation so the
          swarm can't converge on a single trick.

        Survivors are cloned with slight mutations; everyone restarts with
        fresh capital. Returns a summary for reporting.
        """
        rng = random.Random(rng_seed if rng_seed is not None
                            else int(datetime.now(timezone.utc).timestamp()))
        board = self.leaderboard()
        eligible = [a for a in board if a.account.n_trades >= min_trades]
        survivors = eligible[:top_k]
        fallback_used = 0
        if len(survivors) < top_k:
            for a in board:
                if len(survivors) >= top_k:
                    break
                if a not in survivors:
                    survivors.append(a)
                    fallback_used += 1
        summary = {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "generation_finished": self.generation,
            "min_trades_gate": min_trades,
            "disqualified_no_trades": len(board) - len(eligible),
            "fallback_survivors": fallback_used,
            "final_standings": [
                {"id": a.genome.id, "strategy": a.genome.strategy,
                 "equity": round(a.equity, 4),
                 "revenue": round(a.equity - self.capital, 4),
                 "realized_pnl": round(a.account.realized_pnl, 4),
                 "trades": a.account.n_trades,
                 "eligible": a.account.n_trades >= min_trades,
                 "holdings": {p: round(pos.qty, 8)
                              for p, pos in a.account.positions.items()},
                 "params": a.genome.params}
                for a in board
            ],
            "survivors": [a.genome.id for a in survivors],
        }
        new_agents: List[Agent] = []
        i = 0
        self.generation += 1
        for survivor in survivors:
            for c in range(clones):
                gid = f"g{self.generation}-{i:02d}"
                strength = 0.05 if c == 0 else 0.15   # keep one near-clone of champion
                child = mutate(survivor.genome, gid, rng, strength=strength)
                new_agents.append(self._fresh_agent(child))
                i += 1
        for _ in range(immigrants):
            if len(new_agents) >= POP_SIZE:
                break
            gid = f"g{self.generation}-{i:02d}"
            g = make_genome(self.strategy, gid, rng)
            g.lineage = ["immigrant"]
            new_agents.append(self._fresh_agent(g))
            i += 1
        self.agents = new_agents[:POP_SIZE]
        # keep last_ts as-is: the new generation starts trading from the
        # selection moment; warmup history is re-fetched by the runner
        self.history.append(summary)
        return summary

    def maybe_rollover(self, top_k: int = TOP_K, clones: int = CLONES_PER_SURVIVOR,
                       immigrants: int = IMMIGRANTS, min_trades: int = MIN_TRADES
                       ) -> Optional[dict]:
        """Perform daily selection if the UTC day has changed."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today <= self.day or not self.agents:
            return None
        summary = self.select_and_repopulate(top_k, clones, immigrants, min_trades)
        self.day = today
        return summary

    # -------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "version": VERSION,
            "generation": self.generation,
            "day": self.day,
            "last_ts": self.last_ts,
            "granularity": self.granularity,
            "pairs": self.pairs,
            "capital": self.capital,
            "strategy": self.strategy,
            "agents": [a.to_dict() for a in self.agents],
            "history": self.history[-14:],
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str, fee_cfg: Optional[dict] = None) -> "Population":
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        pop = cls(pairs=d["pairs"], granularity=d["granularity"],
                  capital=d.get("capital", DEFAULT_CAPITAL), fee_cfg=fee_cfg,
                  strategy=d.get("strategy", "mean_reversion"))
        pop.generation = d.get("generation", 0)
        pop.day = d.get("day")
        pop.last_ts = d.get("last_ts", 0)
        pop.history = d.get("history", [])
        pop.agents = [Agent.from_dict(a, pop.fee_cfg) for a in d.get("agents", [])]
        return pop

    @classmethod
    def load_or_seed(cls, path: str, pairs: List[str], granularity: str,
                     capital: float, fee_cfg: dict, n: int = POP_SIZE,
                     strategy: str = "mean_reversion",
                     seeds: Optional[List[dict]] = None) -> "Population":
        if os.path.exists(path):
            return cls.load(path, fee_cfg)
        pop = cls(pairs=pairs, granularity=granularity, capital=capital, fee_cfg=fee_cfg)
        pop.seed(n=n, strategy=strategy, seeds=seeds)
        pop.save(path)
        return pop