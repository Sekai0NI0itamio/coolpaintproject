"""Zoo roster: one bot per community-classic model, fixed canonical
parameters, each starting with the same capital. No evolution here —
the zoo measures how published strategies perform AS-IS on live data.
"""
from __future__ import annotations

from typing import List, Tuple

from bot.paper.account import PaperAccount
from bot.swarm.genome import Genome
from bot.swarm.population import Agent, Population

# (bot id, strategy name, params, account overrides)
ROSTER: List[Tuple[str, str, dict, dict]] = [
    ("momentum", "momentum", {}, {}),
    ("mean_reversion", "mean_reversion", {}, {}),
    ("macd_cross", "macd_cross", {}, {}),
    ("golden_cross", "golden_cross", {}, {}),
    ("donchian_breakout", "donchian_breakout", {}, {}),
    ("rsi2", "rsi2", {}, {}),
    ("stochastic_reversion", "stochastic_reversion", {}, {}),
    ("bbands_breakout", "bbands_breakout", {}, {}),
    ("grid_trader", "grid_trader", {}, {}),
    ("dca_bot", "dca_bot", {}, {"allow_averaging": True}),
]


def build_zoo_population(pairs: List[str], granularity: str,
                         capital: float, fee_cfg: dict) -> Population:
    pop = Population(pairs=pairs, granularity=granularity,
                     capital=capital, fee_cfg=fee_cfg, strategy="zoo")
    pop.agents = []
    for bot_id, strategy_name, params, overrides in ROSTER:
        genome = Genome(id=bot_id, strategy=strategy_name, params=dict(params))
        acc = PaperAccount(capital=capital, **fee_cfg)
        for key, value in overrides.items():
            setattr(acc, key, value)
        pop.agents.append(Agent(genome=genome, account=acc, equity=capital))
    return pop
