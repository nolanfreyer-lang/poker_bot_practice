"""OptimalBot: ExploitBot with the best-performing search settings.

Test alongside PokerBot with:
    from optimalbot import OptimalBot
    ("OPTIMAL", OptimalBot(seed=seed))

Override settings per instance as needed:
    OptimalBot(seed=seed, overrides={"BLUFF_FACTOR": 2.0})
"""
from bot_exploit import ExploitBot


OPTIMAL_OVERRIDES = {
    "BLUFF_FACTOR": 1.5,
    "TIGHT_VPIP": 0.20,
    "BLUFF_MULT_VS_TIGHT": 1.0,
    "CBET_BONUS_VS_TIGHT": 0.35,
}


class OptimalBot(ExploitBot):
    """Search-tuned ExploitBot; explicit per-instance overrides take precedence."""

    def __init__(self, cfg=None, overrides=None, seed=None):
        if cfg is None:
            super().__init__(overrides={**OPTIMAL_OVERRIDES, **(overrides or {})}, seed=seed)
        else:
            super().__init__(cfg=cfg, overrides={**OPTIMAL_OVERRIDES, **(overrides or {})}, seed=seed)
