"""DEEP_THOUGHT_V9: DEEP_THOUGHT_V7 with weak hands folding to heavy bettors who fold.

    from deep_thought_v9 import DeepThoughtV9Bot
    ("DEEP9", DeepThoughtV9Bot(seed=seed))

This is v7 with exploit 4 switched on (FOLD_WEAK_VS_BARRELER, see deep_thought_v7.py).
Facing one opponent's first bet on the flop or turn, a weak made hand folds instead of
calling when that bettor bets at least 1.3x the reference rate, folds to our bets at
least 45% of the time, and has shown down at most 45% bluffs. That is Chipzen's
slopUpBotV2: it bets 72% of the time when first in and half its river bets are weak
hands, but those weak hands still beat ours and it keeps betting, so v7's calls with
them added up to -66 bb/100 heads-up against a mimic of it. A MANIAC or LAG shows its
air at showdown and doesn't count.

Paired against v7 from an empty start: +10.7 +- 3.0 bb/100 at 100 BB heads-up against
the slopUpBotV2 mimic (+9.9 +- 1.8 at 40 BB with a 35% limit), small gains at 6-max
tables with it, and -0.04 / -0.12 heads-up against the usual field (-0.6 / -2.6 against
LAG). Its saved profile (37 hands, 38% bluffs) passes the 45% limit, so the rule is on
against it from the first hand. Everything else is v7, and v9 reads the same saved
Chipzen profiles (chipzen_profiles.py --bot v9).
"""
import config as default_config
from deep_thought_v7 import DeepThoughtV7Bot

DEEP9_CONFIG = {
    "FOLD_WEAK_VS_BARRELER": True,
}


class DeepThoughtV9Bot(DeepThoughtV7Bot):
    def __init__(self, cfg=default_config, overrides=None, seed=None):
        super().__init__(cfg, {**DEEP9_CONFIG, **(overrides or {})}, seed)
