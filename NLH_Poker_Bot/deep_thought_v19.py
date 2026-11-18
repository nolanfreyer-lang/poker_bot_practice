"""DEEP_THOUGHT_V19: DEEP_THOUGHT_V18 that calls with strong hands far ahead of the price, however big the pot.

    from deep_thought_v19 import DeepThoughtV19Bot
    ("DEEP19", DeepThoughtV19Bot(seed=seed))

v4 caps how many chips each class of hand may have in the pot after a call (STRONG 60 BB,
MEDIUM 30 BB, ...), and the cap overrides the equity check. Against MercuryRetrograde
(Chipzen, Nov 18, hand 3) v17 bet 1.5x pot on the flop and turn with trip aces on A-A-7-5,
then folded to a 4,400 river shove: calling would have put 84 BB in, over the STRONG cap,
although it rated itself 79-92% against a 26% price.

v19 skips the cap when our equity beats the breakeven by at least CALL_LIMIT_EDGE, for
STRONG hands only (MONSTER has no cap; MEDIUM and weaker keep it). The option lives in
deep_thought_v4.py, off by default, so earlier versions are unchanged. Hand 3 now calls with
every trip ace; 88 and K7 still fold.

Tested against v18 (paired, bb/100), STRONG only / any class:
  heads-up field 100 BB  +3.7 +- 1.6 / +3.6 +- 1.7      250 BB  +5.1 +- 2.1 / +2.0 +- 2.7
  MANIAC                 +16.9 +- 4.4 / +18.5 +- 4.9
  Choas_incarnate mimic  +8.3 +- 4.6 / -5.3 +- 7.5     Chaos mimic +5.3 +- 2.9 / -1.5 +- 5.1
  strong heads-up 100 BB +0.5 +- 1.4 / -1.9 +- 2.4
  6-max weak 100 BB      +2.1 +- 1.3 / +2.7 +- 1.4      250 BB  +2.1 +- 2.8 / +3.4 +- 2.9
  6-max strong 100 BB    +4.5 +- 2.1 / +3.3 +- 2.3      250 BB  +0.7 +- 2.8 / -1.6 +- 3.4
STRONG only never tested below zero. An edge of 0.45 did almost nothing (+-0.5 heads-up).
"""
import config as default_config
from deep_thought_v18 import DeepThoughtV18Bot

DEEP19_CONFIG = {
    "CALL_LIMIT_EDGE": 0.30,
    "CALL_LIMIT_EDGE_BUCKETS": ("STRONG",),
}


class DeepThoughtV19Bot(DeepThoughtV18Bot):
    def __init__(self, cfg=default_config, overrides=None, seed=None):
        super().__init__(cfg, {**DEEP19_CONFIG, **(overrides or {})}, seed)
