"""DEEP_THOUGHT_V12: DEEP_THOUGHT_V11 (v10 with refreshed profiles) that stays pot committed.

    from deep_thought_v12 import DeepThoughtV12Bot
    ("DEEP12", DeepThoughtV12Bot(seed=seed))

v4's stack-off limit folds a call once the chips we'd have in the pot pass a cap for the
hand's strength (60 BB for STRONG). It counted chips already in the pot, so in hand 34
against Shuck1e1-Ex (Nov 9) v11 called off 54 BB with top pair, then folded the river
for its last 6.9 BB into a 126 BB pot: it needed 5.7% equity, had 38% by its own
estimate, and held the winner. v12 drops the limit when the call needs less than
CALL_LIMIT_MAX_PRICE of the final pot; the usual equity check still decides those calls.
"""
import config as default_config
from deep_thought_v10 import DeepThoughtV10Bot

DEEP12_CONFIG = {
    "CALL_LIMIT_MAX_PRICE": 0.20,       # calls needing under 20% equity skip the stack-off limit
}


class DeepThoughtV12Bot(DeepThoughtV10Bot):
    def __init__(self, cfg=default_config, overrides=None, seed=None):
        super().__init__(cfg, {**DEEP12_CONFIG, **(overrides or {})}, seed)
