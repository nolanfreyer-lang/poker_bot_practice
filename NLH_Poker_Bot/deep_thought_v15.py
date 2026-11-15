"""DEEP_THOUGHT_V15: DEEP_THOUGHT_V14 that needs shown weak hands before it widens big bets and raises.

    from deep_thought_v15 import DeepThoughtV15Bot
    ("DEEP15", DeepThoughtV15Bot(seed=seed))

v14 believes a frequent bettor's widened range for a kind of bet unless their shown bets of
that kind are *stronger* than the field's. Chaos_Incarnate (Chipzen, Nov 12-14) raises
often and its shown raises are 29% weak, the field's share (28%), so v14 kept the full raise
read (cut only to 0.62 by its fold rate). It called a 64 BB all-in check-raise with bottom
pair (K4 on Q43) thinking it had 62% equity; Chaos held AQ.

A bettor whose bets are field-typical in strength, however often they bet, is not betting
wider. So v15, for big bets and raises, also asks how much *weaker* than the field their
shown bets of that kind are:

    excess = (weak share - field share) / (FULL - field share),   FULL = p + EXCESS_SPAN * (1 - p)

clamped to 0-1 and raised to EXCESS_POWER. The weak share starts at FULL (counted as
EXCESS_WEIGHT shown bets), so nothing changes before they show down; each strong hand shown
pulls it down. The lowest of this, v14's shown trust and v14's fold-rate trust is used. A
maniac shows weak raises and keeps full trust. Small and mid bets keep v14's rule.

Everything else is v14. Profiles are the same as v14's.
"""
import config as default_config
from deep_thought_v14 import DeepThoughtV14Bot

DEEP15_CONFIG = {
    "EXCESS_TRUST": True,
    "EXCESS_KINDS": ("big", "raise"),
    "EXCESS_SPAN": 0.5,        # full trust once their weak share is halfway from the field's to 100%
    "EXCESS_WEIGHT": 3,        # the starting share counts like this many shown bets
    "EXCESS_POWER": 2.0,
}


class DeepThoughtV15Bot(DeepThoughtV14Bot):
    def __init__(self, cfg=default_config, overrides=None, seed=None):
        super().__init__(cfg, {**DEEP15_CONFIG, **(overrides or {})}, seed)

    def _excess_trust(self, seat, what):
        c, sb = self.cfg, self.shown_bets
        p = c.WEAK_PRIOR[what]
        full = p + c.EXCESS_SPAN * (1 - p)
        share = (sb.weak.get((seat, what), 0) + full * c.EXCESS_WEIGHT) \
            / (sb.shown.get((seat, what), 0) + c.EXCESS_WEIGHT)
        return min(1.0, max(0.0, (share - p) / (full - p))) ** c.EXCESS_POWER

    def _bet_trust(self, seat, what):
        trust = super()._bet_trust(seat, what)
        if self.cfg.EXCESS_TRUST and what in self.cfg.EXCESS_KINDS:
            trust = min(trust, self._excess_trust(seat, what))
        return trust
