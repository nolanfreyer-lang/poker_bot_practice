"""DEEP_THOUGHT_V17: DEEP_THOUGHT_V15 that judges raises against how weak the field's raises are.

    from deep_thought_v17 import DeepThoughtV17Bot
    ("DEEP17", DeepThoughtV17Bot(seed=seed))

v1 weighs a raise by the size of the bet it was made over, never by its own size, so its
starting range for a raise (before any reads) is far too weak: against Choas_incarnate's
64 BB all-in flop raise (Chipzen, Nov 16) it held 55% weak hands. Across every heads-up
Chipzen replay with all cards shown (2,629 postflop raises), raises were weak hands or
draws 40% of the time on the flop, 25% on the turn and 13% on the river, and all-in
raises 21% against 35% for the rest.

v17 builds the range a second time with the bluff and frequency reads neutral. When we
face a raise and that neutral range's weak/draw share is above the field's for the street
(x ALLIN_RAISE_MULT for an all-in), every weak hand and draw is scaled by the same factor,
in the read-adjusted range too, so the reads still move it from a sound starting point.

Tested against v16 (paired, bb/100): +1.5 +- 0.95 against the heads-up field at 100 BB
(+8.6 against v4, +7.6 against GTO, +0.1 against MANIAC), -0.8 +- 0.7 against a MANIAC
alone, +2.3 +- 3.4 against a Choas_incarnate mimic.

Not used, kept as options: capping how far the reads may stack (WIDEN_CAP) does fold
K-high against that shove, but a real maniac does raise that much air: -56 bb/100 against
MANIAC with a cap of 3. Capping only the frequency reads (CAP_KEEP_SHOWN) still lost 35,
and judging the raise share against the shown-bluff range without a cap lost 5.6.

Everything else is v15 (v16 is v15's code with refreshed profiles).
"""
import config as default_config
from deep_thought_v15 import DeepThoughtV15Bot

DEEP17_CONFIG = {
    "WIDEN_CAP": None,              # most the reads together may scale a weak hand's weight (tested 3); None = off
    "CAP_KEEP_SHOWN": False,        # True: the shown bluff rate counts in full, only frequency reads are capped
    "CAP_MIN_FOLD": None,           # cap only bettors who fold to our bets at least this often
    # Facing a raise, the neutral range's weak/draw share is brought down to the field's.
    # Every heads-up Chipzen replay with all cards (2,629 postflop raises): flop 40%, turn
    # 25%, river 13% weak or draws; all-in raises 21% against 35% for the rest.
    "RAISE_WEAK_SHARE": {"flop": 0.40, "turn": 0.25, "river": 0.13},    # None = off
    "ALLIN_RAISE_MULT": 0.6,
}
POSTFLOP_STATS = ("bet", "raise", "check_raise", "all_in")


class DeepThoughtV17Bot(DeepThoughtV15Bot):
    def __init__(self, cfg=default_config, overrides=None, seed=None):
        super().__init__(cfg, {**DEEP17_CONFIG, **(overrides or {})}, seed)
        self._neutral = False

    def _freq_ratio(self, s, seat, stat):
        if self._neutral and stat in POSTFLOP_STATS:
            return 1.0
        return super()._freq_ratio(s, seat, stat)

    def _neutral_range(self, s, seat):
        """The opponent's range with the bluff, frequency and street reads at their defaults."""
        c, sd, streets = self.cfg, self.sd, self.street_reads
        if not c.CAP_KEEP_SHOWN:
            sd.bluff_rate = lambda *a, **k: c.BLUFF_PRIOR
        streets.ratio = lambda *a, **k: 1.0
        self._neutral = True
        try:
            return super()._opponent_range(s, seat)
        finally:
            self._neutral = False
            del streets.ratio
            if not c.CAP_KEEP_SHOWN:
                del sd.bluff_rate

    def _raise_scale(self, s, seat, neutral):
        """Factor on weak/draw weights that brings a raise's neutral weak share down to the field's."""
        c = self.cfg
        if not c.RAISE_WEAK_SHARE or not s.to_call or not self._facing_raise(s) or seat != self._aggressor(s):
            return 1.0
        target = c.RAISE_WEAK_SHARE[s.street] * (c.ALLIN_RAISE_MULT if s.players[seat].all_in else 1.0)
        total = sum(h[2] for h in neutral)
        weak = sum(h[2] for h in neutral if h[3] in ("WEAK", "DRAW"))
        share = weak / total if total else 0.0
        if share <= target or share >= 1:
            return 1.0
        return target * (1 - share) / (share * (1 - target))

    def _opponent_range(self, s, seat):
        hands = super()._opponent_range(s, seat)
        c = self.cfg
        if s.street == "preflop" or not (c.WIDEN_CAP or c.RAISE_WEAK_SHARE):
            return hands
        if not c.WIDEN_CAP and not (s.to_call and self._facing_raise(s)):
            return hands            # only raises are rescaled: skip the second range
        neutral = self._neutral_range(s, seat)
        k = self._raise_scale(s, seat, neutral)
        cap = c.WIDEN_CAP
        if cap and c.CAP_MIN_FOLD is not None and self.sd.fold_rate(seat, c) < c.CAP_MIN_FOLD:
            cap = None              # they rarely fold: their big money can be air
        if not cap and k == 1.0:
            return hands
        base = {(h[0], h[1]): h[2] * k for h in neutral if h[3] in ("WEAK", "DRAW")}
        out = []
        for h in hands:
            if h[3] in ("WEAK", "DRAW"):
                w0 = base.get((h[0], h[1]))
                if w0 is None:
                    continue        # negligible without the reads: don't let them revive it
                h[2] *= k
                if cap:
                    h[2] = min(cap * w0, max(w0 / cap, h[2]))
            out.append(h)
        return out
