"""DEEP_THOUGHT_V3: DEEP_THOUGHT_V2 with min-raise check-raise bluffs and raise reads.

    from deep_thought_v3 import DeepThoughtV3Bot
    ("DEEP3", DeepThoughtV3Bot(seed=seed))

After checking and facing a bet heads-up, weak hands consider a min-raise bluff, only
on the street and stack depths where it tested positive. It starts at a reduced rate,
then each opponent's fold-to-raise rate moves it up or down.
"""
import config as default_config
from bot import Action, AGGRESSIVE
from deep_thought_v2 import DeepThoughtV2Bot
from poker_core import classify_hand

DEEP3_CONFIG = {
    # --- min-raise check-raise bluffs ---------------------------------------------
    "CHECK_RAISE_BLUFF": True,
    "CR_BUCKETS": ("WEAK", "DRAW"),
    "CR_FOLD_SCALE": 0.7,           # they fold less to a small raise than to a bet of that size
    "CR_MAX_SHARE": 0.6,            # at most this share of the effective stack in the pot
    # where it tested positive, measured bluff by bluff; turns and rivers lost chips
    "CR_STREETS": ("flop",),
    "CR_MIN_EFF_BB": 12,
    "CR_POSITIVE_BB": ((12, 20), (25, 50)),     # other depths above the minimum need a read first
    # bluff rate: starts reduced, reaches full as we see how this opponent reacts to raises
    "CR_BASE_FREQUENCY": 0.25,
    "CR_FULL_FREQUENCY": 0.5,
    "CR_READ_HANDS": 8,             # raises seen before the read has its full effect
    "CR_FOLD_PRIOR": 0.55,          # assumed fold-to-raise rate before any data
    "CR_FOLD_PRIOR_WEIGHT": 4,
    "CR_MIN_FOLD_RATE": 0.45,       # below this, stop bluffing them
}


class FoldToRaiseStats:
    """How often each opponent folds to a postflop raise, from every raise they face."""

    def __init__(self):
        self.folds = {}
        self.chances = {}

    def observe(self, history):
        for street in ("flop", "turn", "river"):
            high, raiser, responded = 0, None, set()
            for st, seat, kind, amt in history:
                if st != street:
                    continue
                if kind in AGGRESSIVE and amt > high:
                    if high > 0:
                        raiser, responded = seat, set()
                    high = amt
                elif raiser is not None and seat != raiser and seat not in responded:
                    responded.add(seat)
                    self.chances[seat] = self.chances.get(seat, 0) + 1
                    if kind == "fold":
                        self.folds[seat] = self.folds.get(seat, 0) + 1

    def seen(self, seat):
        return self.chances.get(seat, 0)

    def rate(self, seat, cfg):
        w = cfg.CR_FOLD_PRIOR_WEIGHT
        return (self.folds.get(seat, 0) + cfg.CR_FOLD_PRIOR * w) / (self.seen(seat) + w)


class DeepThoughtV3Bot(DeepThoughtV2Bot):
    def __init__(self, cfg=default_config, overrides=None, seed=None):
        super().__init__(cfg, {**DEEP3_CONFIG, **(overrides or {})}, seed)
        self.last_hands = []
        self.raise_reads = FoldToRaiseStats()

    def on_hand_end(self, history, seats_dealt, showdown=None, board=None, my_seat=None):
        super().on_hand_end(history, seats_dealt, showdown=showdown, board=board, my_seat=my_seat)
        self.raise_reads.observe(history)

    def _hand_equities(self, s, hands):
        self.last_hands = super()._hand_equities(s, hands)
        return self.last_hands

    def _bluff_frequency(self, seat, effective_bb):
        """Reduced until we have a read, higher against folders, zero against callers."""
        c = self.cfg
        if self.raise_reads.rate(seat, c) < c.CR_MIN_FOLD_RATE:
            return 0.0
        positive = any(low < effective_bb <= high for low, high in c.CR_POSITIVE_BB)
        if not positive and self.raise_reads.seen(seat) < c.CR_READ_HANDS:
            return 0.0
        confidence = min(1.0, self.raise_reads.seen(seat) / c.CR_READ_HANDS)
        return c.CR_BASE_FREQUENCY + confidence * (c.CR_FULL_FREQUENCY - c.CR_BASE_FREQUENCY)

    def _check_raise_bluff(self, s, a):
        """Min-raise instead of `a` (a call or fold) when it is +EV as a bluff."""
        c = self.cfg
        opps = [p for p in s.players if p.active and p.seat != s.my_seat]
        checked = any(st == s.street and seat == s.my_seat and kind == "check" for st, seat, kind, _ in s.history)
        if not (c.CHECK_RAISE_BLUFF and s.street in c.CR_STREETS and checked and s.to_call
                and len(opps) == 1 and not opps[0].all_in
                and s.min_raise_to < s.max_raise_to
                and classify_hand(s.hole_cards, s.board, self.good_kicker) in c.CR_BUCKETS):
            return a
        cost = s.min_raise_to - s.me.bet
        spare = self._spare_behind(s)
        in_pot = self._invested(s, s.my_seat) + cost
        effective = self._invested(s, s.my_seat) + spare
        if cost >= spare or in_pot > c.CR_MAX_SHARE * effective \
                or effective / s.big_blind <= c.CR_MIN_EFF_BB:
            return a
        matched = s.to_call
        pot_after = s.pot + matched
        bet = cost - matched
        total = ev = 0.0
        for h in self.last_hands:
            fold, call, rais = self._response(opps[0].seat, h[3], bet / max(1, pot_after))
            read = self.raise_reads.rate(opps[0].seat, c) / c.CR_FOLD_PRIOR
            lost = fold * (1 - min(1.0, c.CR_FOLD_SCALE * read))     # folds that become calls
            fold, call = fold - lost, call + lost
            ev += h[2] * (fold * pot_after + call * (h[4] * (pot_after + 2 * bet) - bet) - rais * bet)
            total += h[2]
        ev = ev / total - matched if total else 0.0
        info = self.last_ev or {}
        call_ev = (info.get("equity", 0) * self._realize(s) - info.get("breakeven", 1)) * pot_after
        if ev > max(0.0, call_ev) and self.rng.random() < self._bluff_frequency(opps[0].seat, effective / s.big_blind):
            return Action("raise", s.min_raise_to)
        return a

    def _postflop(self, s):
        self.last_hands = []
        a = super()._postflop(s)
        return self._check_raise_bluff(s, a) if a.kind in ("call", "fold") else a
