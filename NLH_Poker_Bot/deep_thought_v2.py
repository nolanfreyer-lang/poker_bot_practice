"""DEEP_THOUGHT_V2: DEEP_THOUGHT_V1 with fixes from its Slumbot and Chipzen losses.

    from deep_thought_v2 import DeepThoughtV2Bot
    ("DEEP2", DeepThoughtV2Bot(seed=seed))

V1 lost by betting too big, so V2 changes what those bets are built on:
  1. Responses come from fitted_tables_v2.py: one table per street, with fold rates
     that rise smoothly with bet size, so a bigger bet is never free in the EV sum.
  2. Each kind of hand has a limit on the chips it may put in by betting or raising.
  3. A postflop bet only becomes an all-in when almost nothing would be left behind.
     V1 shoved any bet of 40% of its stack, which turned pot-sized bets into all-ins.
  4. Bets are sized to what the opponent can actually call.
  5. With other players still in, a shove is only called with a tight range. V1 priced
     the call against the shover alone and ignored the players left to act.
  6. No betting into an empty side pot: folds win nothing there.
"""
from bisect import bisect_right

import config as default_config
from bot import Action
from deep_thought_v1 import DeepThoughtV1Bot
from fitted_tables_v2 import FIT_RESPONSE_V2, SIZE_POINTS
from poker_core import classify_hand, hand_class, parse_range

DEEP2_CONFIG = {
    # --- stack-off limits: chips a hand may have in after betting or raising ----
    # The smaller of big blinds and share of the effective stack applies.
    "STACK_OFF_BB": {"STRONG": 40, "STRONG_DRAW": 25, "MEDIUM": 20, "DRAW": 15, "WEAK": 12},
    "STACK_OFF_SHARE": {"STRONG": 1.0, "STRONG_DRAW": 0.6, "MEDIUM": 0.5, "DRAW": 0.35, "WEAK": 0.25},
    # --- planning ---------------------------------------------------------------
    "PLAN_MIN_EQUITY": 0.8,         # V1 used 0.6, which stacked off with one pair
    "PLAN_MAX_SPR": 6,              # above this, three bets are too big a plan
    # --- all-ins ----------------------------------------------------------------
    "JAM_LEFTOVER": 0.2,            # jam if less than this share of the pot would be left behind
    "MULTIWAY_CALL_SHOVE": "TT+, AQs+, AQo+",
    "DRY_SIDE_POT_EQUITY": 0.85,    # bet into an empty side pot only with this much equity
}


class DeepThoughtV2Bot(DeepThoughtV1Bot):
    def __init__(self, cfg=default_config, overrides=None, seed=None):
        super().__init__(cfg, {**DEEP2_CONFIG, **(overrides or {})}, seed)
        self.multiway_call_shove = parse_range(self.cfg.MULTIWAY_CALL_SHOVE)
        self.street = "flop"

    # ------------------------------------------------------------ chip helpers
    @staticmethod
    def _invested(s, seat):
        """Chips this seat has put in over the whole hand."""
        street_max = {}
        for st, who, kind, amt in s.history:
            if who == seat and kind not in ("check", "fold"):
                street_max[st] = max(street_max.get(st, 0), amt)
        return sum(street_max.values())

    def _spare_behind(self, s):
        """Chips we can actually lose: more than the biggest opposing stack can't be called."""
        others = [p.stack + p.bet for p in s.players if p.active and p.seat != s.my_seat and not p.all_in]
        return min(s.me.stack, max(0, max(others) - s.me.bet)) if others else 0

    def _stack_off_limit(self, s, bucket):
        c = self.cfg
        if bucket not in c.STACK_OFF_BB:
            return float("inf")
        behind_start = self._invested(s, s.my_seat) + self._spare_behind(s)
        return min(c.STACK_OFF_BB[bucket] * s.big_blind, c.STACK_OFF_SHARE[bucket] * behind_start)

    def _dry_side_pot(self, s):
        """True when an opponent is all-in, others remain, and no chips are in a side pot."""
        shoved = [p for p in s.players if p.active and p.all_in and p.seat != s.my_seat]
        live = [p for p in s.players if p.active and not p.all_in and p.seat != s.my_seat]
        if not shoved or not live:
            return False
        cap = max(self._invested(s, p.seat) for p in shoved)
        return all(self._invested(s, p.seat) <= cap for p in s.players if p.active)

    # ------------------------------------------------------------ EV helpers
    def _response(self, seat, bucket, frac):
        """(fold, call, raise) for this street, read between the fitted sizes."""
        c = self.cfg
        row = FIT_RESPONSE_V2[self.street][bucket]
        i = bisect_right(SIZE_POINTS, frac)
        if i == 0:
            mix = row[0]
        elif i == len(SIZE_POINTS):
            mix = row[-1]
        else:
            t = (frac - SIZE_POINTS[i - 1]) / (SIZE_POINTS[i] - SIZE_POINTS[i - 1])
            mix = [a + t * (b - a) for a, b in zip(row[i - 1], row[i])]
        read = self.sd.fold_rate(seat, c) / c.FOLD_PRIOR
        fold = min(0.95, mix[0] * min(1.5, max(0.5, read)))
        rest = (1 - fold) / max(1e-9, mix[1] + mix[2])
        return fold, mix[1] * rest, mix[2] * rest

    def _sizes(self, s, eq):
        """Pot fractions to try: within the stack-off limit and what they can call."""
        c = self.cfg
        spare = self._spare_behind(s)
        fractions = list(c.BET_SIZES)
        streets_left = {"flop": 3, "turn": 2, "river": 1}[s.street]
        if c.PLANNING and eq >= c.PLAN_MIN_EQUITY and 0 < spare <= c.PLAN_MAX_SPR * s.pot:
            fractions.append((((s.pot + 2 * spare) / s.pot) ** (1 / streets_left) - 1) / 2)
        bucket = classify_hand(s.hole_cards, s.board, self.good_kicker)
        room = self._stack_off_limit(s, bucket) - self._invested(s, s.my_seat)
        return [f for f in fractions if f * s.pot <= room and (f * s.pot <= spare or spare == 0)] \
            or ([spare / s.pot] if 0 < spare <= room else [])

    # ------------------------------------------------------------ decisions
    def _preflop(self, s):
        a = super()._preflop(s)
        if a.kind not in ("call", "allin") or s.to_call == 0:
            return a
        ctx = self._preflop_context(s)
        raiser = s.players[ctx.last_raiser] if ctx.last_raiser is not None else None
        others = [p for p in s.players if p.active and p.seat != s.my_seat]
        if raiser is not None and raiser.all_in and ctx.raises >= 2 and len(others) >= 2 \
                and hand_class(*s.hole_cards) not in self.multiway_call_shove:
            return Action("fold")
        return a

    def _postflop(self, s):
        c = self.cfg
        self.street = s.street
        a = super()._postflop(s)
        if a.kind not in ("bet", "raise", "allin"):
            return a
        ev = self.last_ev or {}
        if self._dry_side_pot(s) and s.to_call == 0 and ev.get("equity", 0) < c.DRY_SIDE_POT_EQUITY:
            return Action("check")
        bucket = classify_hand(s.hole_cards, s.board, self.good_kicker)
        cost = s.me.stack if a.kind == "allin" else a.amount - s.me.bet
        if self._invested(s, s.my_seat) + cost <= self._stack_off_limit(s, bucket):
            return a
        if s.to_call == 0:
            return Action("check")
        be = ev.get("breakeven")
        return Action("call") if be is None or ev.get("equity", 0) >= be else Action("fold")

    def _legalize(self, s, a):
        if s.street == "preflop" or a.kind not in ("bet", "raise"):
            return super()._legalize(s, a)
        amt = max(int(a.amount), s.min_raise_to)
        cost = amt - s.me.bet
        left = s.me.stack - cost
        spare = self._spare_behind(s)
        if s.min_raise_to >= s.max_raise_to or amt >= s.max_raise_to or left <= 0 \
                or (spare > 0 and spare - cost <= self.cfg.JAM_LEFTOVER * (s.pot + 2 * cost)):
            return Action("allin", s.max_raise_to)
        return Action("bet" if s.me.bet + s.to_call == 0 else "raise", amt)
