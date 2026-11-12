"""ChaosMimic: a stand-in for Chipzen's Chaos_Incarnate, for the simulator.

    from chaos_mimic import ChaosMimic
    ("CHAOS", ChaosMimic(seed))

Built from its first 16 matches (541 heads-up hands, Nov 12, every hand's cards shown):

  * Opens about 80% of small blinds to 2.5 BB, 3-bets about 30% of opens to 4x, and folds
    about 79% of the time to a 3-bet.
  * Bets about 90% of flops and turns when it can bet first. The size gives the hand away:
    a third of the pot with weak and medium hands (0 strong hands in 212 small bets), and
    the pot or more with strong ones (ahead 82-87% of the time).
  * Raises are strong about 72% of the time and large (around 5x the bet).
  * Gives up easily: folds about half the time to a flop bet or raise and 86% to a river bet.

It beat bots that read every bet as a likely bluff because it bets so often.
"""
import random

from bot import Action
from deep_thought_v1 import _PCT
from opponent_bots import preflop_raises, raise_to, check_fold, stack_bb
from poker_core import classify_hand, hand_class

STRONG = ("MONSTER", "STRONG")


class ChaosMimic:
    OPEN_SHARE = 0.80            # best share of hands opened from the small blind
    OPEN_SIZE_BB = 2.5
    THREEBET_SHARE = 0.30        # best share of hands 3-bet over an open
    THREEBET_MULT = 4.0
    DEFEND_SHARE = 0.85          # best share of hands that continue against an open
    CONTINUE_VS_3BET = 0.20      # best share that calls a 3-bet; the rest fold
    JAM_BELOW_BB = 12            # stack depth where an open becomes a shove
    SMALL, BIG = 0.33, 1.0       # bet sizes, as shares of the pot
    BET_FREQ = {"flop": 0.90, "turn": 0.85, "river": 0.30}   # weak or medium hand, betting first
    BIG_WHEN_NOT_STRONG = {"MEDIUM": 0.20, "STRONG_DRAW": 0.20, "DRAW": 0.05, "WEAK": 0.05}
    RAISE_MULT = 5.0             # a raise goes to this many times the bet
    RAISE_BLUFF = 0.05           # raises with a weak hand this often
    CALL_WEAK = 0.12             # floats a bet with a weak hand this often
    CALL_PRICE = 0.34            # medium hands and draws call up to this price

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def _preflop(self, s, r):
        pct = _PCT.get(hand_class(*s.hole_cards), 1.0)       # 0 = best hand
        raises, high, _ = preflop_raises(s)
        if raises == 0:
            if s.to_call == 0:                               # big blind after a limp
                return raise_to(s, 3 * s.big_blind) if pct <= self.THREEBET_SHARE else Action("check")
            if pct > self.OPEN_SHARE:
                return check_fold(s)
            if stack_bb(s) <= self.JAM_BELOW_BB:
                return Action("allin")
            return raise_to(s, self.OPEN_SIZE_BB * s.big_blind)
        if raises == 1:
            if pct <= self.THREEBET_SHARE:
                return raise_to(s, high * self.THREEBET_MULT)
            return Action("call") if pct <= self.DEFEND_SHARE else check_fold(s)
        if pct <= 0.04:
            return Action("allin")
        return Action("call") if pct <= self.CONTINUE_VS_3BET else check_fold(s)

    def _postflop(self, s, r):
        bucket = classify_hand(s.hole_cards, s.board)
        if s.to_call == 0:
            if bucket in STRONG:
                return raise_to(s, s.me.bet + self.BIG * s.pot)
            if r >= self.BET_FREQ[s.street]:
                return Action("check")
            big = self.rng.random() < self.BIG_WHEN_NOT_STRONG.get(bucket, 0.05)
            return raise_to(s, s.me.bet + (self.BIG if big else self.SMALL) * s.pot)
        price = s.to_call / (s.pot + s.to_call)
        if bucket in STRONG:
            if s.me.stack <= s.to_call:
                return Action("call")
            return raise_to(s, self.RAISE_MULT * (s.me.bet + s.to_call))
        if s.street == "river":
            return Action("call") if bucket == "MEDIUM" and r < 0.2 else Action("fold")
        if bucket in ("MEDIUM", "STRONG_DRAW", "DRAW"):
            return Action("call") if price <= self.CALL_PRICE else Action("fold")
        if r < self.RAISE_BLUFF and s.me.stack > s.to_call:
            return raise_to(s, self.RAISE_MULT * (s.me.bet + s.to_call))
        return Action("call") if r < self.RAISE_BLUFF + self.CALL_WEAK and price <= self.CALL_PRICE else Action("fold")

    def act(self, s):
        r = self.rng.random()
        return self._preflop(s, r) if s.street == "preflop" else self._postflop(s, r)
