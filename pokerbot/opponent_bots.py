"""
=====================================================================
  OPPONENT BOTS  --  sparring partners with different, real strategies
=====================================================================
Each bot uses the same interface as PokerBot: act(GameState) -> Action.
All of them are fast table lookups (well under the 1 ms limit).

  NitBot        very tight, raises only premium hands, gives up without a strong hand
  LagBot        loose-aggressive: wide opens, light 3-bets, frequent c-bets and barrels
  PushFoldBot   tournament shove-or-fold preflop based on stack depth
  ChenBot       preflop by Chen formula score, postflop by estimated equity vs pot odds
  TrapperBot    loose-passive: limps and calls a lot, slow-plays and check-raises monsters

TWO WAYS TO USE THEM IN local_sim.py
  A) Import (easiest): add this near the top of local_sim.py
         from opponent_bots import NitBot, LagBot, PushFoldBot, ChenBot, TrapperBot
  B) Copy-paste: copy the "SHARED HELPERS" block once, then any bot classes
     you want, into local_sim.py above default_lineup(), and add
         from poker_core import hand_class, parse_range, classify_hand
     to its imports.

Then put them in default_lineup(), e.g.:
    return [
        ("TEMPLATE", PokerBot(seed=seed)),
        ("TEMPLATE2", PokerBot(seed=seed + 2)),
        ("NIT", NitBot()),
        ("LAG", LagBot(seed + 3)),
        ("PUSHFOLD", PushFoldBot()),
        ("CHEN", ChenBot()),
        ("TRAPPER", TrapperBot(seed + 4)),
        ("STATION", CallingStation()),
    ]
Each bot's tunable numbers are class attributes at the top of the class.
"""
import random

from bot import Action
from poker_core import hand_class, parse_range, classify_hand


# =====================================================================
# SHARED HELPERS  (copy once if pasting bots into local_sim.py)
# =====================================================================
def position(s):
    """Our position label: BTN, SB, BB, UTG, UTG1, MP, HJ, CO (fewer players -> early ones drop)."""
    dealt = [p.seat for p in s.players if p.dealt]
    i = dealt.index(s.button_seat) if s.button_seat in dealt else 0
    order = dealt[i:] + dealt[:i]
    if len(order) == 2:
        return "BTN" if s.my_seat == order[0] else "BB"
    k = len(order) - 3
    mids = (["UTG"] * max(0, k - 5) + ["UTG", "UTG1", "MP", "HJ", "CO"])[-k:] if k > 0 else []
    return (["BTN", "SB", "BB"] + mids)[order.index(s.my_seat)]


def preflop_raises(s):
    """-> (number of preflop raises so far, current high bet, seat of last raiser)."""
    high, raises, last = s.big_blind, 0, None
    for street, seat, kind, amt in s.history:
        if street == "preflop" and kind in ("bet", "raise", "allin") and amt > high:
            raises, high, last = raises + 1, amt, seat
    return raises, high, last


def street_raises(s):
    return sum(1 for st, _, k, _ in s.history if st == s.street and k in ("bet", "raise", "allin"))


def raise_to(s, amount):
    """Legal raise-to; shoves if the size reaches our whole stack."""
    amount = max(int(amount), s.min_raise_to)
    if amount >= s.max_raise_to:
        return Action("allin")
    return Action("raise" if s.to_call or s.me.bet else "bet", amount)


def check_fold(s):
    return Action("check") if s.to_call == 0 else Action("fold")


def stack_bb(s):
    return (s.me.stack + s.me.bet) / s.big_blind


def opponents(s):
    return [p for p in s.players if p.active and p.seat != s.my_seat]


# =====================================================================
# NitBot -- only premium hands, folds to pressure without a strong hand
# =====================================================================
class NitBot:
    OPEN = "88+, AJs+, KQs, AQo+"
    THREEBET = "QQ+, AKs, AKo"
    CALL_RAISE = "TT-88, AQs-AJs"
    SHOVE_UNDER_BB = 10          # shove OPEN range when this short
    OPEN_SIZE_BB = 3.0
    BET_POT = 0.75               # value bet size

    def __init__(self):
        self.open, self.three, self.call = map(parse_range, (self.OPEN, self.THREEBET, self.CALL_RAISE))

    def act(self, s):
        if s.street == "preflop":
            hc = hand_class(*s.hole_cards)
            raises, high, _ = preflop_raises(s)
            if stack_bb(s) <= self.SHOVE_UNDER_BB:
                return Action("allin") if hc in self.open else check_fold(s)
            if raises == 0:
                return raise_to(s, self.OPEN_SIZE_BB * s.big_blind) if hc in self.open else check_fold(s)
            if hc in self.three:
                return raise_to(s, high * 3)
            if raises == 1 and hc in self.call:
                return Action("call")
            return check_fold(s)

        bucket = classify_hand(s.hole_cards, s.board)
        if bucket == "MONSTER":
            return raise_to(s, s.me.bet + s.to_call + self.BET_POT * (s.pot + s.to_call))
        if bucket == "STRONG":
            if s.to_call == 0:
                return raise_to(s, self.BET_POT * s.pot)
            return Action("call") if street_raises(s) <= 1 else Action("fold")
        return check_fold(s)


# =====================================================================
# LagBot -- loose-aggressive: pressure with wide ranges and barrels
# =====================================================================
class LagBot:
    OPEN = ("22+, A2s+, K2s+, Q6s+, J7s+, T7s+, 96s+, 85s+, 75s+, 64s+, 54s, "
            "A5o+, K9o+, QTo+, JTo, T9o")
    THREEBET = "99+, A9s+, A5s-A2s, KTs+, QJs, JTs, T9s, 98s, AJo+, KQo"
    CALL_RAISE = "88-22, A8s-A6s, K9s, Q9s+, J9s+, T8s+, 97s+, 86s+, 76s, 65s, ATo, KJo, QJo"
    FOURBET_JAM = "TT+, AQs+, AKo, A5s"
    OPEN_SIZE_BB = 2.5
    CBET_FREQ = 0.80             # flop bet when checked to
    BARREL_FREQ = 0.55           # turn/river bet when checked to, any hand
    BET_POT = 0.66
    RAISE_DRAWS_FREQ = 0.40      # semi-bluff raise with draws
    CALL_PRICE = 0.30            # call one-pair+ if pot odds better than this

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.open, self.three, self.call, self.jam = map(
            parse_range, (self.OPEN, self.THREEBET, self.CALL_RAISE, self.FOURBET_JAM))

    def act(self, s):
        r = self.rng.random()
        if s.street == "preflop":
            hc = hand_class(*s.hole_cards)
            raises, high, _ = preflop_raises(s)
            if stack_bb(s) <= 15:
                return Action("allin") if hc in self.open else check_fold(s)
            if raises == 0:
                return raise_to(s, self.OPEN_SIZE_BB * s.big_blind) if hc in self.open else check_fold(s)
            if raises == 1:
                if hc in self.three:
                    return raise_to(s, high * 3)
                return Action("call") if hc in self.call else check_fold(s)
            return Action("allin") if hc in self.jam else check_fold(s)

        bucket = classify_hand(s.hole_cards, s.board)
        if s.to_call == 0:
            freq = self.CBET_FREQ if s.street == "flop" else self.BARREL_FREQ
            if bucket in ("MONSTER", "STRONG") or r < freq:
                return raise_to(s, self.BET_POT * s.pot)
            return Action("check")
        if bucket in ("MONSTER", "STRONG"):
            return raise_to(s, s.me.bet + s.to_call + s.pot)
        if bucket in ("STRONG_DRAW", "DRAW") and r < self.RAISE_DRAWS_FREQ:
            return raise_to(s, s.me.bet + s.to_call + s.pot)
        price = s.to_call / (s.pot + s.to_call)
        if bucket in ("MEDIUM", "STRONG_DRAW", "DRAW") and price <= self.CALL_PRICE:
            return Action("call")
        return Action("fold")


# =====================================================================
# PushFoldBot -- tournament push/fold: every preflop decision is shove or fold
# =====================================================================
class PushFoldBot:
    # key = max stack in BB; the smallest key >= our stack is used
    SHOVE_BY_STACK = {
        6:   "*",
        10:  "22+, A2s+, A2o+, K2s+, K6o+, Q5s+, Q9o+, J7s+, J9o+, T7s+, T9o, 97s+, 86s+, 76s, 65s",
        15:  "22+, A2s+, A7o+, K8s+, KTo+, QTs+, QJo, JTs",
        25:  "55+, A9s+, ATo+, KQs",
        999: "TT+, AQs+, AKo",
    }
    CALL_SHOVE = "77+, ATs+, AJo+, KQs"
    POSTFLOP_CONTINUE = ("MONSTER", "STRONG")   # only if somehow postflop without being all-in

    def __init__(self):
        self.shove = sorted((k, parse_range(v)) for k, v in self.SHOVE_BY_STACK.items())
        self.call = parse_range(self.CALL_SHOVE)

    def act(self, s):
        if s.street == "preflop":
            hc = hand_class(*s.hole_cards)
            raises, _, _ = preflop_raises(s)
            bb = stack_bb(s)
            if raises == 0:
                rng = next(r for limit, r in self.shove if bb <= limit)
                return Action("allin") if hc in rng else check_fold(s)
            return Action("allin") if hc in self.call else check_fold(s)
        bucket = classify_hand(s.hole_cards, s.board)
        return Action("allin") if bucket in self.POSTFLOP_CONTINUE else check_fold(s)


# =====================================================================
# ChenBot -- Chen formula preflop, equity-vs-pot-odds postflop
# =====================================================================
class ChenBot:
    # Minimum Chen score to open-raise by position
    OPEN_SCORE = {"UTG": 10, "UTG1": 10, "MP": 9, "HJ": 9, "CO": 8, "BTN": 7, "SB": 8, "BB": 9}
    THREEBET_SCORE = 12
    CALL_RAISE_SCORE = 9
    SHORT_STACK_BB = 12
    SHOVE_SCORE = 7
    # Rough equity per hand bucket (flop, turn/river) -- edit to taste
    EQUITY = {
        "MONSTER":     (0.85, 0.90), "STRONG": (0.70, 0.72), "MEDIUM": (0.50, 0.48),
        "STRONG_DRAW": (0.48, 0.32), "DRAW":   (0.33, 0.18), "WEAK":   (0.12, 0.08),
    }
    VALUE_BET_EQUITY = 0.65
    BET_POT = 0.6

    @staticmethod
    def chen(c1, c2):
        r1, r2 = sorted((c1 >> 2, c2 >> 2), reverse=True)
        high = {12: 10, 11: 8, 10: 7, 9: 6}.get(r1, (r1 + 2) / 2)
        if r1 == r2:
            return max(5, high * 2)
        score = high + (2 if (c1 & 3) == (c2 & 3) else 0)
        gap = r1 - r2 - 1
        score -= {0: 0, 1: 1, 2: 2, 3: 4}.get(gap, 5)
        if gap <= 1 and r1 < 10:
            score += 1
        return -(-score // 1)  # round up

    def act(self, s):
        if s.street == "preflop":
            score = self.chen(*s.hole_cards)
            raises, high, _ = preflop_raises(s)
            if stack_bb(s) <= self.SHORT_STACK_BB:
                return Action("allin") if score >= self.SHOVE_SCORE else check_fold(s)
            if raises == 0:
                if score >= self.OPEN_SCORE.get(position(s), 9):
                    return raise_to(s, 2.5 * s.big_blind)
                return check_fold(s)
            if score >= self.THREEBET_SCORE:
                return raise_to(s, high * 3)
            if raises == 1 and score >= self.CALL_RAISE_SCORE:
                return Action("call")
            return check_fold(s)

        bucket = classify_hand(s.hole_cards, s.board)
        eq = self.EQUITY[bucket][0 if s.street == "flop" else 1]
        eq *= 0.85 ** (len(opponents(s)) - 1)          # less equity multiway
        if s.to_call == 0:
            return raise_to(s, self.BET_POT * s.pot) if eq >= self.VALUE_BET_EQUITY else Action("check")
        price = s.to_call / (s.pot + s.to_call)
        if eq >= 0.80:
            return raise_to(s, s.me.bet + s.to_call + s.pot)
        return Action("call") if eq >= price else Action("fold")


# =====================================================================
# TrapperBot -- loose-passive caller that slow-plays and check-raises big hands
# =====================================================================
class TrapperBot:
    LIMP = ("22+, A2s+, K5s+, Q7s+, J7s+, T7s+, 96s+, 85s+, 75s+, 64s+, 54s, "
            "A7o+, K9o+, Q9o+, J9o+, T9o")
    RAISE = "JJ+, AKs, AKo"
    CALL_RAISE = "22+, A2s+, K9s+, Q9s+, J9s+, T8s+, 97s+, 86s+, 75s+, 65s, 54s, ATo+, KTo+, QJo"
    SLOWPLAY_FREQ = 0.70         # check MONSTER when first to act
    CHECK_RAISE_FREQ = 0.80      # raise MONSTER when bet into
    CALL_PRICE = {"STRONG": 0.45, "MEDIUM": 0.33, "STRONG_DRAW": 0.33, "DRAW": 0.25, "WEAK": 0.0}
    BET_POT = 0.5

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.limp, self.raise_, self.call = map(parse_range, (self.LIMP, self.RAISE, self.CALL_RAISE))

    def act(self, s):
        r = self.rng.random()
        if s.street == "preflop":
            hc = hand_class(*s.hole_cards)
            raises, high, _ = preflop_raises(s)
            if hc in self.raise_:
                return raise_to(s, max(3 * s.big_blind, high * 3))
            if raises == 0:
                return Action("call") if hc in self.limp or s.to_call == 0 else Action("fold")
            return Action("call") if hc in self.call and s.to_call <= 0.15 * stack_bb(s) * s.big_blind \
                else check_fold(s)

        bucket = classify_hand(s.hole_cards, s.board)
        if s.to_call == 0:
            if bucket == "MONSTER" and r < self.SLOWPLAY_FREQ:
                return Action("check")
            if bucket in ("MONSTER", "STRONG"):
                return raise_to(s, self.BET_POT * s.pot)
            return Action("check")
        if bucket == "MONSTER":
            return raise_to(s, s.me.bet + s.to_call + s.pot) if r < self.CHECK_RAISE_FREQ else Action("call")
        price = s.to_call / (s.pot + s.to_call)
        return Action("call") if price <= self.CALL_PRICE[bucket] else Action("fold")
