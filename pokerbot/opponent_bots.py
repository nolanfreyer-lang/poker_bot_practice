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
  GtoApproxBot  solver-style mixed ranges + balanced bluffing and minimum-defense postflop

TWO WAYS TO USE THEM IN local_sim.py
  A) Import (easiest): add this near the top of local_sim.py
         from opponent_bots import NitBot, LagBot, PushFoldBot, ChenBot, TrapperBot, GtoApproxBot
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


# =====================================================================
# GtoApproxBot -- solver-inspired approximation of GTO (NOT a true equilibrium)
#   * Preflop: solver-style 8-max ranges; "MIX" hands are played MIX_FREQ of the time
#   * Postflop: balanced by the standard formulas
#       bluff share of a bet = s / (1 + 2s)      (s = bet as fraction of pot)
#       minimum defense      = 1 / (1 + s)       (fraction of range that must continue)
#     Our hand is placed at a percentile of our range via its bucket, with a random
#     spot inside the bucket so frequencies mix instead of being all-or-nothing.
#   * Short stacks: Nash-style push/fold
# =====================================================================
class GtoApproxBot:
    # --- preflop: (always, mixed) per position, open-raise first in
    RFI = {
        "UTG":  ("77+, A9s+, KTs+, QTs+, JTs, AJo+, KQo",             "66-55, A8s-A6s, A5s-A4s, K9s, T9s, 98s, ATo"),
        "UTG1": ("66+, A8s+, A5s, K9s+, QTs+, JTs, T9s, AJo+, KQo",    "55-44, A7s-A6s, A4s-A3s, Q9s, J9s, 98s, 87s, ATo, KJo"),
        "MP":   ("55+, A2s+, K9s+, Q9s+, J9s+, T9s, 98s, ATo+, KJo+",   "44-33, K8s-K7s, T8s, 87s, 76s, A9o, QJo"),
        "HJ":   ("33+, A2s+, K8s+, Q9s+, J9s+, T8s+, 98s, 87s, A9o+, KJo+, QJo",
                 "22, K7s-K5s, Q8s, J8s, 97s, 76s, 65s, A8o, KTo, QTo, JTo"),
        "CO":   ("22+, A2s+, K5s+, Q8s+, J8s+, T8s+, 97s+, 87s, 76s, 65s, A7o+, KTo+, QTo+, JTo",
                 "K4s-K2s, Q7s-Q6s, J7s, T7s, 86s, 75s, 54s, A6o-A5o, K9o, Q9o, J9o, T9o"),
        "BTN":  ("22+, A2s+, K2s+, Q5s+, J7s+, T7s+, 96s+, 86s+, 75s+, 65s, 54s, A2o+, K8o+, Q9o+, J9o+, T9o",
                 "Q4s-Q2s, J6s-J4s, T6s, 95s, 85s, 74s, 64s, 53s, 43s, K7o-K5o, Q8o, J8o, T8o, 98o, 87o"),
        "SB":   ("22+, A2s+, K5s+, Q8s+, J8s+, T8s+, 97s+, 87s, 76s, A7o+, KTo+, QJo",
                 "K4s-K2s, Q7s-Q5s, J7s, T7s, 86s, 75s, 65s, 54s, A6o-A2o, K9o, QTo, JTo"),
        "BB":   ("TT+, AJs+, KQs, AQo+",                                "99-77, ATs, KJs, AJo"),
    }
    # facing one raise: 3-bet (value, bluff-mix) and flat-call ranges by our position
    THREEBET = {"default": ("QQ+, AKs, AKo",                "JJ, AQs, A5s-A4s, KQs"),
                "late":    ("JJ+, AQs+, AKo",               "TT, AJs, ATs, A5s-A2s, KJs, QJs, AQo"),
                "blinds":  ("JJ+, AQs+, AKo",               "TT, AJs, A5s-A3s, KQs, K9s, AQo")}
    CALL = {"default": "TT-66, AQs-ATs, KJs+, QJs, JTs, AQo",
            "late":    "99-22, AJs-A6s, KTs+, Q9s+, J9s+, T9s, 98s, 87s, AJo-ATo, KQo, KJo",
            "BB":      ("99-22, A9s-A2s, K2s+, Q5s+, J7s+, T7s+, 96s+, 85s+, 75s+, 64s+, 54s, "
                        "AJo-A2o, K8o+, Q9o+, J9o+, T9o, 98o, 87o")}
    FOURBET = ("KK+, AKs", "QQ, AKo, A5s-A4s")
    CALL_3BET = "QQ-99, AQs-AJs, KQs, QJs, JTs, AQo"
    JAM_VS_4BET = ("KK+, AKs, AKo", "QQ")
    MIX_FREQ = 0.5
    OPEN_SIZE_BB = {"default": 2.2, "SB": 3.0, "BB": 4.0}
    # Nash-style push/fold (first in) by stack depth, and calls vs a shove
    PUSH_STACK_BB = 12
    PUSH = {6: "*",
            9: "22+, A2s+, A2o+, K2s+, K5o+, Q4s+, Q9o+, J7s+, J9o+, T7s+, T9o, 96s+, 86s+, 75s+, 65s, 54s",
            12: "22+, A2s+, A4o+, K7s+, K9o+, Q9s+, QTo+, J9s+, JTo, T9s, 98s"}
    CALL_PUSH = "55+, A8s+, ATo+, KTs+, KQo"

    # --- postflop (per street): bet size as pot fraction, share of range we value-bet
    BET_SIZE = {"flop": 0.33, "turn": 0.66, "river": 0.75}
    VALUE_SHARE = {"flop": 0.40, "turn": 0.30, "river": 0.22}
    RAISE_VALUE_SHARE = 0.08          # top of range that raises a bet for value
    # where each bucket sits in our range (low, high percentile)
    PCT = {"WEAK": (0.00, 0.40), "DRAW": (0.40, 0.52), "STRONG_DRAW": (0.52, 0.58),
           "MEDIUM": (0.58, 0.80), "STRONG": (0.80, 0.94), "MONSTER": (0.94, 1.00)}
    DRAW_MASS = 0.18                  # share of range that is draws (preferred bluffs)

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        P = parse_range
        pair = lambda t: (P(t[0]), P(t[1]))
        self.rfi = {k: pair(v) for k, v in self.RFI.items()}
        self.three = {k: pair(v) for k, v in self.THREEBET.items()}
        self.call = {k: P(v) for k, v in self.CALL.items()}
        self.four, self.jam4 = pair(self.FOURBET), pair(self.JAM_VS_4BET)
        self.call3, self.call_push = P(self.CALL_3BET), P(self.CALL_PUSH)
        self.push = sorted((k, P(v)) for k, v in self.PUSH.items())

    def _in(self, hc, ranges, r):
        """(always, mixed) pair -> True if we play this hand on this random draw."""
        return hc in ranges[0] or (hc in ranges[1] and r < self.MIX_FREQ)

    # ------------------------------------------------------------ preflop
    def _preflop(self, s, r):
        hc = hand_class(*s.hole_cards)
        pos = position(s)
        raises, high, last = preflop_raises(s)
        bb = stack_bb(s)
        if bb <= self.PUSH_STACK_BB:
            if raises == 0:
                rng = next(x for limit, x in self.push if bb <= limit)
                return Action("allin") if hc in rng else check_fold(s)
            return Action("allin") if hc in self.call_push else check_fold(s)
        if last is not None and s.players[last].all_in:
            return Action("call") if hc in self.call_push else check_fold(s)
        if raises == 0:
            if self._in(hc, self.rfi[pos], r):
                size = self.OPEN_SIZE_BB.get(pos, self.OPEN_SIZE_BB["default"])
                return raise_to(s, size * s.big_blind)
            return check_fold(s)
        group = "blinds" if pos in ("SB", "BB") else "late" if pos in ("CO", "BTN") else "default"
        if raises == 1:
            if self._in(hc, self.three[group], r):
                return raise_to(s, high * (4.0 if group == "blinds" else 3.0))
            call = self.call["BB"] if pos == "BB" else self.call["late" if group == "late" else "default"]
            return Action("call") if hc in call else check_fold(s)
        if raises == 2:
            if self._in(hc, self.four, r):
                return raise_to(s, high * 2.3)
            return Action("call") if hc in self.call3 else check_fold(s)
        return Action("allin") if self._in(hc, self.jam4, r) else check_fold(s)

    # ------------------------------------------------------------ postflop
    def _postflop(self, s, r):
        bucket = classify_hand(s.hole_cards, s.board)
        lo, hi = self.PCT[bucket]
        pct = lo + (hi - lo) * self.rng.random()
        n_opp = len(opponents(s))

        if s.to_call == 0:
            size = self.BET_SIZE[s.street]
            value = self.VALUE_SHARE[s.street] / n_opp          # bet thinner range multiway
            if pct >= 1 - value:
                return raise_to(s, size * s.pot)
            if n_opp == 1:                                       # bluff only heads-up
                alpha = size / (1 + 2 * size)
                bluff_mass = value * alpha / (1 - alpha)
                if bucket in ("DRAW", "STRONG_DRAW"):
                    if r < min(1.0, bluff_mass / self.DRAW_MASS):
                        return raise_to(s, size * s.pot)
                elif bucket == "WEAK" and bluff_mass > self.DRAW_MASS:
                    if r < (bluff_mass - self.DRAW_MASS) / self.PCT["WEAK"][1]:
                        return raise_to(s, size * s.pot)
            return Action("check")

        # facing a bet of size s (fraction of the pot before the bet)
        bet_frac = s.to_call / max(1, s.pot - s.to_call)
        mdf = 1 / (1 + bet_frac)
        defend = mdf / n_opp                                     # others share the defending
        if street_raises(s) > 1:
            defend *= 0.6                                        # vs raises, continue tighter
        if pct >= 1 - self.RAISE_VALUE_SHARE:
            return raise_to(s, s.me.bet + s.to_call + (s.pot + s.to_call))
        if bucket in ("DRAW", "STRONG_DRAW") and n_opp == 1 and s.street != "river" \
                and r < self.RAISE_VALUE_SHARE * 0.5 / self.DRAW_MASS:
            return raise_to(s, s.me.bet + s.to_call + (s.pot + s.to_call))   # balancing semi-bluff
        if pct >= 1 - defend:
            return Action("call")
        return Action("fold")

    def act(self, s):
        r = self.rng.random()
        return self._preflop(s, r) if s.street == "preflop" else self._postflop(s, r)
