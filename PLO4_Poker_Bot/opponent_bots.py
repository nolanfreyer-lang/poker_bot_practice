"""
=====================================================================
  OPPONENT BOTS  --  PLO sparring partners with different, real strategies
=====================================================================
Same interface as PokerBot: act(GameState) -> Action.

These read hands the way a quickly-built bot would: by absolute made-hand
category (naive_bucket), not by how close to the nuts it is. That is the common
leak in PLO, and the field on event day is mostly bots built in 60 minutes.

  NitBot        very tight, raises premium hands, gives up without a strong hand
  LagBot        loose-aggressive: wide opens, frequent c-bets and barrels
  PushFoldBot   pots it or folds preflop, based on stack depth
  ChenBot       Hutchinson-style points preflop, bucket equity vs pot odds postflop
  TrapperBot    loose-passive: limps and calls a lot, slow-plays big hands
  GtoApproxBot  balanced bluffing and minimum-defense postflop with mixed preflop ranges
  NlhPortBot    a Hold'em bot pasted onto PLO: plays only its best two cards preflop
  CallingStation, RandomBot, Maniac, CheckFoldBot   naive or crashed bots
"""
import random
from itertools import combinations

from bot import Action
from poker_core import PREFLOP, hand_key, plo_value


# =====================================================================
# SHARED HELPERS
# =====================================================================
def position(s):
    dealt = [p.seat for p in s.players if p.dealt]
    i = dealt.index(s.button_seat) if s.button_seat in dealt else 0
    order = dealt[i:] + dealt[:i]
    if len(order) == 2:
        return "BTN" if s.my_seat == order[0] else "BB"
    k = len(order) - 3
    mids = (["UTG"] * max(0, k - 5) + ["UTG", "UTG1", "MP", "HJ", "CO"])[-k:] if k > 0 else []
    return (["BTN", "SB", "BB"] + mids)[order.index(s.my_seat)]


def preflop_raises(s):
    high, raises, last = s.big_blind, 0, None
    for street, seat, kind, amt in s.history:
        if street == "preflop" and kind in ("bet", "raise", "allin") and amt > high:
            raises, high, last = raises + 1, amt, seat
    return raises, high, last


def street_raises(s):
    return sum(1 for st, _, k, _ in s.history if st == s.street and k in ("bet", "raise", "allin"))


def raise_to(s, amount):
    amount = min(max(int(amount), s.min_raise_to), s.max_raise_to)
    if amount >= s.me.stack + s.me.bet:
        return Action("allin")
    return Action("raise" if s.to_call or s.me.bet else "bet", amount)


def pot_raise(s):
    return raise_to(s, s.max_raise_to)


def check_fold(s):
    return Action("check") if s.to_call == 0 else Action("fold")


def stack_bb(s):
    return (s.me.stack + s.me.bet) / s.big_blind


def opponents(s):
    return [p for p in s.players if p.active and p.seat != s.my_seat]


def pct_of(s):
    return PREFLOP[hand_key(s.hole_cards)][0]


def naive_bucket(hole, board):
    """Absolute made-hand reading, no nut awareness: a low flush is a MONSTER."""
    v = plo_value(hole, board)
    cat = v >> 20
    top = max(c >> 2 for c in board)
    pair_rank = (v >> 16) & 15
    if cat >= 4 or cat == 3:
        return "MONSTER"
    if cat == 2:
        return "STRONG"
    if cat == 1:
        return "STRONG" if pair_rank >= top else "MEDIUM"
    if len(board) < 5:
        suits = [c & 3 for c in board]
        for s in range(4):
            if suits.count(s) >= 2 and sum(1 for c in hole if c & 3 == s) >= 2:
                return "DRAW"
    return "WEAK"


# =====================================================================
class NitBot:
    OPEN, THREEBET, CALL_RAISE, SHOVE_UNDER_BB, BET_POT = 0.08, 0.02, 0.05, 8, 0.75

    def act(self, s):
        if s.street == "preflop":
            pct = pct_of(s)
            raises, _, _ = preflop_raises(s)
            if stack_bb(s) <= self.SHOVE_UNDER_BB:
                return Action("allin") if pct <= self.OPEN else check_fold(s)
            if raises == 0:
                return raise_to(s, 3.5 * s.big_blind) if pct <= self.OPEN else check_fold(s)
            if pct <= self.THREEBET:
                return pot_raise(s)
            return Action("call") if raises == 1 and pct <= self.CALL_RAISE else check_fold(s)
        b = naive_bucket(s.hole_cards, s.board)
        if b == "MONSTER":
            return raise_to(s, s.me.bet + s.to_call + self.BET_POT * (s.pot + s.to_call))
        if b == "STRONG":
            if s.to_call == 0:
                return raise_to(s, self.BET_POT * s.pot)
            return Action("call") if street_raises(s) <= 1 else Action("fold")
        return check_fold(s)


class LagBot:
    OPEN, THREEBET, CALL_RAISE = 0.50, 0.10, 0.35
    CBET_FREQ, BARREL_FREQ, BET_POT, RAISE_DRAWS_FREQ, CALL_PRICE = 0.80, 0.50, 0.75, 0.40, 0.33

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def act(self, s):
        r = self.rng.random()
        if s.street == "preflop":
            pct = pct_of(s)
            raises, _, _ = preflop_raises(s)
            if stack_bb(s) <= 10:
                return Action("allin") if pct <= self.OPEN else check_fold(s)
            if raises == 0:
                return raise_to(s, 3 * s.big_blind) if pct <= self.OPEN else check_fold(s)
            if raises == 1 and pct <= self.THREEBET:
                return pot_raise(s)
            return Action("call") if pct <= self.CALL_RAISE else check_fold(s)
        b = naive_bucket(s.hole_cards, s.board)
        if s.to_call == 0:
            freq = self.CBET_FREQ if s.street == "flop" else self.BARREL_FREQ
            return raise_to(s, self.BET_POT * s.pot) if b in ("MONSTER", "STRONG") or r < freq else Action("check")
        if b == "MONSTER":
            return pot_raise(s)
        if b in ("STRONG", "DRAW") and r < self.RAISE_DRAWS_FREQ:
            return pot_raise(s)
        price = s.to_call / (s.pot + s.to_call)
        return Action("call") if b != "WEAK" and price <= self.CALL_PRICE else Action("fold")


class PushFoldBot:
    SHOVE_BY_STACK = {6: 1.0, 10: 0.5, 15: 0.3, 25: 0.15, 999: 0.04}
    CALL_SHOVE = 0.10

    def act(self, s):
        if s.street == "preflop":
            pct = pct_of(s)
            raises, _, _ = preflop_raises(s)
            if raises == 0:
                limit = next(v for k, v in sorted(self.SHOVE_BY_STACK.items()) if stack_bb(s) <= k)
                return pot_raise(s) if pct <= limit else check_fold(s)
            return pot_raise(s) if pct <= self.CALL_SHOVE else check_fold(s)
        b = naive_bucket(s.hole_cards, s.board)
        return pot_raise(s) if b in ("MONSTER", "STRONG") else check_fold(s)


class ChenBot:
    """Hutchinson-style points: high pairs, suits to high cards, connectedness."""
    OPEN_SCORE = {"UTG": 40, "UTG1": 40, "MP": 38, "HJ": 36, "CO": 34, "BTN": 30, "SB": 34, "BB": 38}
    THREEBET_SCORE, CALL_RAISE_SCORE, SHOVE_SCORE, SHORT_STACK_BB = 50, 36, 30, 8
    EQUITY = {"MONSTER": (0.85, 0.85), "STRONG": (0.65, 0.65), "MEDIUM": (0.45, 0.40),
              "DRAW": (0.40, 0.20), "WEAK": (0.12, 0.08)}
    VALUE_BET_EQUITY, BET_POT = 0.60, 0.65

    @staticmethod
    def points(hole):
        ranks = sorted((c >> 2 for c in hole), reverse=True)
        score = 0.0
        for r in set(ranks):
            n = ranks.count(r)
            if n == 2:
                score += 6 + r * 1.5
            elif n >= 3:
                score -= 4
        suits = {}
        for c in hole:
            suits.setdefault(c & 3, []).append(c >> 2)
        for cards in suits.values():
            if len(cards) == 2:
                score += 3 + max(cards)
            elif len(cards) == 3:
                score += 1 + max(cards) / 2
        uniq = sorted(set(ranks), reverse=True)
        for a, b in zip(uniq, uniq[1:]):
            score += {1: 5, 2: 3, 3: 1}.get(a - b, 0)
        return score + sum(ranks) / 4

    def act(self, s):
        if s.street == "preflop":
            score = self.points(s.hole_cards)
            raises, _, _ = preflop_raises(s)
            if stack_bb(s) <= self.SHORT_STACK_BB:
                return Action("allin") if score >= self.SHOVE_SCORE else check_fold(s)
            if raises == 0:
                return raise_to(s, 3 * s.big_blind) if score >= self.OPEN_SCORE.get(position(s), 38) else check_fold(s)
            if score >= self.THREEBET_SCORE:
                return pot_raise(s)
            return Action("call") if raises == 1 and score >= self.CALL_RAISE_SCORE else check_fold(s)
        b = naive_bucket(s.hole_cards, s.board)
        eq = self.EQUITY[b][0 if s.street == "flop" else 1] * 0.85 ** (len(opponents(s)) - 1)
        if s.to_call == 0:
            return raise_to(s, self.BET_POT * s.pot) if eq >= self.VALUE_BET_EQUITY else Action("check")
        price = s.to_call / (s.pot + s.to_call)
        if eq >= 0.80:
            return pot_raise(s)
        return Action("call") if eq >= price else Action("fold")


class TrapperBot:
    LIMP, RAISE, CALL_RAISE = 0.60, 0.03, 0.40
    SLOWPLAY_FREQ, CHECK_RAISE_FREQ, BET_POT = 0.70, 0.80, 0.5
    CALL_PRICE = {"STRONG": 0.45, "MEDIUM": 0.33, "DRAW": 0.33, "WEAK": 0.0}

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def act(self, s):
        r = self.rng.random()
        if s.street == "preflop":
            pct = pct_of(s)
            raises, high, _ = preflop_raises(s)
            if pct <= self.RAISE:
                return pot_raise(s)
            if raises == 0:
                return Action("call") if pct <= self.LIMP or s.to_call == 0 else Action("fold")
            return Action("call") if pct <= self.CALL_RAISE and s.to_call <= 0.15 * stack_bb(s) * s.big_blind \
                else check_fold(s)
        b = naive_bucket(s.hole_cards, s.board)
        if s.to_call == 0:
            if b == "MONSTER" and r < self.SLOWPLAY_FREQ:
                return Action("check")
            return raise_to(s, self.BET_POT * s.pot) if b in ("MONSTER", "STRONG") else Action("check")
        if b == "MONSTER":
            return pot_raise(s) if r < self.CHECK_RAISE_FREQ else Action("call")
        return Action("call") if s.to_call / (s.pot + s.to_call) <= self.CALL_PRICE[b] else Action("fold")


class GtoApproxBot:
    """Mixed percentile ranges preflop; bluff ratio and minimum defense postflop."""
    RFI = {"UTG": 0.14, "UTG1": 0.16, "MP": 0.20, "HJ": 0.25, "CO": 0.33, "BTN": 0.50, "SB": 0.30, "BB": 0.12}
    MIX_WIDTH = 0.04               # hands within this of the edge are mixed 50/50
    THREEBET, CALL = 0.05, 0.22
    FOURBET, CALL_3BET = 0.015, 0.10
    PUSH_BB, PUSH_PCT, CALL_PUSH = 8, 0.40, 0.20
    BET_SIZE = {"flop": 0.5, "turn": 0.66, "river": 0.75}
    VALUE_SHARE = {"flop": 0.40, "turn": 0.30, "river": 0.22}
    RAISE_VALUE_SHARE = 0.08
    PCT = {"WEAK": (0.00, 0.40), "DRAW": (0.40, 0.60), "MEDIUM": (0.60, 0.80),
           "STRONG": (0.80, 0.94), "MONSTER": (0.94, 1.00)}
    DRAW_MASS = 0.20

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def _preflop(self, s, r):
        pct = pct_of(s)
        raises, _, last = preflop_raises(s)
        if stack_bb(s) <= self.PUSH_BB:
            limit = self.PUSH_PCT if raises == 0 else self.CALL_PUSH
            return Action("allin") if pct <= limit else check_fold(s)
        if raises == 0:
            edge = self.RFI[position(s)] + (self.MIX_WIDTH if r < 0.5 else -self.MIX_WIDTH)
            return raise_to(s, 3 * s.big_blind) if pct <= edge else check_fold(s)
        if raises == 1:
            if pct <= self.THREEBET:
                return pot_raise(s)
            return Action("call") if pct <= self.CALL else check_fold(s)
        if pct <= self.FOURBET:
            return pot_raise(s)
        return Action("call") if raises == 2 and pct <= self.CALL_3BET else check_fold(s)

    def _postflop(self, s, r):
        b = naive_bucket(s.hole_cards, s.board)
        lo, hi = self.PCT[b]
        pct = lo + (hi - lo) * self.rng.random()
        n_opp = len(opponents(s))
        if s.to_call == 0:
            size = self.BET_SIZE[s.street]
            value = self.VALUE_SHARE[s.street] / n_opp
            if pct >= 1 - value:
                return raise_to(s, size * s.pot)
            if n_opp == 1 and b == "DRAW":
                alpha = size / (1 + 2 * size)
                if r < min(1.0, value * alpha / (1 - alpha) / self.DRAW_MASS):
                    return raise_to(s, size * s.pot)
            return Action("check")
        bet_frac = s.to_call / max(1, s.pot - s.to_call)
        defend = 1 / (1 + bet_frac) / n_opp
        if street_raises(s) > 1:
            defend *= 0.6
        if pct >= 1 - self.RAISE_VALUE_SHARE:
            return pot_raise(s)
        return Action("call") if pct >= 1 - defend else Action("fold")

    def act(self, s):
        r = self.rng.random()
        return self._preflop(s, r) if s.street == "preflop" else self._postflop(s, r)


class NlhPortBot:
    """Treats its best two cards like a Hold'em hand, then calls on any pair or better."""
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    @staticmethod
    def best_two(hole):
        best = 0
        for a, b in combinations(hole, 2):
            ra, rb = sorted((a >> 2, b >> 2), reverse=True)
            score = ra * 2 + rb + (12 if ra == rb else 0) + (3 if a & 3 == b & 3 else 0)
            best = max(best, score)
        return best

    def act(self, s):
        if s.street == "preflop":
            score = self.best_two(s.hole_cards)
            raises, _, _ = preflop_raises(s)
            if raises == 0:
                return raise_to(s, 2.5 * s.big_blind) if score >= 30 else check_fold(s)
            if score >= 44:
                return pot_raise(s)
            return Action("call") if score >= 34 else check_fold(s)
        b = naive_bucket(s.hole_cards, s.board)
        if s.to_call == 0:
            return raise_to(s, 0.6 * s.pot) if b in ("MONSTER", "STRONG") else Action("check")
        if b == "MONSTER":
            return pot_raise(s)
        return Action("call") if b in ("STRONG", "MEDIUM", "DRAW") else Action("fold")


class CallingStation:
    def act(self, s):
        return Action("call") if s.to_call else Action("check")


class RandomBot:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def act(self, s):
        r = self.rng.random()
        if r < 0.15 and s.to_call:
            return Action("fold")
        if r < 0.80:
            return Action("call") if s.to_call else Action("check")
        return Action("raise", s.min_raise_to) if r < 0.97 else Action("allin")


class Maniac:
    def act(self, s):
        return Action("allin")


class CheckFoldBot:
    """Stand-in for bots that crash or exceed 1 ms: the engine checks/folds for them."""
    def act(self, s):
        return Action("check") if s.to_call == 0 else Action("fold")
