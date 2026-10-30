"""PLO4 decision logic. Reads every knob from config.py; edit strategy there first.

Interface (engine-agnostic -- agent_adapter.py converts to/from the starter kit):
    bot = PokerBot()
    action = bot.act(state)          # GameState -> Action, must finish in < 1 ms
    bot.on_hand_end(history, seats, showdown=..., board=..., my_seat=...)

Everything that won in the NLH bot is here in one class, each part switchable
with overrides for ablation tests:
    BLUFF_FACTOR          bluff more than the baseline policy
    tight-player exploits steal, respect raises, c-bet more, thin-value less
    EV calls              call/fold by equity vs bluff-rate-weighted ranges
    showdown tracking     per-opponent bluff and fold-to-bet rates
    ICM                   break-even from points whenever a call risks 25%+
    gc pause              no garbage collection inside a decision
"""
import gc
import math
import random
from dataclasses import dataclass, field
from types import SimpleNamespace

import config as default_config
from icm import icm_points, np
from poker_core import MEDIUM, MONSTER, PREFLOP, STRONG, equities, hand_key, read_hand
from poker_core import DRAW, STRONG_DRAW, WEAK

STREETS = ("preflop", "flop", "turn", "river")
AGGRESSIVE = ("bet", "raise", "allin")
STREET_INDEX = {"flop": 0, "turn": 1, "river": 2}
BOARD_LEN = {"flop": 3, "turn": 4, "river": 5}
LATER_POSITION = {"UTG": "UTG1", "UTG1": "MP", "MP": "HJ", "HJ": "CO", "CO": "BTN",
                  "BTN": "BTN", "SB": "SB", "BB": "BB"}
EARLIER_POSITION = {"UTG": "UTG", "UTG1": "UTG", "MP": "UTG1", "HJ": "MP", "CO": "HJ",
                    "BTN": "CO", "SB": "HJ", "BB": "HJ"}
PREFLOP_ORDER = ["UTG", "UTG1", "MP", "HJ", "CO", "BTN", "SB", "BB"]


# =====================================================================
# Data types
# =====================================================================
@dataclass
class PlayerInfo:
    seat: int
    stack: int              # chips behind (not counting bet this street)
    bet: int = 0            # chips put in on the current street
    active: bool = True     # still in the hand (not folded)
    all_in: bool = False
    dealt: bool = True      # False for busted / empty seats


@dataclass
class Action:
    kind: str               # "fold" | "check" | "call" | "bet" | "raise" | "allin"
    amount: int = 0         # for bet/raise: TOTAL street bet ("raise to")


@dataclass
class GameState:
    my_seat: int
    hole_cards: list        # four ints (see poker_core.parse_cards)
    board: list
    street: str             # "preflop" | "flop" | "turn" | "river"
    pot: int                # all chips in the middle incl. this street's bets
    players: list           # PlayerInfo for every seat, indexed by seat
    button_seat: int
    small_blind: int
    big_blind: int
    to_call: int
    min_raise_to: int
    max_raise_to: int       # POT-LIMIT cap on our street total (<= stack + bet)
    # (street, seat, kind, amount) -- amount = that player's street total after acting
    history: list = field(default_factory=list)
    hand_number: int = 0
    legal_actions: tuple = ()

    @property
    def me(self):
        return self.players[self.my_seat]


def pot_limit_to(high, my_bet, pot, stack):
    """Largest legal raise-to: call, then raise the size of the pot after calling."""
    return min(stack + my_bet, high + pot + (high - my_bet))


# =====================================================================
# Opponent tracking (state resets per table per the rules)
# =====================================================================
class OpponentStats:
    def __init__(self):
        self.hands, self.vpip, self.pfr = {}, {}, {}
        self.bets_shown, self.bluffs_shown = {}, {}
        self.fold_chances, self.folds = {}, {}

    def observe(self, history, seats):
        for s in seats:
            self.hands[s] = self.hands.get(s, 0) + 1
        vp, pr = set(), set()
        for street, seat, kind, _ in history:
            if street != "preflop":
                continue
            if kind in ("call",) + AGGRESSIVE:
                vp.add(seat)
            if kind in AGGRESSIVE:
                pr.add(seat)
        for s in vp:
            self.vpip[s] = self.vpip.get(s, 0) + 1
        for s in pr:
            self.pfr[s] = self.pfr.get(s, 0) + 1

    def _rate(self, table, seat, cfg):
        n = self.hands.get(seat, 0)
        return None if n < cfg.MIN_HANDS_FOR_READS else table.get(seat, 0) / n

    def is_loose(self, seat, cfg):
        r = self._rate(self.vpip, seat, cfg)
        return r is not None and r > cfg.LOOSE_VPIP

    def is_aggro(self, seat, cfg):
        r = self._rate(self.pfr, seat, cfg)
        return r is not None and r > cfg.AGGRO_PFR

    def is_tight(self, seat, cfg):
        r = self._rate(self.vpip, seat, cfg)
        return r is not None and r < cfg.TIGHT_VPIP

    def observe_fold_to_bet(self, history, me):
        for street in ("flop", "turn", "river"):
            bet_seen, responded = False, set()
            for st, seat, kind, _ in history:
                if st != street:
                    continue
                if bet_seen and seat != me and seat not in responded and kind != "check":
                    self.fold_chances[seat] = self.fold_chances.get(seat, 0) + 1
                    if kind == "fold":
                        self.folds[seat] = self.folds.get(seat, 0) + 1
                    responded.add(seat)
                if kind in ("bet", "raise") or (kind == "allin" and not bet_seen):
                    bet_seen = True
                    responded.add(seat)

    def bluff_rate(self, seat, cfg):
        w = cfg.BLUFF_PRIOR_WEIGHT
        return (self.bluffs_shown.get(seat, 0) + cfg.BLUFF_PRIOR * w) / (self.bets_shown.get(seat, 0) + w)

    def fold_rate(self, seat, cfg):
        w = cfg.FOLD_PRIOR_WEIGHT
        return (self.folds.get(seat, 0) + cfg.FOLD_PRIOR * w) / (self.fold_chances.get(seat, 0) + w)


# =====================================================================
# Bot
# =====================================================================
class PokerBot:
    WANTS_SHOWDOWN = True   # simulator / adapter pass showdown cards to on_hand_end

    def __init__(self, cfg=default_config, overrides=None, seed=None):
        values = {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()}
        values.update(overrides or {})
        self.cfg = SimpleNamespace(**values)
        self.rng = random.Random(seed if seed is not None else self.cfg.RANDOM_SEED)
        self.stats = OpponentStats()
        self._my_seat = None
        self.last_ev = None

    # ------------------------------------------------------------ public
    def act(self, s: GameState) -> Action:
        self._my_seat = s.my_seat
        was_enabled = gc.isenabled()
        gc.disable()         # no collection pause inside the 1 ms decision
        try:
            try:
                a = self._preflop(s) if s.street == "preflop" else self._postflop(s)
            except Exception:  # never crash: the engine would auto check/fold anyway
                a = Action("check") if s.to_call == 0 else Action("fold")
            return self._legalize(s, a)
        finally:
            if was_enabled:
                gc.enable()

    def on_hand_end(self, history, seats_dealt, showdown=None, board=None, my_seat=None):
        c = self.cfg
        if not c.TRACK_OPPONENTS:
            return
        me = self._my_seat if my_seat is None else my_seat
        self.stats.observe(history, seats_dealt)
        self.stats.observe_fold_to_bet(history, me)
        if not showdown or not board or len(board) < 5:
            return
        for seat, cards in showdown.items():
            if seat == me:
                continue
            for street in ("flop", "turn", "river"):
                high, aggressive = 0, False
                for st, x, kind, amt in history:
                    if st == street and kind in AGGRESSIVE and amt > high:
                        aggressive = aggressive or x == seat
                        high = amt
                if not aggressive:
                    continue
                bucket = self._bucket(list(cards), list(board[:BOARD_LEN[street]]))[0]
                self.stats.bets_shown[seat] = self.stats.bets_shown.get(seat, 0) + 1
                weight = 1.0 if bucket in (WEAK, DRAW) else c.SEMI_BLUFF_WEIGHT if bucket == STRONG_DRAW else 0.0
                self.stats.bluffs_shown[seat] = self.stats.bluffs_shown.get(seat, 0) + weight

    # ------------------------------------------------------------ hand reading
    def _bucket(self, hole, board):
        """-> (bucket, (eq vs any, eq vs value, eq vs raise))."""
        c = self.cfg
        _, _, key = read_hand(hole, board)
        eqs = equities(key)
        any_eq, val_eq, _ = eqs
        if val_eq >= c.BUCKET_MONSTER:
            return MONSTER, eqs
        if val_eq >= c.BUCKET_STRONG:
            return STRONG, eqs
        if any_eq >= c.BUCKET_MEDIUM:
            return MEDIUM, eqs
        if len(board) < 5:
            if val_eq >= c.BUCKET_STRONG_DRAW:
                return STRONG_DRAW, eqs
            if val_eq >= c.BUCKET_DRAW:
                return DRAW, eqs
        return WEAK, eqs

    # ------------------------------------------------------------ helpers
    def position(self, s, seat=None):
        seat = s.my_seat if seat is None else seat
        dealt = [p.seat for p in s.players if p.dealt]
        n = len(dealt)
        i = dealt.index(s.button_seat) if s.button_seat in dealt else 0
        order = dealt[i:] + dealt[:i]          # BTN first, then SB, BB, ...
        if n == 2:
            return "BTN" if seat == order[0] else "BB"
        middle = ["UTG", "UTG1", "MP", "HJ", "CO"]
        k = n - 3
        mids = (["UTG"] * max(0, k - 5) + middle)[-k:] if k > 0 else []
        return (["BTN", "SB", "BB"] + mids)[order.index(seat)]

    def _preflop_context(self, s):
        high, raises, limpers, callers, last_raiser = s.big_blind, 0, 0, 0, None
        for street, seat, kind, amt in s.history:
            if street != "preflop":
                continue
            if kind in AGGRESSIVE and amt > high:
                raises, high, last_raiser, callers = raises + 1, amt, seat, 0
            elif kind in ("call", "allin"):
                if raises == 0:
                    limpers += 1
                else:
                    callers += 1
        return SimpleNamespace(raises=raises, high=high, limpers=limpers,
                               callers=callers, last_raiser=last_raiser)

    def _effective_stack(self, s):
        mine = s.me.stack + s.me.bet
        others = [p.stack + p.bet for p in s.players if p.active and p.seat != s.my_seat]
        return min(mine, max(others)) if others else mine

    def _players_left(self, s):
        return sum(1 for p in s.players if p.dealt)

    def _check_fold(self, s):
        return Action("check") if s.to_call == 0 else Action("fold")

    def _all_tight(self, seats):
        seats = list(seats)
        return bool(seats) and all(self.stats.is_tight(x, self.cfg) for x in seats)

    def _players_behind(self, s):
        """Seats still to act after us preflop (dealt, not folded, not all-in)."""
        mine = PREFLOP_ORDER.index(self.position(s))
        return [p.seat for p in s.players
                if p.active and not p.all_in and p.seat != s.my_seat
                and PREFLOP_ORDER.index(self.position(s, p.seat)) > mine]

    def _in_position(self, s, opponents):
        """True when nobody left in the hand acts after us postflop."""
        n = len(s.players)
        offset = lambda seat: (seat - s.button_seat - 1) % n
        return all(offset(s.my_seat) > offset(p.seat) for p in opponents)

    def _street_has_aggression(self, s):
        return any(st == s.street and k in AGGRESSIVE for st, _, k, _ in s.history)

    def _street_aggressors(self, s):
        return {seat for st, seat, k, _ in s.history if st == s.street and k in AGGRESSIVE}

    def _aggressor(self, s):
        high, agg = 0, None
        for st, seat, kind, amt in s.history:
            if st == s.street and kind in AGGRESSIVE and amt > high and seat != s.my_seat:
                high, agg = amt, seat
        if agg is None:   # fall back to the biggest bet in front of us
            others = [p for p in s.players if p.active and p.seat != s.my_seat]
            agg = max(others, key=lambda p: p.bet).seat if others else None
        return agg

    def _facing_raise(self, s):
        high, n = (s.big_blind if s.street == "preflop" else 0), 0
        for st, seat, kind, amt in s.history:
            if st == s.street and kind in AGGRESSIVE and amt > high:
                n += 1
                high = amt
        return n >= 2

    def _pot_raise(self, s, fraction=1.0):
        """Raise-to for `fraction` of the way from a min-raise to the pot-limit cap."""
        lo, hi = s.min_raise_to, s.max_raise_to
        return Action("raise", int(lo + (hi - lo) * fraction))

    def _breakeven(self, s, agg):
        """-> (break-even equity, 'ICM' or 'chips')."""
        me, c = s.me, self.cfg
        matched = min(s.to_call, me.stack)
        agg_bet = s.players[agg].bet if agg is not None else 0
        uncalled = max(0, agg_bet - (me.bet + matched))
        pot_after = s.pot - uncalled + matched
        chip_be = matched / pot_after if pot_after else 0.0
        risk = matched / max(1, me.stack + me.bet)
        if not (c.USE_ICM and np is not None and agg is not None and risk >= c.ICM_MIN_RISK_FRACTION):
            return chip_be, "chips"
        base = {p.seat: p.stack for p in s.players if p.dealt}
        fold, win, lose = dict(base), dict(base), dict(base)
        fold[agg] += s.pot
        win[s.my_seat] += pot_after - matched
        win[agg] += uncalled
        lose[s.my_seat] -= matched
        lose[agg] += pot_after + uncalled
        P = c.PAYOUT_POINTS
        ef, ew, el = (icm_points(x, P)[s.my_seat] for x in (fold, win, lose))
        if ew <= el:
            return chip_be, "chips"
        return min(1.0, max(0.0, (ef - el) / (ew - el))), "ICM"

    # ------------------------------------------------------------ preflop
    def _shover_index(self, s, agg):
        """-> index into a PREFLOP row for the range we assume a big raiser has."""
        c = self.cfg
        if agg is None:
            return 2
        p = s.players[agg]
        bb = (p.stack + p.bet) / s.big_blind
        k = 1 if bb <= c.SHOVER_WIDE_UNDER_BB else 2 if bb <= c.SHOVER_MEDIUM_UNDER_BB else 3
        if self._preflop_context(s).raises >= 3:
            k = 3
        if self.stats.is_tight(agg, c):
            k = min(3, k + 1)
        elif self.stats.is_loose(agg, c) or self.stats.is_aggro(agg, c):
            k = max(1, k - 1)
        return k

    def _preflop(self, s):
        c = self.cfg
        row = PREFLOP[hand_key(s.hole_cards)]
        a = self._preflop_range(s, row[0])
        if not c.PREFLOP_EV or s.to_call == 0:
            return a
        risk = min(s.to_call, s.me.stack) / max(1, s.me.stack + s.me.bet)
        if risk < c.PREFLOP_EV_MIN_RISK or a.kind not in ("fold", "call", "allin"):
            return a
        agg = self._aggressor(s)
        ctx = self._preflop_context(s)
        eq = row[self._shover_index(s, agg)] ** (1 + c.MULTIWAY_EQ_EXP * ctx.callers)
        be, _ = self._breakeven(s, agg)
        self.last_ev = {"equity": eq, "breakeven": be}
        if a.kind == "fold":
            return Action("call") if eq >= be + c.PREFLOP_CALL_MARGIN else a
        if a.kind == "call" or s.me.stack <= s.to_call:
            return a if eq >= be else Action("fold")
        return a if eq >= be - c.SHOVE_FOLD_ALLOWANCE else Action("fold")   # our re-shove

    def _preflop_range(self, s, pct):
        c = self.cfg
        pos = self.position(s)
        bb = s.big_blind
        ctx = self._preflop_context(s)
        eff_bb = self._effective_stack(s) / bb

        # --- short stack: push / fold
        if eff_bb <= c.SHOVE_STACK_BB:
            if ctx.raises == 0:
                limit = next((v for k, v in sorted(c.SHOVE_PCT_BY_STACK.items()) if eff_bb <= k),
                             min(c.SHOVE_PCT_BY_STACK.values()))
                if pct <= limit:
                    return Action("allin")
                return self._check_fold(s)
            bubble = self._players_left(s) <= c.BUBBLE_PLAYERS_LEFT
            return Action("allin") if pct <= (c.CALL_SHOVE_PCT_BUBBLE if bubble else c.CALL_SHOVE_PCT) \
                else self._check_fold(s)

        # --- facing a shove from someone
        raiser = s.players[ctx.last_raiser] if ctx.last_raiser is not None else None
        if raiser is not None and raiser.all_in and s.to_call >= s.me.stack * 0.5:
            bubble = self._players_left(s) <= c.BUBBLE_PLAYERS_LEFT
            return Action("call") if pct <= (c.CALL_SHOVE_PCT_BUBBLE if bubble else c.CALL_SHOVE_PCT) \
                else self._check_fold(s)

        # --- unopened / limped
        if ctx.raises == 0:
            limit = c.OPEN_PCT.get(pos, 0.0)
            if ctx.limpers == 0 and pos != "BB" and self._all_tight(self._players_behind(s)):
                if pos in ("BTN", "SB"):
                    limit = c.STEAL_PCT_VS_TIGHT
                else:
                    later = pos
                    for _ in range(c.STEAL_SHIFT_VS_TIGHT):
                        later = LATER_POSITION.get(later, later)
                    limit = c.OPEN_PCT.get(later, limit)
            if pct <= limit:
                size = c.OPEN_SIZE_BB.get(pos, c.OPEN_SIZE_BB["default"])
                return Action("raise", int((size + c.LIMPER_ADD_BB * ctx.limpers) * bb))
            if pos == "SB" and pct <= c.SB_COMPLETE_PCT and eff_bb > c.SB_COMPLETE_MAX_STACK_BB:
                return Action("call")
            if s.to_call == 0:
                return Action("check")
            if ctx.limpers > 0 and pct <= c.OVERLIMP_PCT:
                return Action("call")
            return Action("fold")

        # --- facing one raise
        if ctx.raises == 1:
            look = pos
            if c.WIDEN_VS_AGGRO_OPENER and self.stats.is_aggro(ctx.last_raiser, c):
                look = LATER_POSITION.get(pos, pos)
            elif c.RESPECT_TIGHT_RAISES and self.stats.is_tight(ctx.last_raiser, c):
                for _ in range(c.RESPECT_SHIFT_VS_TIGHT):
                    look = EARLIER_POSITION.get(look, look)
            if pct <= c.THREEBET_PCT.get(look, 0.0):
                return self._pot_raise(s, c.THREEBET_POT_FRACTION)
            if pct <= c.CALL_OPEN_PCT.get(look, 0.0):
                return Action("call")
            return self._check_fold(s)

        # --- facing a 3-bet
        if ctx.raises == 2:
            if pct <= c.FOURBET_PCT:
                return self._pot_raise(s, c.FOURBET_POT_FRACTION)
            if pct <= c.CALL_3BET_PCT:
                return Action("call")
            return self._check_fold(s)

        # --- facing a 4-bet+
        if pct <= c.JAM_4BET_PCT:
            return Action("allin")
        if pct <= c.CALL_4BET_PCT:
            return Action("call")
        return self._check_fold(s)

    # ------------------------------------------------------------ postflop
    def _bluff_mult(self, bucket, opponents):
        c = self.cfg
        if bucket not in c.BLUFF_BUCKETS:
            return 1.0
        m = c.BLUFF_FACTOR
        if self._all_tight(p.seat for p in opponents):
            m *= c.BLUFF_MULT_VS_TIGHT
        return m

    def _trap_freq(self, s, bucket):
        return self.cfg.TRAP_FREQ.get(s.street, 0.0) if bucket == MONSTER else None

    def _bet(self, s, frac):
        return Action("bet", int(s.me.bet + frac * s.pot))

    def _postflop(self, s):
        c = self.cfg
        bucket, (eq_any, eq_val, eq_raise) = self._bucket(s.hole_cards, s.board)
        pol = c.POSTFLOP_POLICY[bucket]
        opponents = [p for p in s.players if p.active and p.seat != s.my_seat]
        multiway = len(opponents) >= 2
        pot = s.pot
        r = self.rng.random()
        bluff_mult = self._bluff_mult(bucket, opponents)
        vs_tight = self._all_tight(p.seat for p in opponents)

        if s.to_call == 0:
            if bucket in (MONSTER, STRONG) and s.me.stack / max(pot, 1) <= c.COMMIT_SPR:
                return Action("allin")
            tf = self._trap_freq(s, bucket)
            if tf is not None:
                return Action("check") if r >= 1.0 - tf else self._bet(s, pol["bet_size"])
            freq = pol["bet_freq"] * c.STREET_BET_FREQ_MULT.get(s.street, 1.0)
            if multiway and bucket != MONSTER:
                freq *= c.MULTIWAY_BET_FREQ_MULT
            freq *= bluff_mult
            if bucket == MEDIUM and vs_tight:
                freq *= c.THIN_VALUE_MULT_VS_TIGHT
            size = pol["bet_size"]
            ctx = self._preflop_context(s)
            if s.street == "flop" and ctx.last_raiser == s.my_seat and not self._street_has_aggression(s):
                cbet = (c.CBET_FREQ_MULTIWAY if multiway else c.CBET_FREQ_HEADSUP) * bluff_mult
                if vs_tight:
                    cbet += c.CBET_BONUS_VS_TIGHT
                if cbet > freq:
                    freq, size = cbet, c.CBET_SIZE
            return self._bet(s, size) if r < min(freq, 1.0) else Action("check")

        # --- facing a bet: raise by policy, then EV decides everything
        raise_freq = pol["raise_freq"] * bluff_mult
        if bucket == MONSTER and any(st == s.street and seat == s.my_seat and k == "check"
                                     for st, seat, k, _ in s.history):
            raise_freq = c.CHECK_RAISE_FREQ
        if r < min(raise_freq, 1.0):
            high = s.me.bet + s.to_call
            a = Action("raise", int(high + pol["raise_size"] * (pot + s.to_call)))
        else:
            a = Action("call")

        i = STREET_INDEX[s.street]
        agg = self._aggressor(s)
        raised = self._facing_raise(s)
        pb = self.stats.bluff_rate(agg, c) * (c.RAISE_BLUFF_SCALE if raised else 1.0) if agg is not None \
            else c.BLUFF_PRIOR
        vs_value = eq_raise if raised else eq_val
        eq = pb * eq_any + (1 - pb) * vs_value
        eq **= 1 + c.MULTIWAY_EQ_EXP * max(0, len(opponents) - 1)
        if i < 2:
            eq *= c.REALIZE_IP if self._in_position(s, opponents) else c.REALIZE_OOP
            if bucket in (STRONG_DRAW, DRAW):
                eq -= c.DRAW_REVERSE_ODDS
        be, _ = self._breakeven(s, agg)
        call_ok = eq >= be + c.CALL_EV_MARGIN
        self.last_ev = {"equity": eq, "breakeven": be, "bluff_rate": pb, "bucket": bucket}

        if a.kind == "call":
            return a if call_ok or bucket == MONSTER and s.street == "river" else Action("fold")
        if call_ok or bucket == MONSTER:
            return a
        cost = a.amount - s.me.bet
        pf = math.prod(self.stats.fold_rate(p.seat, c) * c.FOLD_TO_RAISE_SCALE for p in opponents)
        ev = pf * pot + (1 - pf) * (eq_raise * (pot + 2 * cost - s.to_call) - cost)
        return a if ev > 0 else Action("fold")

    # ------------------------------------------------------------ legality
    def _legalize(self, s, a):
        """Pot-limit: "allin" means the biggest legal raise, which may be less than our stack."""
        me = s.me
        if a.kind == "fold":
            return Action("check") if s.to_call == 0 else a
        if a.kind == "check":
            return a if s.to_call == 0 else Action("fold")
        if a.kind == "call":
            if s.to_call == 0:
                return Action("check")
            return Action("allin", me.stack + me.bet) if s.to_call >= me.stack else a
        street_high = me.bet + s.to_call
        everything = me.stack + me.bet
        if s.max_raise_to <= street_high:      # no raise is possible
            if s.to_call == 0:
                return Action("check")
            return Action("allin", everything) if s.to_call >= me.stack else Action("call")
        kind = "bet" if street_high == 0 else "raise"
        if a.kind == "allin":
            amt = s.max_raise_to
        else:
            amt = min(max(int(a.amount), s.min_raise_to), s.max_raise_to)
            if amt >= self.cfg.JAM_IF_COMMIT_FRACTION * everything:
                amt = s.max_raise_to
        return Action("allin", amt) if amt >= everything else Action(kind, amt)
