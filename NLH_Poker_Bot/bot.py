"""Decision logic. Reads every knob from config.py; edit strategy there first.

Interface (engine-agnostic -- agent_adapter.py converts to/from the starter kit):
    bot = PokerBot()
    action = bot.act(state)          # GameState -> Action, must finish in < 1 ms
    bot.on_hand_end(history)         # optional, feeds opponent stats
"""
import random
from dataclasses import dataclass, field
from types import SimpleNamespace

import config as default_config
from poker_core import classify_hand, hand_class, parse_range, MONSTER, STRONG, MEDIUM

STREETS = ("preflop", "flop", "turn", "river")
AGGRESSIVE = ("bet", "raise", "allin")


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
    amount: int = 0         # for bet/raise: TOTAL street bet ("raise to"), like Exhibit B's "R"


@dataclass
class GameState:
    my_seat: int
    hole_cards: list        # ints (see poker_core.parse_cards)
    board: list
    street: str             # "preflop" | "flop" | "turn" | "river"
    pot: int                # all chips in the middle incl. this street's bets
    players: list           # PlayerInfo for every seat, indexed by seat
    button_seat: int
    small_blind: int
    big_blind: int
    to_call: int
    min_raise_to: int
    max_raise_to: int       # our stack + our current street bet
    # (street, seat, kind, amount) -- amount = that player's street total after acting
    history: list = field(default_factory=list)
    hand_number: int = 0
    legal_actions: tuple = ()  # optional; empty = assume standard NLHE legality

    @property
    def me(self):
        return self.players[self.my_seat]


# =====================================================================
# Opponent tracking (state resets per table per the rules)
# =====================================================================
class OpponentStats:
    def __init__(self):
        self.hands = {}
        self.vpip = {}
        self.pfr = {}

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


# =====================================================================
# Bot
# =====================================================================
LATER_POSITION = {"UTG": "UTG1", "UTG1": "MP", "MP": "HJ", "HJ": "CO", "CO": "BTN",
                  "BTN": "BTN", "SB": "SB", "BB": "BB"}


class PokerBot:
    def __init__(self, cfg=default_config, overrides=None, seed=None):
        # cfg can be the config module or any object with the same attributes;
        # overrides={"OPEN_SIZE_BB": {...}} lets the simulator test variants.
        values = {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()}
        values.update(overrides or {})
        self.cfg = SimpleNamespace(**values)
        self.rng = random.Random(seed if seed is not None else self.cfg.RANDOM_SEED)
        self.stats = OpponentStats()
        self._compile_ranges()

    # Parse every range once, so a decision is only set lookups.
    def _compile_ranges(self):
        c = self.cfg
        pr = parse_range
        self.R = SimpleNamespace(
            open={k: pr(v) for k, v in c.OPEN_RANGE.items()},
            threebet={k: pr(v) for k, v in c.THREEBET_RANGE.items()},
            call_open={k: pr(v) for k, v in c.CALL_VS_OPEN_RANGE.items()},
            overlimp=pr(c.OVERLIMP_RANGE),
            sb_complete=pr(c.SB_COMPLETE_RANGE),
            fourbet=pr(c.FOURBET_RANGE),
            call_3bet=pr(c.CALL_VS_3BET_RANGE),
            jam_4bet=pr(c.JAM_VS_4BET_RANGE),
            call_4bet=pr(c.CALL_VS_4BET_RANGE),
            shove=sorted((bb, pr(v)) for bb, v in c.SHOVE_RANGE_BY_STACK.items()),
            call_shove=pr(c.CALL_SHOVE_RANGE),
            call_shove_bubble=pr(c.CALL_SHOVE_RANGE_BUBBLE),
        )
        self.good_kicker = "23456789TJQKA".index(c.TOP_PAIR_GOOD_KICKER)

    # ------------------------------------------------------------ public
    def act(self, s: GameState) -> Action:
        try:
            a = self._preflop(s) if s.street == "preflop" else self._postflop(s)
        except Exception:  # never crash: the engine would auto check/fold anyway
            a = Action("check") if s.to_call == 0 else Action("fold")
        return self._legalize(s, a)

    def on_hand_end(self, history, seats_dealt):
        if self.cfg.TRACK_OPPONENTS:
            self.stats.observe(history, seats_dealt)

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
        labels = ["BTN", "SB", "BB"] + mids
        return labels[order.index(seat)]

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

    def _raise_to(self, s, amount):
        return Action("raise", int(amount))

    def _check_fold(self, s):
        return Action("check") if s.to_call == 0 else Action("fold")

    # ------------------------------------------------------------ preflop
    def _preflop(self, s):
        c, R = self.cfg, self.R
        hc = hand_class(*s.hole_cards)
        pos = self.position(s)
        bb = s.big_blind
        ctx = self._preflop_context(s)
        eff_bb = self._effective_stack(s) / bb

        # --- short stack: push / fold
        if eff_bb <= c.SHOVE_STACK_BB:
            if ctx.raises == 0:
                rng = next((r for limit, r in R.shove if eff_bb <= limit), R.shove[-1][1])
                if hc in rng:
                    return Action("allin")
                if pos == "BB" and s.to_call == 0:
                    return Action("check")
                return self._check_fold(s)
            bubble = self._players_left(s) <= c.BUBBLE_PLAYERS_LEFT
            return Action("allin") if hc in (R.call_shove_bubble if bubble else R.call_shove) \
                else self._check_fold(s)

        # --- facing a shove from someone: use call-shove ranges
        raiser = s.players[ctx.last_raiser] if ctx.last_raiser is not None else None
        if raiser is not None and raiser.all_in and s.to_call >= s.me.stack * 0.5:
            bubble = self._players_left(s) <= c.BUBBLE_PLAYERS_LEFT
            return Action("call") if hc in (R.call_shove_bubble if bubble else R.call_shove) \
                else self._check_fold(s)

        # --- unopened / limped
        if ctx.raises == 0:
            if hc in R.open.get(pos, frozenset()):
                size = c.OPEN_SIZE_BB.get(pos, c.OPEN_SIZE_BB["default"])
                return self._raise_to(s, (size + c.LIMPER_ADD_BB * ctx.limpers) * bb)
            if pos == "SB" and hc in R.sb_complete and eff_bb > c.SB_COMPLETE_MAX_STACK_BB:
                return Action("call")
            if s.to_call == 0:
                return Action("check")
            if ctx.limpers > 0 and hc in R.overlimp:
                return Action("call")
            return Action("fold")

        # --- facing one raise
        if ctx.raises == 1:
            look_pos = pos
            if (c.WIDEN_VS_AGGRO_OPENER and ctx.last_raiser is not None
                    and self.stats.is_aggro(ctx.last_raiser, c)):
                look_pos = LATER_POSITION.get(pos, pos)
            if hc in R.threebet.get(look_pos, frozenset()):
                mult = c.THREEBET_MULT_OOP if pos in ("SB", "BB") else c.THREEBET_MULT_IP
                return self._raise_to(s, ctx.high * (mult + c.COLD_CALLER_ADD_MULT * ctx.callers))
            if hc in R.call_open.get(look_pos, frozenset()):
                return Action("call")
            return self._check_fold(s)

        # --- facing 3-bet
        if ctx.raises == 2:
            if hc in R.fourbet:
                return self._raise_to(s, ctx.high * c.FOURBET_MULT)
            if hc in R.call_3bet:
                return Action("call")
            return self._check_fold(s)

        # --- facing 4-bet+
        if hc in R.jam_4bet:
            return Action("allin")
        if hc in R.call_4bet:
            return Action("call")
        return self._check_fold(s)

    # ------------------------------------------------------------ postflop
    def _postflop(self, s):
        c = self.cfg
        bucket = classify_hand(s.hole_cards, s.board, self.good_kicker)
        pol = c.POSTFLOP_POLICY[bucket]
        opponents = [p for p in s.players if p.active and p.seat != s.my_seat]
        multiway = len(opponents) >= 2
        strong_plus = bucket in (MONSTER, STRONG)
        pot = s.pot
        r = self.rng.random()

        # Low SPR: get it in with strong hands
        spr = s.me.stack / max(pot, 1)
        if strong_plus and spr <= c.COMMIT_SPR:
            return Action("allin")

        if s.to_call == 0:
            freq = pol["bet_freq"] * c.STREET_BET_FREQ_MULT.get(s.street, 1.0)
            if multiway and bucket != MONSTER:
                freq *= c.MULTIWAY_BET_FREQ_MULT
            size = pol["bet_size"]
            ctx = self._preflop_context(s)
            if (s.street == "flop" and ctx.last_raiser == s.my_seat
                    and not self._street_has_aggression(s)):
                cbet = c.CBET_FREQ_MULTIWAY if multiway else c.CBET_FREQ_HEADSUP
                if cbet > freq:
                    freq, size = cbet, c.CBET_SIZE
            if r < freq:
                return Action("bet", int(s.me.bet + size * pot))
            return Action("check")

        # Facing a bet
        price = s.to_call / (pot + s.to_call)
        allowed = pol["call_price"]
        if multiway and bucket != MONSTER:
            allowed *= c.MULTIWAY_CALL_PRICE_MULT
        allowed -= c.CALL_PRICE_PENALTY_PER_RAISE * max(0, self._street_raises(s) - 1)
        if any(self.stats.is_aggro(seat, c) or self.stats.is_loose(seat, c)
               for seat in self._street_aggressors(s)):
            allowed += c.CALL_PRICE_BONUS_VS_AGGRO

        if r < pol["raise_freq"]:
            high = s.me.bet + s.to_call
            return self._raise_to(s, high + pol["raise_size"] * (pot + s.to_call))
        if price <= allowed:
            return Action("call")
        invested = self._invested_this_hand(s)
        start_chips = invested + s.me.stack
        if (bucket in (MONSTER, STRONG, MEDIUM) and start_chips
                and invested / start_chips >= c.POT_COMMITTED_FRACTION and price <= 0.4):
            return Action("call")
        return Action("fold")

    def _street_has_aggression(self, s):
        return any(st == s.street and k in AGGRESSIVE for st, _, k, _ in s.history)

    def _street_raises(self, s):
        return sum(1 for st, _, k, _ in s.history if st == s.street and k in AGGRESSIVE)

    def _street_aggressors(self, s):
        return {seat for st, seat, k, _ in s.history if st == s.street and k in AGGRESSIVE}

    def _invested_this_hand(self, s):
        per_street = {}
        for st, seat, _, amt in s.history:
            if seat == s.my_seat:
                per_street[st] = max(per_street.get(st, 0), amt)
        per_street[s.street] = max(per_street.get(s.street, 0), s.me.bet)
        return sum(per_street.values())

    # ------------------------------------------------------------ legality
    def _legalize(self, s, a):
        me = s.me
        if a.kind == "fold":
            return Action("check") if s.to_call == 0 else a
        if a.kind == "check":
            return a if s.to_call == 0 else Action("fold")
        if a.kind == "call":
            if s.to_call == 0:
                return Action("check")
            return Action("allin") if s.to_call >= me.stack else a
        if a.kind == "allin":
            return Action("allin", s.max_raise_to)
        # bet / raise
        if s.min_raise_to >= s.max_raise_to:
            return Action("allin", s.max_raise_to)
        amt = max(int(a.amount), s.min_raise_to)
        if amt >= s.max_raise_to or amt >= self.cfg.JAM_IF_COMMIT_FRACTION * s.max_raise_to:
            return Action("allin", s.max_raise_to)
        street_high = me.bet + s.to_call
        return Action("bet" if street_high == 0 else "raise", amt)
