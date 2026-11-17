"""DEEP_THOUGHT_V18: DEEP_THOUGHT_V17 with cash-game preflop charts for 3 to 6 players.

    from deep_thought_v18 import DeepThoughtV18Bot
    ("DEEP18", DeepThoughtV18Bot(seed=seed))

With three or more players dealt in, v17 played config.py's 8-handed tournament ranges.
In a 6-max cash game (Open Poker) it entered about 17% of hands from each blind and 28%
from the button, and its blinds lost against strong opponents (-12 and -47 bb/100 at six
players, -30 big blind three-handed).

v18 swaps in charts for three spots, with 3+ players dealt and above push/fold depth:

  * Opening first in, by seat: about 15% from the first seat at six-max, 24% hijack,
    27% cutoff, 50% button, 34-40% small blind; 2.5 BB (3 BB from the small blind).
  * The blinds facing an open: 3-bet or call by the opener's seat and size. The big blind
    defends wide against small late opens and tighter against early or big ones; the
    small blind mostly 3-bets or folds; callers in between make it a squeeze.
  * After opening and facing a 3-bet, between 60 and 150 BB effective: 4-bet, call
    (wider in position and against late 3-bets) or fold. A cold 3-bet gets premiums only.

and counts a seat that acts last among the players left as in position for how much
equity a call keeps (v1 only counted the button).

Paired tests (bb/100, 3+ players, stacks reset each hand). Opens, 3-bet charts and the
position fix against v17: +8.8 +- 4.1 against the usual 6-max field at 100 BB, +5.7 +- 3.0
three-handed, +3.1 +- 3.0 against strong bots four-handed, within +-2.7 of zero at 6-max
against strong bots and at 50 and 250 BB. Adding the blinds' defence on top: +2.1 +- 3.3
(strong 6-max), +0.8 +- 4.8 (usual 6-max), +4.8 +- 3.5 (strong 3-handed), +13.1 +- 5.3 and
+11.5 +- 4.2 (usual 4- and 3-handed), +4.6 +- 3.4 (strong 4-handed), +1.3 to +3.6 at 50 BB,
+2.3 +- 6.9 and +31 +- 12 at 250 BB. Heads-up play is identical to v17's.

Tested with no clear effect and left as they were: the multiway postflop rules (no
multiway bluffs, bluff factor 1.0, equity cuts of 0.75 or 0.95 per extra opponent all
within +-2 bb/100), since flops with three or more players are only 3-5% of hands.

Charts written but off, because they tested worse: raising limpers (-11.8 +- 3.9 against
weak bots), every seat's defence against opens (-5.2 +- 6.1; the blinds alone are on), and stacking
off by depth in 4-bet and all-in pots (-93 +- 18 at 250 BB against bots that shove light).
The 3-bet charts were -3.0 +- 2.1 at 50 BB, hence the depth window.
"""
import copy

import config as default_config
from bot import Action, AGGRESSIVE
from deep_thought_v17 import DeepThoughtV17Bot
from poker_core import hand_class, parse_range

# --- opening ranges by seat (labels from bot.position: 6 players = MP HJ CO BTN SB BB) ---
RFI = {
    "UTG": "77+, A9s+, A5s-A4s, KTs+, QTs+, JTs, T9s, AJo+, KQo",
    "UTG1": "66+, A8s+, A5s-A4s, KTs+, QTs+, JTs, T9s, 98s, AJo+, KQo",
    "MP": "66+, A8s+, A5s-A2s, K9s+, QTs+, J9s+, T9s, 98s, 87s, ATo+, KJo+",
    "HJ": "44+, A2s+, K9s+, Q9s+, J9s+, T8s+, 98s, 87s, 76s, ATo+, KJo+, QJo",
    "CO": "22+, A2s+, K6s+, Q8s+, J8s+, T8s+, 97s+, 86s+, 76s, 65s, 54s, A8o+, KTo+, QTo+, JTo",
    "BTN": ("22+, A2s+, K2s+, Q4s+, J6s+, T6s+, 96s+, 85s+, 75s+, 64s+, 54s, 43s, A2o+, K7o+, Q8o+, J8o+, "
            "T8o+, 98o, 87o"),
    "SB": ("22+, A2s+, K2s+, Q5s+, J7s+, T7s+, 96s+, 86s+, 75s+, 65s, 54s, A2o+, K8o+, Q9o+, J9o+, T9o"),
}
# raising limpers (by seat; the big blind raises limpers from here too)
ISO = {
    "early": "88+, ATs+, KJs+, QJs, AQo+",
    "late": "66+, A8s+, A5s, KTs+, QTs+, JTs, ATo+, KJo+",
    "blinds": "77+, A9s+, KTs+, QTs+, JTs, ATo+, KQo",
}
OVERLIMP = "22-66, A2s-A9s, K9s, QTs-Q9s, J9s+, T9s, 98s, 87s, 76s, 65s"

# --- facing one raise: (3-bet, call) by our seat group and the opener's -----------------
# opener groups: "early" (UTG..HJ), "late" (CO, BTN), "sb" (small blind, so we are the BB)
VS_OPEN = {
    ("ip", "early"): ("QQ+, AKs, AKo, A5s",
                      "JJ-55, AQs-ATs, KQs, KJs, QJs, JTs, T9s, 98s, AQo"),
    ("ip", "late"): ("JJ+, AQs+, AKo, KQs, A5s-A4s, 76s, 65s",
                     "TT-22, AJs-A8s, KJs-KTs, QJs-QTs, JTs, T9s, 98s, 87s, AJo-ATo, KQo"),
    ("sb", "early"): ("QQ+, AKs, AKo, AQs, A5s", "JJ-99, AJs, KQs"),
    ("sb", "late"): ("TT+, AJs+, KQs, AQo+, A5s-A3s, K9s, 76s, 65s", "99-66, ATs, KJs, QJs, JTs"),
    ("bb", "early"): ("QQ+, AKs, AKo, A5s",
                      "JJ-22, AQs-A2s, K9s+, Q9s+, J9s+, T8s+, 97s+, 86s+, 76s, 65s, 54s, AQo-ATo, KJo+, QJo"),
    ("bb", "late"): ("TT+, AJs+, KQs, AQo+, A5s-A3s, K9s, Q9s, J9s, 87s, 76s, 65s",
                     "99-22, ATs-A2s, K2s+, Q5s+, J7s+, T7s+, 96s+, 85s+, 75s+, 64s+, 54s, 43s, AJo-A2o, "
                     "K8o+, Q9o+, J9o+, T9o, 98o"),
    ("bb", "sb"): ("TT+, ATs+, KJs+, AJo+, KQo, A5s-A2s, K8s, Q9s, J9s, 87s, 76s",
                   "99-22, A9s-A6s, K2s+, Q4s+, J6s+, T6s+, 96s+, 85s+, 74s+, 63s+, 53s+, 43s, ATo-A2o, "
                   "K7o+, Q8o+, J8o+, T8o+, 98o, 87o, 76o"),
}
# against an open bigger than BIG_OPEN_BB, or with callers in between (a squeeze spot)
VS_BIG_OPEN = ("QQ+, AKs, AKo", "JJ-77, AQs-ATs, KQs, KJs, QJs, JTs, AQo")
SQUEEZE = ("QQ+, AKs, AKo, A5s", "JJ-22, AQs-A9s, KQs, KJs, QJs, JTs, T9s, 98s, 87s, 76s")

# --- after opening, facing a 3-bet: (4-bet, call) --------------------------------------
VS_3BET = {
    "ip": ("QQ+, AKs, AKo, A5s", "JJ-66, AQs-ATs, KQs, KJs, QJs, JTs, T9s, 98s, AQo"),
    "oop": ("QQ+, AKs, AKo, A5s", "JJ-88, AQs-AJs, KQs, AQo"),
    "late": ("JJ+, AQs+, AKo, A5s-A4s", "TT-55, AJs-A9s, KQs, KJs, QJs, JTs, T9s, 98s, AQo-AJo, KQo"),
}
COLD_VS_3BET = ("KK+, AKs", "QQ-JJ, AKo")

SIX_CONFIG = {
    "SIX_PREFLOP": True,
    # which spots use the charts. Tested against v17 (paired, 6-max, bb/100): "limped" -11.8 +- 3.9
    # and "vs_open" -5.2 +- 6.1 against weak bots; "big" (all-ins and 4-bets by depth) -93 +- 18
    # at 250 BB against weak bots, who shove light deep.
    "SIX_PARTS": ("open", "vs_3bet", "vs_open"),
    "SIX_3BET_DEPTH": (60, 150),
    "SIX_VS_OPEN_SEATS": ("BB", "SB"),  # facing an open: blinds use the charts; other seats keep v17's    # effective stack (BB) where the 3-bet charts apply; else v17's
    "SIX_OPEN_BB": 2.5,
    "SIX_SB_OPEN_BB": 3.0,
    "SIX_LIMPER_ADD_BB": 1.0,
    "SIX_ISO_BB": 3.5,
    "SIX_3BET_IP": 3.0,          # x the open
    "SIX_3BET_OOP": 3.8,
    "SIX_SQUEEZE_ADD": 1.0,      # + this x the open per caller
    "SIX_4BET_MULT": 2.3,
    "BIG_OPEN_BB": 4.0,          # opens above this get the tight defence
    "DEEP_SET_BB": 150,          # deeper than this, call raises with any pair for set value
    "SIX_IP_REALIZE": True,
}


def _ranges(table):
    return {k: tuple(parse_range(r) for r in v) if isinstance(v, tuple) else parse_range(v)
            for k, v in table.items()}


class DeepThoughtV18Bot(DeepThoughtV17Bot):
    def __init__(self, cfg=default_config, overrides=None, seed=None):
        super().__init__(cfg, {**SIX_CONFIG, **(overrides or {})}, seed)
        self.rfi = _ranges(RFI)
        self.iso = _ranges(ISO)
        self.overlimp = parse_range(OVERLIMP)
        self.vs_open = _ranges(VS_OPEN)
        self.vs_big_open = tuple(parse_range(r) for r in VS_BIG_OPEN)
        self.squeeze = tuple(parse_range(r) for r in SQUEEZE)
        self.vs_3bet = _ranges(VS_3BET)
        self.cold_vs_3bet = tuple(parse_range(r) for r in COLD_VS_3BET)
        self.pairs = parse_range("22+")

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _group(pos):
        return "sb" if pos == "SB" else "late" if pos in ("CO", "BTN") else "bb" if pos == "BB" else "early"

    @staticmethod
    def _button(s):
        """The button, read from who posted the small blind (the dealt seat just before it).

        Platforms report dealer seats differently (Chipzen's points at the big blind once a
        6-max table is down to two), but the blinds in the history are always right. Falls
        back to the reported seat if the blinds aren't there."""
        dealt = [p.seat for p in s.players if p.dealt]
        blinds = [seat for st, seat, kind, _ in s.history if st == "preflop" and kind == "blind"]
        if len(dealt) < 3 or len(blinds) < 2 or blinds[0] not in dealt:
            return s.button_seat
        return dealt[(dealt.index(blinds[0]) - 1) % len(dealt)]

    def position(self, s, seat=None):
        if sum(1 for p in s.players if p.dealt) >= 3:
            b = self._button(s)
            if b != s.button_seat:
                s = copy.copy(s)
                s.button_seat = b
        return super().position(s, seat)

    def _acts_last(self, s):
        """True if we act after every other player still in the hand on this street."""
        if s.street == "preflop":
            return self.position(s) == "BB"
        n = len(s.players)
        button = self._button(s)
        order = [(button + k) % n for k in range(1, n + 1)]      # SB first ... button last
        live = [x for x in order if s.players[x].active and not s.players[x].all_in or x == s.my_seat]
        return bool(live) and live[-1] == s.my_seat

    def _realize(self, s):
        c = self.cfg
        if not c.SIX_IP_REALIZE or self._players_left(s) <= 2 or not c.PLANNING or s.street == "river":
            return super()._realize(s)
        ip = 1 if self._acts_last(s) else 0
        return (c.REALIZE_FLOP if s.street == "flop" else c.REALIZE_TURN)[ip]

    def _stack_off(self, s, hc, eff_bb):
        return hc in self._by_stack(self.hu_stack_off, eff_bb)

    def _my_open(self, s):
        """True if our first voluntary preflop action was a raise."""
        for st, seat, kind, _ in s.history:
            if st == "preflop" and seat == s.my_seat and kind != "blind":
                return kind in AGGRESSIVE
        return False

    # ------------------------------------------------------------ preflop
    def _preflop_action(self, s):
        c = self.cfg
        eff_bb = self._effective_stack(s) / s.big_blind
        if not c.SIX_PREFLOP or self._players_left(s) <= 2 or eff_bb <= c.SHOVE_STACK_BB:
            return super()._preflop_action(s)
        hc = hand_class(*s.hole_cards)
        pos = self.position(s)
        bb = s.big_blind
        ctx = self._preflop_context(s)
        raiser = s.players[ctx.last_raiser] if ctx.last_raiser is not None else None
        part = ("big" if (raiser is not None and raiser.all_in and s.to_call >= 0.4 * s.me.stack) or ctx.raises >= 3
                else "open" if ctx.raises == 0 and ctx.limpers == 0 else "limped" if ctx.raises == 0
                else "vs_open" if ctx.raises == 1 else "vs_3bet")
        if part == "vs_open" and pos not in c.SIX_VS_OPEN_SEATS:
            return super()._preflop_action(s)
        if part not in c.SIX_PARTS or (part == "vs_3bet" and not
                                       c.SIX_3BET_DEPTH[0] <= eff_bb <= c.SIX_3BET_DEPTH[1]):
            return super()._preflop_action(s)

        # all-in to us for a big part of our stack: stack off by depth
        if raiser is not None and raiser.all_in and s.to_call >= 0.4 * s.me.stack:
            depth = eff_bb / max(1.0, min(3.0, self._freq_ratio(s, ctx.last_raiser, "all_in")))
            return Action("call") if self._stack_off(s, hc, depth) else self._check_fold(s)

        if ctx.raises == 0:
            if ctx.limpers == 0:
                if pos == "BB":
                    return Action("check")
                if hc in self.rfi.get(pos, frozenset()):
                    size = c.SIX_SB_OPEN_BB if pos == "SB" else c.SIX_OPEN_BB
                    return self._raise_to(s, size * bb)
                return self._check_fold(s)
            grp = "blinds" if pos in ("SB", "BB") else "late" if pos in ("CO", "BTN") else "early"
            if hc in self.iso[grp]:
                return self._raise_to(s, (c.SIX_ISO_BB + c.SIX_LIMPER_ADD_BB * ctx.limpers) * bb)
            if pos == "BB":
                return Action("check")
            if hc in self.overlimp and (pos in ("CO", "BTN", "SB") or ctx.limpers >= 2):
                return Action("call")
            return self._check_fold(s)

        if ctx.raises == 1:
            opener = self.position(s, ctx.last_raiser)
            open_bb = ctx.high / bb
            ours = "bb" if pos == "BB" else "sb" if pos == "SB" else "ip"
            if ctx.callers > 0:
                three, call = self.squeeze
            elif open_bb > c.BIG_OPEN_BB:
                three, call = self.vs_big_open
            else:
                key = (ours, "sb" if opener == "SB" and ours == "bb" else
                       "early" if self._group(opener) in ("early", "bb") else "late")
                three, call = self.vs_open.get(key) or self.vs_open[(ours, "late")]
            if hc in three:
                mult = c.SIX_3BET_OOP if ours in ("sb", "bb") else c.SIX_3BET_IP
                return self._raise_to(s, ctx.high * (mult + c.SIX_SQUEEZE_ADD * ctx.callers))
            if hc in call or (eff_bb > c.DEEP_SET_BB and hc in self.pairs):
                return Action("call")
            return self._check_fold(s)

        if ctx.raises == 2:
            if self._my_open(s):
                three_bettor = self.position(s, ctx.last_raiser)
                if self._group(three_bettor) in ("late", "sb") and pos in ("CO", "BTN", "SB"):
                    four, call = self.vs_3bet["late"]
                else:
                    ip = self._group(three_bettor) in ("sb", "bb") or pos == "BTN"
                    four, call = self.vs_3bet["ip" if ip else "oop"]
            else:
                four, call = self.cold_vs_3bet
            if hc in four:
                if self._stack_off(s, hc, eff_bb) or eff_bb <= 60:
                    return self._raise_to(s, ctx.high * c.SIX_4BET_MULT)
                return Action("call") if hc in call or hc in self.pairs else self._check_fold(s)
            if hc in call or (eff_bb > c.DEEP_SET_BB and hc in self.pairs and s.to_call <= 0.08 * s.me.stack):
                return Action("call")
            return self._check_fold(s)

        # 4-bets and up
        if self._stack_off(s, hc, eff_bb):
            return Action("allin") if eff_bb <= 150 else self._raise_to(s, ctx.high * c.SIX_4BET_MULT)
        return self._check_fold(s)
