"""DEEP_THOUGHT_V8: DEEP_THOUGHT_V7 with a solved heads-up preflop at short stacks.

    from deep_thought_v8 import DeepThoughtV8Bot
    ("DEEP8", DeepThoughtV8Bot(seed=seed))

Heads-up with an effective stack of SOLVED_MIN_BB to SOLVED_MAX_BB big blinds, every
preflop decision comes from the CFR+ strategies in hu_preflop_tables.py
(train_hu_preflop_cfr.py): the small blind folds, limps, min-raises or shoves; the big
blind checks, isolates to 4 BB, calls, 3-bets to 3x or shoves; and each side answers the
other's raise or shove. v7 played these depths from hand-written ranges and called
shoves only with 99+, AJs+, AQo+ below 40 BB.

The tables come in two layouts. In 2-seat matches the small blind is the button and acts
last after the flop ("hu"); at a Chipzen 6-max table down to two the big blind acts last
on every street after the flop ("short"), where dealer_seat sits on the big blind, so the
small blind plays tighter and shoves more. A spot the tables don't cover (a 4-bet that
isn't all-in, a raise after a limp-check) goes back to v7's play, as does everything
after the flop.

The tables were solved giving our bot an extra 10% of flop pots (HERO_EDGE 0.1), so it
keeps seeing flops where it outplays the field. Paired against v7 in 2-seat matches
(40 x 200 hands against each of 14 opponents) they won +9.3, +9.4 and +16.9 bb/100 at
15, 20 and 25 BB; without MANIAC, +4.8, +1.7 and +8.7. In the short layout every edge
lost to v7 (-0.6 to -3.4), so SOLVED_LAYOUTS leaves that one to v7.

Facing an all-in from a bot that shoves 1.5x more or less than the field, v7 decides
instead: it calls wider against frequent shovers, which the tables can't (+7 to +10 bb/100
against PUSHFOLD). That made the totals +9.7, +10.1 and +17.5 bb/100. Handing back the
other spots on v7's reads (3-bet rate, aggressive or loose) cost 4-9 bb/100, mostly
against MANIAC, where the tables gain the most.
"""
import config as default_config
from bot import Action, AGGRESSIVE
from deep_thought_v7 import DeepThoughtV7Bot
from hu_preflop_tables import HU_CLASSES, HU_LEVELS, HU_PREFLOP
from poker_core import hand_class

DEEP8_CONFIG = {
    "SOLVED_PREFLOP": True,
    "SOLVED_MIN_BB": 6,                 # below this, v7's push/fold table
    "SOLVED_MAX_BB": 25,
    "SOLVED_OPEN_BB": 2.0,              # the sizes the strategies were solved for
    "SOLVED_ISO_BB": 4.0,
    "SOLVED_3BET_MULT": 3.0,
    "SOLVED_SPOTS": None,               # spots the tables play (None = every spot they cover)
    "SOLVED_LAYOUTS": ("hu",),          # "short" lost to v7 in paired tests
    "SOLVED_READ_FALLBACK": True,       # facing a shove, v7 decides vs bots that shove off the field
    "SOLVED_READ_RATIO": 1.5,           # their all-in rate this far from the field (either way)
}
# the spots where we face an all-in: call or fold, no flop for the model to guess at
ALLIN_SPOTS = frozenset({"bb_vs_shove", "sb_vs_shove_limp", "sb_vs_shove_open", "bb_vs_shove_iso",
                         "bb_vs_shove_3bet"})

_INDEX = {h: i for i, h in enumerate(HU_CLASSES)}
_N = len(HU_CLASSES)

# (the small blind's and big blind's actions so far) -> the spot in the tables
SPOTS = {
    (): "sb_open",
    ("call",): "bb_vs_limp",
    ("call", "raise"): "sb_vs_iso",
    ("call", "allin"): "sb_vs_shove_limp",
    ("call", "raise", "allin"): "bb_vs_shove_iso",
    ("raise",): "bb_vs_open",
    ("raise", "raise"): "sb_vs_3bet",
    ("raise", "raise", "allin"): "bb_vs_shove_3bet",
    ("raise", "allin"): "sb_vs_shove_open",
    ("allin",): "bb_vs_shove",
}


class DeepThoughtV8Bot(DeepThoughtV7Bot):
    def __init__(self, cfg=default_config, overrides=None, seed=None):
        super().__init__(cfg, {**DEEP8_CONFIG, **(overrides or {})}, seed)
        self.preflop_tables = HU_PREFLOP    # a subclass can play other solved tables

    def _preflop(self, s):
        a = self._solved_preflop(s)
        return a if a is not None else super()._preflop(s)

    def solved_spot(self, s):
        """(layout, depth, spot) when the tables cover this decision, else None."""
        c = self.cfg
        if not c.SOLVED_PREFLOP or self._players_left(s) != 2:
            return None
        depth = self._effective_stack(s) / s.big_blind
        if not c.SOLVED_MIN_BB <= depth < c.SOLVED_MAX_BB + 0.5:
            return None
        blinds = [seat for street, seat, kind, _ in s.history if street == "preflop" and kind == "blind"]
        if len(blinds) != 2:
            return None
        sb = blinds[0]
        line = []
        for street, seat, kind, amt in s.history:
            if street != "preflop" or kind == "blind":
                continue
            if kind in AGGRESSIVE:
                p = s.players[seat]
                covers = amt >= s.me.stack + s.me.bet if seat != s.my_seat else False
                kind = "allin" if kind == "allin" or p.all_in or covers else "raise"
            elif kind not in ("call", "check"):
                return None
            line.append(kind)
        if (len(line) % 2 == 0) != (s.my_seat == sb):
            return None
        layout = "hu" if s.button_seat == sb else "short"
        if layout not in c.SOLVED_LAYOUTS:
            return None
        table = self.preflop_tables[layout][max(6, min(25, round(depth)))]
        spot = SPOTS.get(tuple(line))
        if spot not in table or c.SOLVED_SPOTS is not None and spot not in c.SOLVED_SPOTS:
            return None
        opp = next((p.seat for p in s.players if p.dealt and p.seat != s.my_seat), None)
        if opp is not None and self._read_off_field(s, opp, spot):
            return None
        return layout, round(depth), spot

    def _read_off_field(self, s, opp, spot):
        """True when we face an all-in from someone who shoves far more or less than the field:
        v7 widens or tightens its calls by that read, which the tables can't."""
        c = self.cfg
        if not c.SOLVED_READ_FALLBACK or spot not in ALLIN_SPOTS:
            return False
        r = self._freq_ratio(s, opp, "all_in")
        return r > c.SOLVED_READ_RATIO or r < 1 / c.SOLVED_READ_RATIO

    def _solved_preflop(self, s):
        found = self.solved_spot(s)
        if found is None:
            return None
        layout, depth, spot = found
        actions, codes = self.preflop_tables[layout][max(6, min(25, depth))][spot]
        i = _INDEX[hand_class(*s.hole_cards)]
        r, total = self.rng.random() * HU_LEVELS, 0
        for k, name in enumerate(actions):
            total += ord(codes[k * _N + i]) - ord("a")
            if r < total:
                return self._solved_action(s, name, spot)
        return self._solved_action(s, actions[-1], spot)

    def _solved_action(self, s, name, spot):
        c, bb = self.cfg, s.big_blind
        if name == "fold":
            return self._check_fold(s)
        if name in ("check", "call"):
            return Action("check") if s.to_call == 0 else Action("call")
        if name == "allin":
            return Action("allin")
        if spot == "sb_open":
            to = c.SOLVED_OPEN_BB * bb
        elif spot == "bb_vs_limp":
            to = c.SOLVED_ISO_BB * bb
        else:
            to = c.SOLVED_3BET_MULT * self._preflop_context(s).high
        return Action("allin") if to >= s.me.stack + s.me.bet else self._raise_to(s, to)
