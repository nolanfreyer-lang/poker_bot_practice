"""HU_UPDATED_OPTIMAL: UpdatedOptimalBot with hard-coded heads-up preflop ranges.

    from hu_updated_optimal import HuUpdatedOptimalBot
    ("HU_UPDATED", HuUpdatedOptimalBot(seed=seed))

Heads-up, nearly every hand is worth playing, so UPDATED_OPTIMAL's 8-handed
ranges fold far too much (57% of small blinds, 65% of big blinds vs Slumbot).
These ranges are used whenever exactly two players are left; with three or more
it plays exactly like UPDATED_OPTIMAL. Postflop play, bluff tracking, EV checks
and ICM are all inherited.

The ranges are hand-written approximations of heads-up play, not solver output.
Range syntax is the same as config.py. BTN = small blind in heads-up.
"""
import config as default_config
from bot import Action
from poker_core import hand_class, parse_range
from pushfold_tables import PUSHFOLD
from updated_optimal import UpdatedOptimalBot

HU_CONFIG = {
    # --- button / small blind ---------------------------------------------
    "HU_OPEN_BB": 2.5,
    "HU_BTN_OPEN": ("22+, A2s+, A2o+, K2s+, K2o+, Q2s+, Q3o+, J2s+, J5o+, T2s+, T6o+, "
                    "92s+, 96o+, 82s+, 86o+, 72s+, 75o+, 62s+, 65o, 52s+, 54o, 42s+, 32s"),
    # facing a 3-bet: the ranges are in updated_optimal.py, shared by every version

    # --- big blind --------------------------------------------------------
    # facing a limp: raise these, check the rest
    "HU_ISO_BB": 3.5,
    "HU_BB_RAISE_LIMP": "88+, A8s+, A5s, KTs+, QTs+, JTs, ATo+, KJo+",
    # facing an open: 3-bet value + suited bluffs
    "HU_3BET_MULT": 3.5,
    "HU_BB_3BET": ("99+, ATs+, A5s-A2s, KTs+, K9s, QTs+, Q9s, J9s+, T8s+, 97s+, 86s+, 75s+, 65s, "
                   "AJo+, KQo"),
    # call opens up to HU_WIDE_CALL_MAX_BB with the wide range, bigger ones with the tight one
    "HU_WIDE_CALL_MAX_BB": 3.5,
    "HU_BB_CALL": ("88-22, A9s-A6s, K8s-K2s, Q8s-Q2s, J8s-J3s, T7s-T4s, 96s-94s, 85s-83s, 74s-73s, "
                   "64s-63s, 54s-52s, 43s-42s, 32s, ATo-A2o, KJo-K4o, QJo-Q7o, JTo-J8o, T9o-T8o, "
                   "98o-97o, 87o-86o, 76o, 65o"),
    "HU_BB_CALL_TIGHT": ("88-22, A9s-A2s, K8s+, Q8s+, J8s+, T7s+, 97s+, 86s+, 75s+, 64s+, 54s, "
                         "ATo-A7o, KTo-K9o, QTo+, JTo"),
    # facing a 4-bet
    "HU_5BET_MULT": 2.2,
    "HU_BB_CALL_4BET": "QQ-TT, AQs-AJs, AKo, KQs",

    # --- all-in pots: smallest key >= effective stack (big blinds) -----------
    # Deep stacks only stack off with the top of the range.
    "HU_STACK_OFF_BY_BB": {40: "99+, AJs+, AQo+", 100: "JJ+, AKs, AKo", 150: "QQ+, AKs", 100000: "KK+"},

    # --- short stacks: shove or fold -----------------------------------------
    "HU_PUSH_FOLD_BB": 12,
    "PUSHFOLD_CFR": True,           # use the solved strategy (train_pushfold_cfr.py), not the ranges below
    "HU_SHOVE_BY_BB": {
        6: "22+, A2s+, A2o+, K2s+, K2o+, Q2s+, Q2o+, J2s+, J4o+, T2s+, T6o+, 92s+, 96o+, 82s+, 86o+, "
           "72s+, 75o+, 62s+, 65o, 52s+, 54o, 42s+, 32s",
        9: "22+, A2s+, A2o+, K2s+, K2o+, Q2s+, Q6o+, J2s+, J8o+, T4s+, T8o+, 95s+, 98o, 85s+, 74s+, "
           "64s+, 53s+",
        12: "22+, A2s+, A2o+, K2s+, K5o+, Q4s+, Q9o+, J7s+, JTo, T7s+, 97s+, 86s+, 76s, 65s",
    },
    "HU_CALL_SHOVE_BY_BB": {
        6: "22+, A2s+, A2o+, K2s+, K3o+, Q2s+, Q7o+, J5s+, J9o+, T6s+, T9o, 97s+, 87s",
        9: "22+, A2s+, A2o+, K4s+, K8o+, Q8s+, QTo+, J9s+, JTo",
        12: "33+, A2s+, A4o+, K8s+, KTo+, QTs+, QJo",
    },

    # Above this depth the range decides alone: the parent's preflop EV check
    # assumes a shover's range that is far too wide for a 100+ BB shove.
    "HU_PREFLOP_EV_MAX_BB": 60,
}


class HuUpdatedOptimalBot(UpdatedOptimalBot):
    def __init__(self, cfg=default_config, overrides=None, seed=None):
        super().__init__(cfg, {**HU_CONFIG, **(overrides or {})}, seed)
        c = self.cfg
        names = ("HU_BTN_OPEN", "HU_BB_RAISE_LIMP", "HU_BB_3BET",
                 "HU_BB_CALL", "HU_BB_CALL_TIGHT", "HU_BB_CALL_4BET")
        self.hu = {n: parse_range(getattr(c, n)) for n in names}
        by_stack = lambda table: sorted((bb, parse_range(r)) for bb, r in table.items())
        self.hu_stack_off = by_stack(c.HU_STACK_OFF_BY_BB)
        self.hu_shove = by_stack(c.HU_SHOVE_BY_BB)
        self.hu_call_shove = by_stack(c.HU_CALL_SHOVE_BY_BB)

    @staticmethod
    def _by_stack(table, eff_bb):
        return next((r for limit, r in table if eff_bb <= limit), table[-1][1])

    def _heads_up(self, s):
        return self._players_left(s) == 2

    def _preflop(self, s):
        if self._heads_up(s) and self._effective_stack(s) / s.big_blind > self.cfg.HU_PREFLOP_EV_MAX_BB:
            return self._preflop_action(s)
        return super()._preflop(s)

    def _preflop_action(self, s):
        if not self._heads_up(s):
            return super()._preflop_action(s)
        c, R = self.cfg, self.hu
        hc = hand_class(*s.hole_cards)
        pos = self.position(s)
        bb = s.big_blind
        ctx = self._preflop_context(s)
        eff_bb = self._effective_stack(s) / bb

        # --- short stacks: shove or fold
        if eff_bb <= c.HU_PUSH_FOLD_BB:
            if c.PUSHFOLD_CFR and pos == "BTN" or c.PUSHFOLD_CFR and ctx.raises > 0:
                depth = max(2, min(15, round(eff_bb)))
                chance = PUSHFOLD[depth]["shove" if ctx.raises == 0 else "call"].get(hc, 0.0)
                return Action("allin") if self.rng.random() < chance else self._check_fold(s)
            table = self.hu_shove if ctx.raises == 0 else self.hu_call_shove
            if hc in self._by_stack(table, eff_bb):
                return Action("allin")
            return self._check_fold(s)

        # --- facing an all-in: stack off by depth
        raiser = s.players[ctx.last_raiser] if ctx.last_raiser is not None else None
        if raiser is not None and raiser.all_in:
            # the more they shove, the more of our range is good enough to call
            depth = eff_bb / max(1.0, min(4.0, self._freq_ratio(s, ctx.last_raiser, "all_in")))
            return Action("call") if hc in self._by_stack(self.hu_stack_off, depth) else self._check_fold(s)

        if ctx.raises == 0:
            if pos == "BTN":
                return self._raise_to(s, c.HU_OPEN_BB * bb) if hc in R["HU_BTN_OPEN"] else Action("fold")
            if hc in R["HU_BB_RAISE_LIMP"]:                   # big blind vs a limp
                return self._raise_to(s, c.HU_ISO_BB * bb)
            return Action("check")

        if ctx.raises == 1:                                   # big blind vs an open
            if hc in R["HU_BB_3BET"]:
                return self._raise_to(s, ctx.high * c.HU_3BET_MULT)
            wide = ctx.high / bb <= c.HU_WIDE_CALL_MAX_BB
            return Action("call") if hc in R["HU_BB_CALL" if wide else "HU_BB_CALL_TIGHT"] \
                else self._check_fold(s)

        if ctx.raises == 2:                                   # button vs a 3-bet
            return self._defend_3bet(s, hc, ctx)

        stack_off = hc in self._by_stack(self.hu_stack_off, eff_bb)
        if ctx.raises == 3:                                   # big blind vs a 4-bet
            if stack_off:
                return self._raise_to(s, ctx.high * c.HU_5BET_MULT)
            return Action("call") if hc in R["HU_BB_CALL_4BET"] else self._check_fold(s)
        return Action("call") if stack_off else self._check_fold(s)   # 5-bet and up
