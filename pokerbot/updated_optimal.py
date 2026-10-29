"""UPDATED_OPTIMAL: OptimalBot + showdown bluff tracking, EV checks and ICM.

    from updated_optimal import UpdatedOptimalBot
    ("UPDATED", UpdatedOptimalBot(seed=seed))

On top of OptimalBot (which it inherits unchanged otherwise):
  1. Bluff tracking -- after each showdown, every bet/raise an opponent made is
     judged by what they actually held on that street (air or a draw = bluff).
     Only showdowns count. Each opponent's bluff rate is a Bayesian estimate
     (BLUFF_PRIOR weighted like BLUFF_PRIOR_WEIGHT hands), so a couple of
     showdowns don't swing it. Fold-to-bet rates are tracked too (every hand).
  2. EV checks -- facing a bet it estimates its equity from the opponent's bluff
     rate and compares it with the break-even equity; it never makes a call it
     estimates as -EV, and calls folds it estimates as +EV. Bluffs are checked
     against estimated fold equity. Preflop all-in calls use a precomputed
     equity table (preflop_equity.json).
  3. ICM -- whenever a call risks ICM_MIN_RISK_FRACTION of our chips, the
     break-even equity is computed from tournament points (config.PAYOUT_POINTS)
     for the fold / win / lose outcomes instead of from chips.

Everything stays far under the 1 ms decision limit (numpy ICM ~0.03 ms).
Rebuild the preflop table (optional) with:  python3 updated_optimal.py --build-equity
"""
import gc
import json
import math
import os
import random
import sys

import config as default_config
from bot import Action, AGGRESSIVE
from optimalbot import OptimalBot
from poker_core import classify_hand, hand_class, hand_rank, parse_range, ALL_CLASSES, MONSTER

try:
    import numpy as np
except ImportError:          # no numpy -> chip EV only
    np = None

EQUITY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preflop_equity.json")

# =====================================================================
# NEW SETTINGS (edit here, or overrides={...} per bot)
# Equity tables are (flop, turn, river). Buckets: see poker_core.classify_hand.
# =====================================================================
UPDATED_CONFIG = {
    # --- 1. bluff / fold tracking --------------------------------------
    "BLUFF_PRIOR": 0.25,          # assumed share of bets that are bluffs before any showdowns
    "BLUFF_PRIOR_WEIGHT": 6,      # prior counts like this many showdown bets
    "SEMI_BLUFF_WEIGHT": 0.5,     # a STRONG_DRAW bet counts as half a bluff
    "FOLD_PRIOR": 0.40,           # assumed fold-to-bet rate before any data
    "FOLD_PRIOR_WEIGHT": 10,

    # --- 2. EV checks --------------------------------------------------
    "EV_CALLS": True,             # fold -EV calls, call +EV folds
    "CALL_EV_MARGIN": 0.0,        # extra equity required on top of break-even
    "EV_BLUFFS": True,            # only bluff when estimated fold equity makes it +EV
    "RAISE_BLUFF_SCALE": 0.7,     # raises are bluffs less often than bets
    "FOLD_TO_RAISE_SCALE": 0.6,   # players fold to raises less than their usual fold rate... x this
    "MULTIWAY_EQ_MULT": 0.85,     # equity x this per extra opponent in the pot
    # our equity vs a bettor's BLUFFS (air / draws)
    "EQ_VS_BLUFF": {"MONSTER": (0.97, 0.98, 1.00), "STRONG": (0.85, 0.90, 0.98),
                    "MEDIUM": (0.75, 0.82, 0.92), "STRONG_DRAW": (0.55, 0.50, 0.50),
                    "DRAW": (0.45, 0.40, 0.50), "WEAK": (0.35, 0.30, 0.50)},
    # our equity vs a bettor's VALUE bets (first bet on the street)
    "EQ_VS_VALUE": {"MONSTER": (0.75, 0.80, 0.85), "STRONG": (0.40, 0.35, 0.30),
                    "MEDIUM": (0.25, 0.20, 0.12), "STRONG_DRAW": (0.45, 0.35, 0.02),
                    "DRAW": (0.30, 0.18, 0.02), "WEAK": (0.12, 0.06, 0.02)},
    # our equity vs a VALUE raise (raise / re-raise / all-in over a bet) -- much stronger
    "EQ_VS_RAISE": {"MONSTER": (0.60, 0.62, 0.65), "STRONG": (0.20, 0.12, 0.05),
                    "MEDIUM": (0.12, 0.08, 0.02), "STRONG_DRAW": (0.40, 0.28, 0.01),
                    "DRAW": (0.25, 0.14, 0.01), "WEAK": (0.08, 0.04, 0.01)},
    # our equity when a bluff/semi-bluff of ours gets called
    "EQ_WHEN_CALLED": {"WEAK": (0.12, 0.06, 0.02), "DRAW": (0.32, 0.18, 0.02),
                       "STRONG_DRAW": (0.45, 0.33, 0.02)},

    # --- 3. ICM --------------------------------------------------------
    "USE_ICM": True,
    "ICM_MIN_RISK_FRACTION": 0.25,  # use ICM when a call risks >= 25% of our chips

    # --- preflop all-in calls -------------------------------------------
    "PREFLOP_EV": True,
    "PREFLOP_EV_MIN_RISK": 0.25,    # only for calls risking >= 25% of our chips
    "PREFLOP_CALL_MARGIN": 0.02,    # call a range-fold only if equity beats break-even by this
    "SHOVE_FOLD_ALLOWANCE": 0.08,   # our own re-shove keeps fold equity: allow this much below break-even
    # assumed range of a big preflop raise/shove, by the raiser's stack (big blinds)
    "SHOVER_RANGES": {"wide": "22+, A2s+, A7o+, K9s+, KTo+, QTs+, JTs",
                      "medium": "77+, ATs+, KQs, AJo+",
                      "tight": "TT+, AQs+, AKo"},
    "SHOVER_WIDE_UNDER_BB": 15,
    "SHOVER_MEDIUM_UNDER_BB": 35,

    # Original rule shoves STRONG+ at low stack-to-pot even facing raises; the EV
    # check now decides those calls, so the rule only fires when we are betting.
    "SPR_RULE_ONLY_WHEN_BETTING": True,
}

STREET_INDEX = {"flop": 0, "turn": 1, "river": 2}
BOARD_LEN = {"flop": 3, "turn": 4, "river": 5}
BLUFF_WEIGHT = {"WEAK": 1.0, "DRAW": 1.0}


# =====================================================================
# Fast ICM (Malmuth-Harville), vectorised with numpy: ~0.03 ms for 8 players
# =====================================================================
_ICM_CACHE = {}


def _icm_layers(n):
    if n not in _ICM_CACHE:
        masks = np.arange(1 << n)
        M = ((masks[:, None] >> np.arange(n)) & 1).astype(float)
        pop = M.sum(1).astype(int)
        layers = []
        for k in range(n):
            frm, ply = np.nonzero((pop[:, None] == k) & (M == 0))
            layers.append((frm, ply, frm | (1 << ply)))
        _ICM_CACHE[n] = (M, layers)
    return _ICM_CACHE[n]


if np is not None:   # build every table size now, so no decision pays the setup cost
    for _n in range(2, 10):
        _icm_layers(_n)



def icm_points(stacks, payouts):
    """stacks: {seat: chips} of players still in. -> {seat: expected points}.
    Players with 0 chips (busting in this scenario) take the lowest places."""
    alive = [s for s, c in stacks.items() if c > 0]
    out = {s: 0.0 for s in stacks}
    busted = [s for s in stacks if stacks[s] <= 0]
    n = len(alive)
    pays = list(payouts[:len(stacks)]) + [0] * max(0, len(stacks) - len(payouts))
    if busted:   # they share the places below the survivors
        share = sum(pays[n:len(stacks)]) / len(busted)
        for s in busted:
            out[s] = share
    if n == 1:
        out[alive[0]] = pays[0]
        return out
    M, layers = _icm_layers(n)
    st = np.asarray([stacks[s] for s in alive], float)
    tot = st.sum()
    S = M @ st
    prob = np.zeros(1 << n)
    prob[0] = 1.0
    eq = np.zeros(n)
    for k, (frm, ply, to) in enumerate(layers):
        c = prob[frm] * st[ply] / (tot - S[frm])
        if pays[k]:
            eq += np.bincount(ply, c * pays[k], n)
        prob += np.bincount(to, c, 1 << n)
    for i, s in enumerate(alive):
        out[s] = float(eq[i])
    return out


# =====================================================================
# Preflop equity table: hand class -> equity vs each SHOVER_RANGES entry
# =====================================================================
def build_preflop_table(ranges, sims=1500, seed=1):
    from multiprocessing import Pool
    jobs = [(name, rng_text, hc, sims, seed) for name, rng_text in ranges.items() for hc in sorted(ALL_CLASSES)]
    with Pool() as pool:
        rows = pool.map(_preflop_job, jobs)
    table = {}
    for name, hc, eq in rows:
        table.setdefault(name, {})[hc] = round(eq, 4)
    return table


def _class_combos(hc):
    r1, r2 = "23456789TJQKA".index(hc[0]), "23456789TJQKA".index(hc[1])
    if len(hc) == 2:
        return [(r1 * 4 + a, r1 * 4 + b) for a in range(4) for b in range(a + 1, 4)]
    if hc[2] == "s":
        return [(r1 * 4 + s, r2 * 4 + s) for s in range(4)]
    return [(r1 * 4 + a, r2 * 4 + b) for a in range(4) for b in range(4) if a != b]


def _preflop_job(job):
    name, rng_text, hc, sims, seed = job
    rng = random.Random(f"{seed}-{name}-{hc}")
    villain = [c for v in parse_range(rng_text) for c in _class_combos(v)]
    mine = _class_combos(hc)
    total = 0.0
    done = 0
    while done < sims:
        h = rng.choice(mine)
        v = rng.choice(villain)
        if set(h) & set(v):
            continue
        deck = [c for c in range(52) if c not in h and c not in v]
        board = rng.sample(deck, 5)
        a, b = hand_rank(list(h) + board), hand_rank(list(v) + board)
        total += 1.0 if a > b else 0.5 if a == b else 0.0
        done += 1
    return name, hc, total / sims


def _load_preflop_table():
    try:
        with open(EQUITY_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


_PREFLOP_TABLE = _load_preflop_table()


# =====================================================================
# Opponent model: showdown bluffs, fold-to-bet
# =====================================================================
class ShowdownStats:
    def __init__(self):
        self.bets_shown = {}     # aggressive postflop actions seen at showdown
        self.bluffs_shown = {}   # ... of which were bluffs (weighted)
        self.fold_chances = {}   # times facing a postflop bet
        self.folds = {}

    def observe(self, history, showdown, board, good_kicker, me, semi_bluff_weight):
        # fold-to-bet: first response of each player to a bet on each street
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
        # bluffs: only players who reached showdown
        if not showdown or not board or len(board) < 5:
            return
        for seat, cards in showdown.items():
            if seat == me:
                continue
            for street in ("flop", "turn", "river"):
                high = 0
                aggressive = False
                for st, s, kind, amt in history:
                    if st != street:
                        continue
                    if kind in AGGRESSIVE and amt > high:
                        if s == seat:
                            aggressive = True
                        high = amt
                if not aggressive:
                    continue
                bucket = classify_hand(list(cards), list(board[:BOARD_LEN[street]]), good_kicker)
                self.bets_shown[seat] = self.bets_shown.get(seat, 0) + 1
                self.bluffs_shown[seat] = self.bluffs_shown.get(seat, 0) + \
                    BLUFF_WEIGHT.get(bucket, semi_bluff_weight if bucket == "STRONG_DRAW" else 0.0)

    def bluff_rate(self, seat, cfg):
        w = cfg.BLUFF_PRIOR_WEIGHT
        return (self.bluffs_shown.get(seat, 0) + cfg.BLUFF_PRIOR * w) / (self.bets_shown.get(seat, 0) + w)

    def fold_rate(self, seat, cfg):
        w = cfg.FOLD_PRIOR_WEIGHT
        return (self.folds.get(seat, 0) + cfg.FOLD_PRIOR * w) / (self.fold_chances.get(seat, 0) + w)


# =====================================================================
class UpdatedOptimalBot(OptimalBot):
    WANTS_SHOWDOWN = True   # simulator / adapter pass showdown cards to on_hand_end

    def __init__(self, cfg=default_config, overrides=None, seed=None):
        super().__init__(cfg, {**UPDATED_CONFIG, **(overrides or {})}, seed)
        self.sd = ShowdownStats()
        self._my_seat = None
        self.last_ev = None
        self.shover_ranges = {k: parse_range(v) for k, v in self.cfg.SHOVER_RANGES.items()}

    # ------------------------------------------------------------ tracking
    def on_hand_end(self, history, seats_dealt, showdown=None, board=None, my_seat=None):
        super().on_hand_end(history, seats_dealt)
        self.sd.observe(history, showdown or {}, board or [], self.good_kicker,
                        self._my_seat if my_seat is None else my_seat, self.cfg.SEMI_BLUFF_WEIGHT)

    def act(self, s):
        # No garbage-collection pauses inside the 1 ms decision; Python collects
        # again as soon as the decision is returned.
        self._my_seat = s.my_seat
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            return super().act(s)
        finally:
            if was_enabled:
                gc.enable()

    # ------------------------------------------------------------ helpers
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
        P = default_config.PAYOUT_POINTS
        ef, ew, el = (icm_points(x, P)[s.my_seat] for x in (fold, win, lose))
        if ew <= el:
            return chip_be, "chips"
        return min(1.0, max(0.0, (ef - el) / (ew - el))), "ICM"

    # ------------------------------------------------------------ postflop
    def _postflop(self, s):
        c = self.cfg
        if c.SPR_RULE_ONLY_WHEN_BETTING and s.to_call > 0:
            saved = c.COMMIT_SPR
            c.COMMIT_SPR = -1.0          # disable the low-SPR shove while facing a bet
            try:
                a = super()._postflop(s)
            finally:
                c.COMMIT_SPR = saved
        else:
            a = super()._postflop(s)

        bucket = classify_hand(s.hole_cards, s.board, self.good_kicker)
        opps = [p for p in s.players if p.active and p.seat != s.my_seat]
        i = STREET_INDEX[s.street]

        if s.to_call == 0:
            # bluff EV: fold equity vs everyone in the pot
            if (c.EV_BLUFFS and a.kind in ("bet", "raise", "allin")
                    and bucket in ("WEAK", "DRAW", "STRONG_DRAW")):
                bet = s.me.stack if a.kind == "allin" else max(0, a.amount - s.me.bet)
                pf = math.prod(self.sd.fold_rate(p.seat, c) for p in opps)
                eq = c.EQ_WHEN_CALLED[bucket][i]
                ev = pf * s.pot + (1 - pf) * (eq * (s.pot + 2 * bet) - bet)
                if ev < 0:
                    return Action("check")
            return a

        if not c.EV_CALLS:
            return a
        agg = self._aggressor(s)
        raised = self._facing_raise(s)
        pb = self.sd.bluff_rate(agg, c) * (c.RAISE_BLUFF_SCALE if raised else 1.0) if agg is not None \
            else c.BLUFF_PRIOR
        vs_value = (c.EQ_VS_RAISE if raised else c.EQ_VS_VALUE)[bucket][i]
        eq = pb * c.EQ_VS_BLUFF[bucket][i] + (1 - pb) * vs_value
        eq *= c.MULTIWAY_EQ_MULT ** max(0, len(opps) - 1)
        be, _ = self._breakeven(s, agg)
        call_ok = eq >= be + c.CALL_EV_MARGIN
        self.last_ev = {"equity": eq, "breakeven": be, "bluff_rate": pb, "bucket": bucket}

        pure_call = a.kind == "call" or (a.kind == "allin" and s.me.stack <= s.to_call)
        if a.kind == "fold":
            return Action("call") if call_ok else a
        if pure_call:
            return a if call_ok or bucket == MONSTER else Action("fold")
        # a raise: fine if calling is +EV anyway, else it needs fold equity
        if call_ok or bucket == MONSTER:
            return a
        cost = (s.me.stack + s.me.bet if a.kind == "allin" else a.amount) - s.me.bet
        pf = math.prod(self.sd.fold_rate(p.seat, c) * c.FOLD_TO_RAISE_SCALE for p in opps)
        eq_called = c.EQ_VS_RAISE[bucket][i]
        ev = pf * s.pot + (1 - pf) * (eq_called * (s.pot + 2 * cost - s.to_call) - cost)
        return a if ev > 0 else Action("fold")

    # ------------------------------------------------------------ preflop
    def _shover_key(self, s, agg):
        c = self.cfg
        if agg is None:
            return "medium"
        p = s.players[agg]
        bb = (p.stack + p.bet) / s.big_blind
        order = ["tight", "medium", "wide"]
        k = 2 if bb <= c.SHOVER_WIDE_UNDER_BB else 1 if bb <= c.SHOVER_MEDIUM_UNDER_BB else 0
        if self._preflop_context(s).raises >= 3:
            k = 0
        if self.stats.is_tight(agg, c):
            k = max(0, k - 1)
        elif self.stats.is_loose(agg, c) or self.stats.is_aggro(agg, c):
            k = min(2, k + 1)
        return order[k]

    def _preflop(self, s):
        a = super()._preflop(s)
        c = self.cfg
        if not c.PREFLOP_EV or _PREFLOP_TABLE is None or s.to_call == 0:
            return a
        risk = min(s.to_call, s.me.stack) / max(1, s.me.stack + s.me.bet)
        if risk < c.PREFLOP_EV_MIN_RISK or a.kind not in ("fold", "call", "allin"):
            return a
        agg = self._aggressor(s)
        key = self._shover_key(s, agg)
        eq = _PREFLOP_TABLE[key][hand_class(*s.hole_cards)]
        eq *= c.MULTIWAY_EQ_MULT ** self._preflop_context(s).callers
        be, _ = self._breakeven(s, agg)
        self.last_ev = {"equity": eq, "breakeven": be, "range": key}
        if a.kind == "fold":
            return Action("call") if eq >= be + c.PREFLOP_CALL_MARGIN else a
        if a.kind == "call" or s.me.stack <= s.to_call:
            return a if eq >= be else Action("fold")
        return a if eq >= be - c.SHOVE_FOLD_ALLOWANCE else Action("fold")   # our re-shove


if __name__ == "__main__" and "--build-equity" in sys.argv:
    table = build_preflop_table(UPDATED_CONFIG["SHOVER_RANGES"])
    with open(EQUITY_FILE, "w") as f:
        json.dump(table, f, indent=0, sort_keys=True)
    print(f"wrote {EQUITY_FILE}: " + ", ".join(f"{k} {len(v)} hands" for k, v in table.items()))
