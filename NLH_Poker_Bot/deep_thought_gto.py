"""DEEP_THOUGHT_GTO: a balanced, read-free bot, to test equilibrium-style play on Chipzen.

    from deep_thought_gto import DeepThoughtGTOBot
    ("GTO", DeepThoughtGTOBot(seed=seed))

It is not a solved strategy (no-limit hold'em is far too big to solve here). It plays by
the indifference rules a solved strategy satisfies, and it never adapts to an opponent:
no reads, no exploits, no saved profiles. Every opponent is assumed to play a default,
hand-set range (v1's LIKELY_* tables, not ones fitted to the Chipzen field).

Preflop
  * Heads-up at 6-25 BB: the CFR+ solution from train_hu_preflop_cfr.py with no edge for
    our bot, in 2-seat matches (gto_preflop_tables.py, rebuilt with
    `train_hu_preflop_cfr.py --edge 0 --out gto_preflop_tables.py --name GTO_PREFLOP`). The small blind folds 11-14% at
    15-25 BB, near published heads-up solutions.
  * Elsewhere: the push/fold tables when short, and the standard ranges from config.py,
    with every frequency read left at its prior (no opponent is ever tracked).

After the flop, heads-up
  Both ranges are built from the actions so far, every hand weighted by how likely it is
  to have taken them. With SOLVE on, every decision then solves the rest of the hand with
  CFR+ (gto_solver.py): the current street with bets of 1/2 and 1x pot and raises, later
  streets with 2/3-pot bets, sampled turn and river cards, exact showdowns, and our hand's
  share of the solution is played. That is what plans across streets: a flop bet is chosen
  knowing the turn and river bets it leads to. On a textbook river spot (nuts or air
  against bluff-catchers, one pot-size bet) it bets the nuts 100% and air 50%, and the
  caller calls 50%, the known equilibrium. A solve takes SOLVE_SECONDS.
  If a solve fails, the rules below decide. Each hand in our range gets its equity against
  theirs (exact on the river, over sampled runouts before it), which ranks our range.
  * Betting (first to act with the initiative, or checked to): bet BET_SIZE of the pot
    with every hand above VALUE_EQUITY, and with bluffs worth value x f/(1+f) of the
    range at size f, the river ratio that makes a call break even (x BLUFF_MULT on the
    flop and turn, where draws still have equity). Draws are bluffed first, then the
    weakest hands. Without the initiative and out of position, check.
  * Facing a bet of f x the pot: continue with the top 1/(1+f) of our range (the minimum
    defense frequency, so bluffing us with any two cards doesn't profit). Raise to
    RAISE_MULT x with the top RAISE_SHARE of the range, plus draws in the flop/turn ratio.
  Multiway pots have no defense rule that holds for every player, so there it bets only
  value and calls only on equity against the bettor's range.
"""
import bisect

import config as default_config
from bot import Action, AGGRESSIVE
from deep_thought_v1 import COMBOS, _PCT, BOARD_LEN
from deep_thought_v8 import DeepThoughtV8Bot
from gto_preflop_tables import GTO_PREFLOP
from gto_solver import solve, chips_of
from poker_core import classify_hand, hand_rank

GTO_CONFIG = {
    "USE_ICM": False,
    "FITTED": False,                        # default opponent model, not one fitted to the field
    "FREQUENCY_READS": False,
    "TRACK_OPPONENTS": False,
    "SOLVED_READ_FALLBACK": False,
    "SOLVED_LAYOUTS": ("hu",),
    "EV_CALLS": False,
    # --- after the flop ---------------------------------------------------------
    "BET_SIZE": {"flop": 0.5, "turn": 0.66, "river": 0.75},
    "VALUE_EQUITY": {"flop": 0.62, "turn": 0.64, "river": 0.66},
    "BLUFF_MULT": {"flop": 2.0, "turn": 1.5, "river": 1.0},
    "RAISE_SHARE": {"flop": 0.08, "turn": 0.07, "river": 0.05},
    "RAISE_BLUFF_RATIO": {"flop": 0.5, "turn": 0.3, "river": 0.0},
    "RAISE_MULT": 3.0,
    "JAM_SHARE": 0.7,                       # a bet this share of our stack goes all-in
    "RUNOUTS": {"flop": 12, "turn": 46},    # sampled boards for equity before the river
    "RANGE_FLOOR": 0.002,                   # hands below this share of the top weight are dropped
    # --- real-time solving (gto_solver.py) ------------------------------------------
    "SOLVE": True,                          # heads-up after the flop: solve the rest of the hand
    "SOLVE_SECONDS": 1.2,                   # per decision (Chipzen allows 5 s at half a CPU)
    "CARRY_RANGES": True,                   # start each solve from the last solve's ranges
}
DRAWS = ("STRONG_DRAW", "DRAW")


def _same(entry):
    """A history entry as compared between decisions: bet, raise and all-in alike, since the
    Chipzen adapter relabels a player's earlier raises 'allin' once that player is all-in."""
    street, seat, kind, amt = entry
    return street, seat, "aggressive" if kind in AGGRESSIVE else kind, amt
FOLD_, CHECK_, CALL_ = "fold", "check", "call"


class DeepThoughtGTOBot(DeepThoughtV8Bot):
    def __init__(self, cfg=default_config, overrides=None, seed=None):
        super().__init__(cfg, {**GTO_CONFIG, **(overrides or {})}, seed)
        self.preflop_tables = GTO_PREFLOP
        self.carry = None       # the last solve this hand: ranges and where we left its tree

    # ------------------------------------------------------------ no reads at all
    def on_hand_end(self, history, seats_dealt, showdown=None, board=None, my_seat=None):
        pass

    def load_profile(self, seat, profile):
        pass

    # ------------------------------------------------------------ ranges
    def _range(self, s, seat, dead):
        """{(a, b): weight} for every hand `seat` could hold, from its actions so far."""
        c = self.cfg
        width = self._preflop_width(s, seat)
        hands = {}
        for a, b, hc in COMBOS:
            if a in dead or b in dead:
                continue
            pct = _PCT[hc]
            w = 1.0 if pct <= width else 1.0 - (pct - width) / 0.1
            if w > 0:
                hands[(a, b)] = w
        pot_before = self._street_pots(s)
        for street in ("flop", "turn", "river"):
            n = BOARD_LEN[street]
            if len(s.board) < n:
                break
            board = s.board[:n]
            bucket = {h: classify_hand(list(h), board, self.good_kicker) for h in hands}
            high, bets, chips = 0, 0, {}
            for st, who, kind, amt in s.history:
                if st != street:
                    continue
                aggressive = kind in AGGRESSIVE and amt > high
                facing = high - chips.get(who, 0)
                frac = facing / max(1, pot_before[street] + sum(chips.values()) - facing)
                if aggressive:
                    high, bets = amt, bets + 1
                if kind in AGGRESSIVE or kind == "call":
                    chips[who] = max(chips.get(who, 0), amt)
                if who != seat or kind in ("fold", "blind"):
                    continue
                for h in hands:
                    hands[h] *= self._likelihood(bucket[h], aggressive, bets, kind, frac)
        top = max(hands.values(), default=0)
        return {h: w for h, w in hands.items() if w > top * c.RANGE_FLOOR}

    def _equities(self, s, ours, theirs):
        """{hand: equity vs `theirs`} for every hand in `ours`, ignoring card removal between them."""
        c = self.cfg
        board, need = list(s.board), 5 - len(s.board)
        if need == 0:
            runouts = [[]]
        else:
            deck = [x for x in range(52) if x not in board]
            k = c.RUNOUTS["turn" if need == 1 else "flop"]
            runouts = [[x] for x in deck] if need == 1 and k >= len(deck) else \
                [self.rng.sample(deck, need) for _ in range(k)]
        total = {h: 0.0 for h in ours}
        seen = {h: 0 for h in ours}
        for extra in runouts:
            full = board + extra
            dead = set(extra)
            ranks, weights = [], []
            opp = sorted((hand_rank([a, b] + full), w) for (a, b), w in theirs.items()
                         if a not in dead and b not in dead)
            if not opp:
                continue
            ranks = [r for r, _ in opp]
            cum = [0.0]
            for _, w in opp:
                cum.append(cum[-1] + w)
            whole = cum[-1]
            for h in ours:
                if h[0] in dead or h[1] in dead:
                    continue
                r = hand_rank(list(h) + full)
                lo, hi = bisect.bisect_left(ranks, r), bisect.bisect_right(ranks, r)
                total[h] += (cum[lo] + 0.5 * (cum[hi] - cum[lo])) / whole
                seen[h] += 1
        return {h: total[h] / seen[h] for h in ours if seen[h]}

    # ------------------------------------------------------------ helpers
    def _initiative(self, s):
        """Seat of the last aggressor before this street."""
        prev = {"flop": "preflop", "turn": "flop", "river": "turn"}[s.street]
        last = None
        for st, seat, kind, _ in s.history:
            if st == prev and kind in AGGRESSIVE:
                last = seat
        return last

    def _sized(self, s, frac):
        """A bet or raise to `frac` of the pot on top of what is owed, all-in when it is most of our stack."""
        c, me = self.cfg, s.me
        target = me.bet + s.to_call + int(frac * (s.pot + s.to_call))
        if target - me.bet >= c.JAM_SHARE * me.stack:
            return Action("allin")
        return Action("bet" if s.to_call == 0 and me.bet == 0 else "raise", max(target, s.min_raise_to))

    @staticmethod
    def _top_share(eq, weights, mine):
        """Share of the range (by weight) with equity at or below our hand's."""
        total = sum(weights[h] for h in eq)
        below = sum(weights[h] for h, e in eq.items() if e <= eq[mine])
        return below / total if total else 0.0

    # ------------------------------------------------------------ postflop
    def _postflop(self, s):
        c = self.cfg
        opps = [p for p in s.players if p.active and p.seat != s.my_seat]
        mine = tuple(sorted(s.hole_cards))
        if not opps:
            return Action("check") if s.to_call == 0 else Action("call")
        agg = self._aggressor(s) if s.to_call else None
        villain = agg if agg is not None else opps[0].seat
        board = set(s.board)
        carried = self._carried(s, villain, mine) if len(opps) == 1 and c.CARRY_RANGES else None
        self.carry = None
        if carried:
            ours, theirs = carried
        else:
            theirs = self._range(s, villain, board | set(s.hole_cards))
            ours = self._range(s, s.my_seat, board)
        ours[mine] = max(ours.get(mine, 0.0), 1e-6)       # our hand is always in our range
        if not theirs:
            return self._check_fold(s)
        eq = self._equities(s, ours, theirs)
        if mine not in eq:
            return self._check_fold(s)
        if len(opps) > 1:
            return self._multiway(s, eq[mine])
        if c.SOLVE:
            a = self._solved(s, ours, theirs, mine, villain)
            if a is not None:
                return a
        street = s.street
        buckets = {h: classify_hand(list(h), s.board, self.good_kicker) for h in eq}
        total = sum(ours[h] for h in eq)

        if s.to_call == 0:
            initiative = self._initiative(s)
            acted = any(st == street and seat == villain for st, seat, _, _ in s.history)
            if initiative != s.my_seat and not acted:
                return Action("check")             # out of position without the initiative
            f = c.BET_SIZE[street]
            value = {h for h, e in eq.items() if e >= c.VALUE_EQUITY[street]}
            budget = sum(ours[h] for h in value) * f / (1 + f) * c.BLUFF_MULT[street]
            rest = [h for h in eq if h not in value]
            # draws first (best equity first), then the weakest hands
            rest.sort(key=lambda h: (0, -eq[h]) if buckets[h] in DRAWS and street != "river" else (1, eq[h]))
            bluffs, used = set(), 0.0
            for h in rest:
                if used >= budget:
                    break
                bluffs.add(h)
                used += ours[h]
            return self._sized(s, f) if mine in value or mine in bluffs else Action("check")

        # facing a bet: defend the top 1 / (1 + f) of our range
        f = s.to_call / max(1, s.pot - s.to_call)
        share = self._top_share(eq, ours, mine)       # 1.0 = our best hand
        mdf = 1 / (1 + f)
        raise_top = c.RAISE_SHARE[street]
        if share > 1 - raise_top and s.me.stack > s.to_call:
            return self._raise_to(s, c.RAISE_MULT * (s.me.bet + s.to_call))
        if buckets[mine] in DRAWS and street != "river" and s.me.stack > s.to_call:
            draw_weight = sum(ours[h] for h in eq if buckets[h] in DRAWS)
            want = raise_top * total * c.RAISE_BLUFF_RATIO[street]
            if draw_weight and self.rng.random() < min(1.0, want / draw_weight):
                return self._raise_to(s, c.RAISE_MULT * (s.me.bet + s.to_call))
        return Action("call") if share >= 1 - mdf else Action("fold")

    def _solved(self, s, ours, theirs, mine, villain):
        """Our action from a CFR+ solve of the rest of the hand, or None to fall back."""
        street_acts = [(seat, kind) for st, seat, kind, _ in s.history if st == s.street]
        first = 0 if not street_acts or street_acts[0][0] == s.my_seat else 1
        raises = max(0, sum(1 for _, kind in street_acts if kind in AGGRESSIVE) - 1)
        opp = s.players[villain]
        spot = dict(street=s.street, board=list(s.board), first=first, to_act=0, pot=s.pot,
                    bets=[s.me.bet, opp.bet], stacks=[s.me.stack, opp.stack], raises=raises)
        try:
            probs, iterations, solver = solve(ours, theirs, mine, spot, self.rng, seconds=self.cfg.SOLVE_SECONDS)
        except Exception:
            return None
        if not probs:
            return None
        self.last_solve = (probs, iterations)
        r, total = self.rng.random(), 0.0
        choice = None
        for name, p in probs.items():
            total += p
            if r < total:
                choice = name
                break
        choice = choice or max(probs, key=probs.get)
        self._remember(s, solver, choice, villain, mine)
        if choice in (FOLD_, CHECK_, CALL_):
            return Action(choice)
        if choice in ("ball", "rall"):
            return Action("allin")
        chips = int(choice[1:])
        return Action("bet" if choice[0] == "b" else "raise", s.me.bet + chips)

    # ------------------------------------------------------------ carried ranges
    def _remember(self, s, solver, choice, villain, mine):
        """Keep both ranges after our action, weighted by how often each hand takes it."""
        root = solver.root
        a = root.actions.index(choice)
        sigma = solver._avg(root)
        ours = [float(w * x) for w, x in zip(solver.w[0], sigma[a])]
        self.carry = dict(hole=mine, prefix=[_same(h) for h in s.history], villain=villain, street=s.street,
                          hands=solver.hands, weights=[ours, [float(x) for x in solver.w[1]]], node=root.children[a],
                          solver=solver, bets={villain: s.players[villain].bet})

    def _carried(self, s, villain, mine):
        """(ours, theirs) carried from this hand's last solve through what happened since, or None."""
        k = self.carry
        if not k or k["hole"] != mine or k["villain"] != villain:
            return None
        n = len(k["prefix"])
        if [_same(h) for h in s.history[:n]] != k["prefix"] or len(s.history) <= n or s.history[n][1] != s.my_seat:
            return None
        hands, (w0, w1) = k["hands"], [list(x) for x in k["weights"]]
        node, solver = k["node"], k["solver"]
        bet = k["bets"].get(villain, 0)
        street = k["street"]
        pots = self._street_pots(s)
        for st, seat, kind, amt in s.history[n + 1:]:
            if st != street:
                street, bet = st, 0
                node = None                         # a new card: the old tree didn't plan for it
            if seat != villain:
                return None
            if node is not None and node.kind == "act" and node.player == 1:
                a = self._map_action(node, kind, amt - bet)
                if a is None:
                    return None
                sigma = solver._avg(node)
                w1 = [float(w * x) for w, x in zip(w1, sigma[a])]
                node = node.children[a]
            else:
                node = None
                w1 = self._likely(s, hands[1], w1, st, seat, kind, amt, pots)
            bet = max(bet, amt)
        dead = set(s.board)
        ours = {h: w for h, w in zip(hands[0], w0) if w > 0 and h[0] not in dead and h[1] not in dead}
        theirs = {h: w for h, w in zip(hands[1], w1) if w > 0 and h[0] not in dead and h[1] not in dead
                  and h[0] not in s.hole_cards and h[1] not in s.hole_cards}
        if not theirs or sum(theirs.values()) <= 0:
            return None
        return ours, theirs

    @staticmethod
    def _map_action(node, kind, added):
        """The tree action nearest to what the opponent did, or None if it has no match."""
        names = node.actions
        if kind in (FOLD_, CHECK_, CALL_):
            return names.index(kind) if kind in names else None
        if kind == "allin":
            for name in ("ball", "rall"):
                if name in names:
                    return names.index(name)
        sized = [(abs(chips_of(n) - added), i) for i, n in enumerate(names) if chips_of(n) is not None]
        if not sized:
            jam = [i for i, n in enumerate(names) if n in ("ball", "rall")]
            return jam[0] if jam else None
        return min(sized)[1]

    def _likely(self, s, hands, weights, street, seat, kind, amt, pots):
        """weights x how likely each hand is to take this action (the hand-set model)."""
        board = s.board[:BOARD_LEN[street]]
        high = max([a for st, who, k, a in s.history if st == street and k in AGGRESSIVE and who != seat] or [0])
        aggressive = kind in AGGRESSIVE and amt > high
        bets = 1 + sum(1 for st, who, k, a in s.history if st == street and k in AGGRESSIVE and who != seat)
        frac = max(0, high) / max(1, pots[street])
        out = []
        for h, w in zip(hands, weights):
            if w <= 0 or h[0] in board or h[1] in board:
                out.append(0.0)
                continue
            out.append(w * self._likelihood(classify_hand(list(h), board, self.good_kicker), aggressive, bets, kind, frac))
        return out

    def _multiway(self, s, equity):
        c = self.cfg
        n = sum(1 for p in s.players if p.active and p.seat != s.my_seat)
        if s.to_call == 0:
            if equity >= c.VALUE_EQUITY[s.street] ** (1 / n):
                return self._sized(s, c.BET_SIZE[s.street])
            return Action("check")
        price = s.to_call / max(1, s.pot + s.to_call)
        return Action("call") if equity ** n >= price else Action("fold")
