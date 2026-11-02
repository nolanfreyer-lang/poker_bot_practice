"""DEEP_THOUGHT_V1: HU_UPDATED_OPTIMAL with real range-based postflop calculations.

    from deep_thought_v1 import DeepThoughtV1Bot
    ("DEEP", DeepThoughtV1Bot(seed=seed))

Uses far more than 1 ms per decision, so it is for Chipzen (5 s clock, 0.5 CPU),
NOT the event's 1 ms limit. Preflop play is inherited. Postflop it:
  1. Builds the opponent's range: every possible hand, weighted by their preflop
     action, then reweighted by each postflop action on the board as it stood.
  2. Calculates equity hand by hand against that range: exact on the river,
     sampled runouts on the flop and turn. This replaces the bucket tables.
  3. Calls or folds by that equity. With DEEP_SIZING on, it also picks between
     checking and several bet sizes by EV, using estimated fold rates.

The likelihood tables are hand-set estimates, not solver output.
"""
import time

import config as default_config
from bot import Action, AGGRESSIVE
from hu_updated_optimal import HuUpdatedOptimalBot
from fitted_tables import FIT_BET, FIT_CHECK, FIT_RESPONSE, SIZE_BINS
from poker_core import classify_hand, hand_class, hand_rank
import updated_optimal as _updated

DEEP_CONFIG = {
    # --- compute budget --------------------------------------------------
    # 20,000 evaluations takes about 30 ms on a fast laptop. DEADLINE is a hard
    # stop for slow machines: what was finished so far is used.
    "DEEP_EVALS": 20000,
    "DEEP_DEADLINE_MS": 1200,
    # --- what opponents do with each kind of hand (relative likelihoods) ------
    "LIKELY_BET": {"MONSTER": 0.90, "STRONG": 0.75, "MEDIUM": 0.35, "STRONG_DRAW": 0.60, "DRAW": 0.40, "WEAK": 0.18},
    "LIKELY_RAISE": {"MONSTER": 0.70, "STRONG": 0.15, "MEDIUM": 0.03, "STRONG_DRAW": 0.20, "DRAW": 0.07, "WEAK": 0.03},
    "LIKELY_CALL": {"MONSTER": 0.40, "STRONG": 0.85, "MEDIUM": 0.70, "STRONG_DRAW": 0.90, "DRAW": 0.70, "WEAK": 0.10},
    "LIKELY_CHECK": {"MONSTER": 0.25, "STRONG": 0.40, "MEDIUM": 0.70, "STRONG_DRAW": 0.50, "DRAW": 0.70, "WEAK": 1.00},
    # --- preflop range width: share of hands for each line ---------------------
    "RANGE_CHECK": 0.90, "RANGE_CALL": 0.55, "RANGE_RAISE": 0.70, "RANGE_3BET": 0.25, "RANGE_4BET": 0.05,
    "RANGE_MULTIWAY_MULT": 0.55,    # three or more players: tighter ranges
    # --- bet sizing --------------------------------------------------------
    # "off": the inherited bet/check and raise choices stand.
    # "size_only": the inherited choice stands; EV only picks how much to bet.
    # "full": EV also decides whether to bet or raise at all.
    "DEEP_SIZING": "full",
    "FOLD_TRUST": 0.75,             # never assume more folds than this x their tracked fold rate
    "FITTED": True,                 # use measured responses from fitted_tables.py
    "BET_SIZES": (0.33, 0.5, 0.75, 1.0, 1.5),
    "REREAD_AFTER_RAISE": 0.6,      # after being raised, we continue with this equity or more
    "RAISE_SIZE_MULT": 3.0,         # assumed size of a raise over our bet
    # --- planning across streets -------------------------------------------
    "PLANNING": True,
    "PLAN_MIN_EQUITY": 0.6,         # strong hands may also use the size that gets stacks in by the river
    # share of equity we keep on later streets when calling, (out of position, in position)
    "REALIZE_FLOP": (0.82, 0.93), "REALIZE_TURN": (0.90, 0.97),
    "RAISE_SIZE": 0.8,              # raise by this fraction of the pot after calling
    "CHECK_REALIZE": 0.9,           # share of our equity kept after checking
    "CALL_P_REF_SIZE": 0.66,        # P(call) falls for bets bigger than this
}

# hand class -> share of hands that are at least as strong (0 = AA, 1 = worst)
_RANKED = sorted(_updated._PREFLOP_TABLE["wide"], key=_updated._PREFLOP_TABLE["wide"].get, reverse=True) \
    if _updated._PREFLOP_TABLE else []
_PCT, _seen = {}, 0
for _hc in _RANKED:
    _seen += 6 if len(_hc) == 2 else 4 if _hc[2] == "s" else 12
    _PCT[_hc] = _seen / 1326
COMBOS = [(a, b, hand_class(a, b)) for a in range(52) for b in range(a + 1, 52)]
BOARD_LEN = {"flop": 3, "turn": 4, "river": 5}


class DeepThoughtV1Bot(HuUpdatedOptimalBot):
    def __init__(self, cfg=default_config, overrides=None, seed=None):
        super().__init__(cfg, {**DEEP_CONFIG, **(overrides or {})}, seed)

    # ------------------------------------------------------------ range model
    def _preflop_width(self, s, seat):
        c = self.cfg
        level, raised_at, called, high = 0, 0, False, s.big_blind   # 1 = open, 2 = 3-bet, 3 = 4-bet
        for st, who, kind, amt in s.history:
            if st != "preflop":
                continue
            if kind in AGGRESSIVE and amt > high:
                high = amt
                level += 1
                if who == seat:
                    raised_at = level
            elif who == seat and kind in ("call", "allin"):
                called = True
        # A raiser's range is as wide as they actually raise: a 3-bet from someone who
        # 3-bets twice as often as the reference covers about twice as many hands.
        if raised_at >= 3:
            width = min(1.0, c.RANGE_4BET * self._freq_ratio(s, seat, "four_bet"))
        elif raised_at == 2:
            width = min(1.0, c.RANGE_3BET * self._freq_ratio(s, seat, "three_bet"))
        elif raised_at == 1:
            width = c.RANGE_RAISE
        elif called:
            width = c.RANGE_CALL
        else:
            width = c.RANGE_CHECK
        if self._players_left(s) > 2:
            width *= c.RANGE_MULTIWAY_MULT
        if self.stats.is_tight(seat, c):
            width *= 0.6
        elif self.stats.is_loose(seat, c):
            width = min(1.0, width * 1.3)
        return width

    @staticmethod
    def _street_pots(s):
        """Chips in the pot when each street began."""
        pots, total = {}, 0
        for street in ("preflop", "flop", "turn", "river"):
            pots[street] = total
            bets = {}
            for st, who, kind, amt in s.history:
                if st == street and kind != "check" and kind != "fold":
                    bets[who] = max(bets.get(who, 0), amt)
            total += sum(bets.values())
        return pots

    def _likelihood(self, bucket, aggressive, bets, kind, frac):
        c = self.cfg
        if not c.FITTED:
            table = (c.LIKELY_RAISE if bets > 1 else c.LIKELY_BET) if aggressive else \
                c.LIKELY_CHECK if kind == "check" else c.LIKELY_CALL
            return table[bucket]
        if kind == "check":
            return FIT_CHECK[bucket]
        if aggressive and bets == 1:
            return FIT_BET[bucket]
        return FIT_RESPONSE[bucket][self._size_bin(frac)][2 if aggressive else 1]

    @staticmethod
    def _size_bin(frac):
        return next((i for i, x in enumerate(SIZE_BINS) if frac <= x), len(SIZE_BINS))

    def _opponent_range(self, s, seat):
        """-> [a, b, weight, bucket on the current board] for every hand they could hold."""
        c = self.cfg
        dead = set(s.hole_cards) | set(s.board)
        width = self._preflop_width(s, seat)
        hands = []
        for a, b, hc in COMBOS:
            if a in dead or b in dead:
                continue
            pct = _PCT[hc]
            w = 1.0 if pct <= width else 1.0 - (pct - width) / 0.1
            if w > 0:
                hands.append([a, b, w, None])
        scale = self.sd.bluff_rate(seat, c, s.street) / c.BLUFF_PRIOR
        pot_before = self._street_pots(s)
        for street in ("flop", "turn", "river"):
            n = BOARD_LEN[street]
            if len(s.board) < n:
                break
            board = s.board[:n]
            for h in hands:
                h[3] = classify_hand([h[0], h[1]], board, self.good_kicker)
            high, bets, street_chips = 0, 0, {}
            pot = pot_before[street]
            for st, who, kind, amt in s.history:
                if st != street:
                    continue
                aggressive = kind in AGGRESSIVE and amt > high
                facing = high - street_chips.get(who, 0)
                frac = facing / max(1, pot + sum(street_chips.values()) - facing)
                if aggressive:
                    high = amt
                    bets += 1
                if kind in AGGRESSIVE or kind == "call":
                    street_chips[who] = max(street_chips.get(who, 0), amt)
                if who != seat or kind in ("fold", "blind"):
                    continue
                if aggressive:          # bets from frequent bettors contain more weak hands
                    stat = "bet" if bets == 1 else "raise"
                    read = min(c.FREQUENCY_MULT_MAX, max(1 / c.FREQUENCY_MULT_MAX, self._freq_ratio(s, seat, stat)))
                else:
                    read = 1.0
                for h in hands:
                    p = self._likelihood(h[3], aggressive, bets, kind, frac)
                    if aggressive and h[3] in ("WEAK", "DRAW"):
                        p *= min(3.5, max(0.3, scale * read))
                    h[2] *= p
        top = max((h[2] for h in hands), default=0)
        return [h for h in hands if h[2] > top * 0.002]

    def _hand_equities(self, s, hands):
        """Add our equity vs each hand as h[4]. Exact on the river, sampled before it."""
        c = self.cfg
        self.rng.shuffle(hands)
        stop = time.perf_counter() + c.DEEP_DEADLINE_MS / 1000
        board, hole = s.board, s.hole_cards
        done = []
        need = 5 - len(board)
        if need == 0:
            mine = hand_rank(hole + board)
            for k, h in enumerate(hands):
                if k % 64 == 0 and time.perf_counter() > stop:
                    break
                theirs = hand_rank([h[0], h[1]] + board)
                done.append(h + [1.0 if mine > theirs else 0.5 if mine == theirs else 0.0])
            return done
        deck = [x for x in range(52) if x not in hole and x not in board]
        per_hand = max(6, c.DEEP_EVALS // (2 * max(1, len(hands))))
        if need == 1:
            runouts = [[x] for x in self.rng.sample(deck, min(len(deck), per_hand))]
        else:
            runouts = [self.rng.sample(deck, 2) for _ in range(per_hand)]
        full = [board + r for r in runouts]
        mine = [hand_rank(hole + f) for f in full]
        for k, h in enumerate(hands):
            if k % 32 == 0 and time.perf_counter() > stop:
                break
            a, b = h[0], h[1]
            won, seen = 0.0, 0
            for r, f, m in zip(runouts, full, mine):
                if a in r or b in r:
                    continue
                theirs = hand_rank([a, b] + f)
                won += 1.0 if m > theirs else 0.5 if m == theirs else 0.0
                seen += 1
            done.append(h + [won / seen if seen else 0.5])
        return done

    # ------------------------------------------------------------ EV helpers
    def _response(self, seat, bucket, frac):
        """-> (fold, call, raise) chances for this hand, scaled by their tracked fold rate."""
        c = self.cfg
        fold, call, rais = FIT_RESPONSE[bucket][self._size_bin(frac)]
        read = self.sd.fold_rate(seat, c) / c.FOLD_PRIOR
        fold = min(0.95, fold * min(1.5, max(0.5, read)))
        rest = (1 - fold) / max(1e-9, call + rais)
        return fold, call * rest, rais * rest

    def _bet_ev(self, hands, seat, bet, pot, players):
        """EV of betting `bet` chips: they fold, call, or raise us."""
        c = self.cfg
        frac = bet / max(1, pot)
        total = ev = 0.0
        for h in hands:
            fold, call, rais = self._response(seat, h[3], frac)
            e = h[4]
            raised = bet * c.RAISE_SIZE_MULT
            if_raised = e * (pot + 2 * raised) - raised if e >= c.REREAD_AFTER_RAISE else -bet
            ev += h[2] * (fold * pot + call * (e * (pot + 2 * bet) - bet) + rais * if_raised)
            total += h[2]
        return ev / total if total else 0.0

    def _sizes(self, s, eq):
        """Pot fractions to try, including the size that gets stacks in by the river."""
        c = self.cfg
        sizes = list(c.BET_SIZES)
        streets_left = 4 - len(("flop", "turn", "river")[:["flop", "turn", "river"].index(s.street)]) - 1
        if c.PLANNING and eq >= c.PLAN_MIN_EQUITY and s.pot > 0:
            sizes.append((((s.pot + 2 * s.me.stack) / s.pot) ** (1 / streets_left) - 1) / 2)
        return sizes

    def _realize(self, s):
        """Share of equity we keep when calling: earlier streets leave more bets to face."""
        c = self.cfg
        if not c.PLANNING or s.street == "river":
            return 1.0
        acted_before = any(st == s.street and seat == s.my_seat for st, seat, _, _ in s.history)
        position = 1 if self.position(s) == "BTN" or not acted_before and self._players_left(s) > 2 else 0
        return (c.REALIZE_FLOP if s.street == "flop" else c.REALIZE_TURN)[position]

    # ------------------------------------------------------------ postflop
    def _postflop(self, s):
        c = self.cfg
        base = super()._postflop(s)
        opps = [p for p in s.players if p.active and p.seat != s.my_seat]
        if not opps or not _PCT:
            return base
        seat = self._aggressor(s) if s.to_call else None
        if seat is None:
            ctx = self._preflop_context(s)
            seat = ctx.last_raiser if ctx.last_raiser in [p.seat for p in opps] else opps[0].seat
        hands = self._hand_equities(s, self._opponent_range(s, seat))
        total = sum(h[2] for h in hands)
        if total <= 0:
            return base
        eq = sum(h[2] * h[4] for h in hands) / total
        eq *= c.MULTIWAY_EQ_MULT ** max(0, len(opps) - 1)
        bucket = classify_hand(s.hole_cards, s.board, self.good_kicker)
        self.last_ev = {"equity": eq, "bucket": bucket, "hands": len(hands)}
        stack = s.me.stack

        if s.to_call == 0:
            if c.DEEP_SIZING == "off" or stack <= 0 or len(opps) > 1:
                return base
            if c.DEEP_SIZING == "size_only" and base.kind == "check":
                return base
            best, best_ev = None, eq * s.pot * c.CHECK_REALIZE if c.DEEP_SIZING == "full" else float("-inf")
            for frac in self._sizes(s, eq):
                bet = min(stack, int(frac * s.pot))
                ev = self._bet_ev(hands, seat, bet, s.pot, len(opps))
                if ev > best_ev:
                    best, best_ev = bet, ev
            if best is None:
                return Action("check")
            return Action("allin") if best >= stack else Action("bet", s.me.bet + best)

        be, _ = self._breakeven(s, seat)
        self.last_ev["breakeven"] = be
        matched = min(s.to_call, stack)
        pot_after = s.pot + matched
        call_ev = (eq * self._realize(s) - be) * pot_after
        if c.DEEP_SIZING == "full" and len(opps) == 1:
            high = s.me.bet + s.to_call
            target = min(s.max_raise_to, max(s.min_raise_to, int(high + c.RAISE_SIZE * pot_after)))
            if target > high and s.max_raise_to > high:
                extra = target - s.me.bet
                raise_ev = self._bet_ev(hands, seat, extra - matched, pot_after, len(opps)) - matched
                if raise_ev > max(0.0, call_ev):
                    return Action("allin") if target >= s.max_raise_to else Action("raise", target)
            return Action("call") if call_ev >= 0 else Action("fold")
        # calls only: keep the inherited raise if calling is +EV anyway
        if base.kind in ("raise", "bet", "allin") and s.me.stack > s.to_call and call_ev >= 0:
            return base
        return Action("call") if call_ev >= 0 else Action("fold")
