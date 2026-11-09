"""Real-time CFR+ solving of a heads-up hand from the current decision to the river.

    from gto_solver import solve
    probs = solve(ours, theirs, mine, spot, rng, seconds=1.0)   # {action: probability} for our hand

Both ranges are {(card, card): weight}. `spot` describes the decision we face: the board,
the street, who acts first after the flop, the chips in the middle, each player's bet on
this street and stack behind, and how many raises this street has seen. The rest of the
hand is played in a small abstraction:

  * the current street: check / bet 1/2 or 1x pot, raise to pot (one raise level), all-in
    when a bet would be most of a stack;
  * later streets: check / bet 2/3 pot, call or fold, all-in as above;
  * future cards: TURNS sampled turn cards, each with RIVERS sampled river cards, the same
    runouts in every branch (so each board's showdown table is built once).

Every showdown is exact on its board: each player's hands are ranked against the other's
reach-weighted range with prefix sums, so a solve costs O(hands) per tree node. Card
removal between the two ranges is ignored (hands that touch a future board card drop out
of that branch). The tree is constant-sum (each player's share of the final pot minus what
they put in after the root), so CFR+ with linear averaging converges to an equilibrium of
the abstraction. Pure Python: no numpy, which the Chipzen sandbox warns against.
"""
import bisect
import time

from poker_core import hand_rank

FOLD, CHECK, CALL = "fold", "check", "call"
CONFIG = {
    "ROOT_BETS": (0.5, 1.0),        # pot shares for a bet on the current street
    "LATER_BETS": (0.66,),          # ... on later streets
    "RAISE_POT": 1.0,               # a raise adds this share of the pot after calling
    "JAM_SHARE": 0.7,               # a bet or raise this share of the stack is an all-in
    "ROOT_RAISES": {"flop": 1, "turn": 2, "river": 2},   # raises the root street can reach (a bet is none)
    "TURNS": 2, "RIVERS": 2,        # sampled cards from the flop: 2 turns x 2 rivers, shared by every branch
    "RIVERS_FROM_TURN": 4,
    "MAX_HANDS": {"flop": 50, "turn": 70, "river": 120},  # each range is sampled down to this many hands
    "MIN_ITERATIONS": 40,
    # Chance sampling: each iteration follows one random card at every chance node instead
    # of all of them (unbiased, so it converges to the same answer), which makes iterations
    # far cheaper and lets the tree hold many more runouts.
    "CHANCE_SAMPLING": False,       # tested worse: 50-60% of the pot exploitable on the flop in 1.2 s
    "NUMPY": True,                  # use NumpySolver when numpy imports (card removal, ~1.5x quality)
}


class _Node:
    __slots__ = ("kind", "player", "actions", "children", "regret", "avg", "pot", "added", "board", "winner", "masks")


class Solver:
    def __init__(self, hands, weights, spot, cfg, rng):
        self.hands, self.cfg, self.rng = hands, cfg, rng      # hands[p] = list of (a, b)
        self.w = weights                                     # weights[p] = list of floats
        self.spot = spot
        self.show = {}                                       # board -> showdown tables
        deck = [c for c in range(52) if c not in set(spot["board"])]
        self.runouts = {}                                    # board so far -> next cards to deal
        if spot["street"] == "flop":
            turns = rng.sample(deck, cfg["TURNS"])
            self.runouts[tuple(spot["board"])] = turns
            for card in turns:
                rest = [c for c in deck if c != card]
                self.runouts[tuple(spot["board"]) + (card,)] = rng.sample(rest, cfg["RIVERS"])
        elif spot["street"] == "turn":
            self.runouts[tuple(spot["board"])] = rng.sample(deck, min(cfg["RIVERS_FROM_TURN"], len(deck)))
        self.root = self._street(spot["street"], spot["board"], spot["first"], spot["to_act"],
                                 spot["pot"], list(spot["bets"]), list(spot["stacks"]), [0, 0],
                                 spot["raises"], True)

    # ------------------------------------------------------------ tree
    def _deal(self, board, street, pot, stacks, added, first):
        """End of a street: deal the next card(s) or show down."""
        allin = min(stacks) <= 0
        if street == "river":
            return self._showdown(board, pot, added)
        node = _Node()
        node.kind, node.children, node.masks = "chance", [], []
        nxt = "turn" if street == "flop" else "river"
        for card in self.runouts[tuple(board)]:
            b = board + [card]
            node.masks.append([[card not in h for h in self.hands[q]] for q in (0, 1)])
            if allin:
                node.children.append(self._deal(b, nxt, pot, stacks, added, first))
            else:
                node.children.append(self._street(nxt, b, first, first, pot, [0, 0], list(stacks), list(added), 0, False))
        return node

    def _street(self, street, board, first, to_act, pot, bets, stacks, added, raises, root, checked=False):
        c = self.cfg
        node = _Node()
        node.kind, node.player, node.pot, node.board = "act", to_act, pot, board
        node.actions, node.children = [], []
        me, other = to_act, 1 - to_act
        owed = bets[other] - bets[me]
        def add(name, chips, closes):
            nb, ns, na = list(bets), list(stacks), list(added)
            nb[me] += chips
            ns[me] -= chips
            na[me] += chips
            if closes:
                child = self._deal(board, street, pot + chips, ns, na, first)
            else:
                child = self._street(street, board, first, other, pot + chips, nb, ns, na,
                                     raises + (1 if name.startswith("r") else 0), root, name == CHECK)
            node.actions.append(name)
            node.children.append(child)
        if owed > 0:
            leaf = _Node()
            leaf.kind, leaf.winner, leaf.pot, leaf.added = "fold", other, pot, list(added)
            node.actions.append(FOLD)
            node.children.append(leaf)
            call = min(owed, stacks[me])
            add(CALL, call, True)
            max_raises = c["ROOT_RAISES"][street] if root else 0
            if raises < max_raises and stacks[me] > owed and stacks[other] > 0:
                size = owed + int(c["RAISE_POT"] * (pot + owed))
                if size >= c["JAM_SHARE"] * stacks[me] or size >= stacks[me]:
                    add("rall", stacks[me], False)
                else:
                    add(f"r{size}", size, False)
        else:
            if checked:
                add(CHECK, 0, True)                 # check behind ends the street
            else:
                add(CHECK, 0, False)
            if stacks[me] > 0 and stacks[other] > 0:
                seen = set()
                for share in (c["ROOT_BETS"] if root else c["LATER_BETS"]):
                    size = max(1, int(share * pot))
                    if size >= c["JAM_SHARE"] * stacks[me] or size >= stacks[me]:
                        size = stacks[me]
                    if size not in seen:
                        seen.add(size)
                        add(f"b{size}" if size < stacks[me] else "ball", size, False)
        n = [len(self.hands[me])] * len(node.actions)
        node.regret = [[0.0] * k for k in n]
        node.avg = [[0.0] * k for k in n]
        return node

    def _showdown(self, board, pot, added):
        node = _Node()
        node.kind, node.board, node.pot, node.added = "show", board, pot, list(added)
        key = tuple(sorted(board))
        if key not in self.show:
            self.show[key] = self._tables(board)
        return node

    def _tables(self, board):
        """Per player: their hands' rank positions in the other's sorted range, and masks."""
        dead = set(board)
        out = []
        ranks = []
        for p in (0, 1):
            ranks.append([None if a in dead or b in dead else hand_rank([a, b] + board) for a, b in self.hands[p]])
        for p in (0, 1):
            o = 1 - p
            order = sorted((r, j) for j, r in enumerate(ranks[o]) if r is not None)
            sorted_ranks = [r for r, _ in order]
            idx = [j for _, j in order]
            pos = []
            for r in ranks[p]:
                if r is None:
                    pos.append(None)
                else:
                    pos.append((bisect.bisect_left(sorted_ranks, r), bisect.bisect_right(sorted_ranks, r)))
            out.append((idx, pos))
        return out

    # ------------------------------------------------------------ CFR+
    @staticmethod
    def _strategy(node):
        k = len(node.actions)
        regret = node.regret            # CFR+ keeps every regret >= 0
        totals = [sum(col) for col in zip(*regret)]
        even = 1.0 / k
        return [[r / tot if tot > 0 else even for r, tot in zip(row, totals)] for row in regret]

    def _value(self, node, reach, p, t):
        """Counterfactual values for player p's hands at this node, given both reaches; updates p."""
        kind = node.kind
        o = 1 - p
        if kind == "fold":
            ro = sum(reach[o])
            gain = (node.pot - node.added[p]) if node.winner == p else -node.added[p]
            return [gain * ro] * len(reach[p])
        if kind == "show":
            idx, pos = self.show[tuple(sorted(node.board))][p]
            cum = [0.0]
            for j in idx:
                cum.append(cum[-1] + reach[o][j])
            whole = cum[-1]
            out = []
            pot, put = node.pot, node.added[p]
            for x in pos:
                if x is None:
                    out.append(0.0)
                else:
                    lo, hi = x
                    share = cum[lo] + 0.5 * (cum[hi] - cum[lo])
                    out.append(pot * share - put * whole)
            return out
        if kind == "chance":
            k = len(node.children)
            pairs = list(zip(node.children, node.masks))
            if self.cfg["CHANCE_SAMPLING"] and t > 0:
                pairs, k = [pairs[self.rng.randrange(k)]], 1
            total = [0.0] * len(reach[p])
            for child, masks in pairs:
                child_reach = [[r if m else 0.0 for r, m in zip(reach[q], masks[q])] for q in (0, 1)]
                v = self._value(child, child_reach, p, t)
                mp = masks[p]
                for i in range(len(total)):
                    if mp[i]:
                        total[i] += v[i]
            return [x / k for x in total]
        # action node
        sigma = self._strategy(node)
        k = len(node.actions)
        if node.player == p:
            rp = reach[p]
            vals = []
            for a in range(k):
                child_reach = [None, None]
                child_reach[p] = [r * s for r, s in zip(rp, sigma[a])]
                child_reach[o] = reach[o]
                vals.append(self._value(node.children[a], child_reach, p, t))
            node_v = [sum(s[i] * v[i] for s, v in zip(sigma, vals)) for i in range(len(rp))]
            for a in range(k):
                va = vals[a]
                node.regret[a] = [x if x > 0 else 0.0 for x in (r + v - nv for r, v, nv in zip(node.regret[a], va, node_v))]
                node.avg[a] = [av + t * r * s for av, r, s in zip(node.avg[a], rp, sigma[a])]
            return node_v
        total = None
        for a in range(k):
            child_reach = [None, None]
            child_reach[o] = [r * s for r, s in zip(reach[o], sigma[a])]
            child_reach[p] = reach[p]
            v = self._value(node.children[a], child_reach, p, t)
            total = v if total is None else [x + y for x, y in zip(total, v)]
        return total

    # ------------------------------------------------------------ exploitability
    def _avg(self, node):
        k = len(node.actions)
        totals = [sum(col) for col in zip(*node.avg)]
        return [[x / tot if tot > 0 else 1.0 / k for x, tot in zip(row, totals)] for row in node.avg]

    def _best(self, node, reach, p, best=True):
        """Values of player p's best response (or, best=False, of p's own average strategy)
        against the other's average strategy."""
        kind, o = node.kind, 1 - p
        if kind in ("fold", "show"):
            return self._value(node, reach, p, 0)
        if kind == "chance":
            k = len(node.children)
            total = [0.0] * len(reach[p])
            for child, masks in zip(node.children, node.masks):
                child_reach = [[r if m else 0.0 for r, m in zip(reach[q], masks[q])] for q in (0, 1)]
                v = self._best(child, child_reach, p, best)
                for i, m in enumerate(masks[p]):
                    if m:
                        total[i] += v[i]
            return [x / k for x in total]
        if node.player == p:
            if best:
                vals = [self._best(child, reach, p, best) for child in node.children]
                return [max(col) for col in zip(*vals)]
            sigma = self._avg(node)
            vals = []
            for a, child in enumerate(node.children):
                child_reach = [None, None]
                child_reach[p] = [r * s for r, s in zip(reach[p], sigma[a])]
                child_reach[o] = reach[o]
                vals.append(self._best(child, child_reach, p, best))
            return [sum(s[i] * v[i] for s, v in zip(sigma, vals)) for i in range(len(reach[p]))]
        sigma = self._avg(node)
        total = None
        for a, child in enumerate(node.children):
            child_reach = [None, None]
            child_reach[o] = [r * s for r, s in zip(reach[o], sigma[a])]
            child_reach[p] = reach[p]
            v = self._best(child, child_reach, p, best)
            total = v if total is None else [x + y for x, y in zip(total, v)]
        return total

    def exploitability(self):
        """What best responses gain over the average strategies' own values, summed over
        both players, as a share of the pot (0 at an exact equilibrium of this abstraction)."""
        reach = [list(self.w[0]), list(self.w[1])]
        gain = 0.0
        for p in (0, 1):
            br = self._best(self.root, reach, p, True)
            own = self._best(self.root, reach, p, False)
            gain += sum(w * (b - v) for w, b, v in zip(self.w[p], br, own))
        return gain / self.spot["pot"]

    def run(self, deadline):
        t = 0
        while t < self.cfg["MIN_ITERATIONS"] or time.perf_counter() < deadline:
            t += 1
            for p in (0, 1):
                self._value(self.root, [list(self.w[0]), list(self.w[1])], p, t)
            if t >= 4000:
                break
        return t

    def root_strategy(self, i):
        """Average strategy of hand i (player at the root) over the root's actions."""
        node = self.root
        total = sum(node.avg[a][i] for a in range(len(node.actions)))
        if total <= 0:
            return {a: 1 / len(node.actions) for a in node.actions}
        return {node.actions[a]: node.avg[a][i] / total for a in range(len(node.actions))}


def _sample(range_, n, keep, rng):
    """At most n hands, drawn by weight (each then weighted equally), always keeping `keep`."""
    items = [(h, w) for h, w in range_.items() if w > 0]
    if len(items) <= n:
        hands = [h for h, _ in items]
        weights = [w for _, w in items]
    else:
        # weighted sampling without replacement (Efraimidis-Spirakis keys)
        keyed = sorted(items, key=lambda hw: rng.random() ** (1.0 / hw[1]), reverse=True)[:n]
        total = sum(w for _, w in items)
        hands = [h for h, _ in keyed]
        weights = [total / n] * len(hands)
    if keep is not None and keep not in hands:
        hands.append(keep)
        weights.append(min(weights) if weights else 1.0)
    s = sum(weights)
    return hands, [w / s for w in weights]


def solve(ours, theirs, mine, spot, rng, seconds=1.0, cfg=None):
    """-> ({action: probability} for our hand at the root, iterations run, the solver)."""
    cfg = {**CONFIG, **(cfg or {})}
    n = cfg["MAX_HANDS"][spot["street"]]
    h0, w0 = _sample(ours, n, mine, rng)
    h1, w1 = _sample(theirs, n, None, rng)
    if not h1:
        return None, 0, None
    cls = NumpySolver if cfg["NUMPY"] and _np is not None else Solver
    solver = cls([h0, h1], [w0, w1], spot, cfg, rng)
    iterations = solver.run(time.perf_counter() + seconds)
    return solver.root_strategy(h0.index(mine)), iterations, solver


def chips_of(name):
    """Chips an abstract bet or raise adds ('b150' -> 150), None for the rest."""
    return int(name[1:]) if name[:1] in ("b", "r") and name[1:].isdigit() else None


# =====================================================================
# numpy version: the same tree and CFR+, vectorized over hands, with card removal
# =====================================================================
def _load_numpy():
    """numpy if it imports, else None (the pure Python Solver above still works). A function,
    not a module-level try, because build_chipzen.py turns top-level try blocks into np = None."""
    try:
        import numpy
        return numpy
    except ImportError:
        return None


_np = _load_numpy()


class NumpySolver(Solver):
    """Solver with every per-hand loop in numpy, and card removal between the two ranges:
    a hand only meets opposing hands that share no card with it, at showdowns and folds."""

    def __init__(self, hands, weights, spot, cfg, rng):
        super().__init__(hands, weights, spot, cfg, rng)
        np = _np
        self.w = [np.array(w, dtype=float) for w in weights]
        # cards[p]: (n, 52) 0/1 incidence; same[p]: for each hand, the index of the identical
        # opposing hand (or -1), which subtracting both its cards would remove twice
        self.cards = []
        for p in (0, 1):
            m = np.zeros((len(hands[p]), 52))
            for i, (a, b) in enumerate(hands[p]):
                m[i, a] = m[i, b] = 1.0
            self.cards.append(m)
        self.card_idx = [np.array(hands[p]) for p in (0, 1)]
        self.same = []
        for p in (0, 1):
            where = {h: j for j, h in enumerate(hands[1 - p])}
            self.same.append(np.array([where.get(h, -1) for h in hands[p]]))
        self._convert(self.root)
        for key in list(self.show):
            self.show[key] = self._np_tables(list(key))

    def _convert(self, node):
        np = _np
        if node.kind == "act":
            node.regret = np.zeros((len(node.actions), len(self.hands[node.player])))
            node.avg = np.zeros_like(node.regret)
        if node.kind == "chance":
            node.masks = [[np.array(m, dtype=float) for m in masks] for masks in node.masks]
        for child in getattr(node, "children", None) or []:
            self._convert(child)

    def _np_tables(self, board):
        np = _np
        dead = set(board)
        ranks = [[None if a in dead or b in dead else hand_rank([a, b] + board) for a, b in self.hands[p]] for p in (0, 1)]
        out = []
        for p in (0, 1):
            o = 1 - p
            order = sorted((r, j) for j, r in enumerate(ranks[o]) if r is not None)
            sorted_ranks = [r for r, _ in order]
            idx = np.array([j for _, j in order], dtype=int)
            lo, hi, live = [], [], []
            for r in ranks[p]:
                if r is None:
                    lo.append(0); hi.append(0); live.append(0.0)
                else:
                    lo.append(bisect.bisect_left(sorted_ranks, r)); hi.append(bisect.bisect_right(sorted_ranks, r)); live.append(1.0)
            out.append((idx, np.array(lo), np.array(hi), np.array(live)))
        return out

    def _blocked(self, p, reach_o):
        """Per hand of p: the opposing reach that shares no card with it."""
        o = 1 - p
        per_card = reach_o @ self.cards[o]                    # (52,) reach holding each card
        a, b = self.card_idx[p][:, 0], self.card_idx[p][:, 1]
        same = self.same[p]
        back = _np.where(same >= 0, reach_o[_np.maximum(same, 0)], 0.0)
        return reach_o.sum() - per_card[a] - per_card[b] + back

    @staticmethod
    def _strategy(node):
        np = _np
        totals = node.regret.sum(axis=0)
        k = node.regret.shape[0]
        return np.where(totals > 0, node.regret / np.where(totals > 0, totals, 1.0), 1.0 / k)

    def _value(self, node, reach, p, t):
        np = _np
        kind, o = node.kind, 1 - p
        if kind == "fold":
            gain = (node.pot - node.added[p]) if node.winner == p else -node.added[p]
            return gain * self._blocked(p, reach[o])
        if kind == "show":
            idx, lo, hi, live = self.show[tuple(sorted(node.board))][p]
            r = reach[o][idx]
            cum = np.concatenate(([0.0], np.cumsum(r)))
            # the same prefix sums per card, to drop opposing hands that share our cards
            per = np.vstack((np.zeros(52), np.cumsum(r[:, None] * self.cards[o][idx], axis=0)))   # (m+1, 52)
            a, b = self.card_idx[p][:, 0], self.card_idx[p][:, 1]
            def below(pos):
                return cum[pos] - per[pos, a] - per[pos, b]
            # an identical opposing hand ties and holds both our cards: added back once
            same = self.same[p]
            sw = np.where(same >= 0, reach[o][np.maximum(same, 0)], 0.0)
            lo_v, hi_v = below(lo), below(hi) + sw
            whole = self._blocked(p, reach[o])
            share = lo_v + 0.5 * (hi_v - lo_v)
            return live * (node.pot * share - node.added[p] * whole)
        if kind == "chance":
            pairs = list(zip(node.children, node.masks))
            if self.cfg["CHANCE_SAMPLING"] and t > 0:
                pairs = [pairs[self.rng.randrange(len(pairs))]]
            total = np.zeros(len(reach[p]))
            for child, masks in pairs:
                v = self._value(child, [reach[0] * masks[0], reach[1] * masks[1]], p, t)
                total += v * masks[p]
            return total / len(pairs)
        sigma = self._strategy(node)
        if node.player == p:
            vals = np.array([self._value(child, [reach[0] * sigma[a], reach[1]] if p == 0 else [reach[0], reach[1] * sigma[a]], p, t)
                             for a, child in enumerate(node.children)])
            node_v = (sigma * vals).sum(axis=0)
            node.regret = np.maximum(node.regret + vals - node_v, 0.0)
            node.avg += t * reach[p] * sigma
            return node_v
        total = 0.0
        for a, child in enumerate(node.children):
            child_reach = [reach[0] * sigma[a], reach[1]] if o == 0 else [reach[0], reach[1] * sigma[a]]
            total = total + self._value(child, child_reach, p, t)
        return total

    def _avg(self, node):
        np = _np
        totals = node.avg.sum(axis=0)
        k = node.avg.shape[0]
        return np.where(totals > 0, node.avg / np.where(totals > 0, totals, 1.0), 1.0 / k)

    def _best(self, node, reach, p, best=True):
        np = _np
        kind, o = node.kind, 1 - p
        if kind in ("fold", "show"):
            return self._value(node, reach, p, 0)
        if kind == "chance":
            total = np.zeros(len(reach[p]))
            for child, masks in zip(node.children, node.masks):
                total += self._best(child, [reach[0] * masks[0], reach[1] * masks[1]], p, best) * masks[p]
            return total / len(node.children)
        sigma = self._avg(node)
        if node.player == p:
            vals = np.array([self._best(child, reach, p, best) if best else
                             self._best(child, [reach[0] * sigma[a], reach[1]] if p == 0 else [reach[0], reach[1] * sigma[a]], p, best)
                             for a, child in enumerate(node.children)])
            return vals.max(axis=0) if best else (sigma * vals).sum(axis=0)
        total = 0.0
        for a, child in enumerate(node.children):
            child_reach = [reach[0] * sigma[a], reach[1]] if o == 0 else [reach[0], reach[1] * sigma[a]]
            total = total + self._best(child, child_reach, p, best)
        return total

    def exploitability(self):
        reach = [self.w[0].copy(), self.w[1].copy()]
        gain = 0.0
        for p in (0, 1):
            gain += float((self.w[p] * (self._best(self.root, reach, p, True) - self._best(self.root, reach, p, False))).sum())
        return gain / self.spot["pot"]

    def run(self, deadline):
        t = 0
        while t < self.cfg["MIN_ITERATIONS"] or time.perf_counter() < deadline:
            t += 1
            for p in (0, 1):
                self._value(self.root, [self.w[0].copy(), self.w[1].copy()], p, t)
            if t >= 20000:
                break
        return t

    def root_strategy(self, i):
        node = self.root
        col = node.avg[:, i]
        total = col.sum()
        if total <= 0:
            return {a: 1 / len(node.actions) for a in node.actions}
        return {a: float(col[k] / total) for k, a in enumerate(node.actions)}
