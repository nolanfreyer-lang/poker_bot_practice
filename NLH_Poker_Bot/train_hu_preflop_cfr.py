"""Solve heads-up preflop at short stacks with CFR+, one solution per stack depth and layout.

    python3 train_hu_preflop_cfr.py

Game (big blinds, S = effective stack):
    small blind:  fold | limp | raise to 2 | all-in
    vs a limp:    big blind checks | raises to 4 | all-in; then the small blind folds, calls or shoves
    vs a raise:   big blind folds | calls | 3-bets to 6 (S >= 14) | all-in; then the small blind
                  folds, calls or shoves, and the big blind calls or folds that shove
Pots that reach the flop are scored with a realization model instead of being played out:
the player with position after the flop keeps eq + STRENGTH * (eq - 0.5) + edge * eq * (1 - eq)
of the pot. STRENGTH gives the better hand more than its equity (the worse one folds
along the way); edge is REALIZE (0.5 equity in position keeps 55%), scaled down as the
stack-to-pot ratio falls below SPR_FULL, since short stacks get it in and realize their
raw equity. The two were set so the small blind folds 11-15% at 15-25 BB, near published
heads-up solutions; the paired tests, not the model, decide whether v8 uses the tables.

Two layouts:
    "hu"     the small blind is the button and acts last after the flop (2-seat matches)
    "short"  the big blind acts last after the flop: a Chipzen 6-max table down to two
Every one of the 169 starting hands gets its own strategy, weighted by combos and card
removal, like train_pushfold_cfr.py. Writes hu_preflop_tables.py, which ships compact.

Deep_Thought plays better after the flop than the bots it meets, so a pot that sees a flop
is worth more to it than the model's even split says. HERO_EDGE gives our bot that much
more of every such pot (scaled by stack-to-pot ratio like position), and each spot is
solved with our bot in that seat: small blind spots from a solve where we are the small
blind, big blind spots from one where we are the big blind. Without it the solved
preflop folded and shoved too much and lost to v7's ranges in paired tests. Paired
against v7 (40 x 200 hands per opponent, 14 opponents), edge 0.10 won by +9.3, +9.4 and
+16.9 bb/100 at 15, 20 and 25 BB in 2-seat matches (0.05 and 0.15 won less), while every
edge lost or broke even in the "short" layout.
Needs numpy here; the bots only read the finished table.
"""
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from train_pushfold_cfr import CLASSES, combos, equity_matrix

HERE = Path(__file__).resolve().parent
EQUITY_CACHE = HERE / "hu_equity_cache.npz"
OUT = HERE / "hu_preflop_tables.py"
DEPTHS = list(range(6, 26))
LAYOUTS = ("hu", "short")
ITERATIONS = 4000
REALIZE = 0.2               # position's edge in pots that see a flop, at full SPR
SPR_FULL = 4.0              # below this stack-to-pot ratio the edge shrinks linearly
STRENGTH = 0.4              # the better hand keeps more than its equity: the worse one folds
OPEN, ISO, THREEBET = 2.0, 4.0, 6.0
THREEBET_MIN_BB = 14
HERO_EDGE = 0.1             # extra share of flop pots for our bot (see above)
LEVELS = 20                 # strategies are stored in steps of 1/LEVELS


class Node:
    def __init__(self, name, player, children):
        self.name, self.player, self.children = name, player, children   # children: [(action, node)]


class Leaf:
    def __init__(self, kind, sb_in, bb_in, folder=None):
        self.kind, self.sb_in, self.bb_in, self.folder = kind, sb_in, bb_in, folder


def build_tree(S):
    """The betting tree for stack S. Nodes are named for the spot the bot looks up."""
    fold = lambda who, sb, bb: Leaf("fold", sb, bb, who)
    call_shove = lambda name, who, sb, bb: Node(name, who, [
        ("fold", fold(who, sb, bb)), ("call", Leaf("allin", S, S))])
    vs_iso = Node("sb_vs_iso", "sb", [
        ("fold", fold("sb", 1, ISO)), ("call", Leaf("flop", ISO, ISO)),
        ("allin", call_shove("bb_vs_shove_iso", "bb", S, ISO))])
    vs_limp = Node("bb_vs_limp", "bb", [
        ("check", Leaf("flop", 1, 1)), ("raise", vs_iso),
        ("allin", call_shove("sb_vs_shove_limp", "sb", 1, S))])
    vs_open = [("fold", fold("bb", OPEN, 1)), ("call", Leaf("flop", OPEN, OPEN))]
    if S >= THREEBET_MIN_BB:
        vs_open.append(("raise", Node("sb_vs_3bet", "sb", [
            ("fold", fold("sb", OPEN, THREEBET)), ("call", Leaf("flop", THREEBET, THREEBET)),
            ("allin", call_shove("bb_vs_shove_3bet", "bb", S, THREEBET))])))
    vs_open.append(("allin", call_shove("sb_vs_shove_open", "sb", OPEN, S)))
    return Node("sb_open", "sb", [
        ("fold", fold("sb", 0.5, 1)), ("call", vs_limp),
        ("raise", Node("bb_vs_open", "bb", vs_open)),
        ("allin", call_shove("bb_vs_shove", "bb", S, 1))])


def nodes(node):
    if isinstance(node, Node):
        yield node
        for _, child in node.children:
            yield from nodes(child)


class Solver:
    def __init__(self, eq, w, S, layout, hero="sb", hero_edge=0.0):
        self.eq, self.w, self.S, self.layout = eq, w, S, layout
        self.hero, self.hero_edge = hero, hero_edge
        self.root = build_tree(S)
        self.regret = {n.name: np.zeros((len(CLASSES), len(n.children))) for n in nodes(self.root)}
        self.avg = {n.name: np.zeros((len(CLASSES), len(n.children))) for n in nodes(self.root)}
        self.leaf_u = {}

    def utility(self, leaf):
        """The small blind's result (big blinds) for every pair of hands, times their combos."""
        if id(leaf) not in self.leaf_u:
            if leaf.kind == "fold":
                u = np.full_like(self.eq, -leaf.sb_in if leaf.folder == "sb" else leaf.bb_in)
            elif leaf.kind == "allin":
                u = (2 * self.eq - 1) * self.S
            else:
                pot = leaf.sb_in + leaf.bb_in
                depth = min(1.0, (self.S - leaf.sb_in) / pot / SPR_FULL)
                edge = REALIZE * depth
                ip = self.eq if self.layout == "hu" else 1 - self.eq
                ip_share = np.clip(ip + STRENGTH * (ip - 0.5) + edge * ip * (1 - ip), 0, 1)
                share = ip_share if self.layout == "hu" else 1 - ip_share
                skill = self.hero_edge * depth
                share = np.clip(share + (skill if self.hero == "sb" else -skill), 0, 1)
                u = share * pot - leaf.sb_in
            self.leaf_u[id(leaf)] = self.w * u
        return self.leaf_u[id(leaf)]

    @staticmethod
    def current(regret):
        total = regret.sum(1, keepdims=True)
        return np.where(total > 0, regret / np.maximum(total, 1e-12), 1.0 / regret.shape[1])

    def cfr(self, node, r_sb, r_bb, t):
        """Counterfactual values for every SB hand and every BB hand below this node."""
        if isinstance(node, Leaf):
            wu = self.utility(node)
            return wu @ r_bb, -(wu.T @ r_sb)
        sigma = self.current(self.regret[node.name])
        mine = r_sb if node.player == "sb" else r_bb
        values, v_sb, v_bb = [], 0, 0
        for a, (_, child) in enumerate(node.children):
            if node.player == "sb":
                c_sb, c_bb = self.cfr(child, r_sb * sigma[:, a], r_bb, t)
                values.append(c_sb)
                v_sb, v_bb = v_sb + sigma[:, a] * c_sb, v_bb + c_bb
            else:
                c_sb, c_bb = self.cfr(child, r_sb, r_bb * sigma[:, a], t)
                values.append(c_bb)
                v_sb, v_bb = v_sb + c_sb, v_bb + sigma[:, a] * c_bb
        node_v = v_sb if node.player == "sb" else v_bb
        self.regret[node.name] = np.maximum(self.regret[node.name] + np.stack(values, 1) - node_v[:, None], 0)
        self.avg[node.name] += t * mine[:, None] * sigma
        return v_sb, v_bb

    def strategy(self):
        out = {}
        for name, a in self.avg.items():
            total = a.sum(1, keepdims=True)
            out[name] = np.where(total > 0, a / np.maximum(total, 1e-12), 1.0 / a.shape[1])
        return out

    def best_response(self, node, r_sb, r_bb, player, strat):
        """Values for `player` best-responding to the other's average strategy."""
        if isinstance(node, Leaf):
            wu = self.utility(node)
            return wu @ r_bb if player == "sb" else -(wu.T @ r_sb)
        sigma = strat[node.name]
        kids = []
        for a, (_, child) in enumerate(node.children):
            if node.player == player:
                kids.append(self.best_response(child, r_sb, r_bb, player, strat))
            elif node.player == "sb":
                kids.append(self.best_response(child, r_sb * sigma[:, a], r_bb, player, strat))
            else:
                kids.append(self.best_response(child, r_sb, r_bb * sigma[:, a], player, strat))
        return np.max(kids, 0) if node.player == player else np.sum(kids, 0)

    def solve(self):
        ones = np.ones(len(CLASSES))
        for t in range(1, ITERATIONS + 1):
            self.cfr(self.root, ones, ones, t)
        strat = self.strategy()
        total = self.w.sum()
        br_sb = self.best_response(self.root, ones, ones, "sb", strat).sum() / total
        br_bb = self.best_response(self.root, ones, ones, "bb", strat).sum() / total
        return strat, (br_sb + br_bb) / 2, br_sb


def solve_one(job):
    eq, w, S, layout, hero, hero_edge = job
    strat, gap, sb_value = Solver(eq, w, S, layout, hero, hero_edge).solve()
    return S, layout, hero, strat, gap, sb_value


def encode(probs):
    """One character per action per hand: 'a' is never, chr(ord('a') + LEVELS) is always."""
    q = np.rint(probs * LEVELS).astype(int)
    for row in q:                       # keep each hand's steps summing to LEVELS
        row[np.argmax(row)] += LEVELS - row.sum()
    return "".join(chr(ord("a") + x) for x in q.T.ravel())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", type=float, default=HERO_EDGE, help="our bot's extra share of flop pots")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--name", default="HU_PREFLOP", help="the table's variable name (bundles merge every module)")
    args = ap.parse_args()
    out = Path(args.out)
    if EQUITY_CACHE.exists():
        data = np.load(EQUITY_CACHE)
        eq, w = data["eq"], data["w"]
    else:
        eq, w = equity_matrix()
        np.savez(EQUITY_CACHE, eq=eq, w=w)
    combos_of = np.array([len(combos(h)) for h in CLASSES])
    heroes = ("sb", "bb") if args.edge else ("sb",)
    jobs = [(eq, w, S, layout, hero, args.edge) for layout in LAYOUTS for S in DEPTHS for hero in heroes]
    table = {layout: {S: {} for S in DEPTHS} for layout in LAYOUTS}
    with Pool() as pool:
        for S, layout, hero, strat, gap, sb_value in sorted(pool.imap_unordered(solve_one, jobs),
                                                            key=lambda r: (r[1], r[0], r[2])):
            if hero == "sb":
                root = strat["sb_open"]
                mix = " ".join(f"{a} {100 * (root[:, i] * combos_of).sum() / 1326:4.1f}%"
                               for i, (a, _) in enumerate(build_tree(S).children))
                print(f"{layout:5s} {S:>2} BB  SB {sb_value:+.3f} BB/hand  exploitability {gap:.4f}  SB root: {mix}")
            table[layout][S].update({n.name: ([a for a, _ in n.children], encode(strat[n.name]))
                                     for n in nodes(build_tree(S)) if len(heroes) == 1 or n.player == hero})
    with open(out, "w") as f:
        f.write('"""GENERATED by train_hu_preflop_cfr.py. Heads-up preflop strategies, 6-25 BB.\n\n'
                f'{args.name}[layout][depth][spot] = (actions, codes): codes holds one character per\n'
                'action per hand (hands in HU_CLASSES order, action by action); the chance of an\n'
                f'action is (ord(char) - ord("a")) / {LEVELS}.\n"""\n')
        f.write(f"HU_CLASSES = {CLASSES!r}\nHU_LEVELS = {LEVELS}\n{args.name} = {table!r}\n")
    print(f"wrote {out.name} ({out.stat().st_size // 1024} KB), hero edge {args.edge}")


if __name__ == "__main__":
    main()
