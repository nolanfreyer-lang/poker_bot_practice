"""Local 8-bot sit-and-go simulator (NLHE) mirroring the official format.

    python local_sim.py                  # 200 tables, default lineup
    python local_sim.py --tables 1000 --seed 7
    python local_sim.py --verbose        # print one table's hands
    python local_sim.py --table 3 --hand 12              # replay hand 12 of table 3
    python local_sim.py --table 3 --hand 12 --show-cards # ... with every bot's hole cards

Use the starter kit's simulator for final testing; this one lets you tune
config.py before the variant is announced. Simplification: any all-in
raise re-opens the betting.
"""
import argparse
import random
import time
from functools import lru_cache

import config as C
from bot import PokerBot, GameState, PlayerInfo, Action
from poker_core import hand_rank, card_str, RANKS


# =====================================================================
# LINEUP -- edit to test config variants against each other
# =====================================================================
class CallingStation:
    def act(self, s):
        return Action("call") if s.to_call else Action("check")


class RandomBot:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def act(self, s):
        r = self.rng.random()
        if r < 0.15 and s.to_call:
            return Action("fold")
        if r < 0.80:
            return Action("call") if s.to_call else Action("check")
        if r < 0.97:
            return Action("raise", s.min_raise_to)
        return Action("allin")


class Maniac:
    def act(self, s):
        return Action("raise", s.min_raise_to * 2)


def default_lineup(seed):
    return [
        ("TEMPLATE", PokerBot(seed=seed)),
        ("TIGHT", PokerBot(seed=seed + 1, overrides={"POSTFLOP_POLICY": {
            **C.POSTFLOP_POLICY, "WEAK": {**C.POSTFLOP_POLICY["WEAK"], "bet_freq": 0.0}}})),
        ("TEMPLATE2", PokerBot(seed=seed + 2)),
        ("STATION", CallingStation()),
        ("RANDOM", RandomBot(seed + 3)),
        ("MANIAC", Maniac()),
        ("RANDOM2", RandomBot(seed + 4)),
        ("STATION2", CallingStation()),
    ]


# =====================================================================
# ICM (Malmuth-Harville) for the 200-hand cap
# =====================================================================
def icm_equity(stacks, payouts):
    n = len(stacks)
    total = sum(stacks)

    @lru_cache(maxsize=None)
    def place_probs(mask, place):
        # prob each remaining player (bits in mask) finishes at `place`, given mask remains
        rem = [i for i in range(n) if mask >> i & 1]
        tot = sum(stacks[i] for i in rem)
        out = [0.0] * n
        for i in rem:
            p = stacks[i] / tot if tot else 1 / len(rem)
            out[i] += p * payouts[place] if place < len(payouts) else 0
            if len(rem) > 1:
                sub = place_probs(mask & ~(1 << i), place + 1)
                for j in range(n):
                    out[j] += p * sub[j]
        return tuple(out)

    full = sum(1 << i for i in range(n) if stacks[i] > 0)
    return list(place_probs(full, 0)) if total else [0.0] * n


def describe(rank):
    """hand_rank tuple -> readable name, e.g. 'two pair, Ks and 7s'."""
    cat, r = rank[0], [RANKS[x] for x in rank[1:]]
    return [
        lambda: f"high card {r[0]}",
        lambda: f"pair of {r[0]}s",
        lambda: f"two pair, {r[0]}s and {r[1]}s",
        lambda: f"three {r[0]}s",
        lambda: f"straight to {r[0]}",
        lambda: f"flush, {r[0]} high",
        lambda: f"full house, {r[0]}s full of {r[1]}s",
        lambda: f"four {r[0]}s",
        lambda: f"straight flush to {r[0]}",
    ][cat]()


# =====================================================================
# Dealer
# =====================================================================
class Table:
    def __init__(self, lineup, rng, verbose=False, only_hand=None, show_cards=False):
        self.names = [n for n, _ in lineup]
        self.bots = [b for _, b in lineup]
        self.n = len(lineup)
        self.stacks = [C.STARTING_STACK] * self.n
        self.rng = rng
        self.verbose = verbose
        self.only_hand = only_hand     # print just this hand number (None = all)
        self.show_cards = show_cards   # print every bot's hole cards
        self.hand_no = 0
        self.hands_played = 0
        self.button = rng.randrange(self.n)
        self.finish_order = []   # seats in order of elimination
        self.decision_times = [[] for _ in range(self.n)]

    def log(self, *a):
        if self.verbose and (self.only_hand is None or self.hand_no == self.only_hand):
            print(*a)

    def alive(self):
        return [s for s in range(self.n) if self.stacks[s] > 0]

    def next_alive(self, seat):
        for k in range(1, self.n + 1):
            t = (seat + k) % self.n
            if self.stacks[t] > 0:
                return t
        return seat

    def play(self):
        for hand_no in range(1, C.HAND_LIMIT + 1):
            if len(self.alive()) <= 1:
                break
            level = min((hand_no - 1) // C.HANDS_PER_LEVEL, len(C.BLIND_SCHEDULE) - 1)
            self.hand_no = self.hands_played = hand_no
            self.play_hand(hand_no, *C.BLIND_SCHEDULE[level])
            self.button = self.next_alive(self.button)
        return self.points()

    def points(self):
        pts = [0.0] * self.n
        alive = self.alive()
        # eliminated players: last eliminated gets place len(alive)+1, etc.
        for k, seat in enumerate(reversed(self.finish_order)):
            place = len(alive) + k
            pts[seat] = C.PAYOUT_POINTS[place] if place < len(C.PAYOUT_POINTS) else 0
        if len(alive) == 1:
            pts[alive[0]] = C.PAYOUT_POINTS[0]
        else:
            eq = icm_equity([self.stacks[s] if s in alive else 0 for s in range(self.n)],
                            C.PAYOUT_POINTS[:len(alive)])
            for s in alive:
                pts[s] = eq[s]
        return pts

    def play_hand(self, hand_no, sb, bb):
        n, rng = self.n, self.rng
        dealt = self.alive()
        deck = list(range(52))
        rng.shuffle(deck)
        holes = {s: [deck.pop(), deck.pop()] for s in dealt}
        P = [PlayerInfo(seat=s, stack=self.stacks[s], active=s in dealt, dealt=s in dealt)
             for s in range(n)]
        contrib = [0] * n
        history, board = [], []
        start = self.stacks[:]

        heads_up = len(dealt) == 2
        sb_seat = self.button if heads_up else self.next_alive(self.button)
        bb_seat = self.next_alive(sb_seat)

        def put(seat, target):  # raise this seat's street bet to `target`
            p = P[seat]
            add = min(target - p.bet, p.stack)
            p.stack -= add
            p.bet += add
            contrib[seat] += add
            if p.stack == 0:
                p.all_in = True

        put(sb_seat, sb)
        history.append(("preflop", sb_seat, "blind", P[sb_seat].bet))
        put(bb_seat, bb)
        history.append(("preflop", bb_seat, "blind", P[bb_seat].bet))
        self.log(f"\n== Hand {hand_no}  blinds {sb}/{bb}  button S{self.button + 1}")
        if self.show_cards:
            for s in dealt:
                self.log(f"  S{s + 1} {self.names[s]:<10} {card_str(holes[s][0])} {card_str(holes[s][1])}"
                         f"   stack {start[s]}")
            self.log("  --")

        for street in ("preflop", "flop", "turn", "river"):
            if street != "preflop":
                board += [deck.pop() for _ in range(1 if board else 3)]
                for p in P:
                    p.bet = 0
                self.log(f"  {street.upper()} {' '.join(card_str(c) for c in board)}")
            first = self.next_alive(bb_seat) if street == "preflop" else \
                (bb_seat if heads_up else self.next_alive(self.button))
            self.betting_round(street, first, P, holes, board, history, contrib, sb, bb, hand_no)
            if sum(p.active for p in P) == 1:
                break

        # award
        live = [p.seat for p in P if p.active]
        while len(board) < 5 and len(live) > 1:
            board.append(deck.pop())
        ranks = {s: hand_rank(holes[s] + board) for s in live}
        if len(live) > 1:
            self.log(f"  SHOWDOWN {' '.join(card_str(c) for c in board)}")
            for s in live:
                self.log(f"  S{s + 1} {self.names[s]:<10} {card_str(holes[s][0])} {card_str(holes[s][1])}"
                         f"   {describe(ranks[s])}")
        remaining = contrib[:]
        winners = live
        while sum(remaining) > 0:
            live_pos = [s for s in live if remaining[s] > 0]
            level = min(remaining[s] for s in live_pos) if live_pos else max(remaining)
            pot = 0
            for s in range(n):
                take = min(remaining[s], level)
                pot += take
                remaining[s] -= take
            if live_pos:
                best = max(ranks[s] for s in live_pos) if len(live) > 1 else None
                winners = [s for s in live_pos if len(live) == 1 or ranks[s] == best]
            share, odd = divmod(pot, len(winners))
            for i, s in enumerate(winners):
                P[s].stack += share + (1 if i < odd else 0)

        for s in range(n):
            self.stacks[s] = P[s].stack
        busted = [s for s in dealt if self.stacks[s] == 0]
        busted.sort(key=lambda s: start[s])  # smaller starting stack finishes lower
        self.finish_order += busted
        for s in dealt:
            if self.stacks[s] != start[s]:
                self.log(f"  S{s + 1} {self.names[s]:<10} {self.stacks[s] - start[s]:+d}")
        for s in dealt:
            hook = getattr(self.bots[s], "on_hand_end", None)
            if hook and getattr(self.bots[s], "WANTS_SHOWDOWN", False):
                shown = {x: list(holes[x]) for x in live} if len(live) > 1 else {}
                hook(history, dealt, showdown=shown, board=list(board), my_seat=s)
            elif hook:
                hook(history, dealt)

    def betting_round(self, street, first, P, holes, board, history, contrib, sb, bb, hand_no):
        n = self.n
        high = max(p.bet for p in P)
        last_raise = bb
        order = [(first + k) % n for k in range(n)]
        pending = {s for s in order if P[s].active and not P[s].all_in}
        idx = 0
        while pending:
            if sum(p.active for p in P) == 1:
                return
            seat = order[idx % n]
            idx += 1
            if seat not in pending:
                continue
            p = P[seat]
            to_call = high - p.bet
            others = [q for q in P if q.active and not q.all_in and q.seat != seat]
            if not others and to_call == 0:
                pending.discard(seat)
                continue
            state = GameState(
                my_seat=seat, hole_cards=list(holes[seat]), board=list(board), street=street,
                pot=sum(contrib), players=[PlayerInfo(**vars(q)) for q in P],
                button_seat=self.button, small_blind=sb, big_blind=bb, to_call=min(to_call, p.stack),
                min_raise_to=min(high + last_raise, p.stack + p.bet), max_raise_to=p.stack + p.bet,
                history=list(history), hand_number=hand_no)
            t0 = time.perf_counter()
            try:
                a = self.bots[seat].act(state)
            except Exception as e:
                self.log(f"  !! S{seat + 1} error: {e!r}")
                a = Action("check" if to_call == 0 else "fold")
            self.decision_times[seat].append(time.perf_counter() - t0)

            kind = a.kind
            if kind in ("bet", "raise"):
                target = int(a.amount)
                if target >= p.stack + p.bet:
                    kind = "allin"
                elif target < high + last_raise:
                    kind = "call" if to_call else "check"
            if kind == "fold" and to_call == 0:
                kind = "check"
            if kind == "check" and to_call > 0:
                kind = "fold"

            if kind == "fold":
                p.active = False
                pending.discard(seat)
            elif kind in ("check", "call"):
                put_to = min(high, p.stack + p.bet)
                self._put(p, contrib, put_to)
                pending.discard(seat)
                if p.stack == 0:
                    kind = "allin"
            else:
                target = p.stack + p.bet if kind == "allin" else int(a.amount)
                self._put(p, contrib, target)
                pending.discard(seat)
                if p.bet > high:
                    last_raise = max(last_raise, p.bet - high)
                    high = p.bet
                    pending = {q.seat for q in P if q.active and not q.all_in and q.seat != seat}
            history.append((street, seat, kind, p.bet))
            self.log(f"  S{seat + 1} {self.names[seat]:<10} {kind:<5} {p.bet if kind != 'fold' else ''}")

    @staticmethod
    def _put(p, contrib, target):
        add = min(target - p.bet, p.stack)
        p.stack -= add
        p.bet += add
        contrib[p.seat] += add
        if p.stack == 0:
            p.all_in = True


# =====================================================================
def make_table(seed, t, **kwargs):
    """Table t (0-based) is fully determined by (seed, t), so any table can be replayed alone."""
    rng = random.Random(seed * 1000 + t)
    lineup = default_lineup(seed * 1000 + t)   # fresh bots: state resets per table
    rng.shuffle(lineup)                         # random seat assignment
    return Table(lineup, rng, **kwargs)


def replay_table(args):
    t = (args.table or 1) - 1
    table = make_table(args.seed, t, verbose=True, only_hand=args.hand, show_cards=args.show_cards)
    print(f"Table {t + 1} (seed {args.seed})  seats: "
          + "  ".join(f"S{s + 1} {nm}" for s, nm in enumerate(table.names)))
    pts = table.play()
    if args.hand and args.hand > table.hands_played:
        print(f"\nTable {t + 1} ended after {table.hands_played} hands; there is no hand {args.hand}.")
    print(f"\nFinal ({table.hands_played} hands): "
          + "  ".join(f"{table.names[s]} {pts[s]:.1f}pts" for s in range(table.n)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--table", type=int, help="replay just this table (1-based) with a hand log")
    ap.add_argument("--hand", type=int, help="only print this hand number")
    ap.add_argument("--show-cards", action="store_true", help="print every bot's hole cards")
    args = ap.parse_args()

    if args.table or args.hand or args.show_cards:
        replay_table(args)
        return

    totals, times, names = None, None, None
    t_start = time.time()
    for t in range(args.tables):
        table = make_table(args.seed, t, verbose=args.verbose and t == 0)
        pts = table.play()
        lineup = list(zip(table.names, table.bots))
        if totals is None:
            names = sorted(n for n, _ in lineup)
            totals = {n: 0.0 for n in names}
            times = {n: [] for n in names}
        for s, nm in enumerate(table.names):
            totals[nm] += pts[s]
            times[nm] += table.decision_times[s]

    print(f"\n{args.tables} tables in {time.time() - t_start:.1f}s   (payouts {C.PAYOUT_POINTS})")
    print(f"{'BOT':<10} {'AVG PTS':>8} {'AVG ms':>8} {'MAX ms':>8} {'>1ms':>6}")
    for nm in sorted(names, key=lambda x: -totals[x]):
        ts = times[nm] or [0]
        slow = sum(1 for x in ts if x > 0.001)
        print(f"{nm:<10} {totals[nm] / args.tables:8.2f} {1000 * sum(ts) / len(ts):8.3f} "
              f"{1000 * max(ts):8.3f} {slow:6d}")


if __name__ == "__main__":
    main()
