"""Cards, PLO hand evaluator and postflop hand reading.

You normally don't need to edit this file; strategy knobs live in config.py.
Cards are ints 0..51: rank = c >> 2 (0 = deuce .. 12 = ace), suit = c & 3.

PLO rule everything here respects: a hand uses EXACTLY two hole cards and three
board cards. All lookups are plain Python tables, no numpy, so a full hand read
(strength plus draws) stays well under the 1 ms decision limit.
"""
from functools import lru_cache
from itertools import combinations
import json
import os

RANKS = "23456789TJQKA"
SUITS = "cdhs"
_SUIT_ALIASES = {"♣": "c", "♦": "d", "♥": "h", "♠": "s"}

MONSTER, STRONG, MEDIUM, STRONG_DRAW, DRAW, WEAK = (
    "MONSTER", "STRONG", "MEDIUM", "STRONG_DRAW", "DRAW", "WEAK")

EQUITY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "postflop_equity.json")
PREFLOP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preflop_table.json")


# ---------------------------------------------------------------- cards
def parse_card(s):
    if isinstance(s, int):
        return s
    s = s.strip().replace("︎", "").replace("️", "")
    if s.startswith("10"):
        s = "T" + s[2:]
    suit = _SUIT_ALIASES.get(s[1], s[1].lower())
    return RANKS.index(s[0].upper()) * 4 + SUITS.index(suit)


def parse_cards(cards):
    """Accepts ["Ah", "Kd"], "Ah Kd", "AhKd", "A♥ K♦" or ints."""
    if not cards:
        return []
    if isinstance(cards, str):
        s = cards.replace("︎", "").replace("️", "").replace("10", "T")
        s = "".join(ch for ch in s if ch not in " ,[]'\"")
        return [parse_card(s[i:i + 2]) for i in range(0, len(s), 2)]
    return [parse_card(c) for c in cards]


def card_str(c):
    return RANKS[c >> 2] + SUITS[c & 3]


# ---------------------------------------------------------------- 5-card values
# Value = category << 20 | up to five 4-bit ranks. Bigger is better.
# Category: 8 SF, 7 quads, 6 full house, 5 flush, 4 straight, 3 trips, 2 two pair, 1 pair, 0 high.
def _pack(cat, *ranks):
    v = cat
    for i in range(5):
        v = v << 4 | (ranks[i] if i < len(ranks) else 0)
    return v


def _straight_high(mask):
    for hi in range(12, 3, -1):
        if (mask >> (hi - 4)) & 31 == 31:
            return hi
    return 3 if mask & 0b1000000001111 == 0b1000000001111 else -1


def _rank_value(ranks):
    """Five ranks, not all the same suit -> value."""
    counts = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    order = sorted(counts, key=lambda r: (counts[r], r), reverse=True)
    shape = sorted(counts.values(), reverse=True)
    if shape[0] == 5:
        return 0            # five of a rank cannot be dealt
    if shape[0] == 4:
        return _pack(7, *order)
    if shape[:2] == [3, 2]:
        return _pack(6, *order)
    if shape[0] == 3:
        return _pack(3, *order)
    if shape[:2] == [2, 2]:
        return _pack(2, *order)
    if shape[0] == 2:
        return _pack(1, *order)
    st = _straight_high(sum(1 << r for r in ranks))
    return _pack(4, st) if st >= 0 else _pack(0, *order)


def _flush_value(mask):
    st = _straight_high(mask)
    if st >= 0:
        return _pack(8, st)
    return _pack(5, *[r for r in range(12, -1, -1) if mask >> r & 1])


FLUSH = {sum(1 << r for r in c): _flush_value(sum(1 << r for r in c)) for c in combinations(range(13), 5)}

# Tables: value of any three board ranks + any two hole ranks.
PAIRS = [(a, b) for a in range(13) for b in range(a + 1)]               # 91 rank pairs, a >= b
PAIR_INDEX = [[0] * 13 for _ in range(13)]
for _i, (_a, _b) in enumerate(PAIRS):
    PAIR_INDEX[_a][_b] = PAIR_INDEX[_b][_a] = _i

_seen = {}


def _row(triple):
    out = []
    for a, b in PAIRS:
        k = tuple(sorted(triple + (a, b)))
        if k not in _seen:
            _seen[k] = _rank_value(k)
        out.append(_seen[k])
    return out


TRIPLE_ROWS = {}
for _a in range(13):
    for _b in range(_a + 1):
        for _c in range(_b + 1):
            TRIPLE_ROWS[(_a, _b, _c)] = _row((_a, _b, _c))
del _seen


@lru_cache(maxsize=8192)
def _board_best(ranks):
    """Sorted board ranks -> best non-flush value per hole rank pair (91 entries)."""
    rows = [TRIPLE_ROWS[t] for t in set(combinations(ranks, 3))]
    return rows[0] if len(rows) == 1 else list(map(max, *rows))


def _sorted_ranks(board):
    return tuple(sorted((c >> 2 for c in board), reverse=True))


def plo_value(hole, board):
    """Best PLO hand (exactly 2 hole + 3 board) -> comparable int. Board needs 3+ cards."""
    best_row = _board_best(_sorted_ranks(board))
    v = max(best_row[PAIR_INDEX[a >> 2][b >> 2]] for a, b in combinations(hole, 2))
    for s in range(4):
        on_board = [c >> 2 for c in board if c & 3 == s]
        if len(on_board) >= 3:
            mine = [c >> 2 for c in hole if c & 3 == s]
            if len(mine) >= 2:
                for t in combinations(on_board, 3):
                    tmask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
                    for a, b in combinations(mine, 2):
                        f = FLUSH[tmask | 1 << a | 1 << b]
                        if f > v:
                            v = f
    return v


def category(value):
    return value >> 20


def describe(value):
    cat = value >> 20
    r = [RANKS[(value >> (16 - 4 * i)) & 15] for i in range(5)]
    return [f"high card {r[0]}", f"pair of {r[0]}s", f"two pair, {r[0]}s and {r[1]}s", f"three {r[0]}s",
            f"straight to {r[0]}", f"flush, {r[0]} high", f"full house, {r[0]}s full of {r[1]}s",
            f"four {r[0]}s", f"straight flush to {r[0]}"][cat]


# ---------------------------------------------------------------- preflop
def hand_key(hole):
    """Four hole cards -> int identical for all suit-swapped versions (16,432 classes)."""
    m = [0, 0, 0, 0]
    for c in hole:
        m[c & 3] |= 1 << (c >> 2)
    m.sort(reverse=True)
    return m[0] << 39 | m[1] << 26 | m[2] << 13 | m[3]


# ---------------------------------------------------------------- straights
_WINDOWS = [(0b1000000001111, 3)] + [(31 << lo, lo + 4) for lo in range(9)]


@lru_cache(maxsize=8192)
def _straight_needs(board_mask):
    """Board rank mask -> [(two-rank hole mask, straight high)], highest straight first."""
    out = []
    for window, high in reversed(_WINDOWS):
        ranks = [r for r in range(13) if window >> r & 1]
        for a, b in combinations(ranks, 2):
            need = 1 << a | 1 << b
            if (window & ~need) & ~board_mask == 0:
                out.append((need, high))
    return out


# ---------------------------------------------------------------- hand reading
def strength(hole, board):
    """-> (my value, share of unseen two-card holdings that beat me on this board).

    Counts real unseen cards, so blockers are included automatically: holding the
    ace of the flush suit removes every holding that would beat us with it.
    """
    v = plo_value(hole, board)
    best_row = _board_best(_sorted_ranks(board))
    left = [4] * 13
    for c in hole:
        left[c >> 2] -= 1
    for c in board:
        left[c >> 2] -= 1
    n = 0
    for (a, b), val in zip(PAIRS, best_row):
        if val > v:
            n += left[a] * left[b] if a != b else left[a] * (left[a] - 1) // 2
    seen = set(hole) | set(board)
    for s in range(4):
        on_board = [c >> 2 for c in board if c & 3 == s]
        if len(on_board) >= 3:
            unseen = [r for r in range(13) if r * 4 + s not in seen]
            triples = [(1 << t[0]) | (1 << t[1]) | (1 << t[2]) for t in combinations(on_board, 3)]
            for a, b in combinations(unseen, 2):
                pair = 1 << a | 1 << b
                if max(FLUSH[t | pair] for t in triples) > v and best_row[PAIR_INDEX[a][b]] <= v:
                    n += 1
    total = 52 - len(seen)
    return v, n / (total * (total - 1) / 2)


def draws(hole, board, value):
    """-> (flush draw 0/1/2 = none/non-nut/nut, straight outs, nut straight outs)."""
    seen = set(hole) | set(board)
    fd = 0
    for s in range(4):
        if sum(1 for c in board if c & 3 == s) == 2:
            mine = [c >> 2 for c in hole if c & 3 == s]
            if len(mine) >= 2:
                top_unseen = next(r for r in range(12, -1, -1)
                                  if r * 4 + s not in seen or r in mine)
                fd = max(fd, 2 if max(mine) == top_unseen else 1)
    board_mask = hole_mask = 0
    for c in board:
        board_mask |= 1 << (c >> 2)
    for c in hole:
        hole_mask |= 1 << (c >> 2)
    made = (value >> 16) & 15 if value >> 20 == 4 else -1    # our current straight, if any
    left = [4] * 13
    for c in seen:
        left[c >> 2] -= 1
    outs = nuts = 0
    for r in range(13):
        if left[r] == 0:
            continue
        needs = _straight_needs(board_mask | 1 << r)
        if not needs:
            continue
        for need, high in needs:
            if hole_mask & need == need:
                if high > made:
                    outs += left[r]
                    if high == needs[0][1]:
                        nuts += left[r]
                break
    return fd, outs, nuts


HS_BINS = (0.0, 0.02, 0.06, 0.15, 0.30, 0.50, 0.70)


def read_hand(hole, board):
    """Full read -> (value, p_beat, key). p_beat = chance one random 4-card hand has us beat
    right now; key = the cell of postflop_equity.json this spot belongs to."""
    v, frac = strength(hole, board)
    p_beat = 1 - (1 - frac) ** 6
    hs = 0
    for edge in HS_BINS:
        if p_beat > edge:
            hs += 1
    if len(board) == 5:
        return v, p_beat, (5, hs, 0, 0, 0, 0)
    fd, outs, nuts = draws(hole, board, v)
    score = nuts + 0.5 * (outs - nuts)
    sb = 0 if score < 3 else 1 if score < 7 else 2 if score < 11 else 3
    suits = [c & 3 for c in board]
    ranks = [c >> 2 for c in board]
    wet = 1 if len(set(ranks)) < len(ranks) or max(suits.count(s) for s in range(4)) >= 3 else 0
    redraw = 1 if v >> 20 in (2, 3) else 0
    return v, p_beat, (len(board), hs, fd, sb, wet, redraw)


def villain_tier(p_beat, key):
    """Used to build the equity table: 0 = any hand, 1 = bets for value, 2 = raises for value."""
    street, _, fd, sb = key[:4]
    if street == 5:
        return 2 if p_beat <= 0.06 else 1 if p_beat <= 0.25 else 0
    if p_beat <= 0.08 or (fd == 2 and sb >= 2):
        return 2
    if p_beat <= 0.30 or fd == 2 or sb >= 2 or (fd == 1 and sb >= 1):
        return 1
    return 0


# ---------------------------------------------------------------- precomputed tables
def _coarser(key):
    """Backoff chain for cells too rare to have a good estimate."""
    street, hs, fd, sb, wet, redraw = key
    yield (street, hs, fd, sb, wet, redraw)
    yield (street, hs, fd, sb, wet, -1)
    yield (street, hs, fd, sb, -1, -1)
    yield (street, hs, fd, -1, -1, -1)
    yield (street, hs, -1, -1, -1, -1)


def load_equity(path=EQUITY_FILE):
    try:
        with open(path) as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return None
    return {tuple(int(x) for x in k.split(",")): tuple(v) for k, v in raw.items()}


def load_preflop(path=PREFLOP_FILE):
    """-> {hand_key: (percentile, equity vs any, vs top 20%, vs top 5%)}. Percentile 0 = AAxx best."""
    try:
        with open(path) as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return None
    return {int(k): tuple(v) for k, v in raw.items()}


EQUITY = load_equity()
PREFLOP = load_preflop()


def equities(key, table=None):
    """-> (eq vs any hand, vs value range, vs raising range) for a read_hand key."""
    t = EQUITY if table is None else table
    for k in _coarser(key):
        e = t.get(k)
        if e is not None:
            return e
    return (0.5, 0.3, 0.2)
