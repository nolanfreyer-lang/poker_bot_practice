"""Cards, hand evaluator, range parser and hand-strength buckets.

You normally don't need to edit this file; strategy knobs live in config.py.
Cards are ints 0..51: rank = c >> 2 (0 = deuce .. 12 = ace), suit = c & 3.
"""
from functools import lru_cache

RANKS = "23456789TJQKA"
SUITS = "cdhs"
_SUIT_ALIASES = {"♣": "c", "♦": "d", "♥": "h", "♠": "s"}

MONSTER, STRONG, MEDIUM, STRONG_DRAW, DRAW, WEAK = (
    "MONSTER", "STRONG", "MEDIUM", "STRONG_DRAW", "DRAW", "WEAK")


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


def hand_class(c1, c2):
    """Two cards -> canonical class like 'AKs', 'T9o', '77'."""
    r1, r2 = c1 >> 2, c2 >> 2
    if r1 < r2:
        r1, r2 = r2, r1
    if r1 == r2:
        return RANKS[r1] * 2
    return RANKS[r1] + RANKS[r2] + ("s" if (c1 & 3) == (c2 & 3) else "o")


ALL_CLASSES = frozenset(hand_class(a, b) for a in range(52) for b in range(52) if a != b)


# ---------------------------------------------------------------- ranges
def _expand(hi, lo_from, lo_to, suffix):
    out = set()
    for lo in range(lo_from, lo_to + 1):
        base = RANKS[hi] + RANKS[lo]
        out.update([base + "s", base + "o"] if suffix == "" else [base + suffix])
    return out


@lru_cache(maxsize=None)
def parse_range(text):
    """Range string -> frozenset of hand classes. See config.py for syntax."""
    out = set()
    for tok in text.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in ("*", "any", "random"):
            return ALL_CLASSES
        t = tok[0].upper() + tok[1].upper() + tok[2:].lower()
        try:
            if "-" in t:
                a, b = (x.strip() for x in t.split("-"))
                a = a[0].upper() + a[1].upper() + a[2:]
                b = b[0].upper() + b[1].upper() + b[2:]
                ra, rb = RANKS.index(a[0]), RANKS.index(b[0])
                if a[0] == a[1]:  # pair span "77-44"
                    lo, hi = sorted((ra, RANKS.index(b[1])))
                    out.update(RANKS[r] * 2 for r in range(lo, hi + 1))
                else:  # "A5s-A2s"
                    assert ra == rb, "range span must share the top card"
                    k1, k2 = sorted((RANKS.index(a[1]), RANKS.index(b[1])))
                    out |= _expand(ra, k1, k2, a[2:3])
                continue
            plus = t.endswith("+")
            t = t.rstrip("+")
            r1, r2 = RANKS.index(t[0]), RANKS.index(t[1])
            suffix = t[2:3]
            if r1 == r2:
                out.update(RANKS[r] * 2 for r in range(r1, 13 if plus else r1 + 1))
            else:
                hi, lo = max(r1, r2), min(r1, r2)
                out |= _expand(hi, lo, hi - 1 if plus else lo, suffix)
        except (ValueError, IndexError, AssertionError) as e:
            raise ValueError(f"Bad range token {tok!r}: {e}") from None
    return frozenset(out)


def range_pct(rng):
    """Fraction of all 1326 combos covered by a parsed range (for sanity checks)."""
    combos = sum(6 if len(h) == 2 else 4 if h[2] == "s" else 12 for h in rng)
    return combos / 1326


# ---------------------------------------------------------------- evaluator
def _straight_high(ranks):
    for hi in range(12, 3, -1):
        if hi in ranks and hi - 1 in ranks and hi - 2 in ranks and hi - 3 in ranks and hi - 4 in ranks:
            return hi
    if 12 in ranks and 0 in ranks and 1 in ranks and 2 in ranks and 3 in ranks:
        return 3
    return -1


def hand_rank(cards):
    """Best hand from any number of cards -> comparable tuple (bigger is better).
    Category: 8 SF, 7 quads, 6 full house, 5 flush, 4 straight, 3 trips, 2 two pair, 1 pair, 0 high."""
    rc = [0] * 13
    sc = [0] * 4
    for c in cards:
        rc[c >> 2] += 1
        sc[c & 3] += 1
    flush = None
    for s in range(4):
        if sc[s] >= 5:
            fr = sorted({c >> 2 for c in cards if c & 3 == s}, reverse=True)
            sf = _straight_high(set(fr))
            if sf >= 0:
                return (8, sf)
            flush = fr
            break
    desc = [r for r in range(12, -1, -1) if rc[r]]
    quads = [r for r in desc if rc[r] == 4]
    trips = [r for r in desc if rc[r] == 3]
    pairs = [r for r in desc if rc[r] == 2]
    if quads:
        q = quads[0]
        return (7, q, next(r for r in desc if r != q)) if len(desc) > 1 else (7, q)
    if trips and (len(trips) > 1 or pairs):
        p = max(trips[1:2] + pairs[:1])
        return (6, trips[0], p)
    if flush:
        return (5, *flush[:5])
    st = _straight_high(set(desc))
    if st >= 0:
        return (4, st)
    if trips:
        t = trips[0]
        return (3, t, *[r for r in desc if r != t][:2])
    if len(pairs) >= 2:
        p1, p2 = pairs[0], pairs[1]
        return (2, p1, p2, *[r for r in desc if r != p1 and r != p2][:1])
    if pairs:
        p = pairs[0]
        return (1, p, *[r for r in desc if r != p][:3])
    return (0, *desc[:5])


# ---------------------------------------------------------------- buckets
def _draws(hole, board):
    """-> (flush_draw, open_ended) using at least one hole card."""
    cards = hole + board
    fd = False
    for s in range(4):
        if sum(1 for c in cards if c & 3 == s) == 4 and any(c & 3 == s for c in hole):
            fd = True
    ranks = {c >> 2 for c in cards}
    if 12 in ranks:
        ranks.add(-1)  # ace low
    hole_ranks = {c >> 2 for c in hole} | ({-1} if any(c >> 2 == 12 for c in hole) else set())
    oe = False
    for lo in range(-1, 9):
        run = {lo, lo + 1, lo + 2, lo + 3}
        if run <= ranks and run & hole_ranks and lo - 1 >= -1 and lo + 4 <= 12:
            oe = True
    return fd, oe


def classify_hand(hole, board, good_kicker_rank=8):
    """Postflop hand -> bucket name. Fast (~2 hand_rank calls)."""
    mine = hand_rank(hole + board)
    on_board = hand_rank(board)
    cat = mine[0]
    board_ranks = sorted({c >> 2 for c in board}, reverse=True)
    h1, h2 = sorted((c >> 2 for c in hole), reverse=True)
    pocket_pair = h1 == h2
    river = len(board) >= 5
    fd, oe = (False, False) if river else _draws(hole, board)

    if mine == on_board:  # board plays
        made = WEAK
    elif cat >= 4:
        made = MONSTER if cat > on_board[0] else MEDIUM
    elif cat == 3:
        t = mine[1]
        if pocket_pair and h1 == t:
            made = MONSTER                     # set
        elif on_board[0] == 3:
            made = MEDIUM if max(h1, h2) >= 10 else WEAK  # trips on board, kicker matters
        else:
            made = STRONG                      # trips using one hole card
    elif cat == 2:
        p1, p2 = mine[1], mine[2]
        board_pairs = {r for r in board_ranks if sum(1 for c in board if c >> 2 == r) >= 2}
        if not ({p1, p2} & board_pairs):
            made = MONSTER                     # both pairs use hole cards
        else:
            hole_pair = p1 if p1 not in board_pairs else p2
            made = _one_pair_bucket(hole_pair, h1, h2, pocket_pair, board_ranks, good_kicker_rank) \
                if hole_pair not in board_pairs else (MEDIUM if h1 >= 11 else WEAK)
    elif cat == 1 and on_board[0] < 1:
        made = _one_pair_bucket(mine[1], h1, h2, pocket_pair, board_ranks, good_kicker_rank)
    else:
        made = WEAK

    if made in (WEAK, MEDIUM) and (fd and oe or (made == MEDIUM and fd)):
        return STRONG_DRAW
    if made == WEAK and (fd or oe):
        return DRAW
    return made


def _one_pair_bucket(p, h1, h2, pocket_pair, board_ranks, good_kicker_rank):
    if pocket_pair:
        if p > board_ranks[0]:
            return STRONG                      # overpair
        return MEDIUM if len(board_ranks) > 1 and p > board_ranks[1] else WEAK
    if p == board_ranks[0]:
        kicker = h2 if h1 == p else h1
        return STRONG if kicker >= good_kicker_rank or p == 12 and kicker >= good_kicker_rank - 2 else MEDIUM
    if len(board_ranks) > 1 and p == board_ranks[1]:
        return MEDIUM
    return WEAK
