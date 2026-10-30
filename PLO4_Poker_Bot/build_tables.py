"""Rebuild the precomputed tables the bot looks up during play.

    python3 build_tables.py                 # both tables (about 5 min, uses every core)
    python3 build_tables.py --preflop       # preflop_table.json only
    python3 build_tables.py --postflop      # postflop_equity.json only

Build-time only: needs numba for the preflop table. The bot itself does not.

preflop_table.json    every one of the 16,432 distinct starting hands -> percentile
                      plus equity vs a random hand, the top 20% and the top 5%.
postflop_equity.json  equity per read_hand() cell vs any hand / a value range / a
                      raising range. Both players are dealt from the top
                      SEEN_FLOP_PCT of hands, as if they had called preflop.
"""
import argparse
import json
import math
import random
import time
from collections import defaultdict
from itertools import combinations, product
from multiprocessing import Pool

import numpy as np

from poker_core import (EQUITY_FILE, FLUSH, PREFLOP_FILE, _coarser, _rank_value, hand_key, load_preflop,
                        plo_value, read_hand, villain_tier)

SEEN_FLOP_PCT = 0.60       # players reaching a flop hold about this share of hands
PREFLOP_SIMS = 20000
POSTFLOP_SAMPLES = 24_000_000
MIN_CELL = 800             # thinner cells than this fall back to a coarser cell


# =====================================================================
# Preflop (numba Monte Carlo over all hands)
# =====================================================================
def _tables():
    nf = np.zeros(13 ** 5, np.int32)
    seen = {}
    for t in product(range(13), repeat=5):
        k = tuple(sorted(t))
        if k not in seen:
            seen[k] = _rank_value(k)
        nf[sum(r * 13 ** i for i, r in enumerate(t))] = seen[k]
    fl = np.zeros(8192, np.int32)
    for m, v in FLUSH.items():
        fl[m] = v
    return nf, fl


def _compile():
    from numba import njit, prange

    @njit(inline="always")
    def best(h, b, nf, fl):
        top = 0
        for i in range(3):
            for j in range(i + 1, 4):
                a, c = h[i], h[j]
                ra, rc = a >> 2, c >> 2
                for x in range(3):
                    for y in range(x + 1, 4):
                        for z in range(y + 1, 5):
                            p, q, r = b[x], b[y], b[z]
                            s = a & 3
                            if s == (c & 3) and s == (p & 3) and s == (q & 3) and s == (r & 3):
                                v = fl[(1 << ra) | (1 << rc) | (1 << (p >> 2)) | (1 << (q >> 2)) | (1 << (r >> 2))]
                            else:
                                v = nf[ra + 13 * rc + 169 * (p >> 2) + 2197 * (q >> 2) + 28561 * (r >> 2)]
                            if v > top:
                                top = v
        return top

    @njit(parallel=True)
    def equity_all(hands, pool, n_opp, sims, nf, fl, seed):
        """hands: (n, 4). pool: villain hands, sampled uniformly. Returns equity per hand."""
        out = np.zeros(len(hands))
        for idx in prange(len(hands)):
            np.random.seed(seed + idx)
            h = hands[idx]
            total = 0.0
            done = 0
            used = np.zeros(52, np.bool_)
            board = np.zeros(5, np.int64)
            opp = np.zeros((n_opp, 4), np.int64)
            while done < sims:
                used[:] = False
                for c in h:
                    used[c] = True
                ok = True
                for k in range(n_opp):
                    v = pool[np.random.randint(len(pool))]
                    for c in v:
                        if used[c]:
                            ok = False
                        used[c] = True
                    if not ok:
                        break
                    opp[k] = v
                if not ok:
                    continue
                n = 0
                while n < 5:
                    c = np.random.randint(52)
                    if not used[c]:
                        used[c] = True
                        board[n] = c
                        n += 1
                mine = best(h, board, nf, fl)
                top = 0
                ties = 1
                for k in range(n_opp):
                    v = best(opp[k], board, nf, fl)
                    if v > top:
                        top, ties = v, 1
                    elif v == top:
                        ties += 1
                if mine > top:
                    total += 1.0
                elif mine == top:
                    total += 1.0 / ties
                done += 1
            out[idx] = total / sims
        return out

    return equity_all


def build_preflop():
    t0 = time.time()
    nf, fl = _tables()
    equity_all = _compile()
    all_hands = np.array(list(combinations(range(52), 4)), np.int64)
    keys = np.array([hand_key(h.tolist()) for h in all_hands], np.int64)
    uniq, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    reps = all_hands[first]
    print(f"{len(all_hands)} hands, {len(uniq)} classes")
    assert len(uniq) == 16432
    weights = np.bincount(inverse)

    eq_hu = equity_all(reps, all_hands, 1, PREFLOP_SIMS, nf, fl, 11)
    print(f"  vs one random  {time.time() - t0:.0f}s", flush=True)
    eq_4way = equity_all(reps, all_hands, 3, PREFLOP_SIMS, nf, fl, 22)
    print(f"  vs three random  {time.time() - t0:.0f}s", flush=True)

    # Rank by the mix of heads-up and 4-way equity: heads-up alone overrates
    # dry high pairs, multiway alone overrates weak connected hands.
    def pct(x):
        order = np.argsort(-x)
        cum = np.zeros(len(x))
        cum[order] = (np.cumsum(weights[order]) - weights[order] / 2) / weights.sum()
        return cum
    score = (pct(eq_hu) + pct(eq_4way)) / 2
    percentile = pct(-score)
    hand_pct = percentile[inverse]
    top20 = all_hands[hand_pct <= 0.20]
    top5 = all_hands[hand_pct <= 0.05]
    eq_20 = equity_all(reps, top20, 1, PREFLOP_SIMS, nf, fl, 33)
    print(f"  vs top 20%  {time.time() - t0:.0f}s", flush=True)
    eq_5 = equity_all(reps, top5, 1, PREFLOP_SIMS, nf, fl, 44)
    print(f"  vs top 5%  {time.time() - t0:.0f}s", flush=True)

    table = {str(int(k)): [round(float(p), 4), round(float(a), 4), round(float(b), 4), round(float(c), 4)]
             for k, p, a, b, c in zip(uniq, percentile, eq_hu, eq_20, eq_5)}
    with open(PREFLOP_FILE, "w") as f:
        json.dump(table, f, separators=(",", ":"))
    print(f"wrote {PREFLOP_FILE}: {len(table)} hands in {time.time() - t0:.0f}s")


# =====================================================================
# Postflop (Monte Carlo with the bot's own hand reading)
# =====================================================================
def _postflop_job(job):
    seed, samples, cutoff_keys = job
    rng = random.Random(seed)
    counts = defaultdict(lambda: [0.0] * 6)      # key -> (win, n) for tier >= 0, 1, 2
    deck = list(range(52))
    done = 0
    while done < samples:
        cards = rng.sample(deck, 13)
        a, b, runout = cards[:4], cards[4:8], cards[8:]
        if hand_key(a) not in cutoff_keys or hand_key(b) not in cutoff_keys:
            continue
        va5, vb5 = plo_value(a, runout), plo_value(b, runout)
        win_a = 1.0 if va5 > vb5 else 0.5 if va5 == vb5 else 0.0
        for n in (3, 4, 5):
            board = runout[:n]
            for hero, villain, w in ((a, b, win_a), (b, a, 1.0 - win_a)):
                _, _, key = read_hand(hero, board)
                _, p_v, key_v = read_hand(villain, board)
                tier = villain_tier(p_v, key_v)
                cell = counts[key]
                for t in range(tier + 1):
                    cell[t * 2] += w
                    cell[t * 2 + 1] += 1
        done += 1
    return dict(counts)


def build_postflop():
    t0 = time.time()
    pre = load_preflop()
    assert pre, "build the preflop table first"
    allowed = frozenset(k for k, v in pre.items() if v[0] <= SEEN_FLOP_PCT)
    workers = 15
    per_job = 100_000
    jobs = [(i + 1, per_job, allowed) for i in range(POSTFLOP_SAMPLES // per_job)]
    total = defaultdict(lambda: [0.0] * 6)
    with Pool(workers) as pool:
        for i, part in enumerate(pool.imap_unordered(_postflop_job, jobs), 1):
            for k, v in part.items():
                cell = total[k]
                for j in range(6):
                    cell[j] += v[j]
            if i % max(1, len(jobs) // 10) == 0:
                print(f"  {i}/{len(jobs)} jobs, {time.time() - t0:.0f}s", flush=True)

    # Roll every cell up its backoff chain, then keep the cells with enough data.
    rolled = defaultdict(lambda: [0.0] * 6)
    for k, v in total.items():
        for parent in _coarser(k):
            cell = rolled[parent]
            for j in range(6):
                cell[j] += v[j]
    out = {}
    for k, v in rolled.items():
        if v[1] < MIN_CELL:
            continue
        any_eq = v[0] / v[1]
        val_eq = v[2] / v[3] if v[3] >= MIN_CELL / 4 else any_eq * 0.7
        rai_eq = v[4] / v[5] if v[5] >= MIN_CELL / 8 else val_eq * 0.7
        out[",".join(map(str, k))] = [round(any_eq, 4), round(val_eq, 4), round(rai_eq, 4)]
    with open(EQUITY_FILE, "w") as f:
        json.dump(out, f, separators=(",", ":"), sort_keys=True)
    print(f"wrote {EQUITY_FILE}: {len(out)} cells in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflop", action="store_true")
    ap.add_argument("--postflop", action="store_true")
    args = ap.parse_args()
    both = not args.preflop and not args.postflop
    if args.preflop or both:
        build_preflop()
    if args.postflop or both:
        build_postflop()
