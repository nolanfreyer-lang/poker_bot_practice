"""Search for better PLO4 settings against the opponent fields.

    python3 local_sim_search.py                 # all 3 stages (~10 min)
    python3 local_sim_search.py --scale 0.2     # quick, noisier version

Every candidate plays the SAME tables, so differences are measured table by table
(paired), which removes most luck. Candidates are scored over SEARCH_FIELDS.

  Stage 1  screen: each setting changed alone vs the current defaults
  Stage 2  combine: every combination of the settings that helped in stage 1
  Stage 3  confirm: the best few on fresh tables. Adopt only what reaches t >= 3 here.
"""
import argparse
import itertools
import statistics
import time
from multiprocessing import Pool

import config as C
from bot import PokerBot
from local_sim_robust import FIELDS, REGISTRY, CHUNK
from local_sim import Table
import random

SEARCH_FIELDS = ("likely", "tough", "tight", "mirror")


def scaled(table, mult):
    return {k: round(v * mult, 4) for k, v in table.items()}


VARIANTS = {
    "open x0.8": {"OPEN_PCT": scaled(C.OPEN_PCT, 0.8)},
    "open x1.25": {"OPEN_PCT": scaled(C.OPEN_PCT, 1.25)},
    "flat x0.75": {"CALL_OPEN_PCT": scaled(C.CALL_OPEN_PCT, 0.75)},
    "flat x1.3": {"CALL_OPEN_PCT": scaled(C.CALL_OPEN_PCT, 1.3)},
    "3bet x1.5": {"THREEBET_PCT": scaled(C.THREEBET_PCT, 1.5)},
    "margin +.03": {"CALL_EV_MARGIN": 0.03},
    "margin -.02": {"CALL_EV_MARGIN": -0.02},
    "oop .85": {"REALIZE_OOP": 0.85},
    "oop 1.0": {"REALIZE_OOP": 1.0},
    "reverse 0": {"DRAW_REVERSE_ODDS": 0.0},
    "reverse .08": {"DRAW_REVERSE_ODDS": 0.08},
    "multiway .4": {"MULTIWAY_EQ_EXP": 0.4},
    "multiway 1.0": {"MULTIWAY_EQ_EXP": 1.0},
    "monster .55": {"BUCKET_MONSTER": 0.55},
    "monster .65": {"BUCKET_MONSTER": 0.65},
    "strong .44": {"BUCKET_STRONG": 0.44},
    "strong .52": {"BUCKET_STRONG": 0.52},
    "bluff 1.0": {"BLUFF_FACTOR": 1.0},
    "bluff 2.0": {"BLUFF_FACTOR": 2.0},
    "cbet .45": {"CBET_FREQ_HEADSUP": 0.45},
    "cbet .75": {"CBET_FREQ_HEADSUP": 0.75},
    "no trap": {"TRAP_FREQ": {"flop": 0.0, "turn": 0.0, "river": 0.0}},
    "spr .6": {"COMMIT_SPR": 0.6},
    "spr 1.5": {"COMMIT_SPR": 1.5},
    "shove 7bb": {"SHOVE_STACK_BB": 7},
    "shove 14bb": {"SHOVE_STACK_BB": 14},
    "prior .15": {"BLUFF_PRIOR": 0.15},
    "prior .35": {"BLUFF_PRIOR": 0.35},
}


def play(overrides, field, seed):
    rng = random.Random(seed)
    names = list(FIELDS[field])
    picks = rng.choices(names, weights=[FIELDS[field][n] for n in names], k=7)
    seats = [(f"{n}#{i}", REGISTRY[n](seed * 10 + i)) for i, n in enumerate(picks)]
    pos = rng.randrange(8)
    seats.insert(pos, ("CANDIDATE", PokerBot(seed=seed * 10 + 9, overrides=overrides)))
    return Table(seats, rng).play()[pos]


def _work(task):
    name, overrides, field, start, stop = task
    return name, field, start, [play(overrides, field, s) for s in range(start, stop)]


def evaluate(cands, tables, first_seed, workers):
    """{name: overrides} -> {name: per-table points, fields concatenated in a fixed order}."""
    end = first_seed + tables
    tasks = [(n, ov, f, s, min(s + CHUNK, end)) for n, ov in cands.items() for f in SEARCH_FIELDS
             for s in range(first_seed, end, CHUNK)]
    parts = {}
    with Pool(workers) as pool:
        for name, field, start, res in pool.imap_unordered(_work, tasks):
            parts[(name, field, start)] = res
    return {n: [x for f in SEARCH_FIELDS for s in range(first_seed, end, CHUNK) for x in parts[(n, f, s)]]
            for n in cands}


def compare(results, ref):
    out = []
    for n, v in results.items():
        if n == ref:
            continue
        d = [a - b for a, b in zip(v, results[ref])]
        m, se = statistics.mean(d), statistics.stdev(d) / len(d) ** 0.5
        out.append((m / se if se else 0.0, m, se, n))
    return sorted(out, reverse=True)


def show(rows):
    for t, m, se, n in rows:
        print(f"  {n:<44}{m:+6.2f} ± {se:.2f}  t={t:+5.1f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    t0 = time.time()

    print("STAGE 1: one setting at a time")
    res = evaluate({"DEFAULT": None, **VARIANTS}, int(3000 * args.scale), 10_000_001, args.workers)
    rows = compare(res, "DEFAULT")
    show(rows)
    good = [n for t, _, _, n in rows if t >= 1.5][:6]
    print(f"carried forward: {good}  ({time.time() - t0:.0f}s)")

    print("\nSTAGE 2: combinations")
    combos = {"DEFAULT": None}
    for k in range(1, len(good) + 1):
        for names in itertools.combinations(good, k):
            merged = {}
            for n in names:
                merged.update(VARIANTS[n])
            combos[" + ".join(names)] = merged
    res = evaluate(combos, int(3000 * args.scale), 20_000_001, args.workers)
    rows = compare(res, "DEFAULT")
    show(rows[:10])

    print(f"\nSTAGE 3: confirm on fresh tables  ({time.time() - t0:.0f}s)")
    final = {"DEFAULT": None, **{n: combos[n] for _, _, _, n in rows[:4]}}
    show(compare(evaluate(final, int(8000 * args.scale), 30_000_001, args.workers), "DEFAULT"))
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
