"""Robustness test: does the PLO4 bot stay ahead across very different opponent fields?

    python3 local_sim_robust.py                  # every field x candidate, 4000 tables
    python3 local_sim_robust.py --tables 1000    # quicker, noisier
    python3 local_sim_robust.py --candidates PLO4,BASELINE --fields likely,tough

Each table seats one candidate + 7 opponents drawn from the field. Every
candidate plays the same tables (same opponents, seats and cards), so the
comparisons are paired table by table. CANDIDATES holds the ablations: PLO4 with
one NLH lesson removed at a time.
"""
import argparse
import random
import statistics
import time
from multiprocessing import Pool

from bot import PokerBot
from local_sim import BASELINE_OVERRIDES, Table
from opponent_bots import (CallingStation, CheckFoldBot, ChenBot, GtoApproxBot, LagBot, Maniac, NitBot,
                           NlhPortBot, PushFoldBot, RandomBot, TrapperBot)

# =====================================================================
# CANDIDATES and FIELDS -- edit freely
# =====================================================================
CANDIDATES = {
    "PLO4": None,
    "BASELINE": BASELINE_OVERRIDES,
    "NO_TRACKING": {"TRACK_OPPONENTS": False},
    "NO_ICM": {"USE_ICM": False},
    "NO_PREFLOP_EV": {"PREFLOP_EV": False},
    "BLUFF_1.0": {"BLUFF_FACTOR": 1.0},
}
REFERENCE = "PLO4"

REGISTRY = {
    "PLO4": lambda seed: PokerBot(seed=seed),
    "BASELINE": lambda seed: PokerBot(seed=seed, overrides=BASELINE_OVERRIDES),
    "GTO": lambda seed: GtoApproxBot(seed),
    "CHEN": lambda seed: ChenBot(),
    "PUSHFOLD": lambda seed: PushFoldBot(),
    "LAG": lambda seed: LagBot(seed),
    "NIT": lambda seed: NitBot(),
    "TRAPPER": lambda seed: TrapperBot(seed),
    "NLHPORT": lambda seed: NlhPortBot(seed),
    "STATION": lambda seed: CallingStation(),
    "RANDOM": lambda seed: RandomBot(seed),
    "MANIAC": lambda seed: Maniac(),
    "CHECKFOLD": lambda seed: CheckFoldBot(),
}

FIELDS = {
    "likely": {"BASELINE": 3, "GTO": 2, "CHEN": 2, "NLHPORT": 2, "PUSHFOLD": 2, "LAG": 2, "NIT": 1,
               "TRAPPER": 1, "STATION": 1, "MANIAC": 1, "CHECKFOLD": 1},
    "tough": {"BASELINE": 3, "PLO4": 2, "GTO": 2, "LAG": 2},
    "loose": {"STATION": 2, "TRAPPER": 2, "LAG": 2, "NLHPORT": 2, "MANIAC": 1, "RANDOM": 1},
    "tight": {"NIT": 2, "PUSHFOLD": 2, "CHEN": 2, "GTO": 1, "CHECKFOLD": 1},
    "mirror": {"PLO4": 5, "BASELINE": 3},
}
CHUNK = 100


# =====================================================================
def play_table(cand, field, seed):
    rng = random.Random(seed)
    names = list(FIELDS[field])
    picks = rng.choices(names, weights=[FIELDS[field][n] for n in names], k=7)
    seats = [(f"{n}#{i}", REGISTRY[n](seed * 10 + i)) for i, n in enumerate(picks)]
    pos = rng.randrange(8)
    seats.insert(pos, ("CANDIDATE", PokerBot(seed=seed * 10 + 9, overrides=CANDIDATES[cand])))
    table = Table(seats, rng)
    pts = table.play()[pos]
    ts = table.decision_times[pos]
    return pts, sum(1 for x in ts if x > 0.001), max(ts, default=0.0), table.errors


def _work(task):
    cand, field, start, stop = task
    return cand, field, start, [play_table(cand, field, s) for s in range(start, stop)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=int, default=4000)
    ap.add_argument("--first-seed", type=int, default=5_000_001)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--candidates", default=",".join(CANDIDATES))
    ap.add_argument("--fields", default=",".join(FIELDS))
    args = ap.parse_args()
    cands, fields = args.candidates.split(","), args.fields.split(",")
    end = args.first_seed + args.tables
    tasks = [(c, f, s, min(s + CHUNK, end)) for f in fields for c in cands
             for s in range(args.first_seed, end, CHUNK)]
    chunks = {}
    t0 = time.time()
    with Pool(args.workers) as pool:
        for done, (c, f, start, res) in enumerate(pool.imap_unordered(_work, tasks), 1):
            chunks.setdefault((c, f), {})[start] = res
            if done % max(1, len(tasks) // 10) == 0:
                print(f"  {done}/{len(tasks)} chunks, {time.time() - t0:.0f}s", flush=True)

    rows = {k: [r for s in sorted(v) for r in v[s]] for k, v in chunks.items()}
    pts = {k: [r[0] for r in v] for k, v in rows.items()}
    w = max(len(c) for c in cands) + 2
    print(f"\n{args.tables} tables per candidate per field, {time.time() - t0:.0f}s")
    for f in fields:
        print(f"\nFIELD {f}: " + ", ".join(f"{n} x{wt}" for n, wt in FIELDS[f].items()))
        print(f"  {'CANDIDATE':<{w}}{'POINTS':>8}   VS {REFERENCE} (paired)        t   >1ms  max ms  errors")
        ref = pts.get((REFERENCE, f))
        for c in sorted(cands, key=lambda c: -statistics.mean(pts[(c, f)])):
            v = pts[(c, f)]
            if c == REFERENCE or ref is None:
                vs, t = "reference", 0.0
            else:
                d = [a - b for a, b in zip(v, ref)]
                dm, se = statistics.mean(d), statistics.stdev(d) / len(d) ** 0.5
                vs, t = f"{dm:+.2f} ± {se:.2f}", dm / se if se else 0.0
            r = rows[(c, f)]
            print(f"  {c:<{w}}{statistics.mean(v):8.2f}   {vs:<20}{t:+6.1f}  {sum(x[1] for x in r):4d}"
                  f"  {1000 * max(x[2] for x in r):6.2f}  {sum(x[3] for x in r):6d}")
    print("\nAn even share of the points is "
          f"{sum(__import__('config').PAYOUT_POINTS) / 8:.2f} per seat. t>=3 real, 2-3 likely, <2 noise.")


if __name__ == "__main__":
    main()
