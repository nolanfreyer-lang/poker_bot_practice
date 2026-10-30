"""Robustness test: does OptimalBot stay ahead across very different opponent fields?

    python3 local_sim_robust.py                  # 5 fields x 4 candidates x 8000 tables (~4-5 min)
    python3 local_sim_robust.py --tables 2000    # quicker, noisier

Each table seats one candidate + 7 opponents drawn from the field. Every
candidate plays the same tables (same opponents, seats and cards), so the
comparisons are paired table by table.
"""
import argparse
import random
import statistics
import time
from multiprocessing import Pool

from bot import PokerBot
from bot_exploit import ExploitBot
from optimalbot import OptimalBot
from updated_optimal import UpdatedOptimalBot
from local_sim import Table, CallingStation, Maniac
from local_sim_search import CheckFoldBot
from opponent_bots import NitBot, LagBot, PushFoldBot, ChenBot, TrapperBot, GtoApproxBot

# =====================================================================
# CANDIDATES and FIELDS -- edit freely
# =====================================================================
# Bluff levels are set explicitly: bot_exploit.py's default BLUFF_FACTOR is 2.0.
CANDIDATES = {
    "UPDATED_OPTIMAL": lambda seed: UpdatedOptimalBot(seed=seed),
    "OPTIMAL": lambda seed: OptimalBot(seed=seed),                                   # BF 1.5
    "EXPLOIT_BF2": lambda seed: ExploitBot(seed=seed, overrides={"BLUFF_FACTOR": 2.0}),
    "TEMPLATE": lambda seed: PokerBot(seed=seed),
}
REFERENCE = "OPTIMAL"

REGISTRY = {
    "TEMPLATE": lambda seed: PokerBot(seed=seed),
    "EXPLOIT": lambda seed: ExploitBot(seed=seed),
    "OPTIMAL": lambda seed: OptimalBot(seed=seed),
    "UPDATED_OPTIMAL": lambda seed: UpdatedOptimalBot(seed=seed),
    "GTO": lambda seed: GtoApproxBot(seed),
    "CHEN": lambda seed: ChenBot(),
    "PUSHFOLD": lambda seed: PushFoldBot(),
    "LAG": lambda seed: LagBot(seed),
    "NIT": lambda seed: NitBot(),
    "TRAPPER": lambda seed: TrapperBot(seed),
    "STATION": lambda seed: CallingStation(),
    "MANIAC": lambda seed: Maniac(),
    "CHECKFOLD": lambda seed: CheckFoldBot(),
}

FIELDS = {
    "likely": {"TEMPLATE": 3, "GTO": 2, "CHEN": 2, "PUSHFOLD": 2, "LAG": 2, "NIT": 1, "TRAPPER": 1,
               "EXPLOIT": 1, "STATION": 1, "MANIAC": 1, "CHECKFOLD": 1},
    "tough": {"TEMPLATE": 2, "EXPLOIT": 2, "OPTIMAL": 1, "GTO": 2, "LAG": 2},
    "loose": {"STATION": 2, "TRAPPER": 2, "LAG": 2, "MANIAC": 1, "TEMPLATE": 1},
    "tight": {"NIT": 2, "PUSHFOLD": 2, "CHEN": 2, "GTO": 1, "CHECKFOLD": 1},
    "mirror": {"EXPLOIT": 3, "OPTIMAL": 3, "TEMPLATE": 2},
    "updated_mirror": {"UPDATED_OPTIMAL": 3, "OPTIMAL": 3, "TEMPLATE": 2},
}
CHUNK = 250


# =====================================================================
def play_table(cand, field, seed):
    rng = random.Random(seed)
    names = list(FIELDS[field])
    picks = rng.choices(names, weights=[FIELDS[field][n] for n in names], k=7)
    seats = [(f"{n}#{i}", REGISTRY[n](seed * 10 + i)) for i, n in enumerate(picks)]
    pos = rng.randrange(8)
    seats.insert(pos, ("CANDIDATE", CANDIDATES[cand](seed * 10 + 9)))
    table = Table(seats, rng)
    return table.play()[pos], sum(1 for x in table.decision_times[pos] if x > 0.001)


def _work(task):
    cand, field, start, stop = task
    return cand, field, start, [play_table(cand, field, s) for s in range(start, stop)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=int, default=8000)
    ap.add_argument("--first-seed", type=int, default=5_000_001)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    end = args.first_seed + args.tables
    tasks = [(c, f, s, min(s + CHUNK, end)) for f in FIELDS for c in CANDIDATES
             for s in range(args.first_seed, end, CHUNK)]
    chunks = {}
    t0 = time.time()
    with Pool(args.workers) as pool:
        for done, (c, f, start, res) in enumerate(pool.imap_unordered(_work, tasks), 1):
            chunks.setdefault((c, f), {})[start] = res
            if done % max(1, len(tasks) // 10) == 0:
                print(f"  {done}/{len(tasks)} chunks, {time.time() - t0:.0f}s", flush=True)

    pts = {k: [p for s in sorted(v) for p, _ in v[s]] for k, v in chunks.items()}
    slow = {k: sum(x for s in v for _, x in v[s]) for k, v in chunks.items()}
    w = max(len(c) for c in CANDIDATES) + 2
    print(f"\n{args.tables} tables per candidate per field, {time.time() - t0:.0f}s")
    summary = {c: [] for c in CANDIDATES}
    for f in FIELDS:
        print(f"\nFIELD {f}: " + ", ".join(f"{n} x{wt}" for n, wt in FIELDS[f].items()))
        print(f"  {'CANDIDATE':<{w}}{'POINTS':>8}   VS {REFERENCE} (paired)        t   >1ms")
        ref = pts[(REFERENCE, f)]
        for c in sorted(CANDIDATES, key=lambda c: -statistics.mean(pts[(c, f)])):
            v = pts[(c, f)]
            m = statistics.mean(v)
            if c == REFERENCE:
                vs, t = "reference", 0.0
            else:
                d = [a - b for a, b in zip(v, ref)]
                dm, se = statistics.mean(d), statistics.stdev(d) / len(d) ** 0.5
                vs, t = f"{dm:+.2f} ± {se:.2f}", dm / se
                summary[c].append((f, dm, t))
            print(f"  {c:<{w}}{m:8.2f}   {vs:<20}{t:+6.1f}  {slow[(c, f)]:4d}")

    print(f"\nSUMMARY: {REFERENCE} minus each candidate, per field (+ = {REFERENCE} better; t>=3 real)")
    print(f"  {'':<{w}}" + "".join(f"{f:>16}" for f in FIELDS))
    for c, rows in summary.items():
        if rows:
            print(f"  {c:<{w}}" + "".join(f"{-dm:+8.2f} (t{-t:+5.1f})" for _, dm, t in rows))


if __name__ == "__main__":
    main()
