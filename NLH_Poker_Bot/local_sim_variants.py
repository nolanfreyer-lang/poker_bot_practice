"""Head-to-head test of ExploitBot setting variants against the TEMPLATE PokerBot.

    python3 local_sim_variants.py                      # 10 runs x 1000 tables (seeds 1-10)
    python3 local_sim_variants.py --runs 3 --tables 500 --first-seed 20

All 8 bots sit at every table (random seats each table), so each variant plays
against all the others. Uses the dealer from local_sim.py.
"""
import argparse
import random
import statistics
import time

from bot import PokerBot
from bot_exploit import ExploitBot
from local_sim import Table

# =====================================================================
# VARIANT SETTINGS -- edit these to change what each bot means
# =====================================================================
TRAP_MONSTER = {"TRAP_FREQ": 0.3}                          # slow-play MONSTER 30% (flop + turn)
TRAP_MONSTER_STRONG = {"TRAP_FREQ": 0.3, "TRAP_FREQ_STRONG": 0.2}
NO_TRAP = {"TRAP_FREQ": 0.0, "TRAP_FREQ_STRONG": 0.0}      # MONSTER and STRONG always bet
RIVER_ON = {"TRAP_ON_RIVER": True}
BLUFF2 = {"BLUFF_FACTOR": 2.0}

VARIANTS = [
    # (name, overrides for ExploitBot) -- None = the TEMPLATE PokerBot
    ("TEMPLATE", None),
    ("BLUFF2", BLUFF2),
    ("TRAP_M", TRAP_MONSTER),
    ("TRAP_MS", TRAP_MONSTER_STRONG),
    ("BLUFF2_TRAP", {**BLUFF2, **TRAP_MONSTER}),
    ("NOTRAP_B1", {**NO_TRAP, "BLUFF_FACTOR": 1.0}),
    ("RIVER_TRAP", {**TRAP_MONSTER, **RIVER_ON}),
    ("RIVER_TRAP_B2", {**TRAP_MONSTER, **RIVER_ON, **BLUFF2}),
]


def lineup(seed):
    return [(name, PokerBot(seed=seed + i) if ov is None else ExploitBot(seed=seed + i, overrides=ov))
            for i, (name, ov) in enumerate(VARIANTS)]


def run(seed, tables):
    """One run of `tables` tables -> {name: avg points}, {name: [decision seconds]}."""
    totals = {name: 0.0 for name, _ in VARIANTS}
    times = {name: [] for name, _ in VARIANTS}
    for t in range(tables):
        rng = random.Random(seed * 1000 + t)
        seats = lineup(seed * 1000 + t)       # fresh bots: state resets per table
        rng.shuffle(seats)
        table = Table(seats, rng)
        pts = table.play()
        for s, name in enumerate(table.names):
            totals[name] += pts[s]
            times[name] += table.decision_times[s]
    return {n: v / tables for n, v in totals.items()}, times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--tables", type=int, default=1000)
    ap.add_argument("--first-seed", type=int, default=1)
    args = ap.parse_args()

    names = [n for n, _ in VARIANTS]
    results = {n: [] for n in names}
    all_times = {n: [] for n in names}
    t0 = time.time()
    seeds = range(args.first_seed, args.first_seed + args.runs)
    for seed in seeds:
        avg, times = run(seed, args.tables)
        for n in names:
            results[n].append(avg[n])
            all_times[n] += times[n]
        best = max(avg, key=avg.get)
        print(f"run seed {seed:>3}: " + "  ".join(f"{n} {avg[n]:.2f}" for n in names) + f"   best: {best}",
              flush=True)

    print(f"\n{args.runs} runs x {args.tables} tables in {time.time() - t0:.0f}s")
    print(f"{'BOT':<14}{'MEAN':>7}{'± SE':>7}{'MIN':>7}{'MAX':>7}{'WINS':>6}{'MAX ms':>8}{'>1ms':>6}")
    wins = {n: 0 for n in names}
    for i in range(args.runs):
        wins[max(names, key=lambda n: results[n][i])] += 1
    for n in sorted(names, key=lambda n: -statistics.mean(results[n])):
        r = results[n]
        se = statistics.stdev(r) / len(r) ** 0.5 if len(r) > 1 else 0.0
        ts = all_times[n] or [0]
        print(f"{n:<14}{statistics.mean(r):7.2f}{se:7.2f}{min(r):7.2f}{max(r):7.2f}{wins[n]:6d}"
              f"{1000 * max(ts):8.3f}{sum(1 for x in ts if x > 0.001):6d}")


if __name__ == "__main__":
    main()
