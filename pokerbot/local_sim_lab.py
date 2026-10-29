"""Test lab: pit ExploitBot variants against any mix of other bots.

    python3 local_sim_lab.py                       # uses the SETUP block below
    python3 local_sim_lab.py --runs 10 --tables 1000 --first-seed 1
    python3 local_sim_lab.py --runs 1 --tables 200  # quick check

Edit only the SETUP block:
  * MY_BOTS     -- the bots you are testing: ExploitBot variants (or TEMPLATE)
  * OPPONENTS   -- who fills the remaining seats, and how often
  * BASELINE    -- each MY_BOT is compared against this one, run by run
Every table seats all MY_BOTS plus opponents drawn at random to fill 8 seats,
like the event's random table assignment. All bots at a table see the same
cards, so the comparisons against BASELINE cancel most of the luck.
"""
import argparse
import random
import statistics
import time

from bot import PokerBot
from bot_exploit import ExploitBot
from local_sim import Table, CallingStation, RandomBot, Maniac
from opponent_bots import NitBot, LagBot, PushFoldBot, ChenBot, TrapperBot, GtoApproxBot
from optimalbot import OPTIMAL_OVERRIDES   # single source of the tuned settings

# =====================================================================
# SETUP -- edit this block
# =====================================================================
# (name, overrides) -- overrides are ExploitBot settings (see EXPLOIT_CONFIG in
# bot_exploit.py). Use None for the original PokerBot (config.py only).
TRAP_MONSTER = {"TRAP_FREQ": 0.3}                          # slow-play MONSTER 30% (flop + turn)
TRAP_MONSTER_STRONG = {"TRAP_FREQ": 0.3, "TRAP_FREQ_STRONG": 0.2}
NO_TRAP = {"TRAP_FREQ": 0.0, "TRAP_FREQ_STRONG": 0.0}      # MONSTER and STRONG always bet
RIVER_ON = {"TRAP_ON_RIVER": True}
BLUFF2 = {"BLUFF_FACTOR": 2.0}

MY_BOTS = [
    
    ("BLUFF2", BLUFF2),
    ("OPTIMAL", OPTIMAL_OVERRIDES),
   
]

# Opponent pool: name -> weight. Empty seats are filled by random draws from
# this pool each table (weight 2 = drawn twice as often as weight 1).
# Set a weight to 0 to leave a bot out.
OPPONENTS = {
    "LAG": 0,
    "GTO": 2,
    "NIT": 1,
    "CHEN": 1,
    "PUSHFOLD": 1,
    "TRAPPER": 1,
    "STATION": 0,
    "RANDOM": 0,
    "MANIAC": 0,
    "EXPLOIT": 2
}

BASELINE = "BLUFF2"     # must be one of MY_BOTS names (or None to skip comparisons)
SEATS = 8

# Every opponent the pool can use. Add your own bots here: name -> factory(seed).
REGISTRY = {
    "LAG": lambda seed: LagBot(seed),
    "GTO": lambda seed: GtoApproxBot(seed),
    "NIT": lambda seed: NitBot(),
    "CHEN": lambda seed: ChenBot(),
    "PUSHFOLD": lambda seed: PushFoldBot(),
    "TRAPPER": lambda seed: TrapperBot(seed),
    "STATION": lambda seed: CallingStation(),
    "RANDOM": lambda seed: RandomBot(seed),
    "MANIAC": lambda seed: Maniac(),
    "TEMPLATE": lambda seed: PokerBot(seed=seed),
    "EXPLOIT": lambda seed: ExploitBot(seed=seed),
}
# =====================================================================


def make_my_bot(overrides, seed):
    return PokerBot(seed=seed) if overrides is None else ExploitBot(seed=seed, overrides=overrides)


def build_table(seed):
    rng = random.Random(seed)
    seats = [(name, make_my_bot(ov, seed + i)) for i, (name, ov) in enumerate(MY_BOTS)]
    pool = [(n, w) for n, w in OPPONENTS.items() if w > 0]
    for i in range(SEATS - len(seats)):
        name = rng.choices([n for n, _ in pool], weights=[w for _, w in pool])[0]
        seats.append((name, REGISTRY[name](seed + 100 + i)))
    rng.shuffle(seats)
    return Table(seats, rng)


def run(seed, tables):
    """-> {name: avg points per seat}, {name: seats played}, slow decisions."""
    pts_sum, seat_count, slow = {}, {}, 0
    for t in range(tables):
        table = build_table(seed * 100_000 + t)
        pts = table.play()
        for s, name in enumerate(table.names):
            pts_sum[name] = pts_sum.get(name, 0.0) + pts[s]
            seat_count[name] = seat_count.get(name, 0) + 1
        slow += sum(1 for ts in table.decision_times for x in ts if x > 0.001)
    return {n: pts_sum[n] / seat_count[n] for n in pts_sum}, seat_count, slow


def check_setup():
    names = [n for n, _ in MY_BOTS]
    assert len(names) == len(set(names)), "MY_BOTS names must be unique"
    assert len(MY_BOTS) <= SEATS, f"at most {SEATS} MY_BOTS"
    assert BASELINE is None or BASELINE in names, "BASELINE must be one of MY_BOTS"
    missing = [n for n, w in OPPONENTS.items() if w > 0 and n not in REGISTRY]
    assert not missing, f"not in REGISTRY: {missing}"
    if len(MY_BOTS) < SEATS:
        assert any(w > 0 for w in OPPONENTS.values()), "OPPONENTS needs at least one weight > 0"
    clash = set(names) & {n for n, w in OPPONENTS.items() if w > 0}
    assert not clash, f"rename MY_BOTS that share a name with an opponent: {clash}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--tables", type=int, default=1000)
    ap.add_argument("--first-seed", type=int, default=1)
    args = ap.parse_args()
    check_setup()

    mine = [n for n, _ in MY_BOTS]
    everyone = mine + [n for n, w in OPPONENTS.items() if w > 0]
    w = max(len(n) for n in everyone) + 2          # name column width
    runs, seats_total, slow = [], {}, 0
    t0 = time.time()

    # ---- per-run results: every bot at the tables
    print(f"PER RUN (average points per seat; * = best that run)")
    print(f"{'SEED':<6}" + "".join(f"{n:>{w}}" for n in everyone))
    for seed in range(args.first_seed, args.first_seed + args.runs):
        avg, seats, s = run(seed, args.tables)
        runs.append(avg)
        slow += s
        for n, c in seats.items():
            seats_total[n] = seats_total.get(n, 0) + c
        best = max(avg, key=avg.get)
        cells = [(f"{avg[n]:.2f}" + ("*" if n == best else " ")) if n in avg else "-" for n in everyone]
        print(f"{seed:<6}" + "".join(f"{c:>{w}}" for c in cells), flush=True)

    # ---- overall results: every bot
    def stats(vals):
        m = statistics.mean(vals)
        return m, (statistics.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else 0.0)

    def vs_baseline(n):
        if not BASELINE or n == BASELINE:
            return "baseline" if n == BASELINE else ""
        d = [r[n] - r[BASELINE] for r in runs if n in r]
        if len(d) < 2:
            return ""
        m, se = stats(d)
        t = m / se if se else 0.0
        verdict = "real" if abs(t) >= 3 else "likely" if abs(t) >= 2 else "noise"
        return f"{m:+6.2f} ± {se:.2f}  t={t:+5.1f}  {verdict}"

    wins = {n: 0 for n in everyone}
    for r in runs:
        wins[max(r, key=r.get)] += 1
    order = sorted(everyone, key=lambda n: -stats([r[n] for r in runs if n in r])[0]
                   if any(n in r for r in runs) else 0)

    print(f"\nOVERALL: {args.runs} runs x {args.tables} tables in {time.time() - t0:.0f}s"
          f"   decisions over 1 ms: {slow}")
    print(f"{'#':<3}{'BOT':<{w}}{'MEAN':>7}{'± SE':>7}{'MIN':>7}{'MAX':>7}{'WINS':>6}{'SEATS':>8}   "
          f"VS {BASELINE or '-'} (paired; t>=3 real, 2-3 likely, <2 noise)")
    for i, n in enumerate(order, 1):
        vals = [r[n] for r in runs if n in r]
        if not vals:
            continue
        m, se = stats(vals)
        tag = "*" if n in mine else " "
        print(f"{i:<3}{n + tag:<{w}}{m:7.2f}{se:7.2f}{min(vals):7.2f}{max(vals):7.2f}{wins[n]:6d}"
              f"{seats_total.get(n, 0):8d}   {vs_baseline(n)}")
    print("* = one of MY_BOTS. Opponents are averaged over all their copies.")
    if len(runs) < 5:
        print("! fewer than 5 runs: SE and t-values are unreliable, use --runs 10 or more")


if __name__ == "__main__":
    main()
