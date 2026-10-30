"""Search for the best ExploitBot settings against a likely competition field.

    python3 local_sim_search.py                    # full 3-stage search (~20-30 min)
    python3 local_sim_search.py --scale 0.2        # quick, noisier version
    python3 local_sim_search.py --workers 8

Every candidate plays the SAME tables (same opponents, seats and cards) with one
seat for the candidate and seven opponents drawn from FIELD, so differences
between candidates are measured table by table (paired), which removes most luck.

  Stage 1  screen: each setting changed alone vs ExploitBot defaults
  Stage 2  combine: every combination of the settings that helped in stage 1
  Stage 3  confirm: the best few plus baselines on fresh tables, many more of them
"""
import argparse
import itertools
import random
import statistics
import time
from multiprocessing import Pool

from bot import Action, PokerBot
from bot_exploit import ExploitBot
from local_sim import Table, CallingStation, Maniac
from opponent_bots import NitBot, LagBot, PushFoldBot, ChenBot, TrapperBot, GtoApproxBot


# =====================================================================
# LIKELY FIELD -- opponent mix (weights) the candidates face
# =====================================================================
class CheckFoldBot:
    """Stand-in for bots that crash or exceed 1 ms: the engine checks/folds for them."""
    def act(self, s):
        return Action("check") if s.to_call == 0 else Action("fold")


FIELD = {
    "TEMPLATE": 3,    # AI-built range bots similar to ours (teams use Codex + a starter kit)
    "GTO": 2,         # solver-style mixed strategies
    "CHEN": 2,        # simple formula / pot-odds bots
    "PUSHFOLD": 2,    # tournament push/fold bots (rising blinds reward this)
    "LAG": 2,         # aggressive bots
    "NIT": 1,         # very tight bots
    "TRAPPER": 1,     # passive callers
    "EXPLOIT": 1,     # other teams' adaptive bots
    "STATION": 1,     # naive always-call bots
    "MANIAC": 1,      # naive always-raise bots
    "CHECKFOLD": 1,   # crashed / timed-out bots
}
REGISTRY = {
    "TEMPLATE": lambda seed: PokerBot(seed=seed),
    "GTO": lambda seed: GtoApproxBot(seed),
    "CHEN": lambda seed: ChenBot(),
    "PUSHFOLD": lambda seed: PushFoldBot(),
    "LAG": lambda seed: LagBot(seed),
    "NIT": lambda seed: NitBot(),
    "TRAPPER": lambda seed: TrapperBot(seed),
    "EXPLOIT": lambda seed: ExploitBot(seed=seed),
    "STATION": lambda seed: CallingStation(),
    "MANIAC": lambda seed: Maniac(),
    "CHECKFOLD": lambda seed: CheckFoldBot(),
}

# =====================================================================
# SEARCH SPACE -- levels to try for each ExploitBot setting (default first is
# the ExploitBot default). Left out as obviously losing: BLUFF_FACTOR 0, and
# trapping STRONG hands at 0.2+ / forcing STRONG to always bet (both lost before).
# =====================================================================
LEVELS = {
    "BLUFF_FACTOR": [0.5, 1.5, 2.0, 3.0],
    "TRAP_FREQ": [0.3],
    "TRAP_FREQ_STRONG": [0.1],
    "TRAP_ON_RIVER": [True],
    "TRAP_STREAK_CORRELATION": [0.0, 0.3],
    "TIGHT_VPIP": [0.10, 0.20],
    "STEAL_SHIFT_VS_TIGHT": [1, 3],
    "RESPECT_TIGHT_RAISES": [False],
    "BLUFF_MULT_VS_TIGHT": [1.0, 2.5],
    "CBET_BONUS_VS_TIGHT": [0.0, 0.35],
    "CALL_PRICE_MULT_VS_TIGHT": [0.6, 1.0],
    "THIN_VALUE_MULT_VS_TIGHT": [0.4, 1.0],
}
STAGE1_TABLES = 6000
STAGE2_TABLES = 10000
STAGE3_TABLES = 30000
MAX_COMBO_FACTORS = 5      # stage 2 tries all 2^k combos of the k best settings
KEEP_IF_T_ABOVE = 1.0      # a stage-1 setting must beat defaults by this many SE
FINALISTS = 5
CHUNK = 250


# =====================================================================
def play_table(overrides, seed):
    """One table: the candidate (overrides=None -> PokerBot) + 7 field bots. -> (points, slow decisions)."""
    rng = random.Random(seed)
    names = list(FIELD)
    picks = rng.choices(names, weights=[FIELD[n] for n in names], k=7)
    seats = [(f"{n}#{i}", REGISTRY[n](seed * 10 + i)) for i, n in enumerate(picks)]
    pos = rng.randrange(8)
    cand = PokerBot(seed=seed * 10 + 9) if overrides is None else \
        ExploitBot(seed=seed * 10 + 9, overrides=overrides)
    seats.insert(pos, ("CANDIDATE", cand))
    table = Table(seats, rng)
    pts = table.play()
    return pts[pos], sum(1 for x in table.decision_times[pos] if x > 0.001)


def _work(task):
    idx, overrides, start, stop = task
    return idx, start, [play_table(overrides, s) for s in range(start, stop)]


def evaluate(pool, candidates, first_seed, tables, label):
    """candidates: [(name, overrides)] -> {name: [points per table, in seed order]}, {name: slow count}."""
    t0 = time.time()
    end = first_seed + tables
    tasks = [(i, ov, s, min(s + CHUNK, end))
             for i, (_, ov) in enumerate(candidates) for s in range(first_seed, end, CHUNK)]
    chunks = {i: {} for i in range(len(candidates))}
    for done, (idx, start, res) in enumerate(pool.imap_unordered(_work, tasks), 1):
        chunks[idx][start] = res
        if done % max(1, len(tasks) // 10) == 0:
            print(f"  [{label}] {done}/{len(tasks)} chunks, {time.time() - t0:.0f}s", flush=True)
    # concatenate chunks in seed order so table k is the same table for every candidate
    pts, slow = {}, {}
    for i, (name, _) in enumerate(candidates):
        rows = [r for start in sorted(chunks[i]) for r in chunks[i][start]]
        pts[name] = [p for p, _ in rows]
        slow[name] = sum(x for _, x in rows)
    return pts, slow


def summarize(results, ref):
    """Paired table-by-table comparison of every candidate against `ref`."""
    rows = []
    base = results[ref]
    for name, vals in results.items():
        mean = statistics.mean(vals)
        d = [a - b for a, b in zip(vals, base)]
        dm = statistics.mean(d)
        dse = statistics.stdev(d) / len(d) ** 0.5 if name != ref else 0.0
        rows.append((name, mean, dm, dse, dm / dse if dse else 0.0))
    rows.sort(key=lambda r: -r[1])
    return rows


def print_rows(rows, ref, slow):
    w = max(len(r[0]) for r in rows) + 2
    print(f"  {'CANDIDATE':<{w}}{'POINTS':>8}   VS {ref:<10}{'t':>7}  >1ms")
    for name, mean, dm, dse, t in rows:
        vs = "reference" if name == ref else f"{dm:+.2f} ± {dse:.2f}"
        print(f"  {name:<{w}}{mean:8.2f}   {vs:<16}{t:+6.1f}  {slow.get(name, 0):4d}")


def label_of(ov):
    if ov is None:
        return "TEMPLATE"
    if not ov:
        return "EXPLOIT_DEFAULT"
    short = {"BLUFF_FACTOR": "BF", "TRAP_FREQ": "TRAP", "TRAP_FREQ_STRONG": "TRAPS",
             "TRAP_ON_RIVER": "RIVER", "TRAP_STREAK_CORRELATION": "STREAK", "TIGHT_VPIP": "TVPIP",
             "STEAL_SHIFT_VS_TIGHT": "STEAL", "RESPECT_TIGHT_RAISES": "RESPECT",
             "BLUFF_MULT_VS_TIGHT": "BMT", "CBET_BONUS_VS_TIGHT": "CBT",
             "CALL_PRICE_MULT_VS_TIGHT": "CPT", "THIN_VALUE_MULT_VS_TIGHT": "THIN"}
    return "+".join(f"{short.get(k, k)}={v}" for k, v in ov.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply all table counts")
    args = ap.parse_args()
    s1, s2, s3 = (max(CHUNK, int(n * args.scale) // CHUNK * CHUNK)
                  for n in (STAGE1_TABLES, STAGE2_TABLES, STAGE3_TABLES))
    t_all = time.time()
    with Pool(args.workers) as pool:
        # ---------------- stage 1
        cands = [("EXPLOIT_DEFAULT", {}), ("TEMPLATE", None)]
        cands += [(label_of({k: v}), {k: v}) for k, levels in LEVELS.items() for v in levels]
        print(f"STAGE 1: {len(cands)} candidates x {s1} tables", flush=True)
        res, slow = evaluate(pool, cands, 1, s1, "stage 1")
        rows = summarize(res, "EXPLOIT_DEFAULT")
        print_rows(rows, "EXPLOIT_DEFAULT", slow)

        by_name = {r[0]: r for r in rows}
        keep = {}
        for k, levels in LEVELS.items():
            best = max(levels, key=lambda v: by_name[label_of({k: v})][2])
            r = by_name[label_of({k: best})]
            if r[4] > KEEP_IF_T_ABOVE:
                keep[k] = (best, r[4])
        top = sorted(keep, key=lambda k: -keep[k][1])[:MAX_COMBO_FACTORS]
        print(f"\nKept for stage 2: " + (", ".join(f"{k}={keep[k][0]}" for k in top) or "none"), flush=True)

        # ---------------- stage 2
        combos = []
        for mask in itertools.product([0, 1], repeat=len(top)):
            ov = {k: keep[k][0] for k, on in zip(top, mask) if on}
            combos.append((label_of(ov), ov))
        if ("TEMPLATE", None) not in combos:
            combos.append(("TEMPLATE", None))
        print(f"\nSTAGE 2: {len(combos)} candidates x {s2} tables", flush=True)
        res2, slow2 = evaluate(pool, combos, 1_000_001, s2, "stage 2")
        rows2 = summarize(res2, "EXPLOIT_DEFAULT")
        print_rows(rows2, "EXPLOIT_DEFAULT", slow2)

        # ---------------- stage 3
        finalists = [r[0] for r in rows2 if r[0] not in ("TEMPLATE", "EXPLOIT_DEFAULT")][:FINALISTS]
        ov_of = dict(combos)
        final = [(n, ov_of[n]) for n in finalists] + [("EXPLOIT_DEFAULT", {}), ("TEMPLATE", None)]
        print(f"\nSTAGE 3: {len(final)} candidates x {s3} fresh tables", flush=True)
        res3, slow3 = evaluate(pool, final, 2_000_001, s3, "stage 3")
        print("\nFINAL vs EXPLOIT_DEFAULT")
        print_rows(summarize(res3, "EXPLOIT_DEFAULT"), "EXPLOIT_DEFAULT", slow3)
        print("\nFINAL vs TEMPLATE")
        print_rows(summarize(res3, "TEMPLATE"), "TEMPLATE", slow3)
        best = summarize(res3, "EXPLOIT_DEFAULT")[0]
        print(f"\nBEST: {best[0]}   overrides = {ov_of.get(best[0], {} if best[0] == 'EXPLOIT_DEFAULT' else None)}")
    print(f"total time {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
