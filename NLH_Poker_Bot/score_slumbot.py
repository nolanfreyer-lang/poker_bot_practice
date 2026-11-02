"""Score Slumbot logs with all-in luck removed, and compare runs.

    python3 score_slumbot.py slumbot_logs/deep_after.jsonl slumbot_logs/deep_before.jsonl

When both players are all-in before the river, the remaining cards are pure luck,
so the hand is scored by our exact equity at that moment instead of who won.
"""
import itertools
import json
import re
import statistics
import sys

from poker_core import hand_rank, parse_cards

BOARD_LEN = {0: 0, 1: 3, 2: 4, 3: 5}
STACK = 20000


def all_in_street(action):
    """Street index where both players got all-in with cards to come, else None."""
    streets = action.split("/")
    if action.endswith("f"):
        return None
    total, street_bet = {0: 50, 1: 100}, {0: 50, 1: 100}
    for i, text in enumerate(streets[:4]):
        high = max(street_bet.values()) if i == 0 else 0
        if i:
            street_bet = {0: 0, 1: 0}
        actor = 0 if i == 0 else 1
        for tok in re.findall(r"b\d+|[kc]", text):
            if tok[0] == "b":
                amt = int(tok[1:])
                total[actor] += amt - street_bet[actor]
                street_bet[actor] = high = amt
            elif tok == "c":
                total[actor] += high - street_bet[actor]
                street_bet[actor] = high
            actor = 1 - actor
        if total[0] >= STACK and total[1] >= STACK:
            return i if i < 3 else None
    return None


def equity(mine, theirs, board):
    dead = set(mine) | set(theirs) | set(board)
    deck = [c for c in range(52) if c not in dead]
    need = 5 - len(board)
    runs = list(itertools.combinations(deck, need))
    if len(runs) > 4000:
        runs = runs[::len(runs) // 4000]
    won = 0.0
    for r in runs:
        full = board + list(r)
        a, b = hand_rank(mine + full), hand_rank(theirs + full)
        won += 1.0 if a > b else 0.5 if a == b else 0.0
    return won / len(runs)


def adjusted(h):
    i = all_in_street(h["action"])
    if i is None or not h.get("bot_hole_cards"):
        return h["winnings"]
    e = equity(parse_cards(h["hole_cards"]), parse_cards(h["bot_hole_cards"]), parse_cards(h["board"][:BOARD_LEN[i]]))
    return (2 * e - 1) * STACK


def score(path):
    H = [json.loads(l) for l in open(path)]
    raw = [h["winnings"] for h in H]
    adj = [adjusted(h) for h in H]
    n = len(H)
    print(f"{path}: {n} hands")
    print(f"  raw               {statistics.mean(raw):+8.1f} ± {statistics.stdev(raw) / n ** 0.5:.1f} bb/100")
    print(f"  all-in adjusted   {statistics.mean(adj):+8.1f} ± {statistics.stdev(adj) / n ** 0.5:.1f} bb/100")
    return statistics.mean(adj), statistics.variance(adj) / n


if __name__ == "__main__":
    runs = [score(p) for p in sys.argv[1:]]
    if len(runs) == 2:
        d = runs[0][0] - runs[1][0]
        print(f"difference {d:+.1f} ± {(runs[0][1] + runs[1][1]) ** 0.5:.1f} bb/100")
