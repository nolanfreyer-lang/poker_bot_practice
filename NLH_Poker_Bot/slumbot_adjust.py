"""Luck-adjusted Slumbot results: all-in pots scored by equity instead of by the runout.

    python3 slumbot_adjust.py slumbot_hands.jsonl [more logs...]

Slumbot shows its hole cards on every hand, so for a hand where both players were all-in
before the river, nobody had a decision left: the result is the pot times our equity when
the money went in, minus what we put in. That replaces the luck of the remaining cards
with its expectation (exact for flop and turn all-ins, 4,000 sampled boards preflop) and
cuts most of the variance of 1,000-hand runs, which a few 200 BB all-ins dominate. Hands
that reached the river, or were folded, keep their real result.
"""
import json
import random
import statistics
import sys

from poker_core import hand_rank, parse_cards

STACK, BB = 20000, 100
STREET_CARDS = (0, 3, 4, 5)


def allin_street(action):
    """Street (0-3) on which both stacks went in, or None."""
    total, street_bet, high = [50, 100], [50, 100], 100     # index 0 = small blind
    actor, street, i = 0, 0, 0
    while i < len(action):
        c = action[i]
        i += 1
        if c == "/":
            street += 1
            street_bet, high, actor = [0, 0], 0, 1
            continue
        if c == "k":
            actor = 1 - actor
        elif c == "c":
            total[actor] += high - street_bet[actor]
            street_bet[actor] = high
            if min(total) >= STACK:
                return street
            actor = 1 - actor
        elif c == "f":
            return None
        elif c == "b":
            j = i
            while i < len(action) and action[i].isdigit():
                i += 1
            to = int(action[j:i])
            total[actor] += to - street_bet[actor]
            street_bet[actor], high = to, to
            actor = 1 - actor
    return None


def equity(mine, theirs, board, rng, samples=4000):
    dead = set(mine) | set(theirs) | set(board)
    deck = [c for c in range(52) if c not in dead]
    need = 5 - len(board)
    if need == 0:
        boards = [[]]
    elif need == 1:
        boards = [[c] for c in deck]
    elif need == 2:
        boards = [[a, b] for k, a in enumerate(deck) for b in deck[k + 1:]]
    else:
        boards = [rng.sample(deck, need) for _ in range(samples)]
    won = 0.0
    for extra in boards:
        full = board + extra
        a, b = hand_rank(mine + full), hand_rank(theirs + full)
        won += 1.0 if a > b else 0.5 if a == b else 0.0
    return won / len(boards)


def adjust(rows, rng):
    real, adjusted, swaps = [], [], 0
    for r in rows:
        w = r["winnings"] / BB
        real.append(w)
        street = allin_street(r["action"])
        if street is None or street == 3 or not r.get("bot_hole_cards"):
            adjusted.append(w)
            continue
        board = parse_cards(r["board"][:STREET_CARDS[street]])
        eq = equity(parse_cards(r["hole_cards"]), parse_cards(r["bot_hole_cards"]), board, rng)
        adjusted.append((eq * 2 * STACK - STACK) / BB)
        swaps += 1
    return real, adjusted, swaps


def per100(x):
    return f"{100 * statistics.mean(x):+7.1f} ± {100 * statistics.stdev(x) / len(x) ** 0.5:5.1f}"


def main():
    rng = random.Random(1)
    for path in sys.argv[1:]:
        rows = [json.loads(line) for line in open(path)]
        real, adjusted, swaps = adjust(rows, rng)
        print(f"{path.split('/')[-1]:28s} {len(rows):5d} hands   real {per100(real)}   "
              f"all-in adjusted {per100(adjusted)} bb/100   ({swaps} all-ins scored by equity)")


if __name__ == "__main__":
    main()
