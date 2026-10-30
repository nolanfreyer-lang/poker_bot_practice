"""Play PLO4 yourself against 7 bots, with the competition settings from config.py.

    python3 play_vs_bot.py                          # you + the 7 bots in DEFAULT_TABLE
    python3 play_vs_bot.py --hands 50 --timeout 60 --seed 7
    python3 play_vs_bot.py --bots PLO4,LAG,NIT      # choose who sits down (1 to 7 bots)
    python3 play_vs_bot.py --bots PLO4              # heads-up
    python3 play_vs_bot.py --list                   # every bot you can seat

Competition settings: 2,000 chips each, blinds 10/20 rising every 25 hands, 200-hand
cap. You get --timeout seconds per decision (default 30); if time runs out you check,
or fold if checking isn't possible, the same as a bot that times out at the event.
Ctrl+C stops the game and shows the stacks.

Actions (type, then Enter):
    f            fold                  x        check
    c            call                  a        biggest legal raise (pot, or all-in if shorter)
    r 120        raise TO 120 (total street bet; "b 120" works for bets too)
    r            minimum raise         r 1/2    raise by half the pot (also 1/3, 2/3 ...)
    r pot        biggest legal raise   h        help

Reminder: PLO hands use EXACTLY two of your four cards and three from the board.
"""
import argparse
import random
import select
import sys
import time

import config as C
from bot import Action, PokerBot
from local_sim import BASELINE_OVERRIDES, Table
from opponent_bots import (CallingStation, CheckFoldBot, ChenBot, GtoApproxBot, LagBot, Maniac, NitBot,
                           NlhPortBot, PushFoldBot, RandomBot, TrapperBot)
from poker_core import RANKS, describe, plo_value

BOTS = {
    "PLO4": ("our tuned competition bot", lambda seed: PokerBot(seed=seed)),
    "BASELINE": ("our bot with the NLH lessons off", lambda seed: PokerBot(seed=seed, overrides=BASELINE_OVERRIDES)),
    "GTO": ("balanced mixed strategy", lambda seed: GtoApproxBot(seed)),
    "LAG": ("loose-aggressive", lambda seed: LagBot(seed)),
    "NIT": ("very tight", lambda seed: NitBot()),
    "CHEN": ("hand points + pot odds", lambda seed: ChenBot()),
    "TRAPPER": ("passive caller, slow-plays", lambda seed: TrapperBot(seed)),
    "NLHPORT": ("Hold'em bot pasted onto PLO", lambda seed: NlhPortBot(seed)),
    "PUSHFOLD": ("pot-raises or folds preflop", lambda seed: PushFoldBot()),
    "STATION": ("always calls", lambda seed: CallingStation()),
    "RANDOM": ("random actions", lambda seed: RandomBot(seed)),
    "MANIAC": ("always raises the maximum", lambda seed: Maniac()),
    "CHECKFOLD": ("a crashed bot", lambda seed: CheckFoldBot()),
}
DEFAULT_TABLE = ("PLO4", "GTO", "LAG", "NIT", "CHEN", "TRAPPER", "NLHPORT")
SUIT_SYMBOL = {0: "♣", 1: "♦", 2: "♥", 3: "♠"}


def pretty(cards):
    return " ".join(RANKS[c >> 2].replace("T", "10") + SUIT_SYMBOL[c & 3] for c in cards) or "-"


def timed_input(prompt, deadline):
    """Read a line before `deadline` (time.time()); None if time runs out."""
    sys.stdout.write(prompt)
    sys.stdout.flush()
    remaining = deadline - time.time()
    if remaining <= 0:
        return None
    ready, _, _ = select.select([sys.stdin], [], [], remaining)
    if not ready:
        return None
    line = sys.stdin.readline()
    return line.strip().lower() if line else None


class HumanPlayer:
    def __init__(self, names, timeout):
        self.names = names
        self.timeout = timeout
        self.helper = PokerBot()   # only for position labels

    def act(self, s):
        me = s.me
        pos = self.helper.position(s)
        print()
        print(f"  ┌ YOUR TURN ({s.street.upper()}, {pos})")
        print(f"  │ Your cards: {pretty(s.hole_cards)}    Board: {pretty(s.board)}"
              + (f"    ({describe(plo_value(s.hole_cards, s.board))})" if s.board else ""))
        print(f"  │ Pot: {s.pot}    To call: {s.to_call}")
        for p in s.players:
            if p.dealt:
                who = "You" if p.seat == s.my_seat else self.names[p.seat]
                state = "folded" if not p.active else "all-in" if p.all_in else ""
                print(f"  │   S{p.seat + 1} {who:<10} stack {p.stack:>5}  bet {p.bet:>4}  {state}")
        opts = ["[f]old", "[c]all " + str(min(s.to_call, me.stack))] if s.to_call else ["[x] check"]
        if s.max_raise_to > me.bet + s.to_call:
            word = "raise" if s.to_call or me.bet else "bet"
            opts.append(f"[r] {word} to <amount> ({s.min_raise_to}-{s.max_raise_to})")
            opts.append(f"[a] {'all-in' if s.max_raise_to == me.stack + me.bet else 'pot'} {s.max_raise_to}")
        print("  │ " + "   ".join(opts))
        deadline = time.time() + self.timeout
        while True:
            left = int(deadline - time.time())
            raw = timed_input(f"  └ Action ({left}s left): ", deadline)
            if raw is None:
                auto = "check" if s.to_call == 0 else "fold"
                print(f"\n  ⏰ Time's up -- auto {auto}.")
                return Action(auto)
            action = self.parse(raw, s)
            if isinstance(action, str):
                print(f"    {action}")
                continue
            return action

    @staticmethod
    def parse(raw, s):
        """-> Action, or an error / help string to show before asking again."""
        if not raw:
            return "Type an action, or h for help."
        parts = raw.split()
        cmd, arg = parts[0], (parts[1] if len(parts) > 1 else None)
        high = s.me.bet + s.to_call
        if cmd in ("h", "help", "?"):
            return ("f fold | x check | c call | a biggest legal raise | r 120 raise/bet TO 120 | "
                    "r min raise | r 1/2 raise by half pot (1/3, 2/3 ...) | r pot biggest legal raise")
        if cmd in ("f", "fold"):
            return Action("fold") if s.to_call else "Nothing to call -- check (x) instead of folding."
        if cmd in ("x", "k", "check"):
            return Action("check") if s.to_call == 0 else f"You can't check: {s.to_call} to call."
        if cmd in ("c", "call"):
            return Action("call") if s.to_call else Action("check")
        if cmd in ("a", "allin", "all-in", "shove", "jam", "r", "b", "raise", "bet") and s.max_raise_to <= high:
            return "You can't raise, only call or fold." if s.to_call else "You can't bet, only check."
        if cmd in ("a", "allin", "all-in", "shove", "jam"):
            return Action("allin", s.max_raise_to)
        if cmd in ("r", "b", "raise", "bet"):
            if arg is None:
                target = s.min_raise_to
            elif arg == "pot":
                target = s.max_raise_to
            elif "/" in arg:
                try:
                    num, den = arg.split("/")
                    target = int(high + float(num) / float(den) * (s.pot + s.to_call))
                except (ValueError, ZeroDivisionError):
                    return f"Couldn't read {arg!r} -- use a fraction like 1/2 or 2/3, or pot."
            else:
                try:
                    target = int(float(arg))
                except ValueError:
                    return f"Couldn't read {arg!r} as an amount."
            if target > s.max_raise_to:
                return f"Pot limit: the most you can make it is {s.max_raise_to} (type a)."
            if target < s.min_raise_to:
                return f"Minimum raise is to {s.min_raise_to}."
            return Action("allin" if target == s.me.stack + s.me.bet else "raise" if high else "bet", target)
        return f"Unknown action {raw!r} -- type h for help."


def parse_bots(text, ap):
    picks = [x.strip().upper() for x in text.split(",") if x.strip()]
    unknown = [x for x in picks if x not in BOTS]
    if unknown:
        ap.error(f"unknown bots {unknown}; choose from {', '.join(BOTS)}")
    if not 1 <= len(picks) <= 7:
        ap.error("--bots takes 1 to 7 bots")
    return picks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bots", default=",".join(DEFAULT_TABLE), help="comma-separated, 1 to 7 bots; repeats allowed")
    ap.add_argument("--hands", type=int, default=C.HAND_LIMIT)
    ap.add_argument("--timeout", type=float, default=30)
    ap.add_argument("--seed", type=int, default=None, help="set for a repeatable deal")
    ap.add_argument("--list", action="store_true", help="list the bots and exit")
    args = ap.parse_args()
    if args.list:
        for n, (about, _) in BOTS.items():
            print(f"  {n:<10} {about}{'  (default table)' if n in DEFAULT_TABLE else ''}")
        return
    picks = parse_bots(args.bots, ap)
    seed = args.seed if args.seed is not None else random.randrange(10**6)
    rng = random.Random(seed)

    names = ["You"] + [n if picks.count(n) == 1 else f"{n}{i}" for i, n in enumerate(picks, 1)]
    human = HumanPlayer(names, args.timeout)
    lineup = [("You", human)] + [(names[i], BOTS[n][1](seed + i)) for i, n in enumerate(picks, 1)]
    table = Table(lineup, rng, verbose=True)
    seats = len(lineup)

    print(f"\n{'=' * 64}\n You vs {', '.join(names[1:])}\n {args.hands} hands, "
          f"{C.STARTING_STACK} chips each, {args.timeout:.0f}s per decision (seed {seed})\n{'=' * 64}")
    played = 0
    try:
        for hand_no in range(1, args.hands + 1):
            if len(table.alive()) <= 1 or table.stacks[0] == 0:
                break
            level = min((hand_no - 1) // C.HANDS_PER_LEVEL, len(C.BLIND_SCHEDULE) - 1)
            table.hand_no = table.hands_played = hand_no
            table.play_hand(hand_no, *C.BLIND_SCHEDULE[level])
            table.button = table.next_alive(table.button)
            played = hand_no
            print("  Stacks: " + "   ".join(f"{names[s]} {table.stacks[s]}" for s in range(seats)))
    except KeyboardInterrupt:      # the hand in progress is abandoned; stacks are from before it
        print("\n\n Stopped.")

    print(f"\n{'=' * 64}\n Done after {played} hands.")
    for s in sorted(range(seats), key=lambda s: -table.stacks[s]):
        print(f"   {names[s]:<12} {table.stacks[s]:>6}  ({table.stacks[s] - C.STARTING_STACK:+d})")
    if table.stacks[0] == 0:
        print(" You busted.")
    elif len(table.alive()) == 1:
        print(" You won every chip!")
    print("=" * 64)


if __name__ == "__main__":
    main()
