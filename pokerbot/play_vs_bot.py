"""Play hands yourself against a bot, with the competition settings from config.py.

    python3 play_vs_bot.py                    # pick a bot from a menu, heads-up, 20 hands
    python3 play_vs_bot.py OPTIMAL            # straight to a bot
    python3 play_vs_bot.py LAG --seats 8      # full 8-seat table: you + 7 copies of the bot
    python3 play_vs_bot.py GTO --hands 50 --timeout 30 --seed 7

Competition settings: 2,000 chips each, blinds 10/20 rising every 25 hands
(config.BLIND_SCHEDULE). You get --timeout seconds per decision (default 30);
if time runs out you check, or fold if checking isn't possible, the same as
a bot that times out at the event.

Actions (type, then Enter):
    f            fold                  x        check
    c            call                  a        all-in
    r 120        raise TO 120 (total street bet; "b 120" works for bets too)
    r            minimum raise         r 1/2    raise by half the pot (also 1/3, 2/3, 3/2 ...)
    r pot        raise by the pot      h        help
"""
import argparse
import random
import select
import sys
import time

import config as C
from bot import Action, PokerBot
from bot_exploit import ExploitBot
from optimalbot import OptimalBot
from updated_optimal import UpdatedOptimalBot
from local_sim import Table, CallingStation, Maniac, describe
from opponent_bots import NitBot, LagBot, PushFoldBot, ChenBot, TrapperBot, GtoApproxBot
from poker_core import RANKS, hand_rank

BOTS = {
    "UPDATED_OPTIMAL": ("OptimalBot + bluff tracking, EV checks, ICM", lambda seed: UpdatedOptimalBot(seed=seed)),
    "OPTIMAL": ("tuned ExploitBot (best in testing)", lambda seed: OptimalBot(seed=seed)),
    "EXPLOIT": ("ExploitBot with bot_exploit.py defaults", lambda seed: ExploitBot(seed=seed)),
    "TEMPLATE": ("original PokerBot from config.py", lambda seed: PokerBot(seed=seed)),
    "GTO": ("solver-style approximation", lambda seed: GtoApproxBot(seed)),
    "LAG": ("loose-aggressive", lambda seed: LagBot(seed)),
    "NIT": ("very tight", lambda seed: NitBot()),
    "CHEN": ("Chen formula + pot odds", lambda seed: ChenBot()),
    "PUSHFOLD": ("all-in or fold preflop", lambda seed: PushFoldBot()),
    "TRAPPER": ("passive caller, slow-plays", lambda seed: TrapperBot(seed)),
    "STATION": ("always calls", lambda seed: CallingStation()),
    "MANIAC": ("always raises", lambda seed: Maniac()),
}
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
              + (f"    ({describe(hand_rank(s.hole_cards + s.board))})" if s.board else ""))
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
        opts.append(f"[a]ll-in {s.max_raise_to}")
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
        if cmd in ("h", "help", "?"):
            return ("f fold | x check | c call | a all-in | r 120 raise/bet TO 120 | "
                    "r min raise | r 1/2 raise by half pot (1/3, 2/3, 3/2 ...) | r pot pot-sized raise")
        if cmd in ("f", "fold"):
            return Action("fold") if s.to_call else "Nothing to call -- check (x) instead of folding."
        if cmd in ("x", "k", "check"):
            return Action("check") if s.to_call == 0 else f"You can't check: {s.to_call} to call."
        if cmd in ("c", "call"):
            return Action("call") if s.to_call else Action("check")
        if cmd in ("a", "allin", "all-in", "shove", "jam"):
            return Action("allin", s.max_raise_to)
        if cmd in ("r", "b", "raise", "bet"):
            high = s.me.bet + s.to_call
            if s.max_raise_to <= high:
                return "You can't raise, only call or fold."
            if arg is None:
                target = s.min_raise_to
            elif "/" in arg or arg == "pot":
                try:
                    num, den = arg.split("/") if "/" in arg else ("1", "1")
                    frac = float(num) / float(den)
                except ValueError:
                    return f"Couldn't read {arg!r} -- use a fraction like 1/2 or 2/3, or pot."
                target = int(high + frac * (s.pot + s.to_call))
            else:
                try:
                    target = int(float(arg))
                except ValueError:
                    return f"Couldn't read {arg!r} as an amount."
            if target >= s.max_raise_to:
                return Action("allin", s.max_raise_to)
            if target < s.min_raise_to:
                return f"Minimum raise is to {s.min_raise_to} (or a for all-in {s.max_raise_to})."
            return Action("raise" if high else "bet", target)
        return f"Unknown action {raw!r} -- type h for help."


def choose_bot():
    print("Choose an opponent:")
    names = list(BOTS)
    for i, n in enumerate(names, 1):
        print(f"  {i:>2}. {n:<9} {BOTS[n][0]}")
    while True:
        raw = input("Number or name: ").strip().upper()
        if raw in BOTS:
            return raw
        if raw.isdigit() and 1 <= int(raw) <= len(names):
            return names[int(raw) - 1]
        print("Not on the list.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bot", nargs="?", type=str.upper, choices=list(BOTS))
    ap.add_argument("--hands", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=30)
    ap.add_argument("--seats", type=int, default=2, help="2 = heads-up, up to 8 (others are copies of the bot)")
    ap.add_argument("--seed", type=int, default=None, help="set for a repeatable deal")
    args = ap.parse_args()
    if not 2 <= args.seats <= 8:
        ap.error("--seats must be 2-8")
    bot_name = args.bot or choose_bot()
    seed = args.seed if args.seed is not None else random.randrange(10**6)
    rng = random.Random(seed)

    names = ["You"] + [bot_name if args.seats == 2 else f"{bot_name}{i}" for i in range(1, args.seats)]
    human = HumanPlayer(names, args.timeout)
    lineup = [("You", human)] + [(names[i], BOTS[bot_name][1](seed + i)) for i in range(1, args.seats)]
    table = Table(lineup, rng, verbose=True)

    print(f"\n{'=' * 64}\n You vs {bot_name}: {args.hands} hands, {args.seats} seats, "
          f"{C.STARTING_STACK} chips each, {args.timeout:.0f}s per decision (seed {seed})\n{'=' * 64}")
    played = 0
    for hand_no in range(1, args.hands + 1):
        if len(table.alive()) <= 1 or table.stacks[0] == 0:
            break
        level = min((hand_no - 1) // C.HANDS_PER_LEVEL, len(C.BLIND_SCHEDULE) - 1)
        table.hand_no = table.hands_played = hand_no
        table.play_hand(hand_no, *C.BLIND_SCHEDULE[level])
        table.button = table.next_alive(table.button)
        played = hand_no
        print("  Stacks: " + "   ".join(f"{names[s]} {table.stacks[s]}" for s in range(args.seats)))

    print(f"\n{'=' * 64}\n Done after {played} hands.")
    for s in range(args.seats):
        print(f"   {names[s]:<12} {table.stacks[s]:>6}  ({table.stacks[s] - C.STARTING_STACK:+d})")
    if table.stacks[0] == 0:
        print(" You busted.")
    elif len(table.alive()) == 1:
        print(" You won every chip!")
    print("=" * 64)


if __name__ == "__main__":
    main()
