"""Play UPDATED_OPTIMAL or HU_UPDATED_OPTIMAL against Slumbot (slumbot.com) through its public API.

    python3 slumbot_test.py --hands 1000
    python3 slumbot_test.py --bot HU_UPDATED_OPTIMAL --hands 1000
    python3 slumbot_test.py --hands 20 --verbose

Slumbot plays blinds 50/100 with 20,000-chip stacks that reset every hand, so
this is a cash game: ICM is off and the score is big blinds per 100 hands.
Every finished hand is appended to --log, so an interrupted run keeps its data.
"""
import argparse
import json
import statistics
import time
import urllib.error
import urllib.request

from bot import GameState, PlayerInfo
from poker_core import parse_cards
from deep_optimal import DeepOptimalBot
from hu_updated_optimal import HuUpdatedOptimalBot
from updated_optimal import UpdatedOptimalBot

HOST = "https://slumbot.com/slumbot/api"
SMALL_BLIND, BIG_BLIND, STACK = 50, 100, 20_000
ME, SLUMBOT = 0, 1
STREETS = ("preflop", "flop", "turn", "river")
BOT_OVERRIDES = {"USE_ICM": False}     # no tournament payouts in a cash game
BOTS = {"UPDATED_OPTIMAL": UpdatedOptimalBot, "HU_UPDATED_OPTIMAL": HuUpdatedOptimalBot,
        "DEEP_OPTIMAL": DeepOptimalBot}


def post(path, data, retries=5):
    body = json.dumps(data).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(f"{HOST}/{path}", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.load(resp)
            if "error_msg" in result:
                raise RuntimeError(f"Slumbot error: {result['error_msg']}")
            return result
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
            print(f"  network error ({e}), retrying", flush=True)


def build_state(action, client_pos, hole_cards, board):
    """Replay Slumbot's action string (e.g. 'b200c/kb400') into a GameState for our seat."""
    sb_seat = ME if client_pos == 1 else SLUMBOT    # client_pos 1 = small blind / button
    bb_seat = 1 - sb_seat
    total = {sb_seat: SMALL_BLIND, bb_seat: BIG_BLIND}
    street_bet = dict(total)
    street, high, last_raise, acted = 0, BIG_BLIND, BIG_BLIND, 0
    actor = sb_seat
    history = [("preflop", sb_seat, "blind", SMALL_BLIND), ("preflop", bb_seat, "blind", BIG_BLIND)]

    def next_street():
        nonlocal street, high, last_raise, acted, actor
        street, high, last_raise, acted, actor = street + 1, 0, BIG_BLIND, 0, bb_seat
        street_bet[ME] = street_bet[SLUMBOT] = 0

    i = 0
    while i < len(action):
        c = action[i]
        i += 1
        name = STREETS[min(street, 3)]
        if c == "/":
            continue
        if c == "k":
            history.append((name, actor, "check", street_bet[actor]))
            acted += 1
            actor = 1 - actor
            if acted == 2:
                next_street()
        elif c == "c":
            total[actor] += high - street_bet[actor]
            street_bet[actor] = high
            history.append((name, actor, "call", high))
            acted += 1
            actor = 1 - actor
            if acted >= 2:
                next_street()
        elif c == "f":
            history.append((name, actor, "fold", street_bet[actor]))
            actor = 1 - actor
        elif c == "b":
            j = i
            while i < len(action) and action[i].isdigit():
                i += 1
            to = int(action[j:i])
            total[actor] += to - street_bet[actor]
            street_bet[actor] = to
            last_raise = max(to - high, BIG_BLIND)
            kind = "allin" if total[actor] >= STACK else "bet" if high == 0 else "raise"
            high = to
            history.append((name, actor, kind, to))
            acted = 1
            actor = 1 - actor
        else:
            raise ValueError(f"unexpected {c!r} in action {action!r}")

    players = []
    for seat in (ME, SLUMBOT):
        behind = STACK - total[seat]
        players.append(PlayerInfo(seat=seat, stack=behind, bet=street_bet[seat], all_in=behind == 0))
    me = players[ME]
    to_call = min(high - street_bet[ME], me.stack)
    max_to = me.stack + me.bet
    state = GameState(
        my_seat=ME, hole_cards=parse_cards(hole_cards), board=parse_cards(board),
        street=STREETS[min(street, 3)], pot=total[ME] + total[SLUMBOT], players=players,
        button_seat=sb_seat, small_blind=SMALL_BLIND, big_blind=BIG_BLIND, to_call=to_call,
        min_raise_to=min(high + last_raise, max_to), max_raise_to=max_to, history=history)
    return state, actor


def to_incr(a, s):
    """Our Action -> Slumbot's incremental action. Bet sizes are street totals, like ours."""
    if a.kind == "fold":
        return "f"
    if a.kind == "check":
        return "k"
    if a.kind == "call":
        return "c"
    high = s.me.bet + s.to_call
    target = s.max_raise_to if a.kind == "allin" else a.amount
    if target <= high:                 # all-in for no more than the call
        return "c" if s.to_call else "k"
    return f"b{target}"


def play_hand(bot, token, verbose):
    r = post("new_hand", {"token": token} if token else {})
    token = r.get("token", token)
    while r.get("winnings") is None:
        state, actor = build_state(r["action"], r["client_pos"], r["hole_cards"], r["board"])
        if actor != ME:
            raise RuntimeError(f"expected our turn after action {r['action']!r}")
        a = bot.act(state)
        incr = to_incr(a, state)
        if verbose:
            print(f"  {' '.join(r['hole_cards'])} | {' '.join(r['board']) or '-'} | "
                  f"{r['action'] or '(first)'} -> {a.kind} {a.amount or ''} ({incr})", flush=True)
        r = post("act", {"token": token, "incr": incr})
        token = r.get("token", token)

    state, _ = build_state(r["action"], r["client_pos"], r["hole_cards"], r["board"])
    # Slumbot reveals its cards on every hand; the bot itself ignores folded hands.
    shown = r.get("bot_hole_cards")
    showdown = {ME: parse_cards(r["hole_cards"]), SLUMBOT: parse_cards(shown)} if shown else {}
    bot.on_hand_end(state.history, [ME, SLUMBOT], showdown=showdown,
                    board=parse_cards(r["board"]), my_seat=ME)
    return token, r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bot", type=str.upper, choices=list(BOTS), default="UPDATED_OPTIMAL")
    ap.add_argument("--hands", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--log", default="slumbot_hands.jsonl")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    bot = BOTS[args.bot](seed=args.seed, overrides=BOT_OVERRIDES)
    token, results = None, []
    t0 = time.time()
    with open(args.log, "a") as log:
        for h in range(1, args.hands + 1):
            token, r = play_hand(bot, token, args.verbose)
            results.append(r["winnings"])
            log.write(json.dumps({k: r.get(k) for k in ("action", "client_pos", "hole_cards",
                                                        "bot_hole_cards", "board", "winnings")}) + "\n")
            log.flush()
            if args.verbose or h % 100 == 0:
                bb100 = sum(results) / len(results) / BIG_BLIND * 100
                print(f"hand {h}: {r['winnings']:+d}   total {sum(results):+d}   {bb100:+.1f} bb/100   "
                      f"{time.time() - t0:.0f}s", flush=True)

    n = len(results)
    mean = statistics.mean(results)
    se = statistics.stdev(results) / n ** 0.5 if n > 1 else 0.0
    print(f"\n{n} hands: {sum(results):+d} chips = {mean / BIG_BLIND * 100:+.1f} ± "
          f"{se / BIG_BLIND * 100:.1f} bb/100")


if __name__ == "__main__":
    main()
