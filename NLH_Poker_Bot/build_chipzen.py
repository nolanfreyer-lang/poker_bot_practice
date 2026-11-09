"""Bundle a bot into one bot.py for the Chipzen SDK's Python starter.

    python3 build_chipzen.py --sdk /path/to/chipzen-sdk                       # HU_UPDATED_OPTIMAL
    python3 build_chipzen.py --sdk /path/to/chipzen-sdk --bot UPDATED_OPTIMAL

The starter's Dockerfile compiles a single bot.py, so this merges the strategy
modules, embeds preflop_equity.json, and appends the Chipzen adapter below. The
starter's Dockerfile, .dockerignore and requirements.txt are copied unchanged.
Rerun it after any strategy change; never edit the generated bot.py by hand.

Chipzen matches have no payout ladder to model, so ICM is off, which also
removes numpy (the SDK warns native packages can crash in its sandbox).
"""
import argparse
import ast
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMON = ["poker_core", "bot", "bot_exploit", "optimalbot", "updated_optimal"]
# name -> (extra modules, strategy class, adapter class, default output folder)
BOTS = {
    "HU_UPDATED_OPTIMAL": (["pushfold_tables", "hu_updated_optimal"], "HuUpdatedOptimalBot", "ChipzenHuBot",
                           "chipzen_hu_bot"),
    "UPDATED_OPTIMAL": ([], "UpdatedOptimalBot", "ChipzenUpdatedBot", "chipzen_updated_bot"),
    "DEEP_THOUGHT_V1": (["pushfold_tables", "hu_updated_optimal", "fitted_tables", "deep_thought_v1"], "DeepThoughtV1Bot",
                     "ChipzenDeepThoughtBot", "chipzen_deep_thought_v1"),
    "DEEP_THOUGHT_V2": (["pushfold_tables", "hu_updated_optimal", "fitted_tables", "deep_thought_v1",
                         "fitted_tables_v2", "deep_thought_v2"], "DeepThoughtV2Bot",
                        "ChipzenDeepThoughtV2Bot", "chipzen_deep_thought_v2"),
    "DEEP_THOUGHT_V3": (["pushfold_tables", "hu_updated_optimal", "fitted_tables", "deep_thought_v1",
                         "fitted_tables_v2", "deep_thought_v2", "deep_thought_v3"], "DeepThoughtV3Bot",
                        "ChipzenDeepThoughtV3Bot", "chipzen_deep_thought_v3"),
    "DEEP_THOUGHT_V4": (["pushfold_tables", "hu_updated_optimal", "fitted_tables", "deep_thought_v1",
                         "fitted_tables_v2", "deep_thought_v2", "deep_thought_v3", "deep_thought_v4"], "DeepThoughtV4Bot",
                        "ChipzenDeepThoughtV4Bot", "chipzen_deep_thought_v4"),
    "DEEP_THOUGHT_V5": (["pushfold_tables", "hu_updated_optimal", "fitted_tables", "deep_thought_v1",
                         "fitted_tables_v2", "deep_thought_v2", "deep_thought_v3", "deep_thought_v4",
                         "deep_thought_v5"], "DeepThoughtV5Bot",
                        "ChipzenDeepThoughtV5Bot", "chipzen_deep_thought_v5"),
    "DEEP_THOUGHT_V6": (["pushfold_tables", "hu_updated_optimal", "fitted_tables", "deep_thought_v1",
                         "fitted_tables_v2", "deep_thought_v2", "deep_thought_v3", "deep_thought_v4",
                         "deep_thought_v5", "deep_thought_v6"], "DeepThoughtV6Bot",
                        "ChipzenDeepThoughtV6Bot", "chipzen_deep_thought_v6"),
    "DEEP_THOUGHT_V7": (["pushfold_tables", "hu_updated_optimal", "fitted_tables", "deep_thought_v1",
                         "fitted_tables_v2", "deep_thought_v2", "deep_thought_v3", "deep_thought_v4",
                         "deep_thought_v5", "deep_thought_v6", "deep_thought_v7"], "DeepThoughtV7Bot",
                        "ChipzenDeepThoughtV7Bot", "chipzen_deep_thought_v7"),
    "DEEP_THOUGHT_V9": (["pushfold_tables", "hu_updated_optimal", "fitted_tables", "deep_thought_v1",
                         "fitted_tables_v2", "deep_thought_v2", "deep_thought_v3", "deep_thought_v4",
                         "deep_thought_v5", "deep_thought_v6", "deep_thought_v7", "deep_thought_v9"],
                        "DeepThoughtV9Bot", "ChipzenDeepThoughtV9Bot", "chipzen_deep_thought_v9"),
    "DEEP_THOUGHT_V10": (["pushfold_tables", "hu_updated_optimal", "fitted_tables", "deep_thought_v1",
                          "fitted_tables_v2", "deep_thought_v2", "deep_thought_v3", "deep_thought_v4",
                          "deep_thought_v5", "deep_thought_v6", "deep_thought_v7", "deep_thought_v9",
                          "deep_thought_v10"], "DeepThoughtV10Bot", "ChipzenDeepThoughtV10Bot",
                         "chipzen_deep_thought_v10"),
    # v11: v10's code, packaged with profiles refreshed after the Nov 8 4pm bracket
    "DEEP_THOUGHT_V11": (["pushfold_tables", "hu_updated_optimal", "fitted_tables", "deep_thought_v1",
                          "fitted_tables_v2", "deep_thought_v2", "deep_thought_v3", "deep_thought_v4",
                          "deep_thought_v5", "deep_thought_v6", "deep_thought_v7", "deep_thought_v9",
                          "deep_thought_v10"], "DeepThoughtV10Bot", "ChipzenDeepThoughtV11Bot",
                         "chipzen_deep_thought_v11"),
    # v12: v11 that stays pot committed (no stack-off fold when the call needs under 20%)
    "DEEP_THOUGHT_V12": (["pushfold_tables", "hu_updated_optimal", "fitted_tables", "deep_thought_v1",
                          "fitted_tables_v2", "deep_thought_v2", "deep_thought_v3", "deep_thought_v4",
                          "deep_thought_v5", "deep_thought_v6", "deep_thought_v7", "deep_thought_v9",
                          "deep_thought_v10", "deep_thought_v12"], "DeepThoughtV12Bot", "ChipzenDeepThoughtV12Bot",
                         "chipzen_deep_thought_v12"),
}
LOCAL = set(COMMON) | {"config", "hu_updated_optimal", "pushfold_tables", "fitted_tables", "deep_thought_v1",
                       "fitted_tables_v2", "deep_thought_v2", "deep_thought_v3", "deep_thought_v4",
                       "deep_thought_v5", "deep_thought_v6", "deep_thought_v7", "deep_thought_v9", "deep_thought_v10",
                       "deep_thought_v12"}
# Bots that start each match from saved opponent profiles (chipzen_profiles.py writes them)
PROFILED = {"DEEP_THOUGHT_V5", "DEEP_THOUGHT_V6", "DEEP_THOUGHT_V7", "DEEP_THOUGHT_V9", "DEEP_THOUGHT_V10", "DEEP_THOUGHT_V11", "DEEP_THOUGHT_V12"}
PROFILES_FILE = HERE / "chipzen_profiles.json"
# Dev-only code left out of the bundle: table building needs multiprocessing,
# which the sandbox blocks, and the table is embedded instead.
DROP = {"build_preflop_table", "_class_combos", "_preflop_job", "_load_preflop_table",
        "EQUITY_FILE", "_PREFLOP_TABLE", "_n"}

HEADER = '''# cython: language_level=3str
"""@NAME@ for Chipzen. GENERATED by build_chipzen.py -- do not edit."""
'''

ADAPTER = r'''
# =====================================================================
# Chipzen adapter
# =====================================================================
# ICM needs a payout ladder; Chipzen matches are elimination. Bluff tracking
# resets each match because opponents differ.
CHIPZEN_OVERRIDES = {"USE_ICM": False}
AGGRESSIVE_STACKLESS = ("bet", "raise")


def convert_history(entries, stacks=None, starting=None, raise_added=False):
    """Chipzen action_history -> (our history, this street's bets, street).

    Chipzen amounts: raise = street total, call = chips added, blind = chips posted.
    Match replays show raises as chips added instead, so raise_added reads them that
    way. Ours are always the player's street total after acting. A raise is an all-in if
    it empties the stack: read `stacks` mid-hand, or `starting` (stacks at hand
    start) after the hand, when the wire no longer says who was all-in.
    """
    history, bets, high, street = [], {}, 0, "preflop"
    spent = {}
    for e in entries:
        phase, seat, act, amt = e.get("phase", street), e["seat"], e["action"], e.get("amount", 0)
        if phase != street:
            street, bets, high = phase, {}, 0
        if act == "post_ante":
            spent[seat] = spent.get(seat, 0) + amt
            continue
        before = bets.get(seat, 0)
        if act in ("post_small_blind", "post_big_blind"):
            spent[seat] = spent.get(seat, 0) + amt
            bets[seat] = bets.get(seat, 0) + amt
            high = max(high, bets[seat])
            history.append((street, seat, "blind", bets[seat]))
        elif act == "raise":
            amt = before + amt if raise_added else amt
            spent[seat] = spent.get(seat, 0) + amt - before
            bets[seat] = amt
            all_in = (stacks is not None and stacks.get(seat, 1) == 0) or \
                (starting is not None and seat in starting and spent[seat] >= starting[seat])
            history.append((street, seat, "allin" if all_in else "bet" if high == 0 else "raise", amt))
            high = max(high, amt)
        elif act == "call":
            spent[seat] = spent.get(seat, 0) + amt
            bets[seat] = bets.get(seat, 0) + amt
            history.append((street, seat, "call", bets[seat]))
        elif act in ("check", "fold"):
            history.append((street, seat, act, bets.get(seat, 0)))
    return history, bets, street


class @ADAPTER@(Bot):
    def __init__(self):
        self.core = @CORE@(overrides=CHIPZEN_OVERRIDES)
        self.small_blind = 0
        self.big_blind = 0
        self.board = []
        self.starting = {}
        self.raise_added = False
        self.errors = 0

    def on_match_start(self, match_info):
        self.core = @CORE@(overrides=CHIPZEN_OVERRIDES)
        config = match_info.get("game_config") or {}
        self.small_blind = config.get("small_blind", 0)
        self.big_blind = config.get("big_blind", 0)
        self.raise_added = False
        self.load_profiles(match_info)

    def load_profiles(self, match_info):
        """Start each known opponent's reads from earlier matches, found by bot name (a bot
        with load_named_profile also tries the name's older versions)."""
        load = getattr(self.core, "load_profile", None)
        named = getattr(self.core, "load_named_profile", None)
        for seat in match_info.get("seats") or []:
            if seat.get("is_self"):
                continue
            if named is not None:
                named(int(seat["seat"]), seat.get("display_name"), _PROFILES)
                continue
            profile = _PROFILES.get(seat.get("display_name"))
            if load is not None and profile:
                load(int(seat["seat"]), profile)

    @staticmethod
    def reads_raises_as_added(state):
        """Which reading of raise amounts adds up to the pot the server reports."""
        def chips(added):
            history = convert_history(state.action_history, raise_added=added)[0]
            street_max = {}
            for street, seat, kind, amt in history:
                if kind not in ("check", "fold"):
                    street_max[street, seat] = max(street_max.get((street, seat), 0), amt)
            return sum(street_max.values())
        return chips(False) != state.pot and chips(True) == state.pot

    def to_game_state(self, state):
        n = len(state.opponent_stacks) + 1
        others = [seat for seat in range(n) if seat != state.your_seat]
        stacks = dict(zip(others, state.opponent_stacks))
        stacks[state.your_seat] = state.your_stack
        self.raise_added = self.raise_added or self.reads_raises_as_added(state)
        history, bets, street = convert_history(state.action_history, stacks, raise_added=self.raise_added)
        if street != state.phase:
            bets = {}
        folded = {seat for _, seat, kind, _ in history if kind == "fold"}
        seen = {seat for _, seat, _, _ in history}
        players = [PlayerInfo(seat=seat, stack=stacks[seat], bet=bets.get(seat, 0),
                              active=seat not in folded and (seat in seen or stacks[seat] > 0),
                              all_in=stacks[seat] == 0 and seat not in folded,
                              dealt=seat in seen or stacks[seat] > 0)
                  for seat in range(n)]
        posted_sb = max((e["amount"] for e in state.action_history if e["action"] == "post_small_blind"), default=0)
        posted_bb = max((e["amount"] for e in state.action_history if e["action"] == "post_big_blind"), default=0)
        me = players[state.your_seat]
        max_to = state.max_raise or me.stack + me.bet
        return GameState(
            my_seat=state.your_seat, hole_cards=parse_cards([str(c) for c in state.hole_cards]),
            board=parse_cards([str(c) for c in state.board]), street=state.phase, pot=state.pot,
            players=players, button_seat=state.dealer_seat,
            small_blind=max(self.small_blind, posted_sb), big_blind=max(self.big_blind, posted_bb),
            to_call=min(state.to_call, state.your_stack), min_raise_to=state.min_raise or max_to,
            max_raise_to=max_to, history=history, hand_number=state.hand_number)

    @staticmethod
    def to_chipzen(a, state):
        valid = state.valid_actions

        def passive():
            for name, make in (("check", ChipzenAction.check), ("call", ChipzenAction.call),
                               ("fold", ChipzenAction.fold)):
                if name in valid:
                    return make()
            return ChipzenAction.fold()

        if a.kind == "fold":
            return ChipzenAction.fold() if "fold" in valid else passive()
        if a.kind == "check" and "check" in valid:
            return ChipzenAction.check()
        if a.kind == "call" and "call" in valid:
            return ChipzenAction.call()
        if a.kind in ("bet", "raise", "allin"):
            if a.kind == "allin" and state.to_call >= state.your_stack and "call" in valid:
                return ChipzenAction.call()         # all-in is just a call here
            if "raise" in valid:
                target = state.max_raise if a.kind == "allin" else a.amount
                return ChipzenAction.raise_to(int(max(state.min_raise, min(target, state.max_raise))))
        return passive()

    def decide(self, state):
        self.board = [str(c) for c in state.board]
        try:
            return self.to_chipzen(self.core.act(self.to_game_state(state)), state)
        except Exception as e:      # a crash here must not end the match
            self.errors += 1
            if self.errors <= 5:
                print(f"decide error: {e!r}", file=sys.stderr)
            return ChipzenAction.check() if "check" in state.valid_actions else ChipzenAction.fold()

    def on_round_start(self, message):
        self.board = []
        stacks = (message.get("state") or {}).get("stacks") or []
        self.starting = dict(enumerate(stacks))
        super().on_round_start(message)

    def on_phase_change(self, message):
        board = (message.get("state") or {}).get("board")
        if board:
            self.board = list(board)

    def on_hand_result(self, result):
        try:
            history, _, _ = convert_history(result.get("action_history") or [], starting=self.starting,
                                         raise_added=self.raise_added)
            seats = sorted({seat for _, seat, _, _ in history})
            shown = {e["seat"]: parse_cards(e["hole_cards"]) for e in result.get("showdown") or []}
            self.core.on_hand_end(history, seats, showdown=shown, board=parse_cards(self.board),
                                  my_seat=self.core._my_seat)
        except Exception as e:
            self.errors += 1
            if self.errors <= 5:
                print(f"hand result error: {e!r}", file=sys.stderr)


def main():
    url = os.environ.get("CHIPZEN_WS_URL") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not url:
        print("error: CHIPZEN_WS_URL not set and no URL passed on the command line", file=sys.stderr)
        sys.exit(1)
    asyncio.run(run_bot(url, @ADAPTER@(), token=os.environ.get("CHIPZEN_TOKEN"),
                        ticket=os.environ.get("CHIPZEN_TICKET")))


if __name__ == "__main__":
    main()
'''


def node_source(lines, node):
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return "\n".join(lines[start - 1:node.end_lineno])


def split_module(name):
    """-> (stdlib import lines, body source) with local imports and dev-only code removed."""
    text = (HERE / f"{name}.py").read_text()
    lines = text.splitlines()
    imports, body = [], []
    for i, node in enumerate(ast.parse(text).body):
        if i == 0 and isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
            continue                                    # module docstring
        if isinstance(node, ast.Import):
            imports += [node_source(lines, node)] if not {a.name for a in node.names} & LOCAL else []
            continue
        if isinstance(node, ast.ImportFrom):
            if node.module not in LOCAL:
                imports.append(node_source(lines, node))
            continue
        if isinstance(node, ast.Try):                   # the numpy import
            body.append("np = None  # ICM is off on Chipzen, so numpy is not shipped")
            continue
        if isinstance(node, ast.If) and "__name__" in ast.get_source_segment(text, node.test):
            continue
        if isinstance(node, ast.If) and getattr(node.test, "left", None) is not None \
                and getattr(node.test.left, "id", "") == "np":
            continue                                    # numpy ICM table setup
        names = {getattr(node, "name", None)} | {getattr(t, "id", None) for t in getattr(node, "targets", [])}
        if names & DROP:
            continue
        body.append(node_source(lines, node))
    return imports, "\n\n".join(body)


def config_class():
    text = (HERE / "config.py").read_text()
    lines = text.splitlines()
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.Assign)]
    rows = ["class default_config:"]
    for n in nodes:
        rows += ["    " + row for row in node_source(lines, n).splitlines()]
    return "\n".join(rows)


def bundle(name):
    extra, core, adapter, _ = BOTS[name]
    imports = {"import asyncio", "import os", "import sys"}
    bodies = [config_class()]
    for module in COMMON + extra:
        mod_imports, body = split_module(module)
        imports.update(mod_imports)
        bodies.append(f"# ---- {module}.py ----\n{body}")
    equity = json.dumps(json.loads((HERE / "preflop_equity.json").read_text()), separators=(",", ":"), sort_keys=True)
    chipzen = ("from chipzen import Action as ChipzenAction, Bot\n"
               "from chipzen.client import run_bot")
    profiles = {}
    if name in PROFILED and PROFILES_FILE.exists():
        profiles = {k: v for k, v in json.loads(PROFILES_FILE.read_text()).items() if not k.startswith("_")}
    text = (HEADER + "\n".join(sorted(imports)) + "\n\n" + chipzen + "\n\n" + bodies[0]
            + "\n\n_PREFLOP_TABLE = " + equity + "\n\n" + "\n\n".join(bodies[1:])
            + "\n\n_PROFILES = " + json.dumps(profiles, separators=(",", ":"), sort_keys=True) + "\n" + ADAPTER)
    return text.replace("_updated._PREFLOP_TABLE", "_PREFLOP_TABLE").replace("@NAME@", name).replace("@ADAPTER@", adapter).replace("@CORE@", core)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdk", help="local clone of chipzen-sdk (needed the first time, for the starter files)")
    ap.add_argument("--bot", type=str.upper, choices=list(BOTS), default="HU_UPDATED_OPTIMAL")
    ap.add_argument("--out", help="output folder (default: next to NLH_Poker_Bot)")
    args = ap.parse_args()
    out = Path(args.out) if args.out else HERE.parent / BOTS[args.bot][3]
    write(args.bot, out, args.sdk)


def write(bot, out, sdk=None):
    """Write out/bot.py, copying the starter's Dockerfile and friends from `sdk` if given."""
    out.mkdir(parents=True, exist_ok=True)
    if sdk:
        starter = Path(sdk) / "packages/python/starters/python"
        for name in ("Dockerfile", ".dockerignore", "requirements.txt"):
            shutil.copyfile(starter / name, out / name)
    elif not (out / "Dockerfile").exists():
        raise SystemExit(f"{out} has no Dockerfile yet: run once with --sdk")
    (out / "bot.py").write_text(bundle(bot))
    print(f"wrote {out / 'bot.py'} ({(out / 'bot.py').stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
