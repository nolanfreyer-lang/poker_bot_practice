"""
=====================================================================
  ENGINE ADAPTER  --  the ONLY file that knows the official interface.
=====================================================================
The starter kit's Agent interface is revealed at the start of the build
phase. When you get it:
  1. Rename `Agent` / its method to match the kit's template.
  2. Fill in `FIELD_MAP` (their key names -> ours) and `to_game_state`.
  3. Fill in `ACTION_FORMAT` / `from_action` for what they expect back.
  4. Run `python local_sim.py` to make sure nothing crashes and stays < 1 ms.

Exhibit D's message trace suggests text like "ACT? CHECK/BET" and cards
like "7♠"; poker_core.parse_cards already handles suit symbols and "10".
"""
from bot import PokerBot, GameState, PlayerInfo, Action
from poker_core import parse_cards

# ---- their observation keys -> our names (EDIT ON EVENT DAY) ----
FIELD_MAP = {
    "my_seat": "seat",
    "hole_cards": "hole_cards",
    "board": "community_cards",
    "street": "street",
    "pot": "pot",
    "players": "players",          # list of dicts, see PLAYER_FIELD_MAP
    "button_seat": "button",
    "small_blind": "small_blind",
    "big_blind": "big_blind",
    "to_call": "to_call",
    "min_raise_to": "min_raise",
    "max_raise_to": "max_raise",
    "history": "action_history",   # list of dicts, see HISTORY_FIELD_MAP
    "hand_number": "hand_number",
    "legal_actions": "legal_actions",
}
PLAYER_FIELD_MAP = {"seat": "seat", "stack": "stack", "bet": "bet",
                    "active": "active", "all_in": "all_in", "dealt": "in_game"}
HISTORY_FIELD_MAP = {"street": "street", "seat": "seat", "kind": "action", "amount": "amount"}

# their action names -> ours (lower-cased); Exhibit A legend: X C B R A F
ACTION_ALIASES = {"x": "check", "c": "call", "b": "bet", "r": "raise", "a": "allin",
                  "f": "fold", "all-in": "allin", "all_in": "allin", "raise_to": "raise",
                  "sb": "blind", "bb": "blind", "post": "blind"}
STREET_ALIASES = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}  # e.g. by board size

# our action -> what the engine wants back (EDIT ON EVENT DAY)
ACTION_FORMAT = {
    "fold":  lambda a: {"action": "fold"},
    "check": lambda a: {"action": "check"},
    "call":  lambda a: {"action": "call"},
    "bet":   lambda a: {"action": "bet", "amount": a.amount},
    "raise": lambda a: {"action": "raise", "amount": a.amount},
    "allin": lambda a: {"action": "all_in", "amount": a.amount},
}


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def to_game_state(obs) -> GameState:
    f = lambda k, d=None: _get(obs, FIELD_MAP[k], d)
    players = []
    for i, p in enumerate(f("players", [])):
        g = lambda k, d=None: _get(p, PLAYER_FIELD_MAP[k], d)
        players.append(PlayerInfo(seat=g("seat", i), stack=int(g("stack", 0)), bet=int(g("bet", 0)),
                                  active=bool(g("active", True)), all_in=bool(g("all_in", False)),
                                  dealt=bool(g("dealt", True))))
    players.sort(key=lambda p: p.seat)
    board = parse_cards(f("board", []))
    street = f("street") or STREET_ALIASES[len(board)]
    street = STREET_ALIASES.get(street, str(street).lower())
    history = []
    for h in f("history", []) or []:
        g = lambda k, d=None: _get(h, HISTORY_FIELD_MAP[k], d)
        kind = str(g("kind", "")).lower()
        history.append((str(g("street", "preflop")).lower(), g("seat"),
                        ACTION_ALIASES.get(kind, kind), int(g("amount", 0) or 0)))
    me = players[f("my_seat")]
    return GameState(
        my_seat=f("my_seat"), hole_cards=parse_cards(f("hole_cards")), board=board,
        street=street, pot=int(f("pot", 0)), players=players, button_seat=f("button_seat"),
        small_blind=int(f("small_blind", 0)), big_blind=int(f("big_blind", 0)),
        to_call=int(f("to_call", 0)),
        min_raise_to=int(f("min_raise_to", 0) or 0),
        max_raise_to=int(f("max_raise_to", 0) or (me.stack + me.bet)),
        history=history, hand_number=int(f("hand_number", 0) or 0),
        legal_actions=tuple(f("legal_actions", ()) or ()),
    )


def from_action(a: Action):
    return ACTION_FORMAT[a.kind](a)


class Agent:
    """Rename / re-sign to match the starter kit's Agent template."""

    def __init__(self, *args, **kwargs):
        self.bot = PokerBot()   # bot state resets at each table (rules 3.8)

    def act(self, observation):
        return from_action(self.bot.act(to_game_state(observation)))

    def on_hand_end(self, observation):
        s = to_game_state(observation)
        seats = [p.seat for p in s.players if p.dealt]
        if getattr(self.bot, "WANTS_SHOWDOWN", False):
            # EDIT ON EVENT DAY: where the engine reports cards shown at showdown,
            # as {seat: cards}. Bots like UpdatedOptimalBot learn bluff rates from it.
            shown = _get(observation, "showdown", {}) or {}
            self.bot.on_hand_end(s.history, seats, showdown={int(k): parse_cards(v) for k, v in shown.items()},
                                 board=s.board, my_seat=s.my_seat)
        else:
            self.bot.on_hand_end(s.history, seats)
