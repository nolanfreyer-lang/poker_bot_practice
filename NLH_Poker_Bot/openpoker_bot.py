"""Deep_Thought v19 on Open Poker (openpoker.ai): a self-hosted WebSocket client.

    export OPEN_POKER_API_KEY=...        # from the openpoker.ai dashboard (Self Host); never in source
    python3 openpoker_bot.py             # connect, join the lobby, play until stopped

Open Poker is a 6-max cash game (10/20 blinds, buy-in 1,000-5,000 chips, 45 s per action,
season score = chips), not Chipzen's heads-up matches. What this file adds around v19:

  * Protocol: hand_start / hole_cards / player_action / community_cards / your_turn /
    hand_result become the bot's GameState and history; its Action goes back as an `action`
    with hand_id, turn_token and a fresh client_action_id. Amounts are read from stack
    changes (contribution_delta, stack_before/after, or the last known stack), so the
    client does not depend on how `amount` is defined. Blinds are synthesized from the
    dealer seat if the server doesn't send them as actions.
  * Seats change hands at 6-max. The bot keys its reads by seat, so when a new bot sits down
    the seat's reads are cleared and refilled from what we remember about that bot.
  * Memory across sessions: every hand is appended to openpoker_data/hands.jsonl. At
    startup it is replayed into a separate "memory" bot keyed by bot name, which keeps
    learning live; a bot that sits down starts from its profile there (v10's
    name-family fallback included). Only showdown cards are known, as in live play.
  * Robustness: decisions run off the event loop; a crash or rejected action falls back
    to check/call/fold; reconnects back off up to 60 s and resync the table.

Options (environment): OPEN_POKER_BUY_IN (default 2000 = 100 BB), OPEN_POKER_URL,
OPEN_POKER_DATA (folder for the hand log), OPEN_POKER_VERBOSE=1.
"""
import asyncio
import json
import os
import random
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bot import Action, GameState, PlayerInfo          # noqa: E402
from deep_thought_v14 import TRACKERS_V14, tracker_counts   # noqa: E402
from deep_thought_v19 import DeepThoughtV19Bot          # noqa: E402
from poker_core import parse_cards                      # noqa: E402

CORE = DeepThoughtV19Bot      # v17 + 3-6 player preflop charts + strong calls past the stack-off cap
OVERRIDES = {"USE_ICM": False}      # cash game: chips are score, no payout ladder
BLIND_ACTIONS = ("small_blind", "big_blind", "post_small_blind", "post_big_blind", "blind", "post_blind")
MAX_SEATS = 6


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# =====================================================================
# Memory: every hand we have seen, by bot name
# =====================================================================
class Memory:
    """A copy of the bot fed every logged hand with seats replaced by one id per bot name."""

    def __init__(self, path):
        self.path = Path(path)
        self.bot = CORE(seed=0, overrides=OVERRIDES)
        self.ids = {}           # name -> virtual seat (0 is us)
        self.hands = {}         # name -> hands seen
        self.loaded = 0
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    try:
                        self.feed(json.loads(line))
                        self.loaded += 1
                    except (ValueError, KeyError, TypeError):
                        continue

    def vid(self, name):
        if name not in self.ids:
            self.ids[name] = len(self.ids) + 1
        return self.ids[name]

    def feed(self, rec):
        names = {int(k): v for k, v in rec["names"].items()}
        me = rec["me"]
        seat_id = {s: (0 if s == me else self.vid(n)) for s, n in names.items()}
        if any(seat not in seat_id for _, seat, _, _ in rec["history"]):
            return
        history = [(st, seat_id[seat], kind, amt) for st, seat, kind, amt in rec["history"]]
        dealt = [seat_id[s] for s in rec["dealt"] if s in seat_id]
        shown = {seat_id[int(s)]: parse_cards(c) for s, c in rec.get("shown", {}).items() if int(s) in seat_id}
        self.bot.on_hand_end(history, dealt, showdown=shown, board=parse_cards(rec["board"]), my_seat=0)
        for s in rec["dealt"]:
            if s != me and s in names:
                self.hands[names[s]] = self.hands.get(names[s], 0) + 1

    def record(self, rec):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self.feed(rec)

    def profiles(self):
        """{name: profile} in the format load_profile / load_named_profile read."""
        return {name: {"hands": self.hands.get(name, 0), "counts": tracker_counts(self.bot, vid)}
                for name, vid in self.ids.items() if self.hands.get(name)}


def clear_seat(bot, seat):
    """Forget everything the bot's trackers hold for this seat."""
    for name in TRACKERS_V14:
        for table in vars(getattr(bot, name)).values():
            if isinstance(table, dict):
                for k in [k for k in table if k == seat or (isinstance(k, tuple) and k and k[0] == seat)]:
                    del table[k]
    for extra in ("forced", "profile_source"):
        getattr(bot, extra, {}).pop(seat, None)


# =====================================================================
# One player's view of the table, message by message
# =====================================================================
class OpenPokerPlayer:
    """Turns Open Poker server messages into the bot's decisions. handle() returns messages to send."""

    def __init__(self, memory, seed=None, buy_in=2000):
        self.memory = memory
        self.seed = seed
        self.buy_in = buy_in
        self.core = None
        self.names = {}             # seat -> bot name at this table
        self.core_names = {}        # seat -> name the core's reads currently belong to
        self.stacks = {}            # seat -> last known chips behind
        self.status = {}            # seat -> table_state status, when known
        self.table_id = None
        self.table_seq = 0
        self.hand_no = 0
        self.errors = 0
        self.decisions = 0
        self.me = 0
        self.new_table()

    # ---------------------------------------------------------------- table
    def new_table(self, table_id=None):
        self.core = CORE(seed=self.seed, overrides=OVERRIDES)
        self.core_names = {}
        self.names, self.stacks, self.status = {}, {}, {}
        self.table_id = table_id
        self.hand = None

    def sync_seats(self):
        """Give each seat's reads to the bot sitting there now, from memory."""
        profiles = None
        for seat in range(MAX_SEATS):
            if seat == self.me:
                continue
            name = self.names.get(seat)
            if self.core_names.get(seat) == name:
                continue
            clear_seat(self.core, seat)
            self.core_names[seat] = name
            if name:
                profiles = self.memory.profiles() if profiles is None else profiles
                source = self.core.load_named_profile(seat, name, profiles)
                if source:
                    log(f"seat {seat}: {name} starts from {source}'s profile ({profiles[source]['hands']} hands)")

    # ---------------------------------------------------------------- hand state
    def start_hand(self, msg):
        self.hand_no += 1
        self.me = msg.get("seat", getattr(self, "me", 0))
        blinds = msg.get("blinds") or {}
        seated = [s for s, n in self.names.items() if n and self.stacks.get(s, 0) > 0
                  and self.status.get(s, "active") not in ("sitting_out", "away", "empty")]
        if self.me not in seated:
            seated.append(self.me)
        self.hand = {
            "id": msg.get("hand_id"), "dealer": msg.get("dealer_seat", 0),
            "sb": blinds.get("small_blind", 10), "bb": blinds.get("big_blind", 20),
            "dealt": sorted(seated), "start": {s: self.stacks.get(s, 0) for s in seated},
            "names": {s: self.names.get(s) for s in seated if self.names.get(s)},
            "hole": [], "board": [], "street": "preflop", "history": [], "bets": {}, "high": 0,
            "blinds_done": False,
        }
        self.names.setdefault(self.me, "me")
        self.sync_seats()

    def blind_seats(self):
        h = self.hand
        order = sorted(h["dealt"])
        if len(order) < 2:
            return []
        i = order.index(h["dealer"]) if h["dealer"] in order else 0
        if len(order) == 2:
            return [order[i], order[(i + 1) % 2]]
        return [order[(i + 1) % len(order)], order[(i + 2) % len(order)]]

    def ensure_blinds(self):
        """If the server didn't send blinds as actions, post them ourselves from the dealer seat."""
        h = self.hand
        if h is None or h["blinds_done"]:
            return
        h["blinds_done"] = True
        for seat, amount in zip(self.blind_seats(), (h["sb"], h["bb"])):
            paid = min(amount, self.stacks.get(seat, amount))
            self.stacks[seat] = self.stacks.get(seat, paid) - paid
            self.add_chips(seat, paid, blind=True)

    def add_chips(self, seat, delta, blind=False, kind_hint=None):
        h = self.hand
        total = h["bets"].get(seat, 0) + max(0, delta)
        h["bets"][seat] = total
        if blind:
            kind = "blind"
        elif kind_hint in ("fold", "check"):
            kind = kind_hint
        elif self.stacks.get(seat, 1) == 0 and delta > 0:
            kind = "allin"          # as the simulator records it, calls that empty the stack included
        elif total > h["high"]:
            kind = "bet" if h["high"] == 0 and h["street"] != "preflop" else "raise"
        else:
            kind = "call"
        h["high"] = max(h["high"], total)
        h["history"].append((h["street"], seat, kind, total))

    def set_street(self, street):
        h = self.hand
        if street and street != h["street"] and street in ("flop", "turn", "river"):
            self.ensure_blinds()
            h["street"], h["bets"], h["high"] = street, {}, 0

    def on_player_action(self, msg):
        h = self.hand
        if h is None:
            return
        seat, act = msg.get("seat"), (msg.get("action") or "").lower()
        if msg.get("name"):
            self.names[seat] = msg["name"]
        self.set_street(msg.get("street"))
        before = msg.get("stack_before")
        after = msg.get("stack_after", msg.get("stack"))
        if act not in BLIND_ACTIONS:
            self.ensure_blinds()        # first, so a blind poster's known stack is net of the blind
        known = self.stacks.get(seat)
        if act in BLIND_ACTIONS:
            h["blinds_done"] = True
            delta = msg.get("contribution_delta") or msg.get("amount") or 0
            if after is not None:
                self.stacks[seat] = after
            self.add_chips(seat, delta, blind=True)
            return
        if act in ("fold", "check"):
            self.add_chips(seat, 0, kind_hint=act)
        else:
            delta = msg.get("contribution_delta")
            if delta is None and before is not None and after is not None:
                delta = before - after
            if delta is None and known is not None and after is not None:
                delta = known - after
            if delta is None:       # last resort: read `amount` as the street total
                delta = (msg.get("amount") or 0) - h["bets"].get(seat, 0)
            if after is not None:
                self.stacks[seat] = after
            self.add_chips(seat, delta)
        if after is not None:
            self.stacks[seat] = after

    def update_players(self, players):
        for p in players or []:
            seat = p.get("seat")
            if seat is None:
                continue
            if p.get("name") is not None:
                self.names[seat] = p["name"]
            if p.get("stack") is not None:
                self.stacks[seat] = p["stack"]
            if p.get("status"):
                self.status[seat] = p["status"]
                if p["status"] == "empty":
                    self.names.pop(seat, None)

    # ---------------------------------------------------------------- decisions
    def game_state(self, msg):
        h = self.hand
        self.ensure_blinds()
        if msg.get("community_cards") is not None and len(msg["community_cards"]) > len(h["board"]):
            h["board"] = list(msg["community_cards"])
            self.set_street({3: "flop", 4: "turn", 5: "river"}.get(len(h["board"])))
        self.update_players(msg.get("players"))
        valid = {a["action"]: a for a in msg.get("valid_actions") or []}
        folded = {seat for _, seat, kind, _ in h["history"] if kind == "fold"}
        n = max([MAX_SEATS] + [s + 1 for s in self.names] + [s + 1 for s in h["dealt"]])
        players = []
        for seat in range(n):
            dealt = seat in h["dealt"]
            stack = self.stacks.get(seat, 0)
            players.append(PlayerInfo(seat=seat, stack=stack, bet=h["bets"].get(seat, 0) if dealt else 0,
                                      active=dealt and seat not in folded, all_in=dealt and stack == 0
                                      and seat not in folded, dealt=dealt))
        me = players[self.me]
        to_call = (valid.get("call") or {}).get("amount") or 0
        to_call = min(to_call, me.stack)
        if "raise" in valid:
            lo, hi = valid["raise"].get("min"), valid["raise"].get("max")
        elif "all_in" in valid:
            lo = hi = valid["all_in"].get("amount") or me.stack + me.bet
        else:
            lo = hi = me.bet + to_call
        return GameState(my_seat=self.me, hole_cards=parse_cards(h["hole"]), board=parse_cards(h["board"]),
                         street=h["street"], pot=msg.get("pot") or sum(h["bets"].values()), players=players,
                         button_seat=h["dealer"], small_blind=h["sb"], big_blind=h["bb"], to_call=to_call,
                         min_raise_to=lo, max_raise_to=hi, history=list(h["history"]), hand_number=self.hand_no)

    @staticmethod
    def to_wire(a, s, valid):
        """The bot's Action -> an Open Poker action that valid_actions allows."""
        def passive():
            for name in ("check", "call", "fold"):
                if name in valid:
                    return {"action": name}
            return {"action": "fold"}

        if a.kind == "fold":
            return {"action": "check"} if "check" in valid else {"action": "fold"}
        if a.kind in ("check", "call"):
            return {"action": a.kind} if a.kind in valid else passive()
        # bet / raise / allin
        if a.kind == "allin" and s.to_call >= s.me.stack:
            return {"action": "call"} if "call" in valid else passive()
        raise_ = valid.get("raise")
        target = None if a.kind == "allin" else int(a.amount)
        if raise_ and (target is None or target >= raise_["max"]):
            return {"action": "all_in"} if "all_in" in valid else {"action": "raise", "amount": int(raise_["max"])}
        if raise_:
            return {"action": "raise", "amount": int(max(raise_["min"], min(target, raise_["max"])))}
        if "all_in" in valid:
            return {"action": "all_in"}
        return passive()

    def decide(self, msg):
        """your_turn -> the action message (runs off the event loop)."""
        valid = {a["action"]: a for a in msg.get("valid_actions") or []}
        try:
            s = self.game_state(msg)
            out = self.to_wire(self.core.act(s), s, valid)
        except Exception as e:      # a crash must never cost the hand
            self.errors += 1
            if self.errors <= 20:
                log(f"decide error: {e!r}")
            out = {"action": "check"} if "check" in valid else {"action": "call"} if "call" in valid \
                else {"action": "fold"}
        self.decisions += 1
        out.update(type="action", hand_id=msg.get("hand_id") or (self.hand or {}).get("id"),
                   turn_token=msg.get("turn_token"), client_action_id=str(uuid.uuid4()))
        return out

    # ---------------------------------------------------------------- hand end
    def end_hand(self, msg):
        h = self.hand
        if h is None:
            return
        self.ensure_blinds()
        shown = {int(s): c for s, c in (msg.get("shown_cards") or {}).items() if c and "??" not in c}
        for s, v in (msg.get("final_stacks") or {}).items():
            self.stacks[int(s)] = v
        try:
            self.core.on_hand_end(h["history"], h["dealt"], showdown={s: parse_cards(c) for s, c in shown.items()},
                                  board=parse_cards(h["board"]), my_seat=self.me)
        except Exception as e:
            self.errors += 1
            if self.errors <= 20:
                log(f"hand end error: {e!r}")
        names = {s: n for s, n in self.names.items() if s in h["dealt"] and n}
        names[self.me] = "me"
        rec = {"ts": int(time.time()), "table_id": self.table_id, "hand_id": h["id"], "me": self.me,
               "names": {str(s): n for s, n in names.items()}, "dealt": h["dealt"], "blinds": [h["sb"], h["bb"]],
               "history": h["history"], "board": h["board"], "shown": {str(s): c for s, c in shown.items()},
               "start": {str(s): v for s, v in h["start"].items()},
               "final": {str(k): v for k, v in (msg.get("final_stacks") or {}).items()}}
        try:
            self.memory.record(rec)
        except Exception as e:
            log(f"hand log error: {e!r}")
        self.hand = None

    # ---------------------------------------------------------------- dispatch
    def handle(self, msg):
        """Apply one server message. Returns (messages to send now, your_turn needing a decision or None)."""
        t = msg.get("type")
        seq = msg.get("table_seq")
        if isinstance(seq, int):
            self.table_seq = max(self.table_seq, seq)
        out, turn = [], None
        if t == "connected":
            log(f"connected as {msg.get('name')}")
            out += [{"type": "join_lobby", "buy_in": self.buy_in}, {"type": "set_auto_rebuy", "enabled": True}]
        elif t == "table_joined":
            self.new_table(msg.get("table_id"))
            self.me = msg.get("seat", 0)
            self.update_players(msg.get("players"))
            log(f"seated at {self.table_id}, seat {self.me}: "
                + ", ".join(f"{p.get('name')}({p.get('stack')})" for p in msg.get("players") or []))
        elif t == "player_joined":
            self.update_players([msg])
        elif t == "player_left":
            self.names.pop(msg.get("seat"), None)
            self.stacks.pop(msg.get("seat"), None)
        elif t == "hand_start":
            self.start_hand(msg)
        elif t == "hole_cards" and self.hand is not None:
            self.hand["hole"] = list(msg.get("cards") or [])
        elif t == "player_action":
            self.on_player_action(msg)
        elif t == "community_cards" and self.hand is not None:
            self.hand["board"] += [c for c in msg.get("cards") or [] if c not in self.hand["board"]]
            self.set_street(msg.get("street"))
        elif t == "your_turn":
            if self.hand is None:       # joined mid-hand: act on what we have
                self.start_hand({"hand_id": msg.get("hand_id"), "seat": getattr(self, "me", 0)})
            turn = msg
        elif t == "hand_result":
            self.end_hand(msg)
        elif t == "table_state":
            self.update_players(msg.get("seats"))
        elif t == "resync_response":
            for ev in sorted(msg.get("replayed_events") or [], key=lambda e: e.get("table_seq") or 0):
                o, _ = self.handle(ev)
                out += o
            snap = msg.get("snapshot") or {}
            self.update_players(snap.get("seats"))
            hero = snap.get("hero") or {}
            if hero.get("turn_token") and hero.get("valid_actions"):
                if self.hand is None:
                    self.start_hand({"hand_id": snap.get("hand_id") or msg.get("hand_id"),
                                     "seat": hero.get("seat", getattr(self, "me", 0)),
                                     "dealer_seat": snap.get("dealer_seat", 0),
                                     "blinds": {"small_blind": snap.get("small_blind", 10),
                                                "big_blind": snap.get("big_blind", 20)}})
                    self.hand["hole"] = list(hero.get("hole_cards") or [])
                turn = {"type": "your_turn", "hand_id": snap.get("hand_id") or msg.get("hand_id"),
                        "valid_actions": hero["valid_actions"], "pot": snap.get("pot"),
                        "community_cards": snap.get("board"), "turn_token": hero["turn_token"]}
        elif t in ("table_closed", "season_ended"):
            log(f"{t}: {msg.get('reason') or msg.get('next_season_number')}; rejoining the lobby")
            self.hand = None
            out.append({"type": "join_lobby", "buy_in": self.buy_in})
        elif t == "busted":
            out.append({"type": "rebuy", "amount": 0})
        elif t == "rebuy_confirmed":
            out.append({"type": "join_lobby", "buy_in": self.buy_in})
        elif t == "action_rejected":
            log(f"action rejected: {msg.get('code')}")
        elif t == "error":
            code = msg.get("code")
            log(f"error {code}: {msg.get('message')}")
            if code == "already_seated" and msg.get("table_id"):
                self.table_id = msg["table_id"]
                out.append({"type": "resync_request", "table_id": self.table_id})
        return out, turn


# =====================================================================
# Network loop
# =====================================================================
async def run(url, api_key, player):
    import websockets
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(url, additional_headers={"Authorization": f"Bearer {api_key}"},
                                          ping_interval=20, max_size=2 ** 22) as ws:
                backoff = 1.0
                if player.table_id:
                    await ws.send(json.dumps({"type": "resync_request", "table_id": player.table_id,
                                              "last_table_seq": player.table_seq}))
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except ValueError:
                        continue
                    if os.environ.get("OPEN_POKER_VERBOSE"):
                        log("<", raw[:300])
                    out, turn = player.handle(msg)
                    if msg.get("type") == "connected" and player.table_id:
                        out = [m for m in out if m["type"] != "join_lobby"]
                    for m in out:
                        await ws.send(json.dumps(m))
                    if turn is not None:
                        action = await asyncio.to_thread(player.decide, turn)
                        await ws.send(json.dumps(action))
                    if msg.get("type") == "hand_result" and player.hand_no % 50 == 0:
                        log(f"{player.hand_no} hands, {player.decisions} decisions, {player.errors} errors")
                    if msg.get("type") == "error" and msg.get("code") == "auth_failed":
                        log("auth failed: check OPEN_POKER_API_KEY")
                        return
        except Exception as e:
            log(f"connection lost ({e!r}); retrying in {backoff:.0f}s")
        await asyncio.sleep(backoff + random.random())
        backoff = min(60.0, backoff * 2)


def main():
    key = os.environ.get("OPEN_POKER_API_KEY")
    if not key:
        sys.exit("set OPEN_POKER_API_KEY (from the openpoker.ai dashboard, Self Host)")
    data = Path(os.environ.get("OPEN_POKER_DATA") or HERE / "openpoker_data")
    memory = Memory(data / "hands.jsonl")
    log(f"memory: {memory.loaded} hands, {len(memory.ids)} bots")
    player = OpenPokerPlayer(memory, buy_in=int(os.environ.get("OPEN_POKER_BUY_IN", 2000)))
    asyncio.run(run(os.environ.get("OPEN_POKER_URL", "wss://openpoker.ai/ws"), key, player))


if __name__ == "__main__":
    main()
