"""DEEP_THOUGHT_V5: DEEP_THOUGHT_V4 with balanced bluffing, adjusted by reads.

    from deep_thought_v5 import DeepThoughtV5Bot
    ("DEEP5", DeepThoughtV5Bot(seed=seed))

  1. Bluff frequency (BLUFF_CONTROL, off: it tested worse than v5 without it, losing
     to opponents that bet into every check). v4 bet nearly every weak hand it was first to act with: about
     two thirds of its big-blind flop leads were WEAK, twice a balanced share. Now
     heads-up weak bets are kept to the game-theory share for their size,
     size / (pot + 2 * size), with more room on the flop and turn where weak hands can
     still improve. Our own value bets in the same spot set how many bluffs that allows.
  2. Exploits. Each opponent's tracked fold-to-bet rate, compared with the rate that
     makes a bluff of this size break even, moves that frequency up (they over-fold)
     or down (they over-call).
  3. Weak leads into the last street's aggressor are sized within LEAD_SIZE_RANGE of the
     pot. On Chipzen, v4's weak leads made +0.44 BB each at 0.4-0.9 pot, lost 2.24 BB
     each above 1.2 pot and 0.15 BB each at 0.4 pot or less. Against opponents who raise
     leads often (Shuck1e1 28%, Pokerbot 73%), weak hands check instead.
  4. Saved profiles. load_profile() starts an opponent's reads from earlier matches, so
     the Chipzen package can remember bots by name (see chipzen_profiles.py).
"""
import config as default_config
from bot import Action, AGGRESSIVE
from deep_thought_v4 import DeepThoughtV4Bot
from poker_core import classify_hand

DEEP5_CONFIG = {
    # --- weak leads into the last street's aggressor, first to act heads-up ----------
    "LEAD_BUCKETS": ("WEAK",),
    "LEAD_SIZE_RANGE": (0.4, 0.9),      # share of the pot; None leaves v4's size
    "LEAD_RAISE_PRIOR": 0.15,           # assumed raise-to-lead rate before any data
    "LEAD_RAISE_WEIGHT": 6,
    "LEAD_MAX_RAISE": 0.3,              # check weak hands against opponents above this
    # --- bluff frequency for heads-up weak bets ------------------------------------
    # off: tested -0.97 to +1.21 bb/100 at 100 BB against +2.50 for v5 without it
    "BLUFF_CONTROL": False,
    "BLUFF_BUCKETS": ("WEAK",),
    "BLUFF_MAX_EQUITY": 0.45,           # weak hands with this much equity are value bets
    "BLUFF_STREET_ROOM": {"flop": 1.6, "turn": 1.3, "river": 1.0},   # x the river share
    "BLUFF_MAX_SHARE": 0.6,
    "BLUFF_VALUE_PRIOR": 0.6,           # our value bets per weak hand, before counting
    "BLUFF_VALUE_WEIGHT": 20,
    "BLUFF_EXPLOIT_POWER": 1.0,         # how hard their fold rate moves the frequency
    "BLUFF_MULT_RANGE": (0.25, 4.0),
    # --- saved profiles: earlier matches count like at most this many hands ----------
    "PROFILE_MAX_HANDS": 150,
}

# opponent trackers that a saved profile fills in, by attribute name
TRACKERS = ("stats", "sd", "freq", "street_reads", "raise_reads", "lead_reads")


def tracker_counts(bot, seat):
    """Every count the trackers hold for `seat`: {tracker: {table: [[subkey, count], ...]}}."""
    out = {}
    for name in TRACKERS:
        for table, values in vars(getattr(bot, name)).items():
            if not isinstance(values, dict):
                continue
            rows = [[list(k[1:]) if isinstance(k, tuple) else [], v] for k, v in values.items()
                    if k == seat or (isinstance(k, tuple) and k[0] == seat)]
            if rows:
                out.setdefault(name, {})[table] = rows
    return out


def led_into(history, street, me):
    """The seat we would lead into: the last street's aggressor, if nobody has acted yet."""
    if any(st == street for st, _, _, _ in history):
        return None
    prev = ("preflop", "flop", "turn", "river")[("flop", "turn", "river").index(street)]
    seat = None
    for st, s, kind, _ in history:
        if st == prev and kind in AGGRESSIVE:
            seat = s
    return seat if seat is not None and seat != me else None


class LeadStats:
    """How each opponent answers when we lead into them."""

    def __init__(self):
        self.raises = {}
        self.chances = {}

    def observe(self, history, me):
        for street in ("flop", "turn", "river"):
            acts = [(i, seat, kind) for i, (st, seat, kind, _) in enumerate(history) if st == street]
            if not acts or acts[0][1] != me or acts[0][2] not in AGGRESSIVE:
                continue
            target = led_into(history[:acts[0][0]], street, me)
            reply = next((kind for _, seat, kind in acts[1:] if seat == target), None)
            if target is not None and reply is not None:
                self.chances[target] = self.chances.get(target, 0) + 1
                if reply in AGGRESSIVE:
                    self.raises[target] = self.raises.get(target, 0) + 1

    def raise_rate(self, seat, cfg):
        w = cfg.LEAD_RAISE_WEIGHT
        return (self.raises.get(seat, 0) + cfg.LEAD_RAISE_PRIOR * w) / (self.chances.get(seat, 0) + w)


class DeepThoughtV5Bot(DeepThoughtV4Bot):
    def __init__(self, cfg=default_config, overrides=None, seed=None):
        super().__init__(cfg, {**DEEP5_CONFIG, **(overrides or {})}, seed)
        self.lead_reads = LeadStats()
        self.value_bets = {}        # (street, role) -> our non-weak bets
        self.weak_spots = {}        # (street, role) -> times we held a weak hand there

    def on_hand_end(self, history, seats_dealt, showdown=None, board=None, my_seat=None):
        super().on_hand_end(history, seats_dealt, showdown=showdown, board=board, my_seat=my_seat)
        self.lead_reads.observe(history, self._my_seat if my_seat is None else my_seat)

    def load_profile(self, seat, profile):
        """Start this seat's reads from a saved profile, scaled to PROFILE_MAX_HANDS."""
        scale = min(1.0, self.cfg.PROFILE_MAX_HANDS / max(1, profile.get("hands", 1)))
        for name, tables in profile.get("counts", {}).items():
            tracker = getattr(self, name, None)
            for table, rows in tables.items():
                values = getattr(tracker, table, None)
                if not isinstance(values, dict):
                    continue
                for sub, count in rows:
                    key = (seat, *sub) if sub else seat
                    values[key] = values.get(key, 0) + count * scale

    # ------------------------------------------------------------ bluff frequency
    def _role(self, s, opp):
        """'lead' into the last street's aggressor, 'aggressor' if that was us, else 'probe'."""
        if led_into(s.history, s.street, s.my_seat) == opp:
            return "lead"
        prev = ("preflop", "flop", "turn", "river")[("flop", "turn", "river").index(s.street)]
        last = None
        for st, seat, kind, _ in s.history:
            if st == prev and kind in AGGRESSIVE:
                last = seat
        return "aggressor" if last == s.my_seat else "probe"

    def _bluff_gate(self, s, a, opp, key):
        """Keep a weak bet with the balanced frequency for its size, moved by their fold rate."""
        c = self.cfg
        size = (s.me.stack if a.kind == "allin" else a.amount - s.me.bet) / max(1, s.pot)
        share = min(c.BLUFF_MAX_SHARE, size / (1 + 2 * size) * c.BLUFF_STREET_ROOM[s.street])
        w = c.BLUFF_VALUE_WEIGHT
        value = (self.value_bets.get(key, 0) + c.BLUFF_VALUE_PRIOR * w) / (self.weak_spots.get(key, 0) + w)
        chance = share / (1 - share) * value
        fold, even = self.sd.fold_rate(opp, c), size / (1 + size)
        low, high = c.BLUFF_MULT_RANGE
        mult = ((fold / (1 - fold)) / (even / (1 - even))) ** c.BLUFF_EXPLOIT_POWER
        chance *= min(high, max(low, mult))
        return a if self.rng.random() < chance else Action("check")

    def _lead(self, s, a):
        """Weak leads: check against opponents who raise leads, else keep the size in range."""
        c = self.cfg
        opps = [p for p in s.players if p.active and p.seat != s.my_seat]
        target = led_into(s.history, s.street, s.my_seat)
        if len(opps) != 1 or target != opps[0].seat or a.kind != "bet" \
                or classify_hand(s.hole_cards, s.board, self.good_kicker) not in c.LEAD_BUCKETS:
            return a
        if self.lead_reads.raise_rate(target, c) > c.LEAD_MAX_RAISE:
            return Action("check")
        if c.LEAD_SIZE_RANGE is None:
            return a
        low, high = c.LEAD_SIZE_RANGE
        bet = min(max(a.amount - s.me.bet, int(low * s.pot)), int(high * s.pot))
        return Action("bet", s.me.bet + bet)

    def _postflop(self, s):
        a = super()._postflop(s)
        opps = [p for p in s.players if p.active and p.seat != s.my_seat]
        if s.to_call or len(opps) != 1:
            return a
        if a.kind in AGGRESSIVE:
            a = self._lead(s, a)
        c = self.cfg
        if not c.BLUFF_CONTROL:
            return a
        bucket = classify_hand(s.hole_cards, s.board, self.good_kicker)
        key = (s.street, self._role(s, opps[0].seat))
        # a weak hand ahead of their range is a thin value bet, not a bluff
        equity = (self.last_ev or {}).get("equity", 0)
        if bucket not in c.BLUFF_BUCKETS or equity >= c.BLUFF_MAX_EQUITY:
            if a.kind in AGGRESSIVE:
                self.value_bets[key] = self.value_bets.get(key, 0) + 1
            return a
        self.weak_spots[key] = self.weak_spots.get(key, 0) + 1
        return self._bluff_gate(s, a, opps[0].seat, key) if a.kind in AGGRESSIVE else a
