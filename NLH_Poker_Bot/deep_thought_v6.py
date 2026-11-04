"""DEEP_THOUGHT_V6: DEEP_THOUGHT_V5 with reads that look at bet size, not just frequency.

    from deep_thought_v6 import DeepThoughtV6Bot
    ("DEEP6", DeepThoughtV6Bot(seed=seed))

  1. Bluff reads by size. v5 pooled every bet an opponent showed down, so against a bot
     whose cheap stabs are bluffs and whose big bets are value, the stabs made its big
     bets look like bluffs too.
     Shown bets are now split into small and big (BIG_BET_SIZE of the pot), and a bet
     we face is judged by the bluff rate of its own size, starting from the pooled rate
     until that size has been seen at showdown.
  2. Big 3-bets (BIG_3BET_READS, off: v5 already calls a narrow range against 3-bets
     above HU_3BET_TIGHT_MULT x, and folding more tested -0.4 +- 0.9 bb/100 at 100 BB
     against a bot that 3-bets 6x with only TT+ and AQ+; the reads are still recorded).
     When an opponent 3-bets our open to more than HU_3BET_TIGHT_MULT x,
     we estimate how much of that range is value: from how rarely they 3-bet at all
     (half the reference rate means about twice the value share) and from the hands
     they have shown after a big 3-bet. Once that estimate reaches TIGHT_VALUE_SHARE,
     the button calls only HU_BTN_CALL_TIGHT_BIG_3BET (4-bets are unchanged).
  3. Bluff reads by line, within each size: a bet by the last street's aggressor, a lead
     into that aggressor, a bet after a check, or a raise. LimitLess, for one, bet the
     turn into the flop bettor with air 6 times in 10 but bet the river after a check
     with value 7 times in 10, both at its usual half pot. Tested +0.59 +- 0.33 bb/100
     at 100 BB and +0.44 +- 0.19 at 30 BB over size reads alone.
  Size reads tested +1.25 +- 0.51 bb/100 at 100 BB and +0.79 +- 0.27 at 30 BB over v5.
  Both reads are saved in Chipzen profiles (see chipzen_profiles.py).
"""
import config as default_config
from bot import AGGRESSIVE
from deep_thought_v5 import DeepThoughtV5Bot, TRACKERS
from poker_core import classify_hand, hand_class, parse_range
from updated_optimal import ShowdownStats, BLUFF_WEIGHT

DEEP6_CONFIG = {
    # --- bluff reads by bet size ---------------------------------------------------
    "SIZE_READS": True,
    "BIG_BET_SIZE": 0.6,                # bets of at least this share of the pot are big
    "SIZE_BLUFF_WEIGHT": 6,             # the pooled bluff rate counts like this many bets
    # --- bluff reads by line, within each size ----------------------------------------
    "LINE_READS": True,
    "LINE_BLUFF_WEIGHT": 6,             # the size's bluff rate counts like this many bets
    # --- big 3-bets -------------------------------------------------------------------
    "BIG_3BET_READS": False,
    "BIG_3BET_VALUE_PRIOR": 0.5,        # value share of a big 3-bet at the reference 3-bet rate
    "BIG_3BET_VALUE_MAX": 0.9,
    "BIG_3BET_VALUE_WEIGHT": 4,         # the prior counts like this many shown hands
    "BIG_3BET_VALUE_RANGE": "88+, ATs+, KQs, AJo+, KQo",
    "TIGHT_VALUE_SHARE": 0.7,
    "HU_BTN_CALL_TIGHT_BIG_3BET": "TT-88, AQs-AJs, KQs, AQo",
}

# opponent trackers that a saved profile fills in, by attribute name
TRACKERS_V6 = TRACKERS + ("big3",)
STREET_LEN = {"flop": 3, "turn": 4, "river": 5}


def tracker_counts(bot, seat):
    """Every count the trackers hold for `seat`: {tracker: {table: [[subkey, count], ...]}}."""
    out = {}
    for name in TRACKERS_V6:
        for table, values in vars(getattr(bot, name)).items():
            if not isinstance(values, dict):
                continue
            rows = [[list(k[1:]) if isinstance(k, tuple) else [], v] for k, v in values.items()
                    if k == seat or (isinstance(k, tuple) and k[0] == seat)]
            if rows:
                out.setdefault(name, {})[table] = rows
    return out


def bet_lines(history):
    """(street, seat, size, line) for every postflop bet or raise.

    size is the raise over the pot after calling. line is "raise" over a bet, "agg" by
    the last street's aggressor, "lead" into that aggressor before they act, or "free"
    (a bet after a check, or after a street with no bets)."""
    out, pot, last = [], 0, None
    for street in ("preflop", "flop", "turn", "river"):
        put, high, acted, agg = {}, 0, set(), None
        for st, seat, kind, amt in history:
            if st != street:
                continue
            if kind in AGGRESSIVE and amt > high:
                if street != "preflop":
                    call = high - put.get(seat, 0)
                    size = (amt - high) / max(1, pot + sum(put.values()) + call)
                    line = ("raise" if high else "agg" if seat == last
                            else "lead" if last is not None and not acted else "free")
                    out.append((street, seat, size, line))
                agg = seat
            if kind != "blind":
                acted.add(seat)
            if kind != "fold":
                put[seat] = max(put.get(seat, 0), amt)
            high = max(high, amt)
        pot += sum(put.values())
        last = agg
    return out


class SizedShowdownStats(ShowdownStats):
    """ShowdownStats that also splits shown bets by size and line, and judges a bet by both."""

    def __init__(self, big_size=0.6, weight=6, line_weight=6):
        super().__init__()
        self.sized_bets = {}      # (seat, river|early, small|big) -> bets shown
        self.sized_bluffs = {}    # ... of which were bluffs (weighted)
        self.line_bets = {}       # (seat, river|early, small|big, line) -> bets shown
        self.line_bluffs = {}
        self.big_size, self.weight, self.line_weight = big_size, weight, line_weight
        self.facing = None        # the size of the bet we face now, set by the bot
        self.facing_line = None   # ... and its line, None when line reads are off
        # Count bets whose cards we see without a showdown too. Replays show every hand,
        # so profiles see each bet, bluffs that won included; live play sees showdowns.
        self.see_all = False

    def _group(self, street, size):
        return ("river" if street == "river" else "early", "big" if size >= self.big_size else "small")

    def observe(self, history, showdown, board, good_kicker, me, semi_bluff_weight):
        super().observe(history, showdown, board, good_kicker, me, semi_bluff_weight)
        folded = {seat for _, seat, kind, _ in history if kind == "fold"}
        shown = dict(showdown or {})
        if not self.see_all:
            shown = {seat: cards for seat, cards in shown.items() if seat not in folded}
            if len(shown) < 2 or not board or len(board) < 5:
                return
        biggest = {}            # (street, seat) -> [largest size, line of their first bet]
        for street, seat, size, line in bet_lines(history):
            if seat in shown and seat != me and len(board or []) >= STREET_LEN[street]:
                top = biggest.setdefault((street, seat), [size, line])
                top[0] = max(top[0], size)
        for (street, seat), (size, line) in biggest.items():
            bucket = classify_hand(list(shown[seat]), list(board[:STREET_LEN[street]]), good_kicker)
            weight = BLUFF_WEIGHT.get(bucket, semi_bluff_weight if bucket == "STRONG_DRAW" else 0.0)
            key = (seat, *self._group(street, size))
            self.sized_bets[key] = self.sized_bets.get(key, 0) + 1
            self.sized_bluffs[key] = self.sized_bluffs.get(key, 0) + weight
            key += (line,)
            self.line_bets[key] = self.line_bets.get(key, 0) + 1
            self.line_bluffs[key] = self.line_bluffs.get(key, 0) + weight

    def bluff_rate(self, seat, cfg, street="flop"):
        pooled = super().bluff_rate(seat, cfg, street)
        if self.facing is None:
            return pooled
        key = (seat, *self._group(street, self.facing))
        w = self.weight
        sized = (self.sized_bluffs.get(key, 0) + pooled * w) / (self.sized_bets.get(key, 0) + w)
        if self.facing_line is None:
            return sized
        key += (self.facing_line,)
        w = self.line_weight
        return (self.line_bluffs.get(key, 0) + sized * w) / (self.line_bets.get(key, 0) + w)


class Big3BetStats:
    """Each opponent's big 3-bets over our open, and what they showed with them."""

    def __init__(self, mult=5.0, value_range=frozenset()):
        self.made = {}       # big 3-bets over our open
        self.shown = {}      # ... seen at showdown
        self.value = {}      # ... of which were in the value range
        self.mult, self.value_range = mult, value_range

    def observe(self, history, showdown, me):
        bb = max([amt for st, _, kind, amt in history if st == "preflop" and kind == "blind"] or [0])
        opens = [(seat, amt) for st, seat, kind, amt in history
                 if st == "preflop" and kind in AGGRESSIVE and amt > bb]
        if len(opens) < 2 or opens[0][0] != me or opens[1][0] == me or opens[1][1] <= opens[0][1] * self.mult:
            return
        seat = opens[1][0]
        self.made[seat] = self.made.get(seat, 0) + 1
        # any cards we are shown count: replays show every hand, live play only showdowns
        cards = (showdown or {}).get(seat)
        if cards:
            self.shown[seat] = self.shown.get(seat, 0) + 1
            if hand_class(*cards) in self.value_range:
                self.value[seat] = self.value.get(seat, 0) + 1

    def value_share(self, seat, freq_ratio, cfg):
        prior = min(cfg.BIG_3BET_VALUE_MAX, cfg.BIG_3BET_VALUE_PRIOR / max(0.01, freq_ratio))
        w = cfg.BIG_3BET_VALUE_WEIGHT
        return (self.value.get(seat, 0) + prior * w) / (self.shown.get(seat, 0) + w)


class DeepThoughtV6Bot(DeepThoughtV5Bot):
    def __init__(self, cfg=default_config, overrides=None, seed=None):
        super().__init__(cfg, {**DEEP6_CONFIG, **(overrides or {})}, seed)
        c = self.cfg
        self.sd = SizedShowdownStats(c.BIG_BET_SIZE, c.SIZE_BLUFF_WEIGHT, c.LINE_BLUFF_WEIGHT)
        self.big3 = Big3BetStats(c.HU_3BET_TIGHT_MULT, parse_range(c.BIG_3BET_VALUE_RANGE))
        self.tight_big_3bet_call = parse_range(c.HU_BTN_CALL_TIGHT_BIG_3BET)

    def on_hand_end(self, history, seats_dealt, showdown=None, board=None, my_seat=None):
        super().on_hand_end(history, seats_dealt, showdown=showdown, board=board, my_seat=my_seat)
        self.big3.observe(history, showdown, self._my_seat if my_seat is None else my_seat)

    def _postflop(self, s):
        if self.cfg.SIZE_READS and s.to_call:
            self.sd.facing = s.to_call / max(1, s.pot - s.to_call)
            if self.cfg.LINE_READS:
                lines = [line for st, seat, _, line in bet_lines(s.history)
                         if st == s.street and seat != s.my_seat]
                self.sd.facing_line = lines[-1] if lines else None
        try:
            return super()._postflop(s)
        finally:
            self.sd.facing = self.sd.facing_line = None

    def _defend_3bet(self, s, hc, ctx):
        a = super()._defend_3bet(s, hc, ctx)
        c = self.cfg
        if not c.BIG_3BET_READS or a.kind != "call" or ctx.last_raiser is None:
            return a
        opens = [amt for st, _, kind, amt in s.history
                 if st == "preflop" and kind in AGGRESSIVE and amt > s.big_blind]
        if len(opens) < 2 or opens[-1] <= opens[0] * c.HU_3BET_TIGHT_MULT:
            return a
        ratio = self._freq_ratio(s, ctx.last_raiser, "three_bet")
        if self.big3.value_share(ctx.last_raiser, ratio, c) < c.TIGHT_VALUE_SHARE:
            return a
        return a if hc in self.tight_big_3bet_call else self._check_fold(s)
