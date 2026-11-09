"""DEEP_THOUGHT_V4: DEEP_THOUGHT_V3 with fixes from its match against a human player.

    from deep_thought_v4 import DeepThoughtV4Bot
    ("DEEP4", DeepThoughtV4Bot(seed=seed))

  1. Bets are tracked on the flop, turn and river separately and compared with each
     street's expected rate. When facing a bet, a strong read on that street adds weak
     hands to the opponent's range, without needing a showdown.
  2. Calls have stack-off limits by hand strength, like bets and raises, unless the
     street read says this opponent bets here far more than expected.
  3. Small 3-bets are defended wider: the fixed ranges ignored how cheap a call was.
"""
import config as default_config
from bot import Action, AGGRESSIVE
from deep_thought_v3 import DeepThoughtV3Bot
from poker_core import classify_hand, parse_range

DEEP4_CONFIG = {
    # --- per-street bluff reads: expected rates are Slumbot's, from slumbot_logs/ -----
    "STREET_BET_RATE": {"flop": 0.22, "turn": 0.26, "river": 0.37},
    "STREET_READ_WEIGHT": 6,        # prior counts like this many chances
    "STREET_READ_POWER": 2.0,       # how strongly the read moves weak hands' weight
    "STREET_READ_MAX": 3.0,         # most it can move them either way
    # --- stack-off limits on calls: chips in the pot after calling, in big blinds -----
    "CALL_LIMIT_BB": {"STRONG": 60, "STRONG_DRAW": 40, "MEDIUM": 30, "DRAW": 20, "WEAK": 15},
    "CALL_LIMIT_READ": 1.5,         # a street read at least this high lifts the limit
    "CALL_LIMIT_MAX_PRICE": None,   # pot committed: no limit when the call needs less equity than this
    # --- small 3-bets: call wider when the price is at most this ----------------------
    "SMALL_3BET_PRICE": 0.22,
    "HU_BTN_CALL_SMALL_3BET": ("22+, A2s+, A2o+, K2s+, K2o+, Q2s+, Q5o+, J4s+, J7o+, T5s+, T7o+, "
                               "95s+, 97o+, 85s+, 87o, 74s+, 64s+, 53s+"),
}


class StreetBetStats:
    """How often each opponent bets when they could, on each street separately."""

    def __init__(self):
        self.bets = {}
        self.chances = {}

    def observe(self, history, me):
        for street in ("flop", "turn", "river"):
            high, acted = 0, set()
            for st, seat, kind, amt in history:
                if st != street:
                    continue
                if high == 0 and seat != me and seat not in acted:
                    key = (seat, street)
                    self.chances[key] = self.chances.get(key, 0) + 1
                    if kind in AGGRESSIVE:
                        self.bets[key] = self.bets.get(key, 0) + 1
                acted.add(seat)
                if kind in AGGRESSIVE:
                    high = max(high, amt)

    def ratio(self, seat, street, cfg):
        base, w = cfg.STREET_BET_RATE[street], cfg.STREET_READ_WEIGHT
        key = (seat, street)
        return (self.bets.get(key, 0) + base * w) / (self.chances.get(key, 0) + w) / base


class DeepThoughtV4Bot(DeepThoughtV3Bot):
    def __init__(self, cfg=default_config, overrides=None, seed=None):
        super().__init__(cfg, {**DEEP4_CONFIG, **(overrides or {})}, seed)
        self.street_reads = StreetBetStats()
        self.small_3bet_call = parse_range(self.cfg.HU_BTN_CALL_SMALL_3BET)

    def on_hand_end(self, history, seats_dealt, showdown=None, board=None, my_seat=None):
        super().on_hand_end(history, seats_dealt, showdown=showdown, board=board, my_seat=my_seat)
        self.street_reads.observe(history, self._my_seat if my_seat is None else my_seat)

    def _opponent_range(self, s, seat):
        hands = super()._opponent_range(s, seat)
        if s.to_call:
            c = self.cfg
            read = self.street_reads.ratio(seat, s.street, c) ** c.STREET_READ_POWER
            read = min(c.STREET_READ_MAX, max(1 / c.STREET_READ_MAX, read))
            for h in hands:
                if h[3] in ("WEAK", "DRAW"):
                    h[2] *= read
        return hands

    def _defend_3bet(self, s, hc, ctx):
        a = super()._defend_3bet(s, hc, ctx)
        price = s.to_call / max(1, s.pot + s.to_call)
        if a.kind == "fold" and price <= self.cfg.SMALL_3BET_PRICE and hc in self.small_3bet_call:
            return Action("call")
        return a

    def _postflop(self, s):
        a = super()._postflop(s)
        if a.kind == "call" or (a.kind == "allin" and s.to_call >= s.me.stack):
            limit = self.cfg.CALL_LIMIT_BB.get(classify_hand(s.hole_cards, s.board, self.good_kicker))
            chips = self._invested(s, s.my_seat) + min(s.to_call, s.me.stack)
            read = self.street_reads.ratio(self._aggressor(s), s.street, self.cfg)
            cost = min(s.to_call, s.me.stack)
            committed = self.cfg.CALL_LIMIT_MAX_PRICE is not None \
                and cost / (s.pot + cost) < self.cfg.CALL_LIMIT_MAX_PRICE
            if limit is not None and chips > limit * s.big_blind and read < self.cfg.CALL_LIMIT_READ \
                    and not committed:
                return Action("fold")
        return a
