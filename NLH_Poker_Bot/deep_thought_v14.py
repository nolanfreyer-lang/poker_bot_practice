"""DEEP_THOUGHT_V14: DEEP_THOUGHT_V12 that checks its frequency reads against shown hands.

    from deep_thought_v14 import DeepThoughtV14Bot
    ("DEEP14", DeepThoughtV14Bot(seed=seed))

Through v12, a bettor who bets or raises often is given more weak hands whenever they do,
in three places (v1's range, v4's street read, the bluff estimate), whatever the bet's
size. Chaos_Incarnate (Chipzen, Nov 12) bets a third of the pot with nothing on about 90%
of flops and turns, and bets the pot or raises with strong hands: 0 strong hands in 212
small bets, ahead 82-87% of the time with big ones. Its betting frequency made v12 read the
big bets and raises as bluffs too. v12 lost 457 BB on hands where it called a big bet or
a raise, and 338 BB in all over 304 hands (6 of 7 matches).

v14 keeps v12's reads but believes them, for the kind of bet it faces (small, mid, big,
or a raise), only as far as two things allow:

  1. What they have shown with that kind of bet. The widening is scaled by (weak share
     of their shown bets of that kind / the field's share for it) ^ TRUST_POWER, capped
     at 1. Nothing changes until an opponent shows strong hands with that kind of bet.
  2. For big bets and raises, how often they fold to a bet, which needs no showdown. A
     bot that folds often when bet into is left with strong hands when it puts in big
     money (Chaos_Incarnate folds about half the time; a maniac never does). Trust falls
     from full at FOLDER_TRUST_FROM to FOLDER_TRUST_MIN at FOLDER_TRUST_TO.

The lower of the two is used. Two simpler versions tested worse: capping the raise read
lost 399 bb/100 to the simulator's maniac, and comparing each size's frequency with the
field's share of that size misread bots that always bet one size (-16 to -28 bb/100
against the GTO and v4 opponents).

v14 also 3-bets lighter against openers who fold to 3-bets. Chaos_Incarnate opens about
80% of small blinds and folded to 29 of 37 3-bets (78%); v12 3-bet 12% of its opens. Once an
opener's fold rate (pulled toward FOLD_3BET_PRIOR) reaches LIGHT_3BET_MIN_FOLD, the big
blind 3-bets LIGHT_3BET_RANGE too: hands it would call with that hold an ace or a king.

Everything else is v12. Profiles save the new counts; older bots ignore them.
"""
import config as default_config
from bot import AGGRESSIVE
from deep_thought_v4 import StreetBetStats
from deep_thought_v7 import TRACKERS_V7
from deep_thought_v12 import DeepThoughtV12Bot
from poker_core import classify_hand, hand_class, parse_range

DEEP14_CONFIG = {
    "SHOWN_TRUST": True,
    "SIZE_EDGES": (0.45, 0.8),          # bets below / above these shares of the pot are small / big
    # weak (WEAK or DRAW) share of each kind of bet across 70,000 heads-up Chipzen hands
    "WEAK_PRIOR": {"small": 0.65, "mid": 0.45, "big": 0.42, "raise": 0.28},
    "WEAK_WEIGHT": 3,                   # the prior counts like this many shown bets
    "TRUST_POWER": 2.0,
    # big bets and raises: trust also falls with their fold rate to bets (after FOLD_PRIOR)
    "FOLDER_TRUST_FROM": 0.42,          # fold rate where trust starts to fall
    "FOLDER_TRUST_TO": 0.52,            # ... and where it reaches the minimum
    "FOLDER_TRUST_MIN": 0.2,
    # --- light 3-bets against openers who fold to them (heads-up, big blind) ----------
    "LIGHT_3BETS": True,
    "LIGHT_3BET_RANGE": "A9s-A6s, K8s-K5s, ATo-A2o, KJo-K8o, QJo",
    "LIGHT_3BET_MIN_FOLD": 0.62,        # their fold rate to 3-bets, after the prior
    "FOLD_3BET_PRIOR": 0.5,
    "FOLD_3BET_WEIGHT": 4,              # the prior counts like this many 3-bets faced
    "LIGHT_3BET_MIN_BB": 30,            # effective stack; shorter stacks keep v12's ranges
}
TRACKERS_V14 = TRACKERS_V7 + ("fold3", "shown_bets")


def tracker_counts(bot, seat):
    """Every count the trackers hold for `seat`: {tracker: {table: [[subkey, count], ...]}}."""
    out = {}
    for name in TRACKERS_V14:
        for table, values in vars(getattr(bot, name)).items():
            if not isinstance(values, dict):
                continue
            rows = [[list(k[1:]) if isinstance(k, tuple) else [], v] for k, v in values.items()
                    if k == seat or (isinstance(k, tuple) and k[0] == seat)]
            if rows:
                out.setdefault(name, {})[table] = rows
    return out


def size_bucket(share, edges):
    return "small" if share < edges[0] else "big" if share > edges[1] else "mid"


class TrustedStreetBetStats(StreetBetStats):
    """StreetBetStats whose read can be scaled back while the bot faces one kind of bet."""

    def __init__(self):
        super().__init__()
        self.trust = 1.0        # set by the bot: how far a high bet rate is believed (0-1)

    def ratio(self, seat, street, cfg):
        pooled = super().ratio(seat, street, cfg)
        return 1 + (pooled - 1) * self.trust if pooled > 1 else pooled


class ShownBetStats:
    """What each opponent's bets turned out to be, by kind, when we saw the cards.

    Kinds: the street's first bet as small, mid or big, and any raise after the flop."""

    def __init__(self, edges=(0.45, 0.8)):
        self.shown = {}         # (seat, kind) -> bets whose cards we saw
        self.weak = {}          # ... of which were WEAK or DRAW on that street
        self.edges = edges

    def observe(self, history, showdown, board, good_kicker, me, see_all):
        folded = {seat for _, seat, kind, _ in history if kind == "fold"}
        cards = {seat: c for seat, c in (showdown or {}).items() if see_all or seat not in folded}
        paid = {}
        for st, seat, kind, amt in history:
            if kind not in ("check", "fold"):
                paid.setdefault(st, {})[seat] = max(paid.get(st, {}).get(seat, 0), amt)
        pot = sum(paid.get("preflop", {}).values())
        for street, n in (("flop", 3), ("turn", 4), ("river", 5)):
            if len(board or []) < n:
                break
            high = 0
            for st, seat, kind, amt in history:
                if st != street or kind not in AGGRESSIVE or amt <= high:
                    continue
                if seat != me and seat in cards and pot > 0:
                    what = "raise" if high > 0 else size_bucket(amt / pot, self.edges)
                    bucket = classify_hand(list(cards[seat]), list(board[:n]), good_kicker)
                    key = (seat, what)
                    self.shown[key] = self.shown.get(key, 0) + 1
                    self.weak[key] = self.weak.get(key, 0) + (bucket in ("WEAK", "DRAW"))
                high = amt
            pot += sum(paid.get(street, {}).values())

    def trust(self, seat, what, cfg):
        """1 = this kind of bet is as weak as a typical bot's or weaker; toward 0 = all value."""
        p, w = cfg.WEAK_PRIOR[what], cfg.WEAK_WEIGHT
        share = (self.weak.get((seat, what), 0) + p * w) / (self.shown.get((seat, what), 0) + w)
        return min(1.0, share / p) ** cfg.TRUST_POWER


class Fold3BetStats:
    """How often each opponent folds their open to a 3-bet (anyone's, in replays)."""

    def __init__(self):
        self.chances = {}
        self.folds = {}

    def observe(self, history):
        bb = max([amt for st, _, kind, amt in history if st == "preflop" and kind == "blind"] or [0])
        raises, opener = 0, None
        for st, seat, kind, amt in history:
            if st != "preflop" or kind == "blind":
                continue
            if raises == 2 and seat == opener:          # the opener answers the 3-bet
                self.chances[seat] = self.chances.get(seat, 0) + 1
                if kind == "fold":
                    self.folds[seat] = self.folds.get(seat, 0) + 1
                return
            if kind in AGGRESSIVE and amt > bb:
                raises += 1
                if raises == 1:
                    opener = seat
                elif seat == opener or raises > 2:
                    return

    def rate(self, seat, prior, weight):
        return (self.folds.get(seat, 0) + prior * weight) / (self.chances.get(seat, 0) + weight)


class DeepThoughtV14Bot(DeepThoughtV12Bot):
    def __init__(self, cfg=default_config, overrides=None, seed=None):
        super().__init__(cfg, {**DEEP14_CONFIG, **(overrides or {})}, seed)
        self.street_reads = TrustedStreetBetStats()
        self.shown_bets = ShownBetStats(self.cfg.SIZE_EDGES)
        self.fold3 = Fold3BetStats()
        self.light_3bet = parse_range(self.cfg.LIGHT_3BET_RANGE)
        self._trust = None      # (seat, trust) while we face that seat's bet

    def on_hand_end(self, history, seats_dealt, showdown=None, board=None, my_seat=None):
        super().on_hand_end(history, seats_dealt, showdown=showdown, board=board, my_seat=my_seat)
        me = self._my_seat if my_seat is None else my_seat
        self.fold3.observe(history)
        self.shown_bets.observe(history, showdown, board, self.good_kicker, me, getattr(self.sd, "see_all", False))

    # ------------------------------------------------------------ light 3-bets
    def _preflop_action(self, s):
        a = super()._preflop_action(s)
        c = self.cfg
        if not c.LIGHT_3BETS or a.kind != "call" or not self._heads_up(s):
            return a
        ctx = self._preflop_context(s)
        if ctx.raises != 1 or ctx.last_raiser is None or s.players[ctx.last_raiser].all_in \
                or self._effective_stack(s) / s.big_blind < c.LIGHT_3BET_MIN_BB \
                or hand_class(*s.hole_cards) not in self.light_3bet:
            return a
        if self.fold3.rate(ctx.last_raiser, c.FOLD_3BET_PRIOR, c.FOLD_3BET_WEIGHT) < c.LIGHT_3BET_MIN_FOLD:
            return a
        return self._raise_to(s, ctx.high * c.HU_3BET_MULT)

    # ------------------------------------------------------------ trusted reads
    def _facing(self, s):
        """The kind of bet we face: small, mid or big for the street's first bet, or 'raise'."""
        bets = [amt for st, _, kind, amt in s.history if st == s.street and kind in AGGRESSIVE]
        if len(bets) != 1:
            return "raise" if bets else None
        before = s.pot - sum({seat: amt for st, seat, kind, amt in s.history
                              if st == s.street and kind not in ("check", "fold")}.values())
        return size_bucket(bets[0] / max(1, before), self.cfg.SIZE_EDGES)

    def _bet_trust(self, seat, what):
        """How far this seat's betting frequency is believed to mean weak hands (0-1)."""
        c = self.cfg
        trust = self.shown_bets.trust(seat, what, c)
        if what in ("big", "raise"):
            t = (self.sd.fold_rate(seat, c) - c.FOLDER_TRUST_FROM) / (c.FOLDER_TRUST_TO - c.FOLDER_TRUST_FROM)
            trust = min(trust, 1 - (1 - c.FOLDER_TRUST_MIN) * min(1.0, max(0.0, t)))
        return trust

    def _postflop(self, s):
        self._trust = None
        if self.cfg.SHOWN_TRUST and s.to_call:
            agg, what = self._aggressor(s), self._facing(s)
            if agg is not None and what is not None:
                self._trust = (agg, self._bet_trust(agg, what))
                self.street_reads.trust = self._trust[1]
        try:
            return super()._postflop(s)
        finally:
            self._trust = None
            self.street_reads.trust = 1.0

    def _freq_ratio(self, s, seat, stat):
        ratio = super()._freq_ratio(s, seat, stat)
        if self._trust is None or seat != self._trust[0] or ratio <= 1 \
                or stat not in ("bet", "raise", "check_raise", "all_in"):
            return ratio
        return 1 + (ratio - 1) * self._trust[1]
