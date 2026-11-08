"""DEEP_THOUGHT_V7: DEEP_THOUGHT_V6 with exploits that switch on per opponent once the
evidence is strong enough.

    from deep_thought_v7 import DeepThoughtV7Bot
    ("DEEP7", DeepThoughtV7Bot(seed=seed))

Each exploit is a read on one opponent that starts at the Chipzen field average and
adapts as hands come in, from saved profiles and from the match itself. An exploit acts
only when its read is confidently past its threshold: the estimate minus (or plus)
EXPLOIT_Z standard errors, so a few lucky hands don't trigger it, while a bot with a
long Chipzen history (a full profile) gets its exploits from the first hand. Against an
average opponent v7 plays exactly like v6.

  1. Raises on our flop bets (CBET_RAISE_READS). v5 checks weak leads against bots that
     raise leads; this does the same for our other first flop bets (continuation bets,
     bets after a check) against bots that confidently raise them more than
     CBET_MAX_RAISE.
  2. Turn barrels (BARREL_READS). After a flop bet gets called, a weak hand or draw bets
     the turn again against bots that confidently fold to that second bet more than both
     the field and what the bet needs, and a weak hand checks against bots that
     confidently fold less than it needs.
  3. Thin turn value (THIN_VALUE_READS). A medium hand bets the turn for value against
     bots whose turn calls are confidently mostly weak hands and draws.
  4. Heavy bettors who fold (FOLD_WEAK_VS_BARRELER, off by default). Facing one opponent's
     first bet on the flop or turn, a weak made hand folds instead of calling when that
     bettor bets BARRELER_RATIO x the reference rate, folds to our bets at least
     BARRELER_MIN_FOLD of the time, and has shown down at most BARRELER_MAX_BLUFF bluffs:
     Chipzen's slopUpBotV2 type, whose bluffs are random weak hands that still beat ours
     and that bet again on the next street. A MANIAC or LAG shows its air at showdown,
     so it doesn't count. Live play shows slopUpBotV2's bets as 15-20% bluffs (only
     called-down hands show) and its replay profile as 38%; a MANIAC shows 53-63%.
     Paired against v7 from an empty start, with a 0.35 limit: +8.9 +- 2.6 bb/100 at
     100 BB and +9.9 +- 1.8 at 40 BB heads-up against a slopUpBotV2 mimic, +0.4 and +0.7
     at 6-max tables with it, -0.05 at 6-max and -0.3 / -0.2 heads-up against the usual
     field (-3 against LAG). The 0.45 limit, which lets its profile count: +10.7 +- 3.0
     against the mimic and -0.04 / -0.12 heads-up against the field. Raising its bets as
     bluffs instead lost 9 bb/100 at 100 BB, and pricing its bluffs higher (extra bets as
     bluffs) lost 14.

Saved profiles fill these reads with up to EXPLOIT_PROFILE_HANDS hands of history, more
than v6's reads take (PROFILE_MAX_HANDS), since an exploit only acts when it is confident
and live hands keep updating it. A profile can also name exploits to force on against
that bot (its "force" list, written by chipzen_profiles.py from its CHIPZEN_EXPLOITS
table), for bots whose hand data shows a flaw the counts alone don't reach yet.

exploit_plan() lists what is switched on against an opponent; chipzen_profiles.py prints
it for every Chipzen bot after a refresh.

Tested against v6 from an empty start (no profiles): +0.58 +- 0.50 bb/100 at 100 BB and
+0.97 +- 0.21 at 30 BB, +7.1 and +2.9 against a bot that raises a quarter of flop bets
and folds the turn after calling. Forcing turn barrels against everyone was +5.19 at
100 BB (calling stations fold the turn eventually) but -0.79 at 30 BB, so barrels are
forced only against named bots.
"""
import math

import config as default_config
from bot import Action, AGGRESSIVE
from deep_thought_v5 import led_into
from deep_thought_v6 import DeepThoughtV6Bot, TRACKERS_V6, bet_lines
from poker_core import classify_hand

DEEP7_CONFIG = {
    "EXPLOIT_Z": 1.0,                   # standard errors a read must clear before it acts
    "EXPLOIT_PROFILE_HANDS": 600,       # saved history these reads take, in hands
    # --- raises on our first flop bet (not leads, which v5 handles) ---------------------
    "CBET_RAISE_READS": True,
    "CBET_RAISE_PRIOR": 0.085,          # Chipzen field: 8.6% of continuation bets raised
    "CBET_RAISE_WEIGHT": 10,
    "CBET_MAX_RAISE": 0.15,             # weak hands check against bots confidently above this
    # --- turn barrels after a called flop bet -------------------------------------------
    "BARREL_READS": True,
    "BARREL_FOLD_PRIOR": 0.39,          # Chipzen field: folds to the turn bet after calling
    "BARREL_FOLD_WEIGHT": 8,
    "BARREL_SIZE": 0.5,                 # share of the pot for a barrel we add
    "BARREL_DRAW_CREDIT": 0.12,         # draws need this much less fold rate (outs pay back)
    # --- thin turn value bets against weak callers --------------------------------------
    "THIN_VALUE_READS": True,
    "CALL_WEAK_PRIOR": 0.53,            # Chipzen field: turn calls with weak hands or draws
    "CALL_WEAK_WEIGHT": 8,
    "THIN_VALUE_SHARE": 0.6,            # bet medium hands when their weak share is confidently above
    "THIN_VALUE_SIZE": 0.5,
    # --- weak hands fold to heavy bettors who fold -------------------------------------
    "FOLD_WEAK_VS_BARRELER": False,
    "BARRELER_RATIO": 1.3,              # first bets per chance vs the reference rate
    "BARRELER_MIN_FOLD": 0.45,          # their fold rate to our bets
    "BARRELER_MAX_BLUFF": 0.45,         # bluff share of the bets they have shown down
}

# opponent trackers that a saved profile fills in, by attribute name
EXPLOIT_TRACKERS = ("cbet_raises", "barrels", "turn_calls")
TRACKERS_V7 = TRACKERS_V6 + EXPLOIT_TRACKERS
# exploits a profile can force on: "barrel_weak", "barrel_draw", "check_weak_flops", "thin_value"


def tracker_counts(bot, seat):
    """Every count the trackers hold for `seat`: {tracker: {table: [[subkey, count], ...]}}."""
    out = {}
    for name in TRACKERS_V7:
        for table, values in vars(getattr(bot, name)).items():
            if not isinstance(values, dict):
                continue
            rows = [[list(k[1:]) if isinstance(k, tuple) else [], v] for k, v in values.items()
                    if k == seat or (isinstance(k, tuple) and k[0] == seat)]
            if rows:
                out.setdefault(name, {})[table] = rows
    return out


def read(hits, chances, prior, weight, z):
    """(estimate, low, high): the rate pulled toward `prior`, +- z standard errors."""
    est = (hits + prior * weight) / (chances + weight)
    err = z * math.sqrt(est * (1 - est) / (chances + weight))
    return est, est - err, est + err


def first_bet(history, street):
    """(seat, index) of the first bet on this street, or (None, None)."""
    acts = [(seat, kind) for st, seat, kind, _ in history if st == street]
    for i, (seat, kind) in enumerate(acts):
        if kind in AGGRESSIVE:
            return seat, i
    return None, None


def called_flop_bet(history, me):
    """The seat that called our first flop bet without raising, or None."""
    seat, i = first_bet(history, "flop")
    if seat is None or seat != me:
        return None
    acts = [(s, kind) for st, s, kind, _ in history if st == "flop"][i + 1:]
    if any(kind in AGGRESSIVE for _, kind in acts):
        return None
    return next((s for s, kind in acts if kind == "call"), None)


class CbetRaiseStats:
    """How often each opponent raises our first flop bet when it isn't a lead."""

    def __init__(self):
        self.raises = {}
        self.chances = {}

    def observe(self, history, me):
        seat, i = first_bet(history, "flop")
        if seat is None or seat != me:
            return
        line = next((l for st, s, _, l in bet_lines(history) if st == "flop" and s == me), None)
        if line == "lead":
            return
        acts = [(s, kind) for st, s, kind, _ in history if st == "flop"][i + 1:]
        reply = next(((s, kind) for s, kind in acts if s != me), None)
        if reply is not None:
            self.chances[reply[0]] = self.chances.get(reply[0], 0) + 1
            if reply[1] in AGGRESSIVE:
                self.raises[reply[0]] = self.raises.get(reply[0], 0) + 1

    def read(self, seat, cfg):
        return read(self.raises.get(seat, 0), self.chances.get(seat, 0),
                    cfg.CBET_RAISE_PRIOR, cfg.CBET_RAISE_WEIGHT, cfg.EXPLOIT_Z)


class BarrelStats:
    """How often each opponent folds to our turn bet after calling our flop bet."""

    def __init__(self):
        self.folds = {}
        self.chances = {}

    def observe(self, history, me):
        seat = called_flop_bet(history, me)
        bettor, i = first_bet(history, "turn")
        if seat is None or bettor is None or bettor != me:
            return
        acts = [(s, kind) for st, s, kind, _ in history if st == "turn"][i + 1:]
        reply = next((kind for s, kind in acts if s == seat), None)
        if reply is not None:
            self.chances[seat] = self.chances.get(seat, 0) + 1
            if reply == "fold":
                self.folds[seat] = self.folds.get(seat, 0) + 1

    def read(self, seat, cfg):
        return read(self.folds.get(seat, 0), self.chances.get(seat, 0),
                    cfg.BARREL_FOLD_PRIOR, cfg.BARREL_FOLD_WEIGHT, cfg.EXPLOIT_Z)


class TurnCallStats:
    """What each opponent holds when they call our turn bets: the share of weak hands and draws."""

    def __init__(self):
        self.calls = {}
        self.weak = {}
        self.see_all = False        # as SizedShowdownStats.see_all

    def observe(self, history, showdown, board, good_kicker, me):
        if not board or len(board) < 4:
            return
        folded = {seat for _, seat, kind, _ in history if kind == "fold"}
        high = 0
        for st, seat, kind, amt in history:
            if st != "turn":
                continue
            if kind == "call" and seat != me and high:
                cards = (showdown or {}).get(seat)
                if cards and (self.see_all or seat not in folded):
                    bucket = classify_hand(list(cards), list(board[:4]), good_kicker)
                    self.calls[seat] = self.calls.get(seat, 0) + 1
                    if bucket in ("WEAK", "DRAW"):
                        self.weak[seat] = self.weak.get(seat, 0) + 1
            if kind in AGGRESSIVE:
                high = max(high, amt)

    def read(self, seat, cfg):
        return read(self.weak.get(seat, 0), self.calls.get(seat, 0),
                    cfg.CALL_WEAK_PRIOR, cfg.CALL_WEAK_WEIGHT, cfg.EXPLOIT_Z)


class DeepThoughtV7Bot(DeepThoughtV6Bot):
    def __init__(self, cfg=default_config, overrides=None, seed=None):
        super().__init__(cfg, {**DEEP7_CONFIG, **(overrides or {})}, seed)
        self.cbet_raises = CbetRaiseStats()
        self.barrels = BarrelStats()
        self.turn_calls = TurnCallStats()
        self.forced = {}            # seat -> exploits a saved profile forces on

    def load_profile(self, seat, profile):
        """v6's profile load, with more history for the exploit reads and any forced exploits."""
        counts = profile.get("counts", {})
        super().load_profile(seat, {**profile, "counts": {k: v for k, v in counts.items()
                                                          if k not in EXPLOIT_TRACKERS}})
        scale = min(1.0, self.cfg.EXPLOIT_PROFILE_HANDS / max(1, profile.get("hands", 1)))
        for name in EXPLOIT_TRACKERS:
            tracker = getattr(self, name)
            for table, rows in counts.get(name, {}).items():
                values = getattr(tracker, table, None)
                if not isinstance(values, dict):
                    continue
                for sub, count in rows:
                    key = (seat, *sub) if sub else seat
                    values[key] = values.get(key, 0) + count * scale
        if profile.get("force"):
            self.forced[seat] = set(profile["force"])

    def position(self, s, seat=None):
        """Heads-up preflop, the small blind plays the button's ranges, whatever button_seat says.

        When a Chipzen 6-max table is down to two, dealer_seat points at the big blind, and
        the big blind does act last on every street after the flop (the small blind acts
        first all hand), so postflop keeps that reading. Preflop it made the small blind
        call itself "BB" and fold 22 and T6s at 15 BB. In 2-seat matches the small blind is
        the button anyway, so nothing changes there.
        """
        if s.street == "preflop" and sum(1 for p in s.players if p.dealt) == 2:
            blinds = [who for street, who, kind, _ in s.history if street == "preflop" and kind == "blind"]
            if len(blinds) == 2:
                seat = s.my_seat if seat is None else seat
                return "BTN" if seat == blinds[0] else "BB"
        return super().position(s, seat)

    def see_everything(self):
        """Count every hand whose cards we are given, not only showdowns (for replays)."""
        self.sd.see_all = self.turn_calls.see_all = True

    def on_hand_end(self, history, seats_dealt, showdown=None, board=None, my_seat=None):
        super().on_hand_end(history, seats_dealt, showdown=showdown, board=board, my_seat=my_seat)
        me = self._my_seat if my_seat is None else my_seat
        self.cbet_raises.observe(history, me)
        self.barrels.observe(history, me)
        self.turn_calls.observe(history, showdown, board, self.good_kicker, me)

    # ------------------------------------------------------------ exploits
    def _forced(self, opp, name):
        return name in self.forced.get(opp, ())

    def _checks_weak_flops(self, opp):
        c = self.cfg
        return c.CBET_RAISE_READS and (self._forced(opp, "check_weak_flops")
                                       or self.cbet_raises.read(opp, c)[1] > c.CBET_MAX_RAISE)

    def _barrel(self, opp, size, bucket):
        """+1 to add a turn barrel, -1 to drop one, 0 to leave it to v6."""
        c = self.cfg
        if not c.BARREL_READS:
            return 0
        if self._forced(opp, "barrel_" + bucket.lower()):
            return 1
        _, low, high = self.barrels.read(opp, c)
        even = size / (1 + size)
        # draws can barrel a bit below break-even, but only against bots that fold more
        # than the field: at the field's rate that choice is v6's, not an exploit
        if low >= even - (c.BARREL_DRAW_CREDIT if bucket == "DRAW" else 0) and low > c.BARREL_FOLD_PRIOR:
            return 1
        return -1 if bucket == "WEAK" and high < even else 0

    def _thin_value(self, opp):
        c = self.cfg
        return c.THIN_VALUE_READS and (self._forced(opp, "thin_value")
                                       or self.turn_calls.read(opp, c)[1] >= c.THIN_VALUE_SHARE)

    def exploit_plan(self, opp):
        """What v7 does differently against this opponent right now, in words."""
        c, plan = self.cfg, []
        if self.lead_reads.raise_rate(opp, c) > c.LEAD_MAX_RAISE:
            plan.append(f"checks weak leads (raises {self.lead_reads.raise_rate(opp, c):.0%} of leads)")
        est = self.cbet_raises.read(opp, c)[0]
        if self._checks_weak_flops(opp):
            plan.append(f"checks weak flops (raises {est:.0%} of flop bets)")
        est = self.barrels.read(opp, c)[0]
        for bucket, words in (("WEAK", "weak hands"), ("DRAW", "draws")):
            move = self._barrel(opp, c.BARREL_SIZE, bucket)
            if move:
                plan.append(f"{'barrels' if move > 0 else 'gives up'} the turn with {words} "
                            f"(folds {est:.0%} to a second bet"
                            + (", forced" if self._forced(opp, "barrel_" + bucket.lower()) else "") + ")")
        if self._thin_value(opp):
            plan.append(f"bets medium hands on the turn ({self.turn_calls.read(opp, c)[0]:.0%} of turn calls weak)")
        return plan

    def _bet(self, s, share):
        amount = s.me.bet + max(s.big_blind, int(share * s.pot))
        return Action("bet", max(s.min_raise_to, min(s.max_raise_to, amount)))

    def _barreler(self, s, opps):
        """The one opponent whose flop or turn bet we face, if it bets heavily and folds."""
        c = self.cfg
        if s.street not in ("flop", "turn") or len(opps) != 1 or self._facing_raise(s):
            return None
        agg = self._aggressor(s)
        if agg is None or self._freq_ratio(s, agg, "bet") < c.BARRELER_RATIO \
                or self.sd.fold_rate(agg, c) < c.BARRELER_MIN_FOLD \
                or self.sd.bluff_rate(agg, c, s.street) > c.BARRELER_MAX_BLUFF:
            return None
        return agg

    def _postflop(self, s):
        a = super()._postflop(s)
        opps = [p for p in s.players if p.active and p.seat != s.my_seat]
        if s.to_call and a.kind == "call" and self.cfg.FOLD_WEAK_VS_BARRELER \
                and classify_hand(s.hole_cards, s.board, self.good_kicker) == "WEAK" \
                and self._barreler(s, opps) is not None:
            return Action("fold")
        if s.to_call or len(opps) != 1 or s.street == "river":
            return a
        opp = opps[0].seat
        bucket = classify_hand(s.hole_cards, s.board, self.good_kicker)
        betting = a.kind in AGGRESSIVE
        if s.street == "flop":
            if betting and bucket == "WEAK" and led_into(s.history, s.street, s.my_seat) != opp \
                    and self._checks_weak_flops(opp):
                return Action("check")
            return a
        if called_flop_bet(s.history, s.my_seat) != opp:
            return a
        if bucket in ("WEAK", "DRAW"):
            size = (a.amount - s.me.bet) / max(1, s.pot) if a.kind == "bet" else self.cfg.BARREL_SIZE
            move = self._barrel(opp, size, bucket)
            if move > 0 and not betting:
                return self._bet(s, self.cfg.BARREL_SIZE)
            if move < 0 and betting:
                return Action("check")
            return a
        if bucket == "MEDIUM" and not betting and self._thin_value(opp):
            return self._bet(s, self.cfg.THIN_VALUE_SIZE)
        return a
