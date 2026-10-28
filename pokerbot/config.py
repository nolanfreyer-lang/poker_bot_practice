"""
=====================================================================
  POKERBOT CONFIG  --  every tunable knob lives in this file.
=====================================================================
Format reference: Exhibit A ("Live Tables") on pokerbot.sequoiacap.com
  * 8 bots per table, No-Limit Hold'em
  * Actions: blind, check, call, bet, raise(-to), all-in, fold
Official rules (sec. 3-4):
  * 8-bot sit-and-go, 2,000 starting chips, blinds 10/20 rising every 25 hands
  * Table ends when one bot has all chips or after 200 hands (ICM split)
  * Points by finishing position, 1 MS PER DECISION (error/timeout = check/fold)
  -> Everything here is precomputed or a table lookup. No Monte Carlo.

RANGE SYNTAX (comma separated, case-insensitive suits):
  "AA"        one pair class          "TT+"      TT, JJ, QQ, KK, AA
  "77-44"     77, 66, 55, 44          "AKs"      suited only
  "AKo"       offsuit only            "AK"       suited + offsuit
  "ATs+"      ATs, AJs, AQs, AKs      "A5s-A2s"  A5s, A4s, A3s, A2s
  "KJo+"      KJo, KQo                "*"        any two cards
  ""          nothing
"""

# =====================================================================
# 1. GAME / TOURNAMENT SETTINGS  (update from the starter kit on event day)
# =====================================================================
NUM_SEATS = 8
STARTING_STACK = 2_000
HAND_LIMIT = 200
HANDS_PER_LEVEL = 25
# (small_blind, big_blind) per level. Level 1 is from the rules; the rest
# are placeholders -- replace with the official blind schedule.
BLIND_SCHEDULE = [
    (10, 20), (15, 30), (25, 50), (40, 80),
    (60, 120), (100, 200), (150, 300), (250, 500),
]
# Points per finishing place (1st..8th). Placeholder: announced at the event.
PAYOUT_POINTS = [100, 60, 40, 25, 15, 8, 4, 0]

RANDOM_SEED = None  # set an int for reproducible bot randomness


# =====================================================================
# 2. PREFLOP RANGES  (position names: UTG, UTG1, MP, HJ, CO, BTN, SB, BB)
#    With fewer players, early positions disappear first.
#    Heads-up: the button is labelled BTN (it also posts the SB).
# =====================================================================

# --- First in (no raise yet). Raise with these; fold the rest.
#     BB entry = hands to raise when limpers just complete to us.
OPEN_RANGE = {
    "UTG":  "66+, A9s+, A5s, KTs+, QTs+, JTs, T9s, AJo+, KQo",
    "UTG1": "55+, A8s+, A5s-A4s, KTs+, QTs+, JTs, T9s, 98s, ATo+, KQo",
    "MP":   "44+, A7s+, A5s-A2s, K9s+, Q9s+, J9s+, T9s, 98s, 87s, ATo+, KJo+",
    "HJ":   "33+, A2s+, K9s+, Q9s+, J9s+, T8s+, 97s+, 87s, 76s, ATo+, KJo+, QJo",
    "CO":   "22+, A2s+, K7s+, Q8s+, J8s+, T8s+, 97s+, 86s+, 76s, 65s, 54s, A8o+, KTo+, QTo+, JTo",
    "BTN":  "22+, A2s+, K2s+, Q5s+, J7s+, T7s+, 96s+, 85s+, 75s+, 64s+, 54s, A2o+, K8o+, Q9o+, J9o+, T9o, 98o",
    "SB":   "22+, A2s+, K5s+, Q7s+, J8s+, T8s+, 97s+, 86s+, 76s, 65s, A5o+, K9o+, QTo+, JTo",
    "BB":   "88+, ATs+, KQs, AQo+",
}
# Limp behind existing limpers with these (if not raising).
OVERLIMP_RANGE = "22-66, A2s-A9s, KTs-K9s, QTs-Q9s, JTs-J9s, T9s, 98s, 87s, 76s, 65s"
# SB completes a limped / unopened pot with these (if not raising).
SB_COMPLETE_RANGE = "22+, A2s+, K2s+, Q5s+, J7s+, T7s+, 97s+, 86s+, 75s+, 65s, 54s, A2o+, K8o+, Q9o+, J9o+, T9o"

# --- Facing ONE raise: 3-bet / call by OUR position.
THREEBET_RANGE = {
    "UTG":  "QQ+, AKs, AKo",
    "UTG1": "QQ+, AKs, AKo",
    "MP":   "QQ+, AKs, AKo, A5s",
    "HJ":   "JJ+, AQs+, AKo, A5s",
    "CO":   "TT+, AJs+, KQs, AQo+, A5s-A4s",
    "BTN":  "TT+, ATs+, KQs, AQo+, A5s-A4s, K9s",
    "SB":   "TT+, AJs+, KQs, AQo+, A5s-A4s",
    "BB":   "TT+, AJs+, KQs, AQo+, A5s-A4s",
}
CALL_VS_OPEN_RANGE = {
    "UTG":  "JJ-99, AQs, AJs",
    "UTG1": "JJ-88, AQs-AJs, KQs",
    "MP":   "JJ-77, AQs-ATs, KQs, AQo",
    "HJ":   "TT-66, AQs-ATs, KJs+, QJs, JTs, AQo",
    "CO":   "99-55, AQs-A9s, KTs+, QTs+, JTs, T9s, AJo, KQo",
    "BTN":  "99-22, A9s-A2s, K9s+, Q9s+, J9s+, T9s, 98s, 87s, 76s, AJo-ATo, KQo, KJo",
    "SB":   "99-66, AQs-ATs, KQs, AQo",
    "BB":   "99-22, A9s-A2s, K6s+, Q8s+, J8s+, T8s+, 97s+, 86s+, 75s+, 65s, 54s, AJo-A8o, KTo+, QTo+, JTo",
}

# --- Facing a 3-bet (2 raises) and a 4-bet+ (3+ raises).
FOURBET_RANGE = "QQ+, AKs, AKo, A5s"
CALL_VS_3BET_RANGE = "JJ-99, AQs-ATs, KQs, QJs, JTs, AQo"
JAM_VS_4BET_RANGE = "QQ+, AKs, AKo"
CALL_VS_4BET_RANGE = ""

# --- Short-stack push/fold (tournament). Used when effective stack <= SHOVE_STACK_BB.
SHOVE_STACK_BB = 12
# key = max effective stack in BB. The smallest key >= our stack is used.
SHOVE_RANGE_BY_STACK = {
    5:  "*",
    8:  "22+, A2s+, A2o+, K2s+, K7o+, Q8s+, QTo+, J8s+, JTo, T8s+, 97s+, 87s, 76s",
    10: "22+, A2s+, A2o+, K6s+, K9o+, Q9s+, QJo, J9s+, T9s, 98s",
    12: "33+, A2s+, A7o+, K9s+, KTo+, QTs+, JTs",
}
# Calling an all-in shove (by players remaining -> range). Tighter near the money.
CALL_SHOVE_RANGE = "77+, ATs+, AJo+, KQs"
CALL_SHOVE_RANGE_BUBBLE = "TT+, AQs+, AKo"
BUBBLE_PLAYERS_LEFT = 4   # treat <= this many players left as "bubble / ICM pressure"


# =====================================================================
# 3. PREFLOP SIZING  (all in big blinds or multipliers)
# =====================================================================
OPEN_SIZE_BB = {"default": 2.2, "SB": 3.0, "BB": 3.5}  # tournament-style min-ish opens
LIMPER_ADD_BB = 1.0         # add this per limper when raising
THREEBET_MULT_IP = 3.0      # 3-bet to N x the open when we are in position
THREEBET_MULT_OOP = 3.6     # ... out of position (SB/BB)
COLD_CALLER_ADD_MULT = 1.0  # add N x open per caller between opener and us
FOURBET_MULT = 2.3
JAM_IF_COMMIT_FRACTION = 0.40  # if a raise would put >= 40% of our chips in, shove instead
SB_COMPLETE_MAX_STACK_BB = 25  # only complete SB when deeper than this (else push/fold)


# =====================================================================
# 4. POSTFLOP STRATEGY  (hand-strength buckets, no simulation)
# Buckets (assigned in poker_core.classify_hand):
#   MONSTER      two pair using both hole cards, set, straight+, (not board-only)
#   STRONG       overpair, top pair with good kicker, trips with one hole card
#   MEDIUM       top pair weak kicker, second pair, pocket pair between board cards
#   STRONG_DRAW  flush draw + straight draw, or pair + flush draw
#   DRAW         flush draw or open-ended straight draw (never on river)
#   WEAK         everything else
#
# bet_freq    : how often we bet when checked to (0-1)
# bet_size    : fraction of pot
# raise_freq  : how often we raise when facing a bet
# raise_size  : raise to (facing bet + this fraction of pot after calling)
# call_price  : max pot odds we accept to call = to_call / (pot + to_call)
#               1/3-pot bet = 0.20, 1/2-pot = 0.25, 2/3-pot = 0.29,
#               pot = 0.33, 2x pot = 0.40, all-in for more = closer to 0.5
# =====================================================================
TOP_PAIR_GOOD_KICKER = "T"   # kicker this rank or higher = STRONG

POSTFLOP_POLICY = {
    "MONSTER":     {"bet_freq": 0.95, "bet_size": 0.75, "raise_freq": 0.85, "raise_size": 0.90, "call_price": 0.50},
    "STRONG":      {"bet_freq": 0.85, "bet_size": 0.60, "raise_freq": 0.15, "raise_size": 0.80, "call_price": 0.38},
    "MEDIUM":      {"bet_freq": 0.35, "bet_size": 0.45, "raise_freq": 0.00, "raise_size": 0.75, "call_price": 0.26},
    "STRONG_DRAW": {"bet_freq": 0.65, "bet_size": 0.60, "raise_freq": 0.30, "raise_size": 0.80, "call_price": 0.36},
    "DRAW":        {"bet_freq": 0.35, "bet_size": 0.50, "raise_freq": 0.05, "raise_size": 0.75, "call_price": 0.22},
    "WEAK":        {"bet_freq": 0.10, "bet_size": 0.50, "raise_freq": 0.00, "raise_size": 0.75, "call_price": 0.00},
}

# Per-street multipliers on bet_freq (river bluffs/draws are handled by buckets).
STREET_BET_FREQ_MULT = {"flop": 1.0, "turn": 0.9, "river": 0.8}
# Continuation bet: we were last preflop raiser, flop, checked to us.
CBET_FREQ_HEADSUP = 0.65
CBET_FREQ_MULTIWAY = 0.25
CBET_SIZE = 0.40
# Multiway: scale down bet_freq / call_price for non-MONSTER hands.
MULTIWAY_BET_FREQ_MULT = 0.6
MULTIWAY_CALL_PRICE_MULT = 0.8
# Each extra raise we face on this street shrinks call_price by this much.
CALL_PRICE_PENALTY_PER_RAISE = 0.08
# Stack-to-pot ratio at or below which STRONG+ hands just get it in.
COMMIT_SPR = 1.5
# Already put in this fraction of our starting-hand stack? Call with any price-correct MEDIUM+.
POT_COMMITTED_FRACTION = 0.45


# =====================================================================
# 5. OPPONENT ADJUSTMENTS (bot state resets every table, opponents anonymised)
# =====================================================================
TRACK_OPPONENTS = True
MIN_HANDS_FOR_READS = 15
LOOSE_VPIP = 0.45        # opponent plays more than this -> treat as loose
AGGRO_PFR = 0.30         # opponent raises preflop more than this -> treat as aggro
# Call price bonus vs loose / aggro bettors (they bluff more).
CALL_PRICE_BONUS_VS_AGGRO = 0.05
# Widen our 3-bet/call-vs-open vs aggro openers by using the next later position's range.
WIDEN_VS_AGGRO_OPENER = True
