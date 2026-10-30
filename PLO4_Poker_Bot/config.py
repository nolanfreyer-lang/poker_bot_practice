"""
=====================================================================
  PLO4 POKERBOT CONFIG  --  every tunable knob lives in this file.
=====================================================================
Official rules (sec. 3-4): 8-bot sit-and-go, 2,000 starting chips, blinds 10/20
rising every 25 hands, one bot left or 200 hands (ICM split), points by finishing
place, 1 MS PER DECISION (error/timeout = check/fold), 100 MB file limit.
The variant's payouts and blind schedule are announced at the start of the build
phase: copy them into section 1.

PLO has no range notation that works, so preflop "ranges" are percentiles of the
16,432 distinct starting hands (preflop_table.json): 0.15 = the best 15% of hands.
"""

# =====================================================================
# 1. GAME / TOURNAMENT SETTINGS  (update from the starter kit on event day)
# =====================================================================
NUM_SEATS = 8
STARTING_STACK = 2_000
HAND_LIMIT = 200
HANDS_PER_LEVEL = 25
HOLE_CARDS = 4
# Level 1 is from the rules; the rest are placeholders.
BLIND_SCHEDULE = [
    (10, 20), (15, 30), (25, 50), (40, 80),
    (60, 120), (100, 200), (150, 300), (250, 500),
]
# Points per finishing place (1st..8th). Placeholder: announced at the event.
PAYOUT_POINTS = [100, 60, 40, 25, 15, 8, 4, 0]

RANDOM_SEED = None  # set an int for reproducible bot randomness


# =====================================================================
# 2. PREFLOP RANGES (percentiles; positions UTG, UTG1, MP, HJ, CO, BTN, SB, BB)
# =====================================================================
# First in: raise this share of hands. PLO hands run close in equity, so these
# are wider than Hold'em, and position decides far more.
OPEN_PCT = {"UTG": 0.13, "UTG1": 0.15, "MP": 0.18, "HJ": 0.22, "CO": 0.30,
            "BTN": 0.45, "SB": 0.28, "BB": 0.12}
OVERLIMP_PCT = 0.35        # limp behind existing limpers with these
SB_COMPLETE_PCT = 0.50     # SB completes an unraised pot
STEAL_PCT_VS_TIGHT = 0.70  # BTN/SB open this wide when everyone behind is tight

# Facing ONE raise, by our position. In PLO 3-bets are mostly value (aces-heavy).
THREEBET_PCT = {"UTG": 0.025, "UTG1": 0.025, "MP": 0.03, "HJ": 0.035, "CO": 0.045,
                "BTN": 0.06, "SB": 0.04, "BB": 0.04}
CALL_OPEN_PCT = {"UTG": 0.08, "UTG1": 0.09, "MP": 0.10, "HJ": 0.12, "CO": 0.16,
                 "BTN": 0.28, "SB": 0.12, "BB": 0.30}

FOURBET_PCT = 0.012        # roughly aces with a good side
CALL_3BET_PCT = 0.10
JAM_4BET_PCT = 0.01
CALL_4BET_PCT = 0.02

# Short stacks: a pot raise commits us at about this depth, so act as push/fold.
SHOVE_STACK_BB = 10
# key = max effective stack in BB; the smallest key >= our stack is used.
SHOVE_PCT_BY_STACK = {3: 1.0, 5: 0.75, 7: 0.50, 10: 0.32}
CALL_SHOVE_PCT = 0.25
CALL_SHOVE_PCT_BUBBLE = 0.12
BUBBLE_PLAYERS_LEFT = 4

# =====================================================================
# 3. PREFLOP SIZING
# =====================================================================
# Pot-limit caps every raise. The first-in pot raise is 3.5 bb.
OPEN_SIZE_BB = {"default": 3.0, "SB": 3.0, "BB": 4.0}
LIMPER_ADD_BB = 1.0
THREEBET_POT_FRACTION = 1.0  # pot-sized 3-bets and 4-bets; share of the max raise
FOURBET_POT_FRACTION = 1.0
JAM_IF_COMMIT_FRACTION = 0.40  # a raise putting in >= 40% of our chips becomes all-in
SB_COMPLETE_MAX_STACK_BB = 20


# =====================================================================
# 4. POSTFLOP STRATEGY
# Buckets come from the measured equity of the hand (poker_core.read_hand +
# postflop_equity.json), so nut-ness is built in: a non-nut flush on a paired
# board lands far lower than the same flush on a dry board.
#   MONSTER      equity vs a value range >= BUCKET_MONSTER
#   STRONG       equity vs a value range >= BUCKET_STRONG
#   MEDIUM       ahead of most hands now: equity vs any hand >= BUCKET_MEDIUM
#   STRONG_DRAW  not MEDIUM+, equity vs value >= BUCKET_STRONG_DRAW (turn card matters)
#   DRAW         not MEDIUM+, equity vs value >= BUCKET_DRAW
#   WEAK         the rest
#
# bet_freq / bet_size / raise_freq / raise_size mean the same as in the NLH bot.
# There is no call_price: every call is decided by EV from the equity tables.
# =====================================================================
BUCKET_MONSTER = 0.60
BUCKET_STRONG = 0.48
BUCKET_MEDIUM = 0.58
BUCKET_STRONG_DRAW = 0.40
BUCKET_DRAW = 0.32

POSTFLOP_POLICY = {
    "MONSTER":     {"bet_freq": 0.95, "bet_size": 0.85, "raise_freq": 0.85, "raise_size": 1.00},
    "STRONG":      {"bet_freq": 0.85, "bet_size": 0.70, "raise_freq": 0.15, "raise_size": 0.90},
    "MEDIUM":      {"bet_freq": 0.35, "bet_size": 0.50, "raise_freq": 0.00, "raise_size": 0.75},
    "STRONG_DRAW": {"bet_freq": 0.65, "bet_size": 0.75, "raise_freq": 0.30, "raise_size": 1.00},
    "DRAW":        {"bet_freq": 0.30, "bet_size": 0.60, "raise_freq": 0.05, "raise_size": 0.75},
    "WEAK":        {"bet_freq": 0.08, "bet_size": 0.50, "raise_freq": 0.00, "raise_size": 0.75},
}
STREET_BET_FREQ_MULT = {"flop": 1.0, "turn": 0.9, "river": 0.8}
CBET_FREQ_HEADSUP = 0.60
CBET_FREQ_MULTIWAY = 0.20
CBET_SIZE = 0.50
MULTIWAY_BET_FREQ_MULT = 0.5
COMMIT_SPR = 1.0           # at this stack-to-pot or lower, MONSTER/STRONG get it in when betting
POT_LIMIT_MIN_EQ_RAISE = 0.0


# =====================================================================
# 5. WHAT WORKED IN THE NLH BOT (tuned values from OptimalBot / UPDATED_OPTIMAL)
# =====================================================================
# Bluff factor: multiplies bets and raises made without a made hand.
BLUFF_FACTOR = 1.5
BLUFF_BUCKETS = ("WEAK", "DRAW", "STRONG_DRAW")

# Tight-player exploits.
TIGHT_VPIP = 0.20
STEAL_SHIFT_VS_TIGHT = 2
RESPECT_TIGHT_RAISES = True
RESPECT_SHIFT_VS_TIGHT = 2
BLUFF_MULT_VS_TIGHT = 1.0
CBET_BONUS_VS_TIGHT = 0.35
THIN_VALUE_MULT_VS_TIGHT = 0.7

# Trapping: never on the river, rarely with the nuts in PLO (too many draws to pay off).
TRAP_FREQ = {"flop": 0.05, "turn": 0.145, "river": 0.0}
CHECK_RAISE_FREQ = 0.85

# Opponent reads.
TRACK_OPPONENTS = True
MIN_HANDS_FOR_READS = 15
LOOSE_VPIP = 0.65          # PLO players see far more flops than NLH players
AGGRO_PFR = 0.30
WIDEN_VS_AGGRO_OPENER = True

# Showdown bluff / fold tracking (Bayesian priors, as in UPDATED_OPTIMAL).
BLUFF_PRIOR = 0.25
BLUFF_PRIOR_WEIGHT = 6
SEMI_BLUFF_WEIGHT = 0.5
FOLD_PRIOR = 0.40
FOLD_PRIOR_WEIGHT = 10

# EV calls.
CALL_EV_MARGIN = 0.0        # extra equity needed on top of break-even
RAISE_BLUFF_SCALE = 0.7     # raises are bluffs less often than bets
FOLD_TO_RAISE_SCALE = 0.6
MULTIWAY_EQ_EXP = 0.7       # equity ^ (1 + this x extra opponents)
REALIZE_OOP = 0.92          # equity realised out of position with cards to come
REALIZE_IP = 1.00
DRAW_REVERSE_ODDS = 0.04    # subtract from non-made hands: draws pay more when they miss

# ICM: whenever a call risks this share of our chips, break-even comes from points.
USE_ICM = True
ICM_MIN_RISK_FRACTION = 0.25

# Preflop all-in calls (equity lookup, preflop_table.json).
PREFLOP_EV = True
PREFLOP_EV_MIN_RISK = 0.25
PREFLOP_CALL_MARGIN = 0.02
SHOVE_FOLD_ALLOWANCE = 0.08
SHOVER_WIDE_UNDER_BB = 15   # stacks under this shove wide, so use equity vs any
SHOVER_MEDIUM_UNDER_BB = 35 # under this use equity vs top 20%; deeper vs top 5%
