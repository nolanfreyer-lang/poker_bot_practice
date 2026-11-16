"""ChoasMimic: a stand-in for Chipzen's Choas_incarnate, for the simulator.

    from choas_mimic import ChoasMimic
    ("CHOAS", ChoasMimic(seed))

Choas_incarnate (first seen Nov 16, 10 heads-up matches) plays like Chaos_Incarnate in
volume: it opens about 73% of small blinds to 2.5 BB, 3-bets or shoves over about 30% of
opens, and bets every flop it can, mostly a third of the pot. Unlike Chaos, its big bets
and raises are often bluffs: at showdown 11 of 25 big bets and 4 of 8 raises were weak
hands or draws (Chaos: about 38% and 28%). So this is ChaosMimic with weak hands and draws
betting big and raising far more often.
"""
from chaos_mimic import ChaosMimic


class ChoasMimic(ChaosMimic):
    OPEN_SHARE = 0.73
    BIG_WHEN_NOT_STRONG = {"MEDIUM": 0.35, "STRONG_DRAW": 0.40, "DRAW": 0.30, "WEAK": 0.20}
    RAISE_BLUFF = 0.12
