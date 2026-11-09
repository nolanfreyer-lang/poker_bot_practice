"""DEEP_THOUGHT_V10: DEEP_THOUGHT_V9 that guesses a new bot's play from its older versions.

    from deep_thought_v10 import DeepThoughtV10Bot
    ("DEEP10", DeepThoughtV10Bot(seed=seed))

Saved Chipzen profiles are found by exact display name, so a bot that uploads a new
version (slopUpBotV3 after slopUpBotV2) starts from the field average. v10 falls back to
its family: the name with the version tag stripped (V3, -v5-1, _v2, 7, -IV), and takes the
newest relative's profile at FAMILY_PROFILE_WEIGHT of its counts, so the match's own
hands take over sooner. The relative's forced exploits are not carried over, since they
came from hand data with that one version. An exact name always wins.

Why only part weight: across the five bot families in the profiles (Deep_Thought v1-v7,
Phineas-III/IV, fermat1/6, Sleight-of-Hand/v3, v003/v004), a relative's rates guess a
new version's flop bet rate far better than the field average (off by 0.10 against
0.29) and its fold to raises about as well (0.13 against 0.14), but its fold to bets
and river bet rate worse (0.15 against 0.10, 0.11 against 0.08): authors change their
defense between versions more than their aggression. slopUpBot and slopUpBotV2 play
very differently, for one.

Everything else is v9. The Chipzen adapter calls load_named_profile when the bot has it.
"""
import re

import config as default_config
from deep_thought_v9 import DeepThoughtV9Bot

DEEP10_CONFIG = {
    "FAMILY_PROFILES": True,
    "FAMILY_PROFILE_WEIGHT": 0.5,       # share of a relative's counts we start from
}

_VERSION = re.compile(r"[-_ ]?(?:v|ver|version)?(\d+(?:[-_.]\d+)*)$", re.I)
_ROMAN = re.compile(r"[-_ ](i{1,3}|iv|vi{0,3}|ix|x)$", re.I)
_ROMAN_VALUE = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10}


def name_family(name):
    """-> (family, version tuple): 'slopUpBotV2' -> ('slopupbot', (2,)); no tag -> version ()."""
    name = (name or "").strip()
    m = _VERSION.search(name)
    if m and m.start() > 0:
        return name[:m.start()].lower().strip("-_ "), tuple(int(x) for x in re.split(r"[-_.]", m.group(1)))
    m = _ROMAN.search(name)
    if m:
        return name[:m.start()].lower().strip("-_ "), (_ROMAN_VALUE[m.group(1).lower()],)
    return name.lower(), ()


def family_relative(name, profiles):
    """The newest other version of this bot among the profiles, or None."""
    family, _ = name_family(name)
    if len(family) < 3:                 # 'v003' would match every v-number bot
        return None
    best = None
    for other, profile in profiles.items():
        if other == name or other.startswith("_"):
            continue
        fam, version = name_family(other)
        if fam == family:
            key = (version, profile.get("hands", 0))
            if best is None or key > best[0]:
                best = (key, other)
    return best[1] if best else None


def scaled_profile(profile, weight):
    """The profile with every count times weight, and no forced exploits."""
    counts = {tracker: {table: [[sub, n * weight] for sub, n in rows] for table, rows in tables.items()}
              for tracker, tables in profile.get("counts", {}).items()}
    return {**{k: v for k, v in profile.items() if k != "force"}, "counts": counts}


class DeepThoughtV10Bot(DeepThoughtV9Bot):
    def __init__(self, cfg=default_config, overrides=None, seed=None):
        super().__init__(cfg, {**DEEP10_CONFIG, **(overrides or {})}, seed)
        self.profile_source = {}    # seat -> name of the profile it started from

    def load_named_profile(self, seat, name, profiles):
        """Start this seat from its own profile, or else part of its newest relative's."""
        if name in profiles:
            self.load_profile(seat, profiles[name])
            self.profile_source[seat] = name
            return name
        if not self.cfg.FAMILY_PROFILES:
            return None
        relative = family_relative(name, profiles)
        if relative is None:
            return None
        self.load_profile(seat, scaled_profile(profiles[relative], self.cfg.FAMILY_PROFILE_WEIGHT))
        self.profile_source[seat] = relative
        return relative
