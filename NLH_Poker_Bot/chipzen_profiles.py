"""Build opponent profiles from Chipzen replays, so DEEP_THOUGHT_V7 remembers bots by name.

    python3 chipzen_profiles.py                 # update chipzen_profiles.json
    python3 chipzen_profiles.py --build         # ...and regenerate chipzen_deep_thought_v7/bot.py
    python3 chipzen_profiles.py --package       # ...and build the Docker image and .tar.gz to upload
    python3 chipzen_profiles.py --package --bot v8   # the same for chipzen_deep_thought_v8
    python3 chipzen_profiles.py --package --bot v9   # ... and chipzen_deep_thought_v9
    python3 chipzen_profiles.py --package --bot v10  # ... and chipzen_deep_thought_v10
    python3 chipzen_profiles.py --package --bot v11  # ... and chipzen_deep_thought_v11 (v10's code)
    python3 chipzen_profiles.py --package --bot v12  # ... and chipzen_deep_thought_v12
    python3 chipzen_profiles.py --package --bot v13  # ... and chipzen_deep_thought_v13 (v12's code)
    python3 chipzen_profiles.py --package --bot v14  # ... and chipzen_deep_thought_v14
    python3 chipzen_profiles.py --package --bot gto  # ... and chipzen_deep_thought_gto (no profiles)

--build and --package add one to the version in chipzen_deep_thought_v7/VERSION (or the
--bot's folder); the Docker tag and upload file carry it (deep-thought-v7:v3,
deep-thought-v7-v3.tar.gz). DEEP_THOUGHT_V8, V9 and V10 read the same profiles as v7
(V10 also falls back to a new bot's older versions).
The profiles also work for DEEP_THOUGHT_V5 and V6, which skip the reads they don't have.
After the table it prints v7's exploit plan for each bot: what it switches on at match start.

Chipzen bots can't keep files between matches (each match gets a fresh, read-only
container), so the memory is built into the package: this script replays past matches
through the bot's own opponent trackers, saves each bot's counts by name, and the build
embeds them. At match start the bot looks up each opponent's display name and starts
its reads from these counts (capped at PROFILE_MAX_HANDS hands) instead of from zero.

Where the matches come from:
  * Season fixtures, which are public: the current season and the two before it by
    default (--seasons 11 12 13 to choose, --seasons all for every season).
  * Your own matches, from chipzen_matches.txt (one match link or ID per line) or the
    command line. Chipzen needs you logged in to list them, so to collect them all,
    open chipzen.ai while logged in, open the browser console, and paste:

    (async()=>{const t=await Clerk.session.getToken(),h={Authorization:'Bearer '+t};const m=await(await fetch('/api/bots/mine',{headers:h})).json();const ids=[];for(const b of(m.bots||m)){let o=0;for(;;){const j=await(await fetch(`/api/bots/${b.id||b.bot_id}/matches?limit=50&offset=${o}`,{headers:h})).json();ids.push(...j.matches.map(x=>x.match_id));if(j.next_offset==null||!j.matches.length)break;o=j.next_offset}}copy(ids.join('\\n'));console.log(ids.length+' match IDs copied')})()

    Then paste the copied IDs into chipzen_matches.txt.

Replays are cached in chipzen_replays/, so reruns only download new matches.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from deep_thought_v7 import DeepThoughtV7Bot
from deep_thought_v14 import DeepThoughtV14Bot, tracker_counts    # profiles use the newest bot's trackers
from poker_core import parse_cards

HERE = Path(__file__).resolve().parent
API = "https://chipzen.ai/api"
CACHE = HERE / "chipzen_replays"
OUT = HERE / "chipzen_profiles.json"
MATCH_FILE = HERE / "chipzen_matches.txt"
BOTS = {"v7": "DEEP_THOUGHT_V7", "v8": "DEEP_THOUGHT_V8", "v9": "DEEP_THOUGHT_V9", "v10": "DEEP_THOUGHT_V10", "v11": "DEEP_THOUGHT_V11", "v12": "DEEP_THOUGHT_V12", "v13": "DEEP_THOUGHT_V13", "v14": "DEEP_THOUGHT_V14", "gto": "DEEP_THOUGHT_GTO"}   # --bot -> build_chipzen name


def package_dir(bot):
    return HERE.parent / f"chipzen_deep_thought_{bot}"   # its VERSION goes +1 on every --build or --package
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
STREETS = ("preflop", "flop", "turn", "river")
# Exploits to force on against named Chipzen bots, from our own hand data with them, for
# flaws the counts don't make v7 confident about yet. Keys are display names; values
# are deep_thought_v7 exploit names. Review when a bot's results change.
CHIPZEN_EXPLOITS = {
    # After calling our weak flop bet: barrelling the turn won 9.0 BB each (12 times),
    # checking lost 5.4 BB each (19 times); it folds 49% to the second bet.
    "Shuck1e1": ["barrel_weak", "barrel_draw"],
    # Same spot: barrelling won 0.8 BB each (35), checking lost 0.7 BB each (20); it folds
    # 39% to a half-pot second bet, which needs 33%. Small edge, small sample.
    "LimitLess": ["barrel_weak"],
}


def get(path, tries=6):
    """GET a Chipzen API path, waiting and retrying when the server says to slow down."""
    req = urllib.request.Request(f"{API}/{path}", headers={"User-Agent": "chipzen-profiles"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == tries - 1:
                raise
            wait = float(e.headers.get("Retry-After") or 5 * 2 ** attempt)
            print(f"  rate limited, waiting {wait:.0f}s")
            time.sleep(wait)


def season_matches(seasons):
    """Match IDs of every played fixture in these seasons."""
    current = get("competitions/current")["number"]
    if seasons == ["all"]:
        numbers = range(1, current + 1)
    elif seasons:
        numbers = [int(n) for n in seasons]
    else:
        numbers = range(max(1, current - 2), current + 1)
    ids = []
    for n in numbers:
        try:
            fixtures = get(f"competitions/{n}/fixtures").get("fixtures", [])
        except Exception as e:
            print(f"season {n}: {e}")
            continue
        found = [f["match_id"] for f in fixtures if f.get("match_id")]
        print(f"season {n}: {len(found)} played fixtures")
        ids += found
    return ids


def listed_matches(args):
    """Match IDs from chipzen_matches.txt and the command line."""
    text = " ".join(args)
    if MATCH_FILE.exists():
        text += " " + MATCH_FILE.read_text()
    return UUID.findall(text)


def replay(match_id):
    path = CACHE / f"{match_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    data = get(f"matches/{match_id}/replay")
    CACHE.mkdir(exist_ok=True)
    path.write_text(json.dumps(data))
    time.sleep(1)                   # be gentle with the server
    return data


def hand_history(hand):
    """Replay hand -> (history, dealt seats). Replays show raises and calls as chips added."""
    start = {s["seat"]: s["starting_stack"] for s in hand["starting_stacks"]}
    history, spent = [], {}
    for street in hand["streets"]:
        name, bets, high = street["street"], {}, 0
        for a in street["actions"]:
            seat, kind, amt = a["seat"], a["action_type"], a.get("amount") or 0
            if kind == "post_ante":
                spent[seat] = spent.get(seat, 0) + amt
                continue
            if kind in ("post_small_blind", "post_big_blind", "raise", "call"):
                spent[seat] = spent.get(seat, 0) + amt
                bets[seat] = bets.get(seat, 0) + amt
            if kind in ("post_small_blind", "post_big_blind"):
                history.append((name, seat, "blind", bets[seat]))
            elif kind == "raise":
                all_in = spent[seat] >= start.get(seat, 0)
                history.append((name, seat, "allin" if all_in else "bet" if high == 0 else "raise", bets[seat]))
            elif kind == "call":
                history.append((name, seat, "call", bets[seat]))
            elif kind in ("check", "fold"):
                history.append((name, seat, kind, bets.get(seat, 0)))
            high = max(high, bets.get(seat, 0))
    dealt = sorted(s for s, chips in start.items() if chips > 0)
    return history, dealt


def merge(into, counts):
    for tracker, tables in counts.items():
        for table, rows in tables.items():
            have = {tuple(sub): n for sub, n in into.setdefault(tracker, {}).get(table, [])}
            for sub, n in rows:
                have[tuple(sub)] = have.get(tuple(sub), 0) + n
            into[tracker][table] = [[list(sub), n] for sub, n in have.items()]


def profile_match(data, profiles):
    """Run each seat's heads-up hands through the bot's trackers, from the other seat's view."""
    if data.get("metadata", {}).get("game_type") != "poker":
        return 0
    names = {p["seat"]: p["name"] for p in data["metadata"]["players"]}
    hands = [(h, *hand_history(h)) for h in data["hands"] if not h.get("variant")]
    used = 0
    for seat, name in names.items():
        bot, n = DeepThoughtV14Bot(seed=0, overrides={"USE_ICM": False}), 0
        bot.see_everything()        # replays show every hand: count bluffs that won too
        for h, history, dealt in hands:
            if len(dealt) != 2 or seat not in dealt:
                continue
            me = next(s for s in dealt if s != seat)
            bot._my_seat = me
            shown = {int(s): parse_cards(c) for s, c in h["hole_cards"].items() if int(s) in dealt}
            bot.on_hand_end(history, dealt, showdown=shown, board=parse_cards(h["community_cards"]), my_seat=me)
            n += 1
        if n:
            p = profiles.setdefault(name, {"hands": 0, "matches": 0, "counts": {}})
            p["hands"] += n
            p["matches"] += 1
            merge(p["counts"], tracker_counts(bot, seat))
            used += n
    return used


def rate(p, tracker, table_hit, table_all, sub=()):
    rows = p["counts"].get(tracker, {})
    hit = sum(n for s, n in rows.get(table_hit, []) if tuple(s) == sub)
    total = sum(n for s, n in rows.get(table_all, []) if tuple(s) == sub)
    return f"{hit / total:4.0%} ({total:.0f})" if total else "    -     "


def summary(profiles, limit):
    print(f"\n{'bot':22s}{'hands':>6s}  {'fold to bet':>12s}  {'flop bets':>12s}  {'river bets':>12s}"
          f"  {'fold to raise':>13s}  {'raise leads':>12s}")
    for name, p in sorted(profiles.items(), key=lambda kv: -kv[1]["hands"])[:limit]:
        print(f"{name[:22]:22s}{p['hands']:6d}  {rate(p, 'sd', 'folds', 'fold_chances'):>12s}"
              f"  {rate(p, 'street_reads', 'bets', 'chances', ('flop',)):>12s}"
              f"  {rate(p, 'street_reads', 'bets', 'chances', ('river',)):>12s}"
              f"  {rate(p, 'raise_reads', 'folds', 'chances'):>13s}"
              f"  {rate(p, 'lead_reads', 'raises', 'chances'):>12s}")


def plans(profiles):
    """v7's exploits at match start against each bot, from its profile alone."""
    print("\nexploit plans (what v7 switches on against each bot at match start):")
    shown = 0
    for name, p in sorted(profiles.items(), key=lambda kv: -kv[1]["hands"]):
        bot = DeepThoughtV7Bot(seed=0)
        bot.load_profile(1, p)
        plan = bot.exploit_plan(1)
        if plan:
            print(f"  {name}: " + "; ".join(plan))
            shown += 1
    print(f"  ({shown} bots with an exploit; the rest start from v6's play)")


def next_version(bot):
    """Add one to the package version in chipzen_deep_thought_<bot>/VERSION and return it."""
    version_file = package_dir(bot) / "VERSION"
    version = int(version_file.read_text().strip() or 0) + 1 if version_file.exists() else 1
    version_file.write_text(f"{version}\n")
    return version


def package(bot, version):
    """Build the Docker image and save the .tar.gz to upload."""
    env = dict(os.environ, PATH=os.path.expanduser("~/.docker/bin") + os.pathsep + os.environ["PATH"])
    tag = f"deep-thought-{bot}:v{version}"
    subprocess.run(["docker", "build", "--platform", "linux/amd64", "-t", tag, "."], cwd=package_dir(bot), env=env,
                   check=True)
    out = package_dir(bot) / f"deep-thought-{bot}-v{version}.tar.gz"
    with open(out, "wb") as f:
        save = subprocess.Popen(["docker", "save", tag], stdout=subprocess.PIPE, env=env)
        subprocess.run(["gzip"], stdin=save.stdout, stdout=f, check=True)
        save.wait()
    print(f"wrote {out} ({out.stat().st_size / 2**20:.0f} MB): upload it on chipzen.ai")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("matches", nargs="*", help="match links or IDs to add")
    ap.add_argument("--seasons", nargs="*", default=[], help="season numbers, or 'all' (default: last three)")
    ap.add_argument("--no-seasons", action="store_true", help="only use your own listed matches")
    ap.add_argument("--build", action="store_true", help="regenerate chipzen_deep_thought_<bot>/bot.py")
    ap.add_argument("--bot", choices=list(BOTS), default="v7", help="which bot --build and --package make")
    ap.add_argument("--package", action="store_true", help="--build, then make the .tar.gz to upload")
    ap.add_argument("--show", type=int, default=40, help="bots to list in the summary")
    args = ap.parse_args()

    ids = listed_matches(args.matches)
    print(f"your listed matches: {len(ids)}")
    if not args.no_seasons:
        ids += season_matches(args.seasons)
    ids = list(dict.fromkeys(ids))
    profiles, hands, failed = {}, 0, 0
    for i, match_id in enumerate(ids, 1):
        try:
            hands += profile_match(replay(match_id), profiles)
        except Exception as e:
            failed += 1
            print(f"  {match_id}: {e}")
        if i % 50 == 0:
            print(f"  {i}/{len(ids)} matches read")
    for name, force in CHIPZEN_EXPLOITS.items():
        if name in profiles:
            profiles[name]["force"] = force
    profiles["_meta"] = {"matches": len(ids) - failed, "heads_up_hands": hands,
                         "updated": time.strftime("%Y-%m-%d %H:%M")}
    OUT.write_text(json.dumps(profiles, separators=(",", ":"), sort_keys=True))
    print(f"\n{len(ids) - failed} matches, {hands} heads-up hands, {len(profiles) - 1} bots -> {OUT.name}")
    summary({k: v for k, v in profiles.items() if not k.startswith("_")}, args.show)
    plans({k: v for k, v in profiles.items() if not k.startswith("_")})
    if args.build or args.package:
        import build_chipzen
        build_chipzen.write(BOTS[args.bot], package_dir(args.bot))
        version = next_version(args.bot)
        print(f"package version {version}")
        if args.package:
            package(args.bot, version)


if __name__ == "__main__":
    main()
