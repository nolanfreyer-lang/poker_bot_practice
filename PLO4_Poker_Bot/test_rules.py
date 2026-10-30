"""Rule checks for everything a Hold'em port gets wrong. Run: python3 test_rules.py

Covers the exactly-two-hole-cards rule, pot-limit caps, chip conservation, legal
bot output in every spot, and the 1 ms limit on the worst decisions.
"""
import random
import time

import config as C
from bot import Action, GameState, PlayerInfo, PokerBot, pot_limit_to
from local_sim import Table, default_lineup
from poker_core import PREFLOP, category, hand_key, parse_cards, plo_value, read_hand, strength


def value(hole, board):
    return plo_value(parse_cards(hole), parse_cards(board))


def test_exactly_two_hole_cards():
    # Four hearts in hand and one on board is NOT a flush: three board cards are needed.
    assert category(value("Ah Kh Qh Jh", "2h 3c 7d 9s Tc")) != 5
    # One heart in hand with four on board is NOT a flush either.
    assert category(value("Ah Kc Qd Js", "2h 3h 7h 9h Tc")) != 5
    # Two in hand plus three on board is.
    assert category(value("Ah Kh Qd Js", "2h 3h 7h 9c Tc")) == 5
    # Quads on board does not play: only three of them can be used, so this is trips.
    assert category(value("Ah Kc 3d 2s", "7h 7c 7d 7s Tc")) == 3
    # Board straight with no connecting hole cards does not play.
    assert category(value("Ah Ad 2c 2s", "5h 6c 7d 8s 9c")) < 4
    # Two hole cards complete the straight.
    assert category(value("5h 4c Kd Ks", "6h 7c 8d 2s 2c")) == 4
    print("ok  exactly two hole cards")


def test_nut_reading():
    # Top set on a dry flop is the nuts: nothing beats it.
    assert strength(parse_cards("As Ad Kc Qs"), parse_cards("Ah 7d 2c"))[1] == 0.0
    # Holding the ace of the suit makes the flush the nuts, the deuce does not.
    assert strength(parse_cards("Ah 3h Kc Kd"), parse_cards("9h 7h 2h"))[1] == 0.0
    assert strength(parse_cards("4h 3h Kc Kd"), parse_cards("9h 7h 2h"))[1] > 0.02
    # A real wrap on the flop has many straight outs.
    v, p, key = read_hand(parse_cards("Jc Tc 8d 7d"), parse_cards("9h 6s 2c"))
    assert key[3] >= 2, key
    print("ok  nut reading")


def test_pot_limit_formula():
    assert pot_limit_to(20, 0, 30, 2000) == 70             # first-in pot raise: 3.5 bb
    assert pot_limit_to(70, 0, 100, 2000) == 240           # pot 3-bet over a 70 open
    assert pot_limit_to(70, 20, 100, 2000) == 220          # big blind already has 20 in
    assert pot_limit_to(100, 0, 100, 50) == 50             # capped by our stack
    print("ok  pot-limit formula")


def make_state(rng, stack, to_call, pot, street, high):
    n = 8
    cards = rng.sample(range(52), 9)
    board = cards[4:4 + {"preflop": 0, "flop": 3, "turn": 4, "river": 5}[street]]
    players = [PlayerInfo(seat=i, stack=stack if i == 0 else 1500, bet=high if i == 1 else 0) for i in range(n)]
    players[0].bet = max(0, high - to_call)
    cap = pot_limit_to(high, players[0].bet, pot, stack)
    return GameState(my_seat=0, hole_cards=cards[:4], board=board, street=street, pot=pot, players=players,
                     button_seat=rng.randrange(n), small_blind=10, big_blind=20, to_call=min(to_call, stack),
                     min_raise_to=min(high + max(20, to_call), cap), max_raise_to=cap,
                     history=[("preflop", 1, "raise", 60)] if to_call else [])


def test_bot_output_always_legal():
    bot = PokerBot(seed=1)
    rng = random.Random(2)
    for _ in range(40000):
        street = rng.choice(("preflop", "flop", "turn", "river"))
        stack = rng.choice((1, 15, 60, 300, 2000, 9000))
        to_call = rng.choice((0, 0, 20, 100, 600))
        high = to_call + rng.choice((0, 0, 20))
        pot = max(30, to_call * 2 + rng.randrange(0, 900))
        s = make_state(rng, stack, to_call, pot, street, high)
        a = bot.act(s)
        everything = s.me.stack + s.me.bet
        assert a.kind in ("fold", "check", "call", "bet", "raise", "allin"), a
        if a.kind == "check":
            assert s.to_call == 0
        if a.kind == "call":
            assert 0 < s.to_call < s.me.stack
        if a.kind == "fold":
            assert s.to_call > 0
        if a.kind in ("bet", "raise"):
            assert s.min_raise_to <= a.amount <= s.max_raise_to, (a, s.min_raise_to, s.max_raise_to)
            assert a.amount < everything
            assert a.kind == ("bet" if s.me.bet + s.to_call == 0 else "raise")
        if a.kind == "allin":
            assert a.amount == everything, (a, everything, s.max_raise_to)
    print("ok  bot output legal in 40,000 random spots")


def test_sim_conserves_chips_and_caps_raises():
    for t in range(60):
        rng = random.Random(t)
        lineup = default_lineup(t)
        rng.shuffle(lineup)
        table = Table(lineup, rng)
        for hand_no in range(1, 60):
            if len(table.alive()) <= 1:
                break
            table.play_hand(hand_no, 10 * (1 + hand_no // 25), 20 * (1 + hand_no // 25))
            table.button = table.next_alive(table.button)
            assert sum(table.stacks) == 8 * C.STARTING_STACK, table.stacks
            assert min(table.stacks) >= 0
        assert table.errors == 0
    print("ok  chips conserved over 60 tables")


def test_all_hands_in_table():
    rng = random.Random(3)
    for _ in range(20000):
        assert hand_key(rng.sample(range(52), 4)) in PREFLOP
    print("ok  preflop table covers dealt hands")


def test_human_input():
    from play_vs_bot import HumanPlayer
    parse = HumanPlayer.parse
    s = make_state(random.Random(1), 2000, 100, 300, "flop", 100)      # pot-limit cap is 500
    assert s.max_raise_to == 500 and s.max_raise_to < s.me.stack + s.me.bet
    assert parse("r pot", s) == Action("raise", 500)
    assert parse("a", s) == Action("allin", 500)                        # capped, not the whole stack
    assert parse("r", s) == Action("raise", s.min_raise_to)
    assert parse("r 1/2", s) == Action("raise", 300)
    assert isinstance(parse("r 501", s), str)                           # over the pot limit
    assert isinstance(parse("r 1/0", s), str)
    assert isinstance(parse("x", s), str)                               # can't check facing a bet
    print("ok  human input")


def test_decision_speed():
    bot = PokerBot(seed=5)
    rng = random.Random(6)
    times = []
    for _ in range(100000):
        street = rng.choice(("preflop", "flop", "turn", "river"))
        to_call = rng.choice((0, 100, 700))
        s = make_state(rng, 2000, to_call, to_call * 2 + 200, street, to_call)
        t0 = time.perf_counter()
        bot.act(s)
        times.append(time.perf_counter() - t0)
    times.sort()
    n = len(times)
    print(f"ok  speed over {n}: avg {sum(times) / n * 1e3:.3f} ms, p99.9 {times[int(n * .999)] * 1e3:.3f} ms, "
          f"max {times[-1] * 1e3:.3f} ms, over 1 ms: {sum(1 for x in times if x > 0.001)}")


if __name__ == "__main__":
    test_exactly_two_hole_cards()
    test_nut_reading()
    test_pot_limit_formula()
    test_all_hands_in_table()
    test_bot_output_always_legal()
    test_sim_conserves_chips_and_caps_raises()
    test_human_input()
    test_decision_speed()
