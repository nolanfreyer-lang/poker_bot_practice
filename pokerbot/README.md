# No Limit pokerbot template

Built around Exhibit A (8 bots, No-Limit Hold'em; actions: blind, check, call, bet, raise-to, all-in, fold) and the official rules: 8-bot sit-and-gos, 2,000 chips, blinds start at 10/20 and rise every 25 hands, 200-hand cap with ICM, and **1 ms per decision**.

| File | Edit it when… |
|---|---|
| `config.py` | **Always.** All ranges, sizings, postflop policy, and tournament settings. |
| `agent_adapter.py` | On event day, to map the starter kit's Agent interface to `GameState`/`Action`. |
| `bot.py` | You want different decision logic than ranges and buckets. |
| `poker_core.py` | Rarely. It holds the card parser, evaluator, range parser and hand buckets. |
| `local_sim.py` | To change the test lineup (`default_lineup`) or the opponent bots. |

```
python3 local_sim.py --tables 300          # points per table plus decision timing (watch the >1ms column)
python3 local_sim.py --tables 1 --verbose  # hand-by-hand log of one table
python3 local_sim.py --table 3 --hand 12               # replay one hand of one table
python3 local_sim.py --table 3 --hand 12 --show-cards  # ... with every bot's hole cards
```
Each table depends only on `--seed` and its table number, so the same command always replays the same hand.

To test ExploitBot variants against other bots, edit the SETUP block at the top of `local_sim_lab.py` (your bots, the opponent pool, the baseline), then:
```
python3 local_sim_lab.py                      # 10 runs x 1000 tables
python3 local_sim_lab.py --runs 3 --tables 200  # quick check
```

Event-day checklist:
1. Copy the blind schedule and points table into `config.py`.
2. Wire up `agent_adapter.py`.
3. If the variant isn't Hold'em, change `classify_hand` and the preflop ranges.
4. Run the sim and confirm the bot never goes over 1 ms.
