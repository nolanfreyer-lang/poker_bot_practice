# PLO4 pokerbot

Pot-Limit Omaha (4 hole cards) bot for the same format as the NLH bot: 8-bot sit-and-gos, 2,000 chips, blinds from 10/20 rising every 25 hands, 200-hand cap with ICM, and **1 ms per decision**.

| File | Edit it when… |
|---|---|
| `config.py` | **Always.** Percentile ranges, sizings, postflop policy, tournament settings. |
| `agent_adapter.py` | On event day, to map the starter kit's Agent interface to `GameState`/`Action`. |
| `bot.py` | You want different decision logic. |
| `poker_core.py` | Rarely. Evaluator (exactly two hole cards + three board cards) and hand reading. |
| `build_tables.py` | To rebuild `preflop_table.json` and `postflop_equity.json` (~6 min, needs numba). |
| `opponent_bots.py` | To change the sparring partners. |
| `local_sim.py`, `local_sim_robust.py`, `local_sim_search.py` | To test: one lineup, every field with ablations, settings search. |
| `test_rules.py` | Run after any change to rules, legality or speed. |

```bash
python3 test_rules.py
```
```bash
python3 local_sim.py --tables 300
```
```bash
python3 local_sim_robust.py --tables 4000
```

## What carried over from the NLH bot

Re-running the NLH robustness test showed the EV/ICM/showdown layer was the real gain and the tuned bluff settings were mostly noise, so this bot is EV-first. Ablation here, 1,000 paired tables per field (points lost when the piece is removed; t >= 3 is real):

| Removed | likely | tough | loose | tight | mirror |
|---|---|---|---|---|---|
| Opponent tracking | -1.93 | -2.43 | -0.81 | **-8.55** | +0.03 |
| ICM | -1.25 | -1.33 | -0.48 | **-5.82** | **-2.53** |
| Everything (BASELINE) | -1.29 | -2.87 | -0.45 | **-9.92** | -0.66 |

These fields are bots written here, so they show the bot beats plausible leaks, not how it will rank on event day.

## What changed for PLO

- Preflop ranges are percentiles of all 16,432 starting hands, from simulated equity.
- Postflop equity is measured, not guessed: each spot maps to a table cell by how much of the deck beats us, flush-draw nut-ness, nut straight outs and board texture. Blockers are counted automatically.
- Every call is an EV decision. There is no fixed call price.
- `max_raise_to` is the pot-limit cap, and `allin` means "raise to the cap".

## Event-day checklist

1. Copy the blind schedule and points table into `config.py`.
2. Wire up `agent_adapter.py`. Confirm `max_raise_to` is the pot-limit cap.
3. Confirm the kit really is pot-limit. If it is not, `pot_limit_to` needs changing.
4. Run `test_rules.py` and the starter kit's simulator. Confirm nothing goes over 1 ms.
