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

Re-running the NLH robustness test showed the EV/ICM/showdown layer was the real gain and the tuned bluff settings were mostly noise, so this bot is EV-first. Ablation here, 4,000 paired tables per field (change in points when the piece is removed; bold = t >= 3):

| Removed | likely | tough | loose | tight | mirror |
|---|---|---|---|---|---|
| Opponent tracking | **-1.87** | -0.56 | **-1.02** | **-6.88** | +0.14 |
| ICM | **-1.61** | **-1.07** | **-0.64** | **-4.61** | **-1.45** |
| Preflop EV | +0.32 | +0.46 | **-0.71** | +0.13 | **+1.28** |
| Everything (BASELINE) | **-1.54** | -0.44 | **-1.00** | **-5.97** | +0.37 |

Bluff factor 1.5 vs 1.0 made no measurable difference in any field. Preflop EV helps against loose fields and hurts in the mirror, so it is a judgment call. `PLO4` scores 49.1 / 38.3 / 66.9 / 55.3 / 30.4 in these fields against an even share of 31.5.

`local_sim_search.py` chose the current settings: a 3% call margin, 0.85 out-of-position realisation, 7 bb push/fold, opens x1.25 and flats x0.75. Together they added +1.48 ± 0.20 points on fresh tables.

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
