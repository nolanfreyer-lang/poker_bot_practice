# Deep_Thought_v4-dealer for Chipzen

A frozen copy of DEEP_THOUGHT_V4 at commit 51355c4. Do not edit or regenerate `bot.py`:
later strategy changes go to DEEP_THOUGHT_V4, not to this bot.

Its one change: when two players are left, it uses the seat that posted the small blind
as the button, in case `dealer_seat` points at the big blind at a shrinking table.

```bash
docker build --platform linux/amd64 -t deep-thought-v4-dealer:v1 .
docker run --rm --entrypoint sh deep-thought-v4-dealer:v1 -c "ls /bot/"
docker save deep-thought-v4-dealer:v1 | gzip > deep-thought-v4-dealer.tar.gz
```
