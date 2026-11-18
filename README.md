# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/upayy9234-art/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/upayy9234-art/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecast GTA VI's launch-day Twitch viewers hour by hour: a deterministic pooled hour-fixed-effects regression over past big launches' release-day curves, anchored on hour 1's print. Pure Python standard library, no network, no clock, no randomness.

## How it works

GTA VI unlocks at midnight local time, rolling (New Zealand 11:00Z on 18 Nov, central Europe 23:00Z, the UK 00:00Z, the US east coast 05:00Z, the west coast 08:00Z), so at 00:00Z only New Zealand / Australia / Asia / Europe are playing and the Americas are still locked. Past launches show the same pattern: when hour 1 is small (only a few regions live yet) the day ramps several-fold into the US evening; when hour 1 is already huge (a simultaneous global unlock, or early access already running) the day stays flat. Across the clean comparables the peak-to-hour-1 ratio runs from ~1.0x (ELDEN RING, hour 1 = 875k) to ~12x (Resident Evil Requiem, hour 1 = 46k) and ~9x (Marvel's Wolverine, hour 1 = 23k).

So, for each hour 2..24, the program fits one pooled regression over all clean comparables' log ratios log(viewers_h / viewers_1): a free intercept per hour, a shared slope on centred log(viewers_1) with a linear hour modulation, plus two round-2 features — log(channels at hour 1) and log(viewers_1 / pre-launch 7-day median baseline), both centred on their training medians. How explosive a launch is relative to its own pre-launch baseline, and how concentrated its hour-1 audience is across channels, predict the day's shape (overnight dip, evening ramp) far better than hour-1 size alone. The forecast adds that hour's median training residual to the fitted log ratio; the spread is the hour's residual standard deviation (floored at 0.20 log units, scaled by 0.5), applied with the standard-normal quantiles. The target's features are clamped to the training range. Fallbacks keep every case valid: if the pooled fit fails, it falls back to round 1's per-hour OLS, then to the house baseline's persistence forecast.

## Local validation

Scored on the four practice cases against SullyGnome's hourly averages (the same source the answers come from), pinball loss on the log scale — lower is better:

| model | mean loss |
| --- | --- |
| house baseline (persistence) | 0.387 |
| round 1 (per-hour OLS) | 0.154 |
| this model (round 2), kit comparables | **0.120** |
| this model (round 2), final-like set* | **0.120** |

Per case (kit protocol): Marvel's Wolverine 0.108, Red Dead Redemption 2 0.103, God of War Ragnarök 0.092, Spider-Man 2 0.177. Typical miss ~21%, 90% coverage ~0.88.

* The final-like set augments each practice case's comparables with the other three practice launches, the way the final case ships them. The kit protocol is the relevant estimate for the dev score: round 1's kit-protocol local number (0.164) matched its dev score (0.164) exactly, so this entry's dev score is expected near 0.120 (round 1: 0.164).

Research anchors: GTA VI releases 19 November 2026 on PS5 and Xbox Series X|S (Rockstar, 6 November 2025; previously dated 26 May 2026); Trailer 1 drew 93m YouTube views in 24 hours (Guinness record), Trailer 2 drew 475m views across platforms in 24 hours; the single-player Twitch peak record is Hogwarts Legacy at ~1.28m concurrent viewers, ahead of Cyberpunk 2077 (~1.14m) and ELDEN RING (~0.91m).

## Run it

```sh
python3 -P -s main.py
```

Reads every `<id>.in` in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR` (else `/output`): `{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]}` for hours 2..24. Runs in ~0.3s per case, far under the 8-second limit; two runs produce byte-identical output.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
