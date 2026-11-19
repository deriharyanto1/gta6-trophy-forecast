# GTA VI Trophy Rarity Forecast

[![gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/deriharyanto1/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/deriharyanto1/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

A forecast for [gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
for every Grand Theft Auto VI trophy on PlayStation 5, five quantiles of its
earn rate on Exophase at 2026-11-26T00:00:00Z.

## The model

An earn rate is a ratio — earners over players tracked — and on launch night
the tracked-player pool jumps in big refreshes (638 tracked at 23:48, 3,868
after the 04:30 refresh), so a rate falls mostly because the denominator grows
with new players who have barely played. This program forecasts the two counts
separately, in log time:

- `ln E_i(t) = a_i + b_i * ln(1 + t)` — earners of trophy i
- `ln N(t) = a_N + b_N * ln(1 + t)` — players tracked
- `rate_i(T) = 100 * E_i(T) / N(T)`

with `t` in hours since the release (2026-11-19T00:00:00Z). Slopes are fitted
per trophy with a Theil-Sen estimator over the last 24 hours of reads and
shrunk toward per-trophy priors calibrated on the public Exophase page at
launch (fast-earned trophies near 0.5–0.6, slow ones near 0.15). When a case's
history is flat (the launch-night practice cases), the priors carry the
forecast and the players are assumed to jump once at the next refresh by the
first refresh's factor (3,868 / 638 = 6.06). Spreads are honest and widen with
the horizon: a fit residual, a random-walk term, and a refresh term.

## Run it

```bash
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Node 22, standard library only, no network. Reads every `*.in` file in the
input directory and writes one `<id>.out` per case to the output directory:
`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`, five
quantiles of each trophy's earn rate in percent, non-decreasing, 0–100.

## Practice

On the kit's two practice cases (p00, p01), scored against proxy answers
reconstructed from the known 04:30 refresh (3,868 players tracked) and the
public Exophase page at midday 19 Nov, this model scores about 0.09–0.15 mean
pinball loss on the logit scale, against about 1.27 for the house baseline's
persistence forecast on the same proxy answers. The practice answers are not
in the kit, so those numbers are against a reconstruction, not the official
practice grade — the live practice board is the check that counts.
