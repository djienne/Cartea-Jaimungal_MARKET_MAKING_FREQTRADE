# Latency x requote-hysteresis sweep

Companion to `latency_hysteresis_sweep.json`. Generated 2026-08-23.

> **Supersession note.** The rows remain valid for this frozen window, but the
> 95-hour latency conclusion originally cited below did not survive the later
> 161.95-hour tape, and the schema-v5 395.69-hour sweep reordered the scenarios
> again. Latency remains coupled to refresh cadence. Use this artifact for
> address-action sensitivity, not for a causal latency ranking.

Twelve deterministic replays over **one frozen CASHCAT window** — 357 shards
copied out of the live collector tape, 174.8 min span, of which replay consumed
a 119-minute calibration window — crossing simulated latency
(`dry_run.decision_latency_ms` = `acknowledgement_latency_ms` =
`cancel_latency_ms`) with `quoting.replace_threshold_bps`. Everything else came
from `config/cashcat.toml` as it existed at companion-JSON build revision
`8fb76781f9a0`, not necessarily the current file. Exchange-time base. All 12 runs reported
`scientifically_valid = true`.

Freezing the tape matters: the collector writes continuously, so runs taken
minutes apart would otherwise see different windows and would not be comparable.

## Results

| latency ms | bps | fills | address actions | actions/fill | equity | realized | mk 100ms | 1s | 5s | 30s | end inv |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0 | 57 | 20,851 | **366** | 994.21 | -5.872 | +3.67 | +1.08 | -0.69 | -1.17 | 1,262 |
| 50 | 2 | 52 | 10,371 | **199** | 1000.19 | -0.037 | +3.72 | +1.00 | -0.50 | -0.28 | 1,431 |
| 50 | 4 | 46 | 5,354 | **116** | 999.60 | -0.534 | +3.65 | +1.13 | -0.33 | -0.52 | 1,317 |
| 100 | 0 | 47 | 20,764 | **442** | 998.51 | -1.859 | +2.91 | -0.27 | -0.35 | -0.70 | 1,601 |
| 100 | 2 | 48 | 10,233 | **213** | 999.73 | -0.579 | +3.29 | -0.05 | -0.69 | +0.43 | 1,550 |
| 100 | 4 | 55 | 5,446 | **99** | 999.03 | -1.102 | +3.47 | +0.07 | -0.48 | +1.52 | 1,404 |
| 200 | 0 | 54 | 20,812 | **385** | 1000.51 | +0.267 | +2.87 | +1.21 | +1.87 | +0.71 | 1,870 |
| 200 | 2 | 48 | 10,400 | **217** | 999.35 | -0.591 | +3.18 | +1.30 | +1.45 | +1.03 | 1,038 |
| 200 | 4 | 70 | 5,486 | **78** | 999.77 | -0.285 | +4.07 | +1.32 | +1.31 | +2.09 | 1,273 |
| 500 | 0 | 51 | 20,887 | **410** | 998.70 | -1.444 | +2.33 | +0.57 | +0.44 | +0.75 | 1,745 |
| 500 | 2 | 42 | 10,273 | **245** | 997.38 | -2.807 | +2.79 | +0.51 | +0.55 | +0.42 | 1,758 |
| 500 | 4 | 51 | 5,463 | **107** | 998.10 | -2.246 | +3.10 | -0.20 | +0.15 | +0.87 | 1,881 |

## What this window can and cannot answer

**It answers the action cost decisively.** Address-action consumption is a
function of the hysteresis and is essentially independent of latency — the
spread across latencies within each `bps` is under 1%:

| bps | mean address actions | mean actions/fill | range across latencies |
|---:|---:|---:|:--|
| 0 | 20,828 | 401 | 20,764–20,887 |
| 2 | 10,319 | 218 | 10,233–10,400 |
| 4 | 5,437 | 100 | 5,354–5,486 |

That is the constraint that binds live (`live_canary_20260823.md`): the venue
allowance is a lifetime `10,000 + 1 per USDC traded` and never resets. `bps = 4`
buys roughly **4x** the runtime per unit of allowance that `bps = 0` does, and
**2.2x** that of the shipped `bps = 2`.

**It cannot answer the latency economics.** Equity across all 12 runs spans just
6.30 USDC (stdev 1.62) on 42–70 fills, and the latency means are
non-monotonic:

| latency ms | mean equity | mean fills | mean 100ms markout |
|---:|---:|---:|---:|
| 50 | 998.00 | 51.7 | +3.68 |
| 100 | 999.09 | 50.0 | +3.23 |
| 200 | 999.88 | 57.3 | +3.37 |
| 500 | 998.06 | 48.0 | +2.74 |

200 ms scoring best and 50 ms mid-pack is noise, not a finding. Worse, the
equity delta is almost entirely **mark-to-market on a large open inventory**
(every run ends 1,038–1,881 units long, and `mtm` tracks the equity delta almost
exactly) — it is measuring an open directional bet, not maker skill.

The only latency-shaped hint is the 100 ms markout drifting down as latency
rises (3.68 → 3.23 → 3.37 → 2.74), which is directionally right but well inside
the noise.

**The later tapes do not isolate latency either.** The 95-hour, 161.95-hour, and
395.69-hour staged sweeps rank the coupled machine scenarios differently.
Because each scenario changes latency and refresh together, none identifies a
causal benefit from latency alone.

## Recommendation

- Keep `replace_threshold_bps = 2.0` as the economic default; the evidence for
  it is `requote_hysteresis_sweep.md`, not this run.
- Treat `4.0` as the **budget-constrained** setting: when the objective is
  runtime per unit of address allowance rather than P&L per fill, it is 2.2x
  more efficient, at the cost of the worst 30s markout in the earlier sweep.
- Draw no latency conclusion until latency is varied independently on several
  representative frozen windows with adequate fills. The collector's 30-day
  retention makes that experiment possible (`DATA_COLLECTION.md`).
