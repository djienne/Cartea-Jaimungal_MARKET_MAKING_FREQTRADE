# CASHCAT - period archived 2026-08-30

Written by `scripts/archive_period.py` because the collector keeps only
30 days of CASHCAT tape and this window would otherwise be deleted. A replay can only score a
window while its Parquet shards exist, so once this period rolls off the tape
these files are the only remaining record of it.

## Provenance

- tape scored: **13.6 days** (2026-08-16 21:59 -> 2026-08-30 12:31 UTC), 12313 shards
- grid curve sliced from: 2026-08-16 21:59 UTC (all history - first archive)
- build: `unknown`

The two windows differ on purpose. The sweep scores the whole tape on disk, so
consecutive archives overlap and a skipped cycle still loses nothing; the grid
curve covers only the period since the last archive. `run_started_ms` preserves
independent run boundaries; archive periods do not overlap.

## Replay sweep

- status: `ok`
- historical pre-schema-v5 sweep; values below are preserved evidence, not current defaults
- search scenario: `good` (100 ms latency, 250 ms refresh)
- train/held-out split at `2026-08-26T10:35:45.136000+00:00`

| configuration | train P&L | held-out P&L | fills |
| --- | ---: | ---: | ---: |
| `eps200/200_klo0.75/0.75|phiKT1000_alphaK0.05_T150_q3` | -83.54 | **-99.12** | 2753 |
| `eps200/200_klo0.75/0.75|phiKT1000_alphaK0.5_T150_q3` | -83.54 | **-99.12** | 2753 |
| `eps200/200_klo0.75/0.75|phiKT1000_alphaK5_T150_q3` | -83.54 | **-99.12** | 2753 |

| latency scenario | P&L | fills |
| --- | ---: | ---: |
| colocated | +130.74 | 3046 |
| good | -99.12 | 2753 |
| mid | -263.89 | 2546 |
| this_stack | -365.01 | 2357 |
| reality | -255.45 | 3523 |

## Dry-run grid

- elapsed: 2.8 h, 0 resume(s)
- feed: 0.09% down, verdict VALID

| variant | net P&L | fills | inventory |
| --- | ---: | ---: | ---: |
| baseline | +0.00 | 0 | 0 |
| q3 | +0.00 | 0 | 0 |
| wide4 | +0.00 | 0 | 0 |
| ... | | | |
| slow15s | -5.09 | 57 | 51 |
| wide16slow30s | -6.82 | 68 | 112 |
| wide24slow30s | -7.85 | 46 | 75 |

## Files

| file | what it is |
| --- | --- |
| `sweep.md` / `sweep.json` | the full staged sweep on the tape above |
| `grid_leaderboard.json` | the grid's ranking at the moment of archiving |
| `grid_equity_curve.csv.zst` | the period's P&L curve, thinned and compressed |
| `grid_pnl_curve.png` | that curve, rendered |
