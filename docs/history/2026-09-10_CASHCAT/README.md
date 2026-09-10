# CASHCAT - period archived 2026-09-10

Written by `scripts/archive_period.py` because the collector keeps only
30 days of CASHCAT tape and this window would otherwise be deleted. A replay can only score a
window while its Parquet shards exist, so once this period rolls off the tape
these files are the only remaining record of it.

## Provenance

- tape scored: **24.6 days** (2026-08-16 21:59 -> 2026-09-10 11:41 UTC), 25319 shards
- grid curve sliced from: 2026-08-30 00:00 UTC
- build: `unknown`

The two windows differ on purpose. The replay scores the whole tape on disk, so
consecutive archives overlap and a skipped cycle still loses nothing; the grid
curve covers only the period since the last archive. Run boundaries remain explicit
through `run_started_ms`; archive periods do not overlap.

## Replay

The same variants the grid ran, scored offline by the same simulator over the
window above. Rows are directly comparable to the grid table below, but they are
not the same measurement: the replay is one continuous pass at an assumed
latency, while a grid row may be stitched across restarts. See
`docs/DRY_RUN_GRID.md` for the fidelity limits.

- scored: 2026-08-18 03:28 -> 2026-09-10 11:41 UTC at 150 ms assumed latency
- train/score split: first 5% fits and sizes only

| variant | net P&L | fills | inventory |
| --- | ---: | ---: | ---: |
| baseline | -193.50 | 5614 | 13 |
| sweep1_flat300 | +953.05 | 2948 | 0 |
| sweep1_flat550 | +642.00 | 2946 | 0 |
| ... | | | |
| sweep3 | -190.18 | 5943 | -591 |
| sweep1_unguarded | -183.90 | 4874 | 1191 |
| sweep1_wide60 | -151.14 | 2678 | 4163 |

## Dry-run grid

- elapsed: 78.9 h, 4 resume(s)
- feed: 0.07% down, event loss no

| variant | net P&L | fills | inventory |
| --- | ---: | ---: | ---: |
| wide60 | -0.07 | 188 | -540 |
| q12 | -40.81 | 2804 | 28 |
| q9 | -63.51 | 2941 | 95 |
| ... | | | |
| sweep1_unguarded | -8.90 | 271 | 219 |
| slow5s | -192.27 | 6148 | -695 |
| slow15s | -195.43 | 6089 | -270 |

## Files

| file | what it is |
| --- | --- |
| `replay_leaderboard.json` | every grid variant, replayed over the tape above |
| `replay-<variant>.json` | that variant's full session report |
| `grid_leaderboard.json` | the grid's ranking at the moment of archiving |
| `grid_equity_curve.csv.zst` | the period's P&L curve, thinned and compressed |
| `grid_pnl_curve.png` | that curve, rendered |

