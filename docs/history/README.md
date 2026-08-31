# Period archive — what survives the 30-day tape

The CASHCAT collector keeps **30 days** (`CASHCAT_RETENTION_MINUTES: 43200`) and
deletes everything older. A replay can only score a window while its Parquet
shards exist, so once a period rolls off the tape **no sweep can ever be run
against it again** — not more cheaply, not at all. The dry-run grid expires too:
its event logs rotate at ~34 days.

Every 21 days `scripts/archive_period.py` attempts to write one directory here
with a fresh full sweep and the grid's P&L curve for the period. A successful
cycle has 9 days of slack against 30-day retention, so an interrupted attempt can
be retried within that margin. A failure is not harmless indefinitely; inspect
`sweep_FAILED.log` and rerun with `--force` before the oldest unarchived shards
expire.

These files are committed. That is the point: they are the only durable record
of a window whose raw data is gone.

## Layout

`<YYYY-MM-DD>_<SYMBOL>/`, dated by when it was archived:

| file | what it is |
|---|---|
| `README.md` | tape span, headline numbers, provenance — **start here** |
| `sweep.md` / `sweep.json` | the staged replay sweep on the tape as it stood |
| `grid_leaderboard.json` | the live grid's ranking at that moment |
| `grid_equity_curve.csv.zst` | the period's P&L curve, 15-min, zstd |
| `grid_pnl_curve.png` | that curve, rendered |

Each period's `README.md` is generated and carries the sweep's held-out table,
the latency ladder and the grid's top and bottom rows — enough to read the
period without decompressing anything.

## Two window conventions, deliberately different

- The **sweep** scores the whole tape on disk (up to 30 days), so consecutive
  archives **overlap** by roughly 9 days. That overlap is the safety margin.
- The **grid curve** covers only the period since the previous archive, so the
  archives **concatenate** into one continuous non-overlapping timeline.

Both spans are recorded in every period README, so neither has to be inferred.

## Reading a curve

```
zstd -d 2026-08-30_CASHCAT/grid_equity_curve.csv.zst -c | head
python scripts/grid_pnl_curve.py --history <the decompressed csv> --out /tmp/curve.png
```

The columns are the grid's own `equity_history.csv` schema, thinned to one row
per variant per 15 minutes. Full resolution is 60 s, about 95 MB/month — too
much to commit every three weeks, and not needed for a month-scale
retrospective. The full-resolution file lives in
`rust_live/reports/grid_live/equity_history.csv`; unlike event logs it is
intentionally not rotated and grows at roughly 95 MB/month.

## Scope

The only supported instrument profile is CASHCAT. The archiver selects any symbol whose tape spans more than 7
days, which cleanly separates the 30-day collector from the 3-day one (ETH, ACE,
CHIP, PENGU, NIL at `RETENTION_MINUTES: 4320`) without this repo reading another
project's compose file. A symbol that qualifies on tape length but has no
instrument profile is skipped rather than archived with CASHCAT's tick size and
inventory base — confident numbers for the wrong asset are worse than none.

## The one manual step

The `mm-archiver` container writes here but **does not commit**: that would mean
mounting an SSH key into a container. It logs an uncommitted-history reminder on
every wake; use `git status` to see the exact paths.

```
git add docs/history && git commit && git push
```
