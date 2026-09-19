# Period archive — what survives the rolling tape

The CASHCAT collector keeps a bounded tape (`CASHCAT_RETENTION_MINUTES` in
`HYPERLIQUID_DATA/docker-compose.yml` — the one place that value is set) and
deletes everything older. A replay can only score a window while its Parquet
shards exist, so once a period rolls off the tape **no replay can ever be run
against it again** — not more cheaply, not at all. The dry-run grid expires too:
its event logs rotate at ~34 days.

Every 21 days `scripts/archive_period.py` attempts to write one directory here
with a fresh full replay and the grid's P&L curve for the period. A successful
cycle leaves retention-minus-cadence days of slack, so an interrupted attempt can
be retried within that margin. A failure is not harmless indefinitely; inspect
`replay_FAILED.log` and rerun with `--force` before the oldest unarchived shards
expire. Due-ness is read from the newest directory on disk, not a sleep timer,
so a reboot cannot reset the countdown.

These files are committed. That is the point: they are the only durable record
of a window whose raw data is gone.

## Layout

`<YYYY-MM-DD>_<SYMBOL>/`, dated by when it was archived:

| file | what it is |
|---|---|
| `README.md` | tape span, headline numbers, provenance — **start here** |
| `replay_leaderboard.json` | every grid variant, replayed over the tape as it stood |
| `replay-<variant>.json` | that variant's full session report |
| `grid_leaderboard.json` | the live grid's ranking at that moment |
| `grid_equity_curve.csv.zst` | the period's P&L histories, 15-min, zstd; `run_started_ms` keeps run boundaries |
| `grid_pnl_curve.png` | that curve, rendered |

Each period's `README.md` is generated and carries the replay's leaderboard and
the grid's top and bottom rows — enough to read the period without decompressing
anything. `2026-08-30_CASHCAT/` predates this layout: it holds only a README and
a grid leaderboard, its Python sweep having been retired with that engine.

## Two window conventions, deliberately different

- The **replay** scores the whole tape on disk (up to the retention window), so consecutive
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
per variant per 15 minutes. Full resolution is 60 s and roughly 0.1 GB/month at
the current grid size—too much to commit every three weeks. Each scientific run
stores its full-resolution file under
`rust_live/reports/grid_live/runs/<run-id>/`; unlike event logs, those files are
not rotated.

## Scope

The only supported instrument profile is CASHCAT. The archiver selects any symbol whose tape spans more than 7
days, which cleanly separates the long-retention collector from the short-retention
controls without this repo reading another
project's compose file. A symbol that qualifies on tape length but has no
instrument profile is skipped rather than archived with CASHCAT's tick size and
inventory base — confident numbers for the wrong asset are worse than none.

## The one manual step

`archive_period.py` writes here but **does not commit**, so nothing pushes on
its own. It logs an uncommitted-history reminder; use `git status` to see the
exact paths.

It also no longer runs itself the way it did. The archiver used to be a compose
service with a restart policy; replay is now a host command, so the cadence is a
Windows scheduled task, **`MM CASHCAT period archive`**, which fires
`archive_period.py` daily at 04:30 and appends to
`rust_live/reports/archive_period.log`. The script decides due-ness itself, so a
daily run is a no-op on 20 days out of 21 and a missed day costs nothing.

```powershell
Get-ScheduledTaskInfo -TaskName "MM CASHCAT period archive"   # LastTaskResult 0 is healthy
```

Nothing else notices if that task is removed. The first sign would be a window
that rolled off unarchived, which is exactly the failure this directory exists
to prevent, so check the task or `git log docs/history` if you are unsure the
last cycle ran.

```
git add docs/history && git commit && git push
```
