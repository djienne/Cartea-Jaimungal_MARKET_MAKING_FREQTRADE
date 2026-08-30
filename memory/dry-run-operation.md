---
name: dry-run-operation
description: How to run the Rust market-making dry run and grid, and the runtime gotchas that bite
metadata:
  type: project
---

**Rewritten 2026-08-25.** This file described the freqtrade stack — two legs on
separate sub-accounts, a param-estimator publishing to Redis, and a docker
compose project. That trader is retired (tag `freqtrade-trader-final`); its
containers are stopped and removed. What follows is the stack that actually
runs. The freqtrade-era gotchas that died with it — the `param_update.lock`
abandoned-lock recovery, the ccxt 4.5.22 pin and its fee invariant, the
gate-battery profiles, `docker compose up -d mm-long mm-short` — are gone with
the code; look them up at the tag if a question about the old bot ever comes up.

## What runs

The trader is `rust_live/`. **The grid is a compose service** as of 2026-08-30
(`docker-compose.yml` at the repo root, container `mm-grid-dryrun`); the other
modes are still run by hand.

```
docker compose up -d                 # the grid, and the only thing this file starts
docker compose logs -f               # watch it
docker compose stop                  # SIGINT, 60 s grace, teardown runs

mm-live --config config/cashcat_dryrun_realistic.toml dry-run
mm-live --config config/cashcat.toml live          # real money, explicit, gated
```

Why it is containerized: on 2026-08-27 a Windows-update reboot at 13:47 killed
the bare-process grid and it stayed down, unnoticed, for 66 h. `restart:
unless-stopped` brings it back when Docker starts (Docker Desktop must be set to
start on login), `stop_signal: SIGINT` reaches the teardown path, and the
`HEALTHCHECK` plus the existing `autoheal` container catch the case
`restart:` cannot see — a process that is alive but blind. It is registered in
`../folder_list.txt` so `start_all.bat` brings it up with the rest of the fleet.

**Container paths are load-bearing.** Configs set
`data_dir = "../../scripts/HL_data"`, resolved relative to the *config file*,
not the CWD. So configs at `/opt/mm/config` require the tape at
`/opt/scripts/HL_data`. Move one without the other and calibration dies with
"no usable price data under ...". The grid calibrates from that tape at startup,
so the mount is required, not optional.

The grid opens **one** WebSocket no matter how many variants it runs. That is
deliberate: Hyperliquid allows ten per IP and the budget is shared with the
collectors and any live session. Bridge networking NATs through the host, so
containerizing does not change that count. The grid never writes Parquet and
never reads credentials.

## Collectors — separate, and must stay that way

Not operated from this repo. `hl-cashcat-collector` (CASHCAT alone, 30-day
retention) and `hl-collector` (ETH, ACE, CHIP, PENGU, NIL, 3 days) live in
`HYPERLIQUID_DATA/docker-compose.yml` and write to the shared `./data/eth_mm`
tree, reachable here as the `scripts/HL_data` junction.

**Their `SYMBOLS` lists must stay disjoint** or every trade lands on disk twice:
`estimator_common._load_parquet_dir()` concatenates every `*.parquet` in a
directory blindly. On 2026-08-16 two collectors shared `scripts/HL_data` and
doubled `n_trades` and λ± (2104 rows / 1055 unique trade_ids).

That specific consequence is now corrected downstream, not upstream:
`normalize_trades()` drops repeated `trade_id`s and reports the count as
`MarketWindow.meta["duplicate_trade_ids_dropped"]`, and `cj-data` does the same
in Rust. So the rule still holds — it wastes disk and the non-trade streams
rely on collapsing by `ts_ms` — but a duplicate no longer silently biases a
calibration.

Expect a *small* non-zero drop count as normal: the venue replays trades after
each reconnect and the collector has no suppression for it, so it appends the
same trade with a new receive timestamp. Measured 2026-08-25 over 211 h:
**1,123 of 752,532 rows, 0.149%**, all identical in price, size, side and
exchange timestamp. A drop count near that scale is backfill; a drop count near
half the rows is two collectors.

**`scripts/` is their Docker build context.** `HYPERLIQUID_DATA/docker-compose.yml`
builds both with `context: ../Cartea-Jaimungal_MARKET_MAKING_FREQTRADE/scripts`,
`dockerfile: Dockerfile`, and the image copies exactly `hyperliquid_data_collector.py`
and `run_collector.py`. Do not move or rename those three paths. The failure is
silent — running containers keep going and only break at the next rebuild — so
after touching `scripts/`, run
`docker compose -f ../HYPERLIQUID_DATA/docker-compose.yml build hl-cashcat-collector`.

Collector websocket gotcha (fixed 2026-08-19, 945648e): Hyperliquid expires a
websocket session about every 3 hours and sends a close frame; the SDK logs it,
its manager thread exits, and nothing in the SDK reconnects. Recovery used to be
the time-based `_watchdog_inactivity`, so every routine expiry cost a full
`INACTIVITY_TIMEOUT_SEC` (180 s) of missing data — over 60.3 h of CASHCAT that
was 20 gaps of 3.1–3.5 min on a clockwork ~3 h cadence, 71% of all missing data.
`_websocket_is_down()` now reads the SDK's own state and the watchdog acts within
~10 s.

## Gotchas that bite

**Rebuilding while the grid runs used to fail** with `Access is denied. (os
error 5)`, because the running `mm-live.exe` held the binary. Containerizing
removed that: the build happens in the image, so `docker compose build` never
contends with a live run. Only the bare-process modes still have the problem.

**Logs come from Docker now**, not a shell redirect — `docker compose logs -f`,
capped at 3 × 10 MB. The old `... > reports/grid_live/run.log` pattern also only
redirected *stdout*; panics go to stderr and would have been lost.

**Feed health is logged on change, not on a timer.** A `grid feed health` line
appears when `reconnects`, `feed_gaps` or `replayed_trades_ignored` moves, plus
a floor of one line per ten minutes. So `trade_prints` in that line is stale by
design; it is a snapshot from the last time a health counter moved, not a live
count.

**An outage that is still open used to be invisible in that line** — those three
counters are only written when a gap *closes*. On 2026-08-26 the feed was down
for 19.65 h and the line printed 117 byte-identical copies of
`reconnects=28 feed_gaps=27 feed_downtime_ms=282835`; the only tell was
`trade_prints` frozen, which you had to diff consecutive lines to see. Fixed
2026-08-30: the line now carries `feed_down_for_ms`, switches to `WARN` and a
60 s floor while down, and fires the moment the feed drops. So a
`grid feed health` at `INFO` genuinely means the feed is up.

**A leaderboard can now disqualify itself.** `feed_failures` and
`feed_down_for_ms` are recomputed into `leaderboard.json` on every write, with
open gaps counted, and ANDed into each row's `scientifically_valid`. Before, the
verdict only existed in the teardown, so a run killed mid-flight left 18 rows
claiming validity after 42.5% downtime. Check `feed_failures` before quoting any
number out of that file.

**Check a run is alive with `mm-live grid-health`**, not by looking for a
process — that is what the container healthcheck runs. Liveness alone is not
enough: through the 19.65 h blackout the process was healthy and rewriting the
leaderboard every five seconds. `grid-health` fails on a stale
`generated_at_ms` *or* a long `feed_down_for_ms`.

**A `tail -f` on `run.log` locks the run directory.** Windows will not rename
or move a directory while any file inside it has an open handle, so a monitor
tailing `reports/grid_live/run.log` makes `mv reports/grid_live reports/archive_x`
fail with *Permission denied* — with no hint that your own monitor is the cause.
Stop the tail first. If the rename still fails, `rm -f` the directory's contents
(that works even while the handle is held) and then `rmdir` it.

**Replayed trades are expected, not a fault.** The venue replays history on
every subscribe — measured 27 prints on connect against 13 live ones, then a
further ~30 on each reconnect. `replayed_trades_ignored` rising while
`feed_gaps` stays flat is the system working.

**Disk.** ~17 MB/h at 18 variants, so ~12 GB/month, with zstd holding 16×.
`quote_decision` events are **99.1%** of the bytes (157k events / 79.7 MB against
2.9k execution events / 0.7 MB), so sampling them is the only lever that matters
if that ever needs to come down.

**`equity_history.csv` is append-only across restarts** and stamps
`run_started_ms`, so a P&L curve survives a relaunch — split on that column
rather than assuming one run per file.

## Calibration

All-Rust now: `mm-live calibrate` solves κ/λ/ε and the HJB surface over Parquet
history. The Python estimators (`estimate_all.py`, `get_{kappa,lambda,epsilon}.py`)
are kept for replay and analysis, not for feeding a live bot — nothing consumes
their snapshots any more. See [[cartea-jaimungal-phi-kappa-trap]] before
touching φ, and [[hjb-alpha-untuned-after-episodic]] for α.
