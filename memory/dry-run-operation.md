---
name: dry-run-operation
description: How to run the Rust market-making dry run and grid, and the runtime gotchas that bite
metadata:
  type: project
---

**Current stack only.** Historical trader code and operating notes are available
at tag `freqtrade-trader-final`; they are intentionally omitted here.

## What runs

The trader is `rust_live/`. **The grid is a compose service** as of 2026-08-30
(`docker-compose.yml` at the repo root, container `mm-grid-dryrun`); the other
modes are still run by hand.

```
docker compose up -d                 # dry-run grid + period archiver; never live trading
docker compose logs -f               # watch it
docker compose stop                  # SIGINT, 60 s grace, teardown runs

mm-live --config config/cashcat_dryrun_realistic.toml dry-run
mm-live --config config/cashcat.toml live          # real money, explicit, gated
```

The second service in that file is **`mm-archiver`**. The CASHCAT collector keeps
30 days and deletes the rest, so a replay window is destroyed on a clock: once a
period rolls off, no sweep can ever score it again. Every 21 days the archiver
attempts a full-search sweep and writes it with the grid's P&L curve under
`docs/history/<date>_<SYMBOL>/`. The 9-day margin is retry time, not a guarantee:
after `sweep_FAILED.log`, rerun with `--force` before the oldest shards expire.

It **writes but does not commit** — that would need an SSH key in a container —
and logs an uncommitted-history reminder on every wake. Confirm the exact paths
with `git status`, then run `git add docs/history && git commit && git push`.

Two things it gets right on purpose, both learned here the hard way: due-ness
comes from the newest directory on disk rather than a sleep timer (a timer
restarts its countdown on every reboot and could never fire), and symbols are
selected by tape length > 7 days rather than a hardcoded list, which separates
the 30-day collector from the 3-day one without reading another project's
compose file.

Why it is containerized: on 2026-08-27 a Windows-update reboot at 13:47 killed
the bare-process grid and it stayed down, unnoticed, for 66 h. `restart:
unless-stopped` brings it back when Docker starts (Docker Desktop must be set to
start on login), `stop_signal: SIGINT` reaches the teardown path, and the
`HEALTHCHECK` plus the existing `autoheal` container catch the case
`restart:` cannot see — a process that is alive but blind. It is registered in
`../folder_list.txt` so `start_all.bat` brings it up with the rest of the fleet.

**A restart continues the run, it does not start a new one.** The grid
checkpoints all 18 variants to `<out-dir>/grid_state.json` every stats tick, and
on startup resumes from it: equity, inventory, fills, drawdown, markouts and the
elapsed clock all carry forward, so a reboot costs a gap rather than the run.

Three things bound that, and they are the point rather than an afterthought:

- **The interruption is counted as feed downtime**, so it erodes the 5%
  `max_feed_downtime_fraction` budget like any other blindness. It is
  deliberately *not* added to `feed_longest_gap_ms` — that limit guards against
  a long hole while *quoting* (stale resting orders, fills never seen), and a
  restart has none of that, since the checkpoint restores a book with no working
  orders. `resumes` and `resumed_downtime_ms` in `leaderboard.json` say how much
  of the downtime was restarts, and the rendered table prints `[RESUMED]`.
- **Beyond `--max-resume-gap-seconds` (default 900) it starts fresh instead.**
  Resuming means marking held inventory at a price whose path was never
  observed; across a long gap that is exactly the mechanism that made the 46.4 h
  run report a 13.2% rally as profit.
- **An edited grid spec starts fresh.** The checkpoint carries a fingerprint of
  every variant's config, and resume is all-or-nothing — a partially-resumed
  grid would have rows that are not comparable, which is the one thing the grid
  exists to provide.

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

**Replayed trades are expected, not a fault.** The venue replays history on
every subscribe — measured 27 prints on connect against 13 live ones, then a
further ~30 on each reconnect. `replayed_trades_ignored` rising while
`feed_gaps` stays flat is the system working.

**Disk is now bounded, and the bound is on the event logs.** Measured over the
46.4 h run: 0.31 MB/h per variant compressed, 5.5 MB/h across eighteen — 3.9
GB/month, and since the grid resumes across restarts that stream had no natural
end. `--log-max-mb` (64) and `--log-keep` (3) roll each log at 64 MB and keep
three generations, so the ceiling is **4.5 GB total, whatever the run length**,
covering ~34 days of history. A rotation logs what it deleted; it never drops a
generation silently. `quote_decision` events are **99.1%** of the bytes (157k
events / 79.7 MB against 2.9k execution events / 0.7 MB), so sampling them is
the lever if the ceiling ever needs to come down further.

**The checkpoint does not grow.** `grid_state.json` is ~27 KB at eighteen
variants, overwritten in place each tick, plus one `.bak` generation — a
constant ~54 KB, not an accumulator. The only part that grows at all is
`fills_by_depth_bps`, one entry per 0.1 bps depth bucket ever filled at, which
tops out in the low thousands of entries.

**`equity_history.csv` is the one thing still unbounded**, at ~129 KB/h = 91
MB/month. Left that way deliberately: it is the P&L curve, the primary result
artifact, and it is two orders of magnitude smaller than the logs were.

**`equity_history.csv` is append-only across restarts** and stamps
`run_started_ms`. Since resume landed, a restart *keeps* the original
`run_started_ms`, so the curve is genuinely continuous rather than two runs in
one file — a change of that value now marks a real boundary (a refused resume:
gap too long, or an edited spec), which is exactly when you do want to split.

## Calibration

`mm-live calibrate` solves κ/λ/ε and the HJB surface over Parquet history. The
Python estimators (`estimate_all.py`, `get_{kappa,lambda,epsilon}.py`) are kept
for replay and independent analysis; the Rust trader does not consume their JSON
snapshots. See `../docs/UNITS.md` before changing φ or α.
