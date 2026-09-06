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

The second service in that file is **`mm-archiver`**: every 21 days it writes a
full sweep plus the grid's P&L curve under `docs/history/<date>_<SYMBOL>/`, and
**does not commit** — `docs/history/README.md` has the why, the failure handling
and the manual `git add docs/history` step.

Containerised after the 2026-08-27 reboot loss (66 h unnoticed); the compose
header explains the three mechanisms (`restart`, `stop_signal`, healthcheck +
autoheal). Registered in `../folder_list.txt` so `start_all.bat` brings it up.

**A restart continues the run, it does not start a new one.** The grid
checkpoints every configured variant to `<out-dir>/grid_state.json` every stats
tick, and on startup resumes from it: equity, inventory, fills, drawdown,
markouts and the elapsed clock all carry forward, so a reboot costs a gap rather
than the run.

Three things bound that, and they are the point rather than an afterthought:

- **The interruption is NOT counted as feed downtime.** The 5%
  `max_feed_downtime_fraction` budget measures blindness *while quoting* (stale
  resting orders, fills never seen), and a stopped process has none of that. It
  lives in `resumed_downtime_ms` instead, is subtracted from the budget's
  denominator, and the rendered table prints `[RESUMED]`.
- **The carry (900 s) and resume (3600 s) windows**, and why they exist:
  `docs/DRY_RUN_GRID.md` "Checkpoint recovery".
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

## Collectors — separate, and must stay that way

Not operated from this repo. `hl-cashcat-collector` (CASHCAT alone, 30-day
retention) and `hl-collector` (ETH, ACE, CHIP, PENGU, NIL, 3 days) live in
`HYPERLIQUID_DATA/docker-compose.yml` and write to the shared `./data/eth_mm`
tree, reachable here as the `scripts/HL_data` junction.

**Their `SYMBOLS` lists must stay disjoint** or every trade lands on disk twice
(2026-08-16). Trade-id dedup now protects calibration, and the expected
replay-backfill drop rate is in `docs/DATA_COLLECTION.md`.

**`scripts/` is their Docker build context** (`hyperliquid_data_collector.py`,
`run_collector.py`, `Dockerfile` — do not move or rename them). After touching
it rebuild both collectors: `scripts/README.md` "Updating".

Collector websocket expiry (a ~3 h close frame the SDK never recovers from) was
fixed 2026-08-19 (945648e): `scripts/README.md` "Websocket expiry".

## Gotchas that bite

**Docker Desktop must run on the WSL 2 engine.** The 4.89 upgrade (2026-09-05)
silently switched the backend to Docker VMM, whose virtiofs sharing wedged under
the fleet's load: every bind mount stalled and `docker stop`/`exec` hung. Fix:
`settings-store.json` `UseLibkrun=false`, `WslEngineEnabled=true`, then
`docker desktop restart`. Keep bind mounts, never named volumes for reports.

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

**Disk is now bounded, and the bound is on the event logs.** A pre-ALO 46.4 h
run measured 0.31 MB/h per variant compressed, about 5 GB/month for the
current 22-row spec; the rate varies with event mix. `--log-max-mb` (64) and
`--log-keep` (3) cap each variant at 256 MB, so the current ceiling is **about
5.5 GiB total** (~34 days at that measured rate). A
rotation logs what it deleted; it never drops a generation silently.

**The checkpoint does not grow with run duration.** `grid_state.json` and its
`.bak` are overwritten in place. Size depends on the configured variants and
diagnostic buckets (about 170 KB per generation for the current spec).

**`equity_history.csv` is the one thing still unbounded**, roughly 0.1 GB/month
at the current grid size. It lives under
`<out-dir>/runs/<grid_state.run_id>/`, alongside that run's event logs; the root
leaderboard remains only the latest pointer. This history is the primary P&L
curve and is intentionally retained.

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
