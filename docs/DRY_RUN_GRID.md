# Dry-run grid

The grid compares 22 independent paper accounts on one public market feed,
over **one** WebSocket regardless of variant count (the venue allows ten per
IP and that budget is shared with the collectors and any live session). It does
not construct a live backend, read trading credentials, open an account socket
or write collector Parquet. It is not a latency benchmark or permission to
trade. Scientific results and limitations are in `CAUSAL_EXECUTION_REVIEW.md`;
this guide describes operation, not a history of parameter changes.

## Run and inspect

From the repository root, operate only the paper service:

```sh
docker compose up -d --no-deps mm-grid-dryrun
docker compose logs -f mm-grid-dryrun
docker compose stop -t 60 mm-grid-dryrun
```

For a bounded native session:

```sh
mm-live --config rust_live/config/cashcat_dryrun_realistic.toml \
  dry-run-grid --grid rust_live/config/grid_cashcat.toml \
  --out-dir rust_live/reports/grid_live --duration-seconds 600
```

`--duration-seconds 0` runs until shutdown. Check feed and output freshness
separately from trading performance:

```sh
python scripts/validate_hl_data.py --symbol CASHCAT --newest-per-stream 25 --max-age-seconds 180 --fail-on-bad-data
docker exec mm-grid-dryrun /usr/local/bin/mm-live grid-health --leaderboard /opt/mm/reports/grid_live/leaderboard.json --max-age-seconds 30 --max-feed-down-seconds 120
python scripts/show_grid_leaderboard.py --markdown
python scripts/grid_pnl_curve.py
```

## Offline comparison

The same Rust paper step can score one existing grid row:

```sh
mm-live --config rust_live/config/cashcat_dryrun_realistic.toml replay \
  --grid rust_live/config/grid_cashcat.toml --variant sweep1 \
  --from 2026-08-30T14:24:06Z --to 2026-09-05T11:09:28Z \
  --train-fraction 0.25 --latency-ms 150 --report sweep1.json
```

`--from/--to` select the tape range (RFC 3339 or epoch ms); without them the
config's `calibration.window_minutes` ending at the newest shard is replayed.
`--latency-ms` overrides all three dry-run latencies. `scripts/replay_latency.py`
wraps this for latency ladders and the `--against-live` fidelity check (see
"Queue model" below). Use separate report paths per variant. Calibration (unless
a saved profile is selected), VPIN volume scale and order sizing use only the
training prefix; replay neither loads nor updates the calibration cache. Scoring
starts flat with cold flow guards and no orders, and stops on terminal risk
invalidation. Reports identify both windows, the consumed time source, fitted
parameters and the last scored event. This is a controlled model comparison,
not evidence that historical venue fills would match paper fills.

## Experimental controls

`rust_live/config/grid_cashcat.toml` defines the variants;
`rust_live/config/cashcat_dryrun_realistic.toml` supplies their common settings.
The 22 slots contain 13 recent-fit controls, three training-ranked sweep
finalists, four targeted combinations and two fixed-fit flatten contenders:

| Row | Model / change |
|---|---|
| `sweep1` | Saved fit A, phi*kappa*T=3000, T=150 s, q max=3 |
| `sweep2` | Saved fit B, phi*kappa*T=3000, T=150 s, q max=6 |
| `sweep3` | Saved fit C, phi*kappa*T=3000, T=150 s, q max=6 |
| `sweep1_unguarded` | First finalist without the flow guard |
| `sweep1_wide60` | First finalist with a 60 bps half-spread floor |
| `sweep1_flat300` | Same 60 bps floor with the 301 ms exit target |
| `sweep1_flat550` | Same 60 bps floor with the 550 ms exit target |
| `contender_flat300` | Saved control fit, phi*kappa*T=300, q max=6, 60 bps floor, 301 ms exit target |
| `contender_flat550` | Same fixed control fit and 60 bps floor, 550 ms exit target |

The four `parameter_profiles` store the six fitted CJ parameters from
`cashcat_sweep.json`, not a new fit with similar settings. Finalist A/B/C
arrival-jump horizons are 1000/1000, 1000/500 and 500/1000 ms; all use the
upper-quartile depth support. The contenders use the 200/200 ms, full-support
training fit, rather than the recent-data fit of `flatten300` and `flatten550`.
They participate in checkpoint identity and appear in each startup log.
Controls use the shared recent-data fit, calculated once at grid startup.
Duplicate effective configurations warn; raise `phi_kappa_t_max` with
`phi_kappa_t` when exceeding the base ceiling.

Common assumptions are 297.88 USDC starting capital, 1.5 bps maker fees,
0.0000125/hour funding, and 150 ms each for decision, acknowledgement and
cancellation. A 2.35 multiplier applies during one exchange-time second in
every 20-second cycle. These configured assumptions are not continuously
measured execution capabilities. Queue position follows the model in "Queue
model" below; twenty recorded levels cannot reconstruct the venue's queue past
them.

All rows retain common paper execution settings, capital-derived order sizing,
risk gates and the current-data VPIN volume scale. The finalists therefore test
the saved models prospectively under the same paper conditions as the controls;
they do not reproduce the Python sweep's fixed sizing or execution assumptions.
The internal HJB timestep is 1/512 s; each current 150 s candidate uses 76,800
steps within the 153,600-step ceiling. Across the three finalists and shared
contender quote model, half-inventory states, prices 0.05--0.30 USDC and sampled
times including the final five seconds, executable quotes change by at most
one venue price increment under both further halvings.
Newton uses a 1e-10 residual tolerance and a 100-update budget. These
numerical settings do not change event-driven quoting or simulated latency.

Lot-age exits and fixed parameter profiles are not supported by live promotion.
Rows using either remain paper-only and ineligible regardless of P&L. Exit
deadlines include lot age plus decision/acknowledgement latency: `flatten300`,
`flatten300w40` and `sweep1_flat300` target 301 ms; `flatten550` and
`sweep1_flat550` target 550 ms, before event discretization. Combinations are
controlled comparisons, not presumed improvements.

## Queue model

The simulator tracks, per resting order, the volume ahead of it (`front`) and
the size of its price level as last reconciled (`seen`), in the style of
nkaz001/hftbacktest's probabilistic queue model:

- **Activation.** `front = seen =` the level's size in the latest `l2Book`
  snapshot (live BBO size at the touch is a fallback). A price strictly inside
  the spread has nobody ahead, `front = 0`. A price beyond the deepest recorded
  level is *unknown-queue*: it cannot fill on a print at its price, only on a
  print strictly beyond it, until a later snapshot shows the level.
- **Print at our price.** The print consumes `front` first; what is left fills
  us. Both `front` and `seen` drop by the print, so the next snapshot does not
  read the trade as a cancellation.
- **Snapshot.** If the level shrank by `chg` beyond what prints explain, a share
  `front^n / (front^n + back^n)` of `chg` is taken off `front`, where
  `back = seen - front`. Growth joins the queue behind us. `n` is
  `dry_run.queue_cancel_power` (shipped 2.0, an uncalibrated prior); `0`
  counts trades only, hftbacktest's risk-averse model. The removed volume is
  reported as `queue_cancel_units`.

Hyperliquid pushes `l2Book` at roughly one snapshot every 5 s for CASHCAT with
20 levels per side, which reach about 40 bps from mid. The harvester keeps every
frame, so this is the venue's limit, not the tape's: a quote at 60 bps half
spread is unknown-queue most of the time under any model, and
`unknown_queue_activations` in the reports says how often.

**Fidelity against the live dry run.** Replay and the grid share this
simulator, so replaying the grid's own window should reproduce its row up to
feed outages the tape did not see:

```
python scripts/replay_latency.py --variant sweep1_flat300 --against-live
```

prints the leaderboard row (net, fills, inventory, resumes, downtime) above a
replay of the same window at the configured latency. A window with heavy
downtime is not a fidelity measurement. The same script sweeps assumed
latencies over any tape range (`--from`, `--to`, `--latency ...`).

## Accounting and validity

The runtime ranks `leaderboard.json` by promotion P&L: remaining inventory
valued at the executable side with the configured exit fee and slippage
(currently 3.5 and 25 bps). Net marked P&L, cash/realized results, inventory,
fees, funding, drawdown and fills remain separate diagnostics.
`show_grid_leaderboard.py` sorts by net P&L by default; this does not change
runtime promotion ordering. Positive marked P&L with open inventory is not
liquidated profit, and fill counts alone do not establish an edge.

Quotes use only the consumed book and a nondecreasing decision clock. Trades
cannot fill orders that were not active at the trade's exchange timestamp.
Post-only acceptance is checked at activation; the queue ahead (see "Queue
model") must be consumed before a print reaches us, and cancellation latency
leaves orders exposed until cancellation arrives.

- Event loss is disqualifying. Feed gaps during execution are recorded separately
  and assessed against `runtime.max_feed_downtime_fraction` (default 5%) and
  `runtime.max_feed_gap_ms` (default 60 seconds).
- The consecutive-loss breaker is terminal. Its cap is 500 in the paper profile
  and 25 in the live profiles; zero would halt immediately. A latched account
  freezes, reports invalidity and cannot continue lot exits or reconciliation.
- The daily-loss gate is non-latching. Its daily accounting survives resume.
- A variant error invalidates that variant without aborting the other accounts.
  Execution-invalid rows have no promotion P&L; all scientifically invalid rows
  are ineligible for promotion.

## Outages and restarts

Missing, stale or disconnected BBO data pauses quoting and withdraws local paper
orders, including deferred replacements. The last mark and account/risk history
are retained; pending markouts spanning the gap are discarded. Trading resumes
on a fresh post-reconnect BBO without resetting cash, inventory or loss limits.
Paper withdrawal is a simulation boundary, not a claim that venue orders were
cancelled during an unobserved interval.

Startup metadata requests, connections and writes have timeouts; missing BBO
updates trigger resubscription
even if heartbeat frames still arrive. Recovery retries with bounded backoff.
An unexpectedly terminated feed task fails the process visibly so Docker can
restart it. Pause/resume and terminal execution-risk stops are logged explicitly;
a recoverable data pause never releases a terminal risk halt.

Scientific validity is separate from availability. A gap exceeding the research
limits continues to disqualify that run even after quoting resumes. The health
command reports both, plus working orders and valid-row count; it does not restart
a functioning trader merely to erase an unfavorable validity flag. The
leaderboard's `quote_pause_reason` identifies a current data pause.

On Windows, Docker Desktop must start at sign-in and its Windows Startup entry
must be enabled. The paper container's `restart: unless-stopped` policy recovers
process exits and daemon restarts; a deliberate stop remains stopped.
The host uses automatic sign-in so Docker Desktop starts unattended (verified: a
reboot cost 127 s). A host without it is not an unattended trading host.

Docker Desktop must run on the WSL 2 engine. The 4.89 upgrade (2026-09-05)
silently switched the backend to Docker VMM, whose virtiofs sharing wedged under
the fleet's load: every bind mount stalled and `docker stop`/`exec` hung. Fix:
`settings-store.json` `UseLibkrun=false`, `WslEngineEnabled=true`, then
`docker desktop restart`. Keep bind mounts, never named volumes for reports.

### Checkpoint recovery

Schema-3 checkpoints contain every variant's accounting, diagnostics, daily risk
and last observed BBO. Startup validates the complete variant set and execution/
configuration fingerprints before adopting the run ID, inventory sizing or output
paths. A rejected checkpoint starts a separate run; it cannot overwrite the
rejected run's artifacts. Incompatible execution models start fresh.

`grid_state.json` is replaced atomically with one `.bak` generation. Decode
failures warn and try the backup. Missing feed-health fields are errors, not an
assumed healthy history; neither an omitted loss flag nor a malformed variant
is silently converted into a resumable clean account.

| Checkpoint gap | Behavior |
|---|---|
| Up to 900 seconds | Restore accounting with no working orders; carry inventory, waiting for a fresh BBO before normal stale-lot exits |
| Over 900 seconds, up to 3600 | Close valid carried inventory at the checkpoint bid/ask with configured promotion fee/slippage before feed startup |
| Over 3600 seconds | Start a fresh run |

`--max-carry-inventory-gap-seconds` and `--max-resume-gap-seconds` control
these limits. Zero resume window disables resume; zero carry window closes
inventory on any resume. Terminal invalid accounts remain frozen. The carry
window exists because resuming with a position intact marks it at a price whose
path was never observed: that mechanism let a 46.4 h run report a 13.2% rally
as profit (2026-08-27).

A gap close updates cash, realized P&L, fees and daily risk together, reducing
equity by spread/exit costs relative to the checkpoint mark. It is a conservative
boundary valuation, **not evidence that a trade executed during the outage**.
Short-gap inventory still experiences unobserved price risk; report the gap
rather than presenting the session as uninterrupted. Pending markouts are not
restored across it.

Process downtime is recorded in `resumed_downtime_ms`, excluded from both
feed-downtime budgets and active-time denominators. `resumes` and `[RESUMED]`
expose the interruption. Total unobserved time is feed downtime plus process
downtime. `run_started_ms` remains constant within the resumed run.

## Reports and logs

The root leaderboard and checkpoint describe the active run. Its artifacts live
under `rust_live/reports/grid_live/runs/<run_id>/`: per-variant reports, logs
and `equity_history.csv`. The history records one row per variant every
`--history-seconds` (default 60; zero disables it), plus a final shutdown sample.
It retains run identity and mid-price, so plotting does not require retained tape.

Each bounded `grid-<variant>.jsonl.zst` log rotates at `--log-max-mb` (64)
and keeps `--log-keep` (3) generations. Restart rotates a nonempty current log
before opening a fresh frame. Size rolls close the frame; a restart may preserve
an interrupted old frame. The plotter reads retained files oldest-first and warns
on decode failure, without letting an old damaged generation hide the new one.
Rotation limits retained history by both volume and restart count; it does not
guarantee a fixed number of days. Flush-boundary rotation can overshoot the size
threshold. With 22 rows, the default current-plus-three-generation log budget
is about 5.5 GiB. Old run directories and unrotated
history are additional; this is not a hard total-disk ceiling.
`--log-max-mb 0` allows unbounded append with interrupted-frame risk.

Use a streaming zstd reader across frames, not one-shot decompression. The
`--from-fills` plotting fallback reconstructs cash and inventory from retained
logs but needs the price tape and cannot recover funding; it is not a substitute
for the recorded equity history. `equity_history.csv` is not rotated.

## The period archive — what outlives the tape

`scripts/archive_period.py` (`mm-archiver`) writes a full sweep, leaderboard and
period P&L under `docs/history/` every 21 days, ahead of the collector's 30-day
retention, and does not commit. Cadence, layout, failure handling and the manual
commit step: `history/README.md`. Collector ownership and data validation:
`DATA_COLLECTION.md`.
