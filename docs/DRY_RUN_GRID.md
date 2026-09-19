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

All dry-run and backtest execution is Rust. Build and validate in Docker; use a
separate image tag and report directory so an experiment cannot replace the
running grid's binary or reports.

```sh
mm-live --config /opt/mm/config/cashcat_dryrun_realistic.toml replay \
  --grid /opt/mm/config/grid_cashcat.toml --all-variants \
  --from 2026-09-13T00:00:00Z --to 2026-09-14T00:00:00Z \
  --scoring-from 2026-09-13T06:00:00Z \
  --inventory-unit sweep1_flat300=636 \
  --board /opt/mm/reports/experiment/leaderboard.json
```

`--variant` and `--inventory-unit variant=positive_units` are repeatable.
Without an explicit size, the training prefix sizes each variant automatically.
Without `--scoring-from`, `--train-fraction` sets the scoring boundary.
Training excludes observations exchanged or received at/after that boundary.
Calibration and VPIN scale are shared; frozen parameter profiles skip fitting.
An omitted `--from` / `--to` uses the configured calibration window.

`--initial-state historical-grid_state.json` restores the accounts, sizes,
daily losses and stopped variants using the grid restart path; orders, pending
operations and guards start cold. Future, incompatible or size-conflicting
checkpoints are rejected. An omitted checkpoint means **flat, incomplete historical
state**, not a reconstruction of the grid account. A restart gap beyond
`--max-carry-inventory-gap-seconds` (default 900) closes carried inventory at
its saved mark with estimated costs, as in the grid.

Replay revision **receipt-flow-v1** retains exchange time for existing model and
execution calculations, but dispatches events in collector receipt order.
Equal receipt times use trades, L2 books, direct BBOs, then each stream's
exchange-sorted order (stable file/row order for remaining ties).
Each accepted L2 book emits book then its derived BBO using the connector's
shared touch validation. Missing receipt times fall back to exchange time and
are counted. Collector receipt time is not the grid process's receipt time.

Fresh direct or L2-derived BBOs keep the market available; transactions alone
do not. A virtual clock runs the grid's periodic freshness checks even through
silence. Shared pause actions withdraw simulated orders and pending exits.
Resumption requires a fresh BBO; an interruption over the carry limit uses the
same last-mark close. These are **inferred data gaps**, not reconstructed network
disconnects. Timer phase, process outages, warm guards and unrecorded messages
cannot be recovered from an independent collector.

Configured latency remains authoritative: CASHCAT grid uses **150/150/150 ms**,
tail multiplier **2.35**, one slow second in twenty. Repeatable `--latency-ms`
overrides all three for sensitivity experiments; `--against-live` does not
replace them. It supplies default bounds/variants and compares changes in
`equity_history.csv` at the scored bounds, with boundary sample ages disclosed.
Never compare a lifetime grid profit to a replay subinterval. Complete matching
account state and feeds are still required for a strict fidelity comparison.

`--board` writes interval gains, fees, funding, fills and drawdown; session
reports retain the final account and an explicit initial-account baseline.
`replay.execution` records sizes, initialization, latencies, receipt fallbacks,
rejected touches, pauses and gap closes. WebSocket `feed_health` stays
unmeasured. The optional comparison report gives the first observable difference
in a bounded prefix of retained quote/fill traces; absent or rotated-away traces
are disclosed. A profit difference is not automatically a latency effect.

The historical execution boundary remains **2026-09-12 20:26:31 UTC**, when
`causal-v5` reached the grid (later commit `142ae3b`). Replaying earlier data
with v5 is counterfactual. `receipt-flow-v1` changes replay event handling,
not that past deployment. Timed exits still require cancellation/reconciliation,
latency and fresh bounded depth; their labels do not promise exact fill times.
Frozen fits are only out-of-sample after their original fitting/selection date.
These simulations do not establish historical venue fills.

See [the 22-variant validation](REPLAY_FIDELITY_20260919.md) for measured agreement and remaining differences.

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
| `sweep1_flat300` | Same 60 bps floor with the historical flat300 trigger |
| `sweep1_flat550` | Same 60 bps floor with the historical flat550 trigger |
| `contender_flat300` | Saved control fit, phi*kappa*T=300, q max=6, 60 bps floor, historical flat300 trigger |
| `contender_flat550` | Same fixed control fit and 60 bps floor, 550 ms exit target |

The four `parameter_profiles` store six fitted CJ parameters each, frozen from
the retired Python sweep and now recorded nowhere else, not a new fit with
similar settings. Finalist A/B/C
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
Timestep, Newton tolerance and the convergence study behind them: `CAUSAL_EXECUTION_REVIEW.md`.

`eligible_for_promotion` is false for rows with a lot-age exit or a fixed
parameter profile: this remains an evidence flag, not a verdict or a rank.
There is no automatic promotion. The live configuration can explicitly reference
`sweep1_flat300` to share its frozen strategy and proportionally scale its saved
sizing unit; venue constraints and execution differ. See `CASHCAT_LIVE_PROMOTION.md`.

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

Replay and the grid share this simulator, so `--against-live` (see "Offline
comparison") should reproduce a live row up to feed outages the tape did not
see -- and the unknown queue above is the first reason it may not.

## Accounting and validity

The runtime ranks `leaderboard.json` by promotion P&L: remaining inventory
valued at the executable side with the configured exit fee and slippage
(currently 3.5 and 25 bps). It sorted every `eligible_for_promotion` row above
every ineligible one until 2026-09-10, because `promote-best` read `rows[0]`
and needed it promotable; since promotability is false for every flatten
variant, the board opened with whichever promotable row existed rather than the
best one. Net marked P&L, cash/realized results, inventory,
fees, funding, drawdown and fills remain separate diagnostics.
`show_grid_leaderboard.py` shows both and sorts by promotion P&L by default, so
its order is the board's own; `--sort net-pnl` restores the previous ranking.
It sorted by net until 2026-09-10 while never displaying promotion P&L at all,
which left the ranking metric invisible in the only tool that reads the board.
Positive marked P&L with open inventory is not liquidated profit, and fill
counts alone do not establish an edge. A row that is not
`scientifically_valid` carries its `invalid_reason` -- all three invalid rows of
the 87 h run breached the liquidation buffer, which is a different claim about
the parameters than a merely bad number.

Quotes use only the consumed book and a nondecreasing decision clock. Trades
cannot fill orders that were not active at the trade's exchange timestamp.
Post-only acceptance is checked at activation; the queue ahead (see "Queue
model") must be consumed before a print reaches us, and cancellation latency
leaves orders exposed until cancellation arrives.

- Nothing about the feed invalidates a run. Gaps, downtime and event loss are
  counted and shown in `feed_health`, not judged. Quoting is withdrawn within
  `runtime.market_stale_ms` and resumes on the first fresh BBO, and inventory
  held through a gap longer than `--max-carry-inventory-gap-seconds` is closed
  at its last mark with promotion exit costs, the same rule a resume applies.
  An outage shortens the measurement; it does not corrupt it.
- A losing streak never stops a row; `consecutive_losses` is a diagnostic only.
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

Startup metadata requests, connections and writes have timeouts. Fresh L2
snapshots refresh the touch even when the change-only BBO channel is quiet;
heartbeats alone cannot keep stale quotes alive. Recovery uses bounded backoff.
An unexpectedly terminated feed task fails the process visibly so Docker can
restart it. Pause/resume and terminal execution-risk stops are logged explicitly;
a recoverable data pause never releases a terminal risk halt.

Scientific validity is separate from availability. A gap exceeding the research
limits continues to disqualify that run even after quoting resumes. The health
command reports both, plus working orders and valid-row count; it does not restart
a functioning trader merely to erase an unfavorable validity flag. The
leaderboard's `quote_pause_reason` identifies a current data pause.

On Windows, Docker Desktop must start at sign-in and its Windows Startup entry
must be enabled. The grid container's `restart: unless-stopped` policy recovers
process exits and daemon restarts; a deliberate stop remains stopped.
The host uses automatic sign-in so Docker Desktop starts unattended (verified: a
reboot cost 127 s). A host without it is not an unattended trading host. It must
also run the WSL 2 engine, not Docker VMM — `memory/dry-run-operation.md`.

### Checkpoint recovery

Schema-3 checkpoints contain every variant's accounting, diagnostics, daily risk
and last observed BBO. **A config change continues the run.** A retuned row
keeps its history and its `config_changes` count goes up once. The roster must
match: new, removed or renamed rows are refused. Stopped rows stay frozen. The leaderboard
prints `[RECONFIGURED]` naming the rows that span more than one configuration.
**Normal startup is resume-only.** `--out-dir` must identify the existing history.
Both checkpoint generations are checked for compatibility and valid accounting;
if neither can be resumed, startup fails before calibration or feed access.
Missing or incompatible checkpoints never create a fresh run.
For an explicit reset, stop the paper service and prepare an empty output
directory. Run `dry-run-grid --initialize --grid <spec> --out-dir <empty-dir>`
with the paper configuration: it prepares all accounts using the existing sizing
and calibration code, saves `initial_state.json` in the new run directory, and
exits without subscribing to a market feed. It refuses any existing history.
Start the service normally afterward; its first resume is from this zero-account
checkpoint. The paper service uses `mm-live:grid-receipt-flow-v1`, independently
of the real-money image tag. Keep the initial checkpoint, actual startup fit and
UTC boundary when comparing subsequent backtests.
The specific v4-to-v5 upgrade preserves the schema-3 ledger while marking new
execution behavior prospectively. Other incompatible execution revisions fail.

`grid_state.json` is replaced atomically with one `.bak` generation. Decode
failures try the backup. Missing feed-health fields are errors, not an
assumed healthy history; neither an omitted loss flag nor a malformed variant
is silently converted into a resumable clean account.

| Checkpoint gap | Behavior |
|---|---|
| Up to 900 seconds | Restore accounting with no working orders or pending IOC; carried lots wait through exit scheduling and fresh depth |
| Over 900 seconds | Close valid carried inventory at the saved bid/ask with the taker fee and promotion haircut before feed startup, then resume |

`--max-carry-inventory-gap-seconds` sets the carry window; zero closes
inventory on any resume. There is no upper limit on the gap: a run resumes
after an outage of any length. Terminal invalid accounts remain frozen. The carry
window exists because resuming with a position intact marks it at a price whose
path was never observed: that mechanism let a 46.4 h run report a 13.2% rally
as profit (2026-08-27).

A gap close updates cash, realized P&L, fees and daily risk together, reducing
equity by spread/exit costs relative to the checkpoint mark. It is a scenario
boundary assumption, **not evidence that a trade executed during the outage**.
Short-gap inventory still experiences unobserved price risk; report the gap
rather than presenting the session as uninterrupted. Pending markouts are not
restored across it.

Process downtime is recorded in `resumed_downtime_ms`, excluded from both
feed-downtime budgets and active-time denominators. `resumes` and `[RESUMED]`
expose the interruption. Total unobserved time is feed downtime plus process
downtime. `run_started_ms` remains constant within the resumed run.

## Reports and logs

The root leaderboard and checkpoint describe the active run. Its artifacts live
under `rust_live/reports/grid_live/runs/<run_id>/`: per-variant reports, logs,
`equity_history.csv` and a copy of its latest checkpoint; a verified copy can
restore the root checkpoint without replacing the history. The
history records one row per variant every
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

`scripts/archive_period.py` writes a full replay of every grid variant, the
leaderboard and the period P&L under `docs/history/` every 21 days, ahead of the
collector's retention (`CASHCAT_RETENTION_MINUTES` in
`HYPERLIQUID_DATA/docker-compose.yml`). Cadence, the scheduled task it now needs, failure
handling and the manual commit step: `history/README.md`.
