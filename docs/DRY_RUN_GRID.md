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

`mm-live replay` scores grid rows over a tape window instead of a live feed. It
is a command, not a service, so it runs natively; only the collectors and the
grid need containers, because only they run continuously.

```sh
cargo build --release                     # in rust_live/

mm-live --config rust_live/config/cashcat_dryrun_realistic.toml replay \
  --grid rust_live/config/grid_cashcat.toml --all-variants \
  --from 2026-08-30T14:24:06Z --to 2026-09-05T11:09:28Z \
  --train-fraction 0.05 --latency-ms 150 \
  --board replay_leaderboard.json
```

`--variant` is repeatable and `--all-variants` takes the whole spec. `--from` /
`--to` select the tape range (RFC 3339 or epoch ms); without them the config's
`calibration.window_minutes` ending at the newest shard is replayed.
`--latency-ms` overrides all three dry-run latencies and is itself repeatable,
one rung of a ladder each. `--against-live <leaderboard.json>` replaces all
three: it takes the window, the latency and -- absent `--variant` -- the
variant list from a live board, and prints each live row beside its replay.
That is the fidelity check between replay and dry run.

**The board is the point.** `--board` writes the live `leaderboard.json` schema,
because a replay row and a grid row both come from
`PaperVariant::leaderboard_row` — same code, same accounting — so
`show_grid_leaderboard.py` renders either and rows compare field for field. A
replay board carries a `replay` key and a live board does not; that is the only
thing distinguishing two deliberately identical shapes, so check it first.

Tape and calibration are loaded once and shared: `[calibration]` is not
overridable, so per-variant refits would be byte-identical work. Calibration
(unless a frozen `parameter_profile` is selected), VPIN volume scale and order
sizing use only the training prefix; replay neither loads nor updates the
calibration cache. Each variant starts flat with cold guards and no orders, and
stops on terminal risk invalidation. This is a controlled model comparison, not
evidence that historical venue fills would match paper fills.

### Fidelity limits, before reading any replay-vs-live table

- **Exit names retain their historical labels, not guaranteed fill times.**
  Execution v5 waits for fill notification, the configured FIFO age, effective
  cancellation, acknowledgement and reconciliation before sending an IOC. Each
  delay uses the configured latency tails. Maker placements are suppressed during
  the exit; orders can still fill before their cancellation becomes effective.
  A new depth snapshot at/after arrival supplies actual prices and bounded size,
  within the sent 25/100/configured-maximum-bps IOC limit. Residual inventory remains
  exposed through the live ten-poll retry cadence. Book events use their own touch;
  prints, BBOs, duplicate and older snapshots cannot manufacture taker liquidity.
- **Frozen profiles bypass prefix calibration.** The prefix still sizes replay
  orders; it does not make a frozen fit out-of-sample. Scoring before that fit's
  original fitting/selection date is in-sample; overlapping replays are dependent.
- **A live row may be stitched, a replay never is.** Live rows cross `resumes`,
  `resumed_downtime_ms` and checkpoint restores; a replay is one continuous
  pass. A window with heavy downtime is not a fidelity measurement.
- **Carried inventory is priced differently.** Gap-carry valuation retains its
  25-bps haircut; timed exits consume visible depth. Both use `flatten_fee_rate`.
  The old fixed-walk and separate promotion-fee settings are compatibility inputs.
- **`feed_health` on a replay board is not a measurement.** A replay consumes a
  tape slice and cannot observe a gap inside it, so those counters read zero
  meaning "not measured". `calibration` is likewise null for `sweep1_*` and
  `contender_*`, which skip the fit entirely.

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
parameter profile: they have no live equivalent as written. That is a property
of the row, not a verdict and not a rank -- there is no automatic promotion, and the shipped
live config in fact runs a timed exit (`live.flatten_after_ms`) by choice. Their
exit deadlines include the round trip; see the fidelity limits above.

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
**The grid is resume-only.** `--out-dir` must identify the existing history.
Both checkpoint generations are checked for compatibility and valid accounting;
if neither can be resumed, startup fails before calibration or feed access.
Missing or incompatible checkpoints never create a fresh run.
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
