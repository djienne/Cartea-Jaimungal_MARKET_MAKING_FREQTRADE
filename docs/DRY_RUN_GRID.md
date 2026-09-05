# Dry-run grid

The grid compares 20 independent paper accounts on one public market feed.
It does not construct a live backend, read trading credentials, open an account
socket or write collector Parquet. It is not a latency benchmark or permission
to trade. Scientific results and limitations are in `CAUSAL_EXECUTION_REVIEW.md`;
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
mm-live --config path/to/replay.toml replay --train-fraction 0.25 \
  --grid rust_live/config/grid_cashcat.toml --variant sweep1 --report sweep1.json
```

Set the replay config's `storage.data_dir` to a frozen tape and
`calibration.window_minutes = 480` for two training hours and six scored hours,
with `storage.retention_minutes` at least 510 to satisfy config validation.
Use separate report paths for `sweep1`, `sweep2` and `sweep3`. Calibration (unless
a saved profile is selected), VPIN volume scale and order sizing use only the
training prefix; replay neither loads nor updates the calibration cache. Scoring
starts flat with cold flow guards and no orders, and stops on terminal risk
invalidation. Reports identify both windows, the consumed time source, fitted
parameters and the last scored event. This is a controlled model comparison,
not evidence that historical venue fills would match paper fills.

## Experimental controls

`rust_live/config/grid_cashcat.toml` defines the variants;
`rust_live/config/cashcat_dryrun_realistic.toml` supplies their common settings.
The 20 slots contain 13 controls, the three training-ranked sweep finalists,
and four targeted combinations of the first finalist:

| Row | Model / change |
|---|---|
| `sweep1` | Saved fit A, phi*kappa*T=3000, T=150 s, q max=3 |
| `sweep2` | Saved fit A, phi*kappa*T=3000, T=300 s, q max=3 |
| `sweep3` | Saved fit B, phi*kappa*T=3000, T=150 s, q max=6 |
| `sweep1_unguarded` | First finalist without the flow guard |
| `sweep1_wide60` | First finalist with a 60 bps half-spread floor |
| `sweep1_flat300` | Same 60 bps floor with the 301 ms exit target |
| `sweep1_flat550` | Same 60 bps floor with the 550 ms exit target |

The two `parameter_profiles` store the six fitted CJ parameters from
`cashcat_sweep_causal_20260904.json`, not a new fit with similar settings.
They participate in checkpoint identity and appear in each startup log.
Controls use the shared recent-data fit, calculated once at grid startup.
Duplicate effective configurations warn; raise `phi_kappa_t_max` with
`phi_kappa_t` when exceeding the base ceiling.

Common assumptions are 297.88 USDC starting capital, 1.5 bps maker fees,
0.0000125/hour funding, and 150 ms each for decision, acknowledgement and
cancellation. A 2.35 multiplier applies during one exchange-time second in
every 20-second cycle. These configured assumptions are not continuously
measured execution capabilities. The model uses visible queue information with no uncalibrated
time decay; finite market-depth data cannot reconstruct a venue's order queue.

All rows retain common paper execution settings, capital-derived order sizing,
risk gates and the current-data VPIN volume scale. The finalists therefore test
the saved models prospectively under the same paper conditions as the controls;
they do not reproduce the Python sweep's fixed sizing or execution assumptions.
The internal HJB timestep is 1/512 s, with a 153,600-step ceiling for the 300 s
horizon. Across the three finalists, half-inventory states, prices 0.05--0.30
USDC and sampled times including the final five seconds, executable quotes
change by at most one venue price increment under both further halvings.
Newton still uses a 1e-8 residual tolerance and a 100-update budget. These
numerical settings do not change event-driven quoting or simulated latency.

Lot-age exits and fixed parameter profiles are not supported by live promotion.
Rows using either remain paper-only and ineligible regardless of P&L. Exit
deadlines include lot age plus decision/acknowledgement latency: `flatten300`,
`flatten300w40` and `sweep1_flat300` target 301 ms; `flatten550` and
`sweep1_flat550` target 550 ms, before event discretization. Combinations are
controlled comparisons, not presumed improvements.

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
Post-only acceptance is checked at activation; queue volume precedes partial
fills, and cancellation latency leaves orders exposed until cancellation arrives.

- Event loss is disqualifying. Feed gaps during execution are recorded separately
  and assessed against `runtime.max_feed_downtime_fraction` (default 5%) and
  `runtime.max_feed_gap_ms` (default 60 seconds).
- The consecutive-loss breaker is terminal. Its cap is 500 in the paper profile
  and 25 in the live profiles; zero would halt immediately. A latched account
  freezes, reports invalidity and cannot continue lot exits or reconciliation.
- The daily-loss gate is non-latching. Its daily accounting survives resume.
- A variant error invalidates that variant without aborting the other accounts.
  Invalid rows have no promotion P&L and are ineligible for promotion.

## A restart continues the run

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
inventory on any resume. Terminal invalid accounts remain frozen.

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
threshold. With 20 rows, the default current-plus-three-generation log budget
is about 5 GiB, unchanged by this roster. Old run directories and unrotated
history are additional; this is not a hard total-disk ceiling.
`--log-max-mb 0` allows unbounded append with interrupted-frame risk.

Use a streaming zstd reader across frames, not one-shot decompression. The
`--from-fills` plotting fallback reconstructs cash and inventory from retained
logs but needs the price tape and cannot recover funding; it is not a substitute
for the recorded equity history. `equity_history.csv` is not rotated.

## The period archive — what outlives the tape

`scripts/archive_period.py` checks hourly and attempts a full-search archive
every 21 days in `mm-archiver`, ahead of the collector's 30-day CASHCAT retention.
It writes scores, leaderboard and period P&L artifacts under `docs/history/`,
not raw market data, and does not commit them. Sweep windows can overlap;
archived grid curves preserve run boundaries.

Inspect `sweep_FAILED.log` and rerun with `--force` before source data expires.
A completed summary does not make deleted raw tape reproducible. Review the
specific generated paths before manually staging an archive. Collector ownership
and data validation are documented in `DATA_COLLECTION.md`.
