# Dry-run grid — several parameter sets, one feed

```
mm-live --config rust_live/config/cashcat_dryrun_realistic.toml \
        dry-run-grid --grid rust_live/config/grid_cashcat.toml \
        --out-dir rust_live/reports/grid_live
```

Runs every variant in the grid spec against **one** shared public market feed,
simulating each independently, and rewrites a leaderboard ranked by net P&L
every stats interval. `--duration-seconds 0` (the default) runs until Ctrl-C.

## Why one process rather than N

Hyperliquid allows **10 simultaneous WebSocket connections per IP**. That budget
is already shared with three collector containers and with any live session,
which needs two. One `dry-run` process per variant would open one connection
each and could take down data collection or block real trading.

Measured on the running **20-variant** grid: **1 connection**, both collectors
still `(healthy)`. The count does not grow with the number of variants — that is
the whole point.

## What a variant may change

A deliberately narrow set, because only these are implicated by the evidence.
The 161.95 h staged sweep (`cashcat_sweep.md`) localised the entire loss to one
six-hour burst: 26 of 27 windows sum to **+35.28**, and `08-22 03:57` alone is
**−241.17** on 1,771 fills. So the question a variant should answer is *how much
of a burst does this take*.

| key | maps to | evidence |
|---|---|---|
| `q_max` | `model.q_max` | replay: q=3 scored −279.11 against q=6's −585.61. **Live reversed this** — see below |
| `phi_kappa_t` | `model.phi_kappa_t` | sweep winner used 300 against a grid topping out at 1000 |
| `min_half_spread_bps` | `quoting.min_half_spread_bps` | the losing window earned +683.89 across 1,771 fills — ~0.39/fill, under its adverse selection |
| `min_order_lifetime_ms`, `replace_threshold_bps` | `quoting.*` | the only positive latency-ladder rung was the 30 s-refresh one |

> **`min_order_lifetime_ms` was inert in the simulator until 2026-08-24.**
> Only the live backend honored it, so every `slow*` variant was a byte-identical
> duplicate of its non-slow twin and every replay ever run ignored the lever.
> Fixing it moved the 185 h spread ladder by −58.73 to +29.84 per rung, so
> results recorded before that date are not comparable with ones after it
> (`spread_ladder_185h.md`). It also showed the lever is **non-monotone in
> width** — slowing helps at 24 bps, breaches the liquidation buffer at 8, and
> destroys the profitable rung at 40 — which retired the `wide8slow30s`,
> `wide8slow60s`, `slow30s` and `slow60s` variants.

**Feed gaps are measured, not latched.** `scientifically_valid` used to be a
one-way latch: any single blip marked the whole run invalid forever. Past about
three hours a run is guaranteed at least one venue connection recycle, so the
flag was false for every long run and carried no information — a 20 h grid with
nine 3-second reconnects reported exactly the same verdict as a broken one.

The four invalidation causes are not the same kind of event, and the fix is to
stop pretending they are:

| cause | kind | treatment |
|---|---|---|
| causal ring saturated | **event loss** — we processed the wrong sequence | always disqualifying |
| reconnect / dead session / late trade | **downtime** — a bounded, knowable hole | counted, judged against a threshold |

Every report now carries `feed_gaps`, `feed_downtime_ms` and
`feed_longest_gap_ms`, and the verdict comes from
`runtime.max_feed_downtime_fraction` (default 5% of wall time) and
`runtime.max_feed_gap_ms` (default 60 s) — one long gap disqualifies even when
the total is small, because a ten-minute hole is a different problem from sixty
blips. When a run does fail, `invalid_reasons` says which limit and by how much
instead of leaving a bare `false`.

**A failing variant no longer kills the grid.** `variant.step(...)?` used to
propagate, so one blown-up hypothesis aborted all the others — and with the
cadence lever working, a liquidation-buffer breach is a realistic way to blow
up. A variant that errors is now invalidated, stops trading, records the reason
in its report's `invalid_reasons`, and the run continues.
| `flow_guard_enabled`, `vpin_threshold`, `fast_move_threshold_bps` | `flow_guard.*` | the toxic-flow guard (`TOXIC_FLOW_GUARD.md`); the shipped spec pairs `guarded`/`unguarded` so the A/B runs on one shared live feed |
| `phi_kappa_t_max` | `model.phi_kappa_t_max` | `hjb.rs` rescales φ so φ·κ·T never exceeds this, so a `phi_kappa_t` above the base ceiling of 300 is silently clamped. Without this lever a variant asking for 1000 quietly runs 300 |

### What the first 20 h changed

The spec is evidence-led and gets re-cut when the evidence moves. After the
first 20 h run:

- **`slow30s` and `wide8slow30s` added.** The 162 h replay's latency ladder has
  exactly one profitable rung — 30 000 ms refresh, +9.36, against −279 to −418
  for every faster one — and the grid's slowest variant was 5 000 ms. The
  single most important result in the sweep was untested live.
- **`wide16`/`wide24` added.** Live spread results were strictly monotone with
  no turning point (baseline −34.06, wide4 −26.99, wide8 −20.00) while
  `max_half_spread_bps` allows 80. Stopping at 8 assumed an answer.
- **`q2` removed.** −141 with a drawdown of 145, ~49% of equity. Live also
  *reversed* the replay here: replay had q3 (−279) beating q6 (−585); live has
  baseline q6 −34.06, q3 −78.36. A smaller `q_max` means a larger per-bucket
  notional, so the "tighter" cap quotes coarser and bigger. `q3` stays only to
  keep watching the disagreement.
- **`defensive` removed.** It bundled `q2` with the good levers, so it could
  never show whether they compose, and it latched its risk limit — 0 working
  orders, 1,357 units frozen — so it stopped measuring a strategy at all.
  `wide8slow30s` replaces it as a clean combined corner.
- **The ladders were then filled out to 20 variants**, since compression and a
  shared socket make the marginal variant nearly free (still one connection;
  `policy.compute` at p99 0.02 ms x 20 x ~50 events/s is a few percent of a
  core). That adds `slow15s`/`slow60s` on the cadence ladder, `wide40` on the
  spread ladder, and a spread x cadence matrix (`wide16slow30s`,
  `wide24slow30s`, `wide8slow60s`) to show whether the two levers keep
  composing or one saturates the other.
- **`q9` and `q12` follow the live gradient upward.** Every `q_max` test so far
  went *down* from the base 6 and got steadily worse (q6 −34.06, q3 −78.36,
  q2 −141.25). The gradient points up and had never been followed: a larger cap
  means a *finer* per-bucket notional, so quotes get smaller and more granular —
  the opposite of what made q2 fail.

Every field is optional and inherits when absent, so a variant with no overrides
is literally the shipped configuration — which is what makes `baseline` an
honest control rather than a copy that can drift.

Each variant is validated independently at startup. An override can create a
combination the base never had (a hold window wider than the widest permitted
quote), and that must fail immediately rather than hours into a run.

## Realism — measured, not assumed

Held identical across every variant, so the comparison isolates the levers:

| setting | value | where it came from |
|---|---|---|
| decision / ack / cancel latency | 150 / 150 / 150 ms | `public_ws_ping_rtt` p50 = **281 ms** measured on this host, split across the round trip |
| `funding_rate_per_hour` | **0.0000125** | venue `metaAndAssetCtxs`, 2026-08-23. It was `0.0`, which silently omitted a real cost |
| `maker_fee_rate` | 0.00015 | venue `userFees.userAddRate` — already correct, confirmed |
| starting equity / capital | **297.88** | the real account value, not the 1000.0 placeholder |

p95 RTT is 661 ms, so 150/150 is the *typical* machine and not the bad-patch
one. The venue **action budget is deliberately not modelled** — it is a lifetime
account allowance (`live_canary_20260823.md`) and simulating it would conflate a
strategy question with an account-history one.

## Reading the leaderboard

`leaderboard.json` plus a printed table, ordered by **net P&L** (equity minus
starting equity). Also recorded per variant, but *not* used for ordering: fills,
realized vs mark-to-market split, fees, funding, inventory, working orders and
max drawdown. Those are there because the staged sweep showed a single window
can be the whole result, so a variant leading on total while carrying that shape
should at least be visible.

## P&L over time

The leaderboard is rewritten in place, so it only ever shows the *current*
state. `equity_history.csv` in the same directory is the time axis: one row per
variant per `--history-seconds` (default 60; `0` disables it).

```
python scripts/grid_pnl_curve.py                    # every variant
python scripts/grid_pnl_curve.py wide8 baseline     # a subset
```

### The event log

Per-variant event logs are **zstd-compressed JSONL**, `grid-<variant>.jsonl.zst`.

Plain JSONL cost 158 MB per variant per 20 h — 1.85 GB/day at ten variants,
78 GB/month at fourteen. Measured on a real grid log, zstd-3 gives **~16x** at
736 MB/s, roughly 350x the actual write rate, so the codec can never be the
bottleneck. **Measured on the 12-variant smoke run: 18.0x.** That turns
78 GB/month into about 5.

Nothing is dropped to achieve it. Only 2.07% of `quote_decision` lines are
reason transitions, so filtering to those would have given another ~48x — but
compression already solves the disk problem while keeping every line, and a log
that silently discarded 98% of its rows is worth much less the day a question
needs it.

Three properties the format was chosen for:

| property | why it matters |
|---|---|
| **frames concatenate** | a restart opens the file with `append` and starts a new frame; readers see one continuous stream. This is what makes a multi-day grid survivable |
| **flushed every 2,000 events** | the file is decodable *while the grid is still writing*, and a kill loses a bounded window rather than the whole tail |
| **truncated tail degrades gracefully** | a reader gets every complete frame and discards only the frame in flight |

A `run_started` event is written at each open, carrying the run's
`started_at_ms`, the variant's overrides and the build — so run boundaries in an
appended file are explicit rather than inferred from a timestamp gap.

> **Never use one-shot `decompress()` on these files.** It stops at the first
> frame and reports success, silently returning a fraction of the data. Use
> `stream_reader(..., read_across_frames=True)`, which is what
> `scripts/grid_pnl_curve.py:open_log` does.

`live` and `replay` logs stay plain `.jsonl`: one is a money audit trail, the
other is bounded by tape length. `scripts/compress_reports.py` converts
historical plain logs, verifying each round trip before deleting the original.

Choices that exist to survive a long run:

| property | why |
|---|---|
| append-only, flushed every write | a kill loses the last row, not the file; the grid is meant to run for days |
| CSV | a dense numeric series whose only consumer is a plotting script — a third the size of the equivalent JSON |
| own interval, coarser than stats | at the 5 s stats tick ten variants write ~15 MB/day; 60 s keeps a year under half a gigabyte |
| `run_started_ms` on every row | a restart appends to the same file, and splicing two runs into one curve would be silently wrong — the plotter keeps the newest |
| `mid` on every row | the curve plots against price without needing the tape, which retention does not keep forever |
| empty `mid`, never zero | a zero would plot as a real price; the field is blank until the first book arrives |
| a forced final sample at shutdown | the curve ends where the run ended, not up to one interval short |

`grid_pnl_curve.py` falls back to reconstructing the curve from the per-variant
fill logs (`pnl = cash + inventory·mid − fees`, mid from the collector tape) for
runs recorded before this file existed, or with `--from-fills`. That path is
slower, needs the tape still within retention, and cannot recover funding.

Each variant also writes a full `SessionReport` (`<name>.json`) and its own
JSONL event log, so a variant can be audited exactly like a single dry run.

An early 3-minute smoke already separated the cadence lever: the fast variants
created 326 orders, `slow5s` and `defensive` 55–58 — a 5.9x difference in action
consumption before any P&L difference appears.

## What it is not

- **Not a latency benchmark.** The grid does not spawn hot-path threads: each
  `HotPathSignal` registers exactly one thread, so N variants would need N
  signals and N isolated cores. It calls the same `policy.compute` the hot path
  calls, on the event loop. Simulated latency dominates real compute by four
  orders of magnitude (`hot_decision` p99 = 0.02 ms vs 150 ms simulated).
  `dry-run` remains the latency-faithful path.
- **Not a route to real money.** Grid mode never constructs the live backend,
  never reads credentials, and never opens an account socket —
  `tests/cli_safety.rs` asserts this holds even when handed a config with
  `live.enabled = true`. Real money is a single explicit config, never a grid.
- **Never a Parquet writer.** Not a flag: the grid cannot contend with the
  reference collector (`DATA_COLLECTION.md`).
