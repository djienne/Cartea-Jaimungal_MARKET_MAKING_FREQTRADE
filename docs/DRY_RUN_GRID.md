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

## Reading the spread ladder: measure per side, never the average

`min_half_spread_bps` is a floor applied to **each quoted side independently**,
and CJ quotes are asymmetric: a non-zero inventory pulls one side toward the
touch and pushes the other away. Averaging bid and ask therefore hides exactly
the thing the lever acts on.

Measured on the 3.01 h grid tape (`grid-baseline-1787508794016.jsonl`,
38,692 sampled quoted sides), the two readings disagree completely:

| statistic | bid/ask **average** | **per side** |
|---|---|---|
| range | 13.6 – 59.6 bps | 1.5 – 80.8 bps |
| median | 38.8 | 40.7 |
| 4 bps floor binds | 0.0% | 4.6% |
| 8 bps | 0.0% | 5.4% |
| 16 bps | 0.0% | 8.2% |
| 24 bps | 3.0% | 16.1% |
| 40 bps | 55.7% | 45.9% |
| 48 bps | — | 74.9% |
| 60 bps | 100% | 94.9% |

Read as an average, the ladder looks **degenerate** — every rung below 40 bps
appears never to bind, which would make `baseline`, `wide4`, `wide8` and
`wide16` the same experiment. Read per side it is not: each rung binds, and the
mechanism is specific. The lever is not "quote wider overall", it is **how far
inventory skew is allowed to drag one quote toward the touch** — which is also
why the ladder is monotone in markout while being noisy in P&L.

A quick check that distinguishes the two: diff the `quote_decision` streams of
two variants by `quote_seq`. Over one 4-minute window `baseline` and `wide8`
agreed on 1,014 of 1,064 decisions and differed on 50 — all of them on the
inside side while inventory was non-zero. Identical averages, different quotes.

### Why `wide48` and `wide60` exist

The 185 h replay's best post-fix rung was **60 bps (+33.11)**, with 40 bps
second (+21.02), but the live ladder stopped at 40 — the winning rung had never
been tested out of sample. `max_half_spread_bps` is 80, so both fit. `wide60`
is close to a constant-60 quoter (binds 94.9% of sides) and `wide48` covers the
band where the floor goes from occasional to dominant (45.9% → 74.9%).

They also answer the question `spread_ladder_185h.md` left open: the replay's
spread60 ended short 2,440 units — 130% of equity — because nothing capped
inventory. In the grid `q_max = 6` binds, so these rungs test whether the wide
edge survives a cap that actually holds.

## The feed churn loop: measured before and after

The fix in `0bfea13` suppressed venue-replayed trades on **every** frame rather
than only the first. The claim was that the interruption rate would collapse,
and that claim is only worth making if measured on the same thing before and
after: a *continuous* grid run, counting feed interruptions per hour.

| | pre-fix | post-fix |
|---|---|---|
| continuous runtime | 3.01 h | 2.84 h |
| feed interruptions | **19** | **0** |
| rate | **6.3 / h** | **0.0 / h** |
| reconnects | — | 0 |
| feed downtime | — | 0 ms |
| longest gap | — | 0 ms |
| replayed trades suppressed | — | 27 |
| live trade prints | — | 37,754 |

At the pre-fix rate a 2.84 h run would be expected to show ~17.9
interruptions. Zero is not a marginal improvement.

**Why this is a test and not just a quiet afternoon.** Zero reconnects could
mean the fix works, or it could mean nothing happened to test it. It is the
former, because of *where* the suppressed prints are: all 27 arrived in the
first 45 seconds, on the initial subscribe, and the counter has not moved
since. That is precisely the loop's ignition point. Pre-fix, that same
subscribe backfill spilled past frame one, hit the lag check, and bailed —
which forced a re-subscribe, which replayed more backfill. The ignition point
was exercised once, and did not ignite.

**The second exercise, which arrived on its own.** At 3.2 h the venue closed
the stream itself — `server closed market stream: CloseFrame { code: Normal }`,
an ordinary server-initiated close, not our own bail. This is the unrelated
reconnect the paragraph above was waiting for, and it is the stronger test,
because it forces a re-subscribe and therefore a fresh backfill burst:

```
23:42:27  reconnects=0 feed_gaps=0 downtime=0ms    replayed_ignored=27  trades=39,639
23:43:39  WARN public market stream interrupted; measuring the gap
23:43:42  reconnects=1 feed_gaps=1 downtime=772ms  replayed_ignored=57  trades=39,741
23:53:47  reconnects=1 feed_gaps=1 downtime=772ms  replayed_ignored=57  trades=41,126
```

Thirty more replayed trades were suppressed on the re-subscribe, the feed was
back in **772 ms**, and then every counter froze for the following ten minutes.
That +30 burst is exactly what pre-fix would have hit the lag check, bailed,
and re-subscribed on — the 6.3/h loop, ignited a second time. It did not
ignite. The 772 ms gap is far inside `max_feed_gap_ms` (60 s), so no variant
was invalidated.

**What is still untested.** The residual trade-lag trip rate — the reason
`max_trade_lag_ms` was given its own key rather than a new default — remains
unmeasured, because in 3.2 h no *genuinely new* trade has ever exceeded the
threshold. It stays at 5,000 ms until data says otherwise. One instrument, one
venue, one window.

### First live ladder result (single window — read with care)

At 2.84 h into the 18-variant run, over a window containing a +9.7% rally with
an 8.28% minute:

```
wide48      +29.07   381 fills        wide8      -15.49  1697
wide24slow30s +15.18 1630             q9         -16.55  1748
wide40      +14.54   576              wide4      -22.60  1755
wide16slow30s +10.82 2085             baseline   -27.16  1804
wide24      +10.73  1085              slow15s    -47.93  2937
wide16       +9.01  1489              slow5s     -49.24  3218
wide60       -2.39   195              q3         -49.34  2200
```

Three things this window says, none of them settled:

- **Fill count runs inverse to P&L.** The winner took 381 fills; the three
  worst took 2,200–3,218. This is the adverse-selection mechanism stated about
  as plainly as this instrument states anything, and it matches the replay's
  markout column rather than its P&L column.
- **60 bps fails live where the replay ranked it best.** `wide60` is −2.39 on
  195 fills with the grid's largest drawdown (39.97). Too few fills to manage
  inventory through a move; it ends the window short 976 units. The replay's
  spread60 had the same shape (−2,440 units) but no binding `q_max` to reveal
  it. The live sweet spot so far is **40–48 bps**, not 60.
- **The flow guard did not trip once.** `guarded` and `unguarded` are identical
  to the cent across 1,804 fills, and the guarded variant logged zero
  `toxic_flow` decisions in 17,147 quotes. An 8.28%-per-minute move never
  approaches an 8%-per-**5s** tripwire. `risk_limit` fired 1,028 times instead.
  See `TOXIC_FLOW_GUARD.md` — this is a live instance of the narrow-trigger
  concern, not a new finding.

One window, one direction, one instrument. The ladder has reordered twice
already inside this run (`wide40` led at 1.5 h, `wide48` at 2.5 h).
