# Dry-run grid — several parameter sets, one feed

Normally run as a container, which is what survives a reboot:

```
docker compose up -d          # from the repo root
docker compose logs -f
```

Directly, for a one-off:

```
mm-live --config rust_live/config/cashcat_dryrun_realistic.toml \
        dry-run-grid --grid rust_live/config/grid_cashcat.toml \
        --out-dir rust_live/reports/grid_live
```

Runs every variant in the grid spec against **one** shared public market feed,
simulating each independently, and rewrites a leaderboard ranked by net P&L
every stats interval. `--duration-seconds 0` (the default) runs until stopped,
and a restart **resumes** the run rather than beginning a new one — see
[A restart continues the run](#a-restart-continues-the-run).

## Why one process rather than N

Hyperliquid allows **10 simultaneous WebSocket connections per IP**. That budget
is already shared with three collector containers and with any live session,
which needs two. One `dry-run` process per variant would open one connection
each and could take down data collection or block real trading.

The current **18-variant** spec uses **1 connection**. Earlier 20-variant runs
used the same single connection while both market-making collectors remained
healthy; the count does not grow with the number of variants.

## What a variant may change

A deliberately narrow set, because only these are implicated by the evidence.
The staged sweep (`cashcat_sweep.md`) localises the entire loss to one six-hour
burst. Re-run 2026-09-02 on 393.77 h under estimator schema v5: 63 of 64
windows sum to **+371.31**, and `08-22 03:57` alone is **−323.70** on 1,473
fills. (The 161.95 h v4 artifact read +35.28 against −241.17 on 1,771 fills.) So the question a variant should answer is *how much
of a burst does this take*.

| key | maps to | evidence |
|---|---|---|
| `q_max` | `model.q_max` | replay: q=3 scored −279.11 against q=6's −585.61. **Live reversed this** — see below |
| `phi_kappa_t` | `model.phi_kappa_t` | sweep winner used 300 against a grid topping out at 1000 |
| `min_half_spread_bps` | `quoting.min_half_spread_bps` | the losing window earned +683.89 across 1,771 fills — ~0.39/fill, under its adverse selection |
| `min_order_lifetime_ms`, `replace_threshold_bps` | `quoting.*` | the only positive latency-ladder rung was the 30 s-refresh one |
| `flow_guard_enabled`, `vpin_threshold`, `fast_move_threshold_bps` | `flow_guard.*` | paired guarded/unguarded variants isolate the toxic-flow guard on one feed |
| `phi_kappa_t_max` | `model.phi_kappa_t_max` | a variant requesting φ·κ·T above the base ceiling of 300 must raise this ceiling too |

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

### A restart continues the run

The grid checkpoints every variant's accounting to `grid_state.json` on each
stats tick, and resumes from it on startup. Equity, inventory, fills, drawdown,
markouts and the elapsed clock all carry across, so a reboot costs a gap rather
than the measurement — which is what the 2026-08-27 Windows update cost before
this existed.

Resuming is bounded, because a resumed run is not the same object as an
uninterrupted one:

| | |
|---|---|
| the interruption | counted in `feed_health.downtime_ms`, so it erodes the 5% budget |
| | **not** added to `feed_longest_gap_ms` — that limit is about a long hole *while quoting*, and a restart leaves no working orders |
| visible as | `resumes`, `resumed_downtime_ms`, and `[RESUMED]` in the rendered table |
| gap > `--max-resume-gap-seconds` (900) | starts fresh instead |
| grid spec edited | starts fresh — the checkpoint fingerprints every variant's config |
| resume failure | all-or-nothing; a half-resumed grid has rows that are not comparable |

The long-gap refusal is the important one. Resuming means marking held inventory
at a price whose path was never observed, and that is precisely the mechanism
that turned the 46.4 h leaderboard into a 13.2% rally reported as trading
profit. Fifteen minutes covers a reboot; hours do not, and are not worth
pretending about.

`run_started_ms` in `equity_history.csv` now stays constant across a restart, so
a *change* in it marks a genuine boundary — a refused resume — rather than
merely a relaunch.

The checkpoint itself does not accumulate: `grid_state.json` is ~27 KB at
eighteen variants, rewritten in place each tick, plus one `.bak` generation — a
constant ~54 KB. The `.bak` is what makes a checkpoint torn by something outside
the process (a full disk, power loss mid-write) cost one stats interval instead
of the whole run; `load` falls back to it automatically.

### The period archive — what outlives the tape

Both of this project's evidence streams expire on a clock:

- the **replay** can only score a window while its Parquet shards exist, and the
  CASHCAT collector keeps **30 days** (`CASHCAT_RETENTION_MINUTES: 43200`);
- the **grid**'s event logs rotate at ~34 days.

So once a period rolls off, no sweep can ever be run against it again — not more
cheaply, not at all. `scripts/archive_period.py` runs every 21 days in the
`mm-archiver` container and attempts to write one directory per period under
[`history/`](history/): a full-search sweep, the grid's leaderboard, the
period's P&L curve (15-min, zstd), and its render. The container does not commit.

21 days against 30 leaves 9 days to retry an interrupted or failed cycle. If a
directory contains `sweep_FAILED.log`, inspect it and rerun with `--force` before
that margin expires; an indefinitely failed cycle does lose raw history.

Two design points worth knowing:

- **Due-ness comes from disk, never a timer.** The loop wakes hourly and asks
  whether the newest directory in `docs/history/` is older than the cadence. A
  `sleep 21d` restarts its countdown on every reboot, and on this machine that
  could plausibly mean never firing — the same failure that cost the 46.4 h run.
- **It writes but does not commit.** Committing from a container would mean an
  SSH key inside it. It logs an uncommitted-history reminder on every wake;
  confirm the exact paths with `git status`, then use
  `git add docs/history && git commit && git push`.

Symbol selection is by tape length (> 7 days), which separates the 30-day
collector from the 3-day one without this repo reading another project's compose
file. A symbol that qualifies on length but has no instrument profile is skipped
rather than scored with CASHCAT's tick size and inventory base.

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

#### Rotation — the disk ceiling

Compression fixed the *rate*; it did not bound the *total*. Once the grid began
resuming across restarts, the append stream had no natural end, and 0.31 MB/h
per variant is 3.9 GB/month across eighteen — forever.

Each log now rolls at `--log-max-mb` (64) and keeps `--log-keep` (3)
generations, `grid-wide8.jsonl.zst` → `.1` → `.2` → `.3`, deleting what falls
off:

| | |
|---|---|
| per variant | 64 MB live + 3 rolled = **256 MB** |
| eighteen variants | **4.5 GB ceiling**, independent of how long the run lasts |
| history retained | ~206 h per file, ~34 days total |

The roll happens at a flush boundary, where the frame is already terminated, so
a rolled generation is complete and readable rather than a truncated frame. Each
rotation logs the size rolled and that the oldest generation was deleted —
nothing is dropped silently. `--log-max-mb 0` disables rotation and restores the
old unbounded behaviour.

`equity_history.csv` is deliberately *not* rotated. At ~129 KB/h it is 91
MB/month, two orders of magnitude smaller, and it is the P&L curve itself — the
artifact the run exists to produce.

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
  `rust_live/tests/cli_safety.rs` asserts this holds even when handed a config with
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

**The residual trade-lag trip rate, now measured.** This is the number
`max_trade_lag_ms` was given its own config key for, rather than a new default.
Over 15.2 h the feed was interrupted nine times, and the causes split cleanly:

| cause | count | shape |
|---|---:|---|
| `server closed market stream` (CloseFrame) | 4 | clockwork, every ~2h50m |
| `live trade arrived more than 5000ms late` | 5 | two self-limiting clusters |

The four server closes are the venue's ~3-hourly session expiry, the same
cadence the collector work documented — 23:43, 02:33, 05:21, 10:37.

The five lag trips are **not** that. They arrive in clusters: two at 07:41:50
and 07:42:24 (34 s apart), then three at 11:51:57, 11:53:06 and 11:54:41 (~4 h
later, spanning 2m44s). Both clusters stopped on their own. That is worth
stating precisely because it is the churn loop's *shape* without its behaviour:
`replayed_trades_ignored` rose 156 → 241 across those reconnects, so backfill
was arriving each time and being suppressed rather than driving the next bail.

**~~The threshold stays at 5,000 ms~~ — raised to 15,000 the same day.** The
paragraph above ended "revisit only if a cluster stops self-limiting or downtime
becomes a material fraction of a run". Within the hour it did both: 37
interruptions in 15.9 h, the last 21 every ~35 s, downtime pinned at the 5%
invalidation threshold.

Two things were wrong with the reasoning, not just the number.

First, **nothing logged the actual lag**, so "these trips are detecting real
lateness" was an assumption. Adding the measured lag to the bail message
settled it in minutes: seven trips at **5336, 5607, 5639, 5966, 6106, 6301,
9852 ms**, each on a trade born 6–130 s *after* the connection — genuinely live,
not replay, but clustered barely above a 5 s line. A delivery tail, not a broken
feed.

Second, I had just fixed a real hole in the replay suppression (a replayed trade
inside the 2 s skew grace could still bail) and predicted it would stop the
churn. It did not — the first bail on the fixed binary was a trade born 6,238 ms
after connect. The fix is correct and has a regression test; it simply was not
what production was hitting.

**15,000 ms, not 10,000.** The first five samples spanned 5.3–6.3 s and 10 s
looked like ample headroom; the sixth was 9,852 ms, 1.5% under that line.
Picking a threshold off a handful of samples is exactly what produced the
original 5,000.

**It is not the quoting guard.** `runtime.market_stale_ms` still decides whether
the top of book is fresh enough to quote from and stays at 5,000; this only
decides when to reconnect a socket. That is why raising it is safe.

**No threshold absorbs this tail.** At 15 s a trip still occurred — 17,754 ms,
34 minutes into a connection — which matches the original 183,344-trade finding
that p99.9 was 79 s. 15 s removes the dense cluster and leaves the rare genuine
outlier; it does not cure the stall.

### Measured over 10 h on the 15 s threshold

| cause | count | what it is |
|---|---:|---|
| `server closed market stream` | 3 | venue session expiry, ~3 h cadence |
| `no inbound frame before idle timeout` | 3 | the venue feed going silent for 45 s |
| `live trade arrived 17754ms late` | 1 | a genuine outlier, correctly caught |
| **total** | **7** | **0.70 / h** |

For comparison on the same instrument: **6.3/h** was the original pre-fix
baseline, and **~80/h** was the degraded window that forced this change.

Two things this settles. The threshold now *discriminates*: one trip in ten
hours, on a real 17.75 s outlier, rather than several hundred on a 5–10 s
delivery tail. And **six of seven interruptions are venue-side**, so what
remains is not something a client-side threshold can fix — the dominant cause
is no longer trade lag at all but the idle watchdog firing on a feed that
simply stops sending. Both detectors are reporting one underlying thing: a
venue feed that intermittently stalls and then delivers a burst of stale
prints.

### Two full runs, 2026-08-25/26 — the ordering does not survive

Two independent runs on the same instrument, days apart, both ~16 h at the
point of comparison. The wide rungs rank in **exactly the opposite order**:

| rung | run A (16.07 h) | run B (16.0 h) | run B final (23.09 h) |
|---|---:|---:|---:|
| wide40 | **+60.15** | +30.12 | +27.00 |
| wide48 | +50.03 | +70.15 | +71.90 |
| wide60 | +29.61 | **+97.63** | +83.81 |

That is stronger than the within-run reversals recorded above: it is not a
timestamp artefact, it is two complete day-length windows disagreeing about
which rung wins. **No ranking of 40 / 48 / 60 is supported by this evidence.**

What *does* survive both runs, and every checkpoint inside them:

| | run B final, 23.09 h, 38,168 fills |
|---|---|
| ≥ 16 bps | wide60 +83.81, wide48 +71.90, wide40 +27.00, wide24 +24.00, wide16 −0.29 |
| ≤ 8 bps | wide8 −22.71, wide4 −27.22, baseline −43.04 |
| slow cadence | wide24slow30s −86.38, slow5s −114.48, slow15s −142.30 |

**Fill count runs inverse to P&L, without exception.** The winner took 173
fills; the three worst took 3,762–5,953. The shipped 1.5 bps baseline is
−43.04 on 2,232 fills.

**The flow guard has never fired.** `guarded`, `unguarded` and `baseline` are
identical to the cent across both runs — 2,232 fills each — so nothing in the
grid has yet exercised the toxic-flow breaker.

**Inventory is the open risk, and it is not what the replay warned about.** All
three wide rungs end run B *long* 1,658–1,909 units, not short; wide60's
59.51 drawdown is the grid's largest. `q_max` binds, but the residual is
material against ~298 USDC of equity.

Feed over the same 23.09 h on the 15 s threshold: **14 interruptions (0.61/h),
of which 2 were lag trips** (17,754 and 18,382 ms — both genuine outliers, both
~3 s clear of the limit rather than grazing it). Total downtime **80.1 s =
0.10%**, 435 replayed trades suppressed against 114,810 live prints, and no
variant invalidated.

### Run B past 24.6 h is void — a 19.65 h blackout, marked at a rally

**The 23.09 h figures above stand.** Downtime to that point was 80.1 s, 0.10%.
Everything in this section is about what happened *after* that checkpoint, and
none of it is evidence.

Run B was not stopped deliberately. It ran on to 46.4 h and died at **2026-08-27
13:47 local**, when the machine was powered off from the Start menu — Windows
event 1074, almost certainly for the update that had been downloading since
13:00. Nobody noticed for 66 h. Its final `leaderboard.json` looks like a
result and is not one:

| rung | 23.09 h | 24.6 h | 46.4 h | inventory at the gap | inv × Δmid |
|---|---:|---:|---:|---:|---:|
| wide60 | +83.81 | +62.74 | **+124.01** | 2,307 | +62.5 vs +60.97 actual |
| wide48 | +71.90 | +53.42 | **+114.09** | 2,285 | +61.9 vs +60.37 actual |
| wide40 | +27.00 | +17.17 | **+60.87** | 1,658 | +44.9 vs +43.42 actual |
| wide24 | +24.00 | +21.45 | +28.16 | 216 | +5.8 vs +4.45 actual |
| baseline | −43.04 | −43.25 | −39.31 | 164 | +4.4 vs +3.04 actual |

The public feed was down from `2026-08-26T14:00:42Z` to `2026-08-27T09:39:52Z` —
**19.65 h, 42.5% of the run** — and CASHCAT's mid went 0.20569 → 0.23276,
**+13.2%**, while the grid was blind. Every rung's gain over that stretch is its
frozen inventory times that move, to within a percent. The rungs finish ranked
in the order of how *long* they happened to be, and each took **one fill** after
hour 23.

Realized P&L, which is what a maker actually earns, had already turned over
before the blackout:

| rung | h4 | h8 | h16 | h23 | h46 |
|---|---:|---:|---:|---:|---:|
| wide60 | 47.27 | 70.24 | **97.90** | 86.71 | 86.47 |
| wide48 | 23.51 | 58.34 | 70.22 | **78.97** | 72.77 |
| wide40 | −0.94 | 26.19 | 30.67 | **31.86** | 22.50 |

So wide60's headline +124.01 is +86.47 realized (below its own h16 peak) plus
+37.54 of mark-to-market on 2,255 units — roughly 515 USDC of notional against
~298 USDC of equity, a 1.7× levered long. Its inventory had also swung from
−1,621 at h16 to +1,909 at h23, a 3,530-unit flip. Whatever that is, it is not
the market-making result the ladder was built to measure.

Four defects had to line up, and all four are now fixed (2026-08-30):

- **`connect_async` had no timeout.** The host network wedged and the call hung
  rather than failing, so the reconnect loop never ran. The backoff caps at 8 s,
  so a loop that was genuinely retrying would have logged ~8,800 failures; it
  logged **zero**, then six in 40 s once the call started returning. Now bounded
  by `runtime.ws_connect_timeout_ms` (10 s), at all three connect sites.
- **An open gap was invisible.** `feed_gaps` / `feed_downtime_ms` /
  `feed_longest_gap_ms` are only written when a gap *closes*, so the health line
  printed 117 identical copies of `reconnects=28 feed_gaps=27
  feed_downtime_ms=282835` across the blackout. Only `trade_prints`, frozen at
  122,337, gave it away. The line now carries `feed_down_for_ms`, logs at `WARN`
  and every 60 s while down, and fires the moment the feed drops.
- **`leaderboard.json` could not be marked invalid.** `FeedHealth` would have
  disqualified this run twice — 42.5% downtime against a 5% limit, a 70,750,528
  ms gap against 60,000 — but it was only evaluated in the teardown, which the
  power-off skipped. The verdict is now recomputed on every write, open gaps
  included, and ANDed into every row.
- **Windows killed the process without teardown.** `wait_for_shutdown_signal`
  handled `CTRL_C_EVENT` alone; a Start-menu shutdown sends
  `CTRL_SHUTDOWN_EVENT`. It now handles shutdown, close, logoff and break — and
  the grid runs in a container, where SIGINT reaches the handler that always
  worked.

The run directory is kept at `rust_live/reports/archive_grid_15s_contaminated/`.
Read the equity curve up to hour 23 and stop there.

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
- **No ordering among the wide rungs is established, and I asserted one
  twice.** At 2.84 h I wrote that 60 bps fails live; at 3.51 h I retracted that
  for an interior optimum at 48; at 4.2 h `wide60` leads outright. The rungs
  tracked through the run:

  | rung | 1.0 h | 2.0 h | 2.84 h | 3.5 h | 4.2 h |
  |---|---:|---:|---:|---:|---:|
  | wide16 | +7.68 | +3.47 | +9.03 | +14.41 | +11.99 |
  | wide24 | +10.56 | +9.56 | +10.49 | +18.31 | +22.90 |
  | wide40 | +18.94 | +17.69 | +14.81 | +24.44 | +35.14 |
  | wide48 | +6.53 | +11.97 | +29.25 | +37.79 | +50.50 |
  | wide60 | −8.37 | −14.70 | −2.71 | +21.51 | **+57.33** |

  `wide60` travels −14.70 → +57.33 on 218-odd fills. Any ranking of 40 / 48 /
  60 read off a single timestamp is a ranking of which rung most recently
  caught a move. **What is stable across every timestamp is only the coarse
  split**: everything ≤ 8 bps loses at every check, everything ≥ 16 bps is
  positive at every check. That is the finding; the fine ordering is not one
  yet, and this table is here so the next reader does not extract one.

  The generalisable error is stating a directional conclusion from a variant
  whose fill count is in the low hundreds — three times, in the same run,
  in both directions.

- **The flow guard did not trip once.** `guarded` and `unguarded` are identical
  to the cent across 1,804 fills, and the guarded variant logged zero
  `toxic_flow` decisions in 17,147 quotes. An 8.28%-per-minute move never
  approaches an 8%-per-**5s** tripwire. `risk_limit` fired 1,028 times instead.
  See `TOXIC_FLOW_GUARD.md` — this is a live instance of the narrow-trigger
  concern, not a new finding.

One window, one direction, one instrument. The ladder has reordered twice
already inside this run (`wide40` led at 1.5 h, `wide48` at 2.5 h).
