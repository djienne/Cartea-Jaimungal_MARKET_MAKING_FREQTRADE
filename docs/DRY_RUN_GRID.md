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

Measured on the running **10-variant** grid: **1 connection**, both collectors
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
| `q_max` | `model.q_max` | q=3 scored −279.11 against q=6's −585.61 on the same calibration |
| `phi_kappa_t` | `model.phi_kappa_t` | sweep winner used 300 against a grid topping out at 1000 |
| `min_half_spread_bps` | `quoting.min_half_spread_bps` | the losing window earned +683.89 across 1,771 fills — ~0.39/fill, under its adverse selection |
| `min_order_lifetime_ms`, `replace_threshold_bps` | `quoting.*` | the only positive latency-ladder rung was the 30 s-refresh one |
| `flow_guard_enabled`, `vpin_threshold`, `fast_move_threshold_bps` | `flow_guard.*` | the toxic-flow guard (`TOXIC_FLOW_GUARD.md`); the shipped spec pairs `guarded`/`unguarded` so the A/B runs on one shared live feed |

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
