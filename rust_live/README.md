# Rust Cartea–Jaimungal Engine

This directory contains the standalone Rust research runtime. The Freqtrade
strategy, Python estimators, replay harness, tests, and evidence remain in their
original folders and are the numerical reference oracle. Both runtimes now use
the same schema-v4 direct-window calibration semantics: there is no EMA or other
cross-window smoothing between the observed parameters and the HJB.

The Rust model is intentionally singular: asymmetric Cartea–Jaimungal arrival
and adverse-selection parameters feed the nonlinear backward-Euler HJB. The
runtime uses the full episodic `(t,q)` surface, fractional-inventory depth
interpolation, one maker-fee cushion per side, bps clamps, and venue-aware price
rounding.

The Cargo workspace enforces dependency direction:

- `cj-core` contains the model, HJB, instrument math, and quote policy and has
  no Tokio, Arrow/Parquet, HTTP, redb, or JSON dependency;
- `cj-data` owns calibration, Parquet collection, and replay;
- `mm-execution` owns venue-neutral interfaces and dry-run simulation;
- `mm-runtime` owns bounded publications, the hot thread, and latency
  observation;
- `hyperliquid-connector` owns transport, signing, account state, and live
  execution; `mm-live` is only the CLI/orchestrator.

## Safety boundary

- `dry-run` consumes real public Hyperliquid data and simulates orders locally.
- `replay` consumes the existing symbol-partitioned Parquet history.
- `live` selects the stateful Hyperliquid backend only when `[live].enabled=true`.
  The tracked profile ships disabled and refuses before credentials or network.
- Production live always enforces latency. `acceptance_test` is the only bypass
  and is hard-capped at 12 USDC directional, 24 USDC gross working, 60 USDC
  turnover, and 0.5 USDC campaign loss.
- `connector-check` loads the ignored four-key credential file but is read-only:
  it exercises account `/info` calls and all account WebSocket subscriptions
  while sending zero actions.
- Real-account acceptance/canary code is compiled only into the separately
  feature-gated `mm-live-acceptance` artifact and is absent from the production
  image.
- Any causal event loss or reconnect invalidates the dry-run evidence and
  withdraws quotes.
- An active legacy collector is detected before Rust becomes a Parquet writer.
  Use `--no-write-parquet` when intentionally running beside the reference
  collector.

Only the `cashcat` instrument profile is scientifically validated. Shared model,
calibration, execution, accounting, and storage code accepts an `InstrumentSpec`;
future markets require a new validated profile and parity fixtures, not a model
rewrite.

## Live connector boundary

The model and hot path publish venue-neutral `DesiredQuotes` containing integer
price/size `OrderIntent`s. Execution is behind `ExecutionBackend`; account reads
are behind `AccountStateProvider`; replay market data is behind
`MarketDataSource`, and the public Hyperliquid stream produces the same canonical
`MarketEvent` values. The implementations are `DryRunBackend` and the stateful
`HyperliquidLiveBackend`.

The pure-Rust live path provides secret-safe credentials, vault-aware signing,
fixed-point actions, crash-safe nonce ranges, normalized transactional
order/fill/funding state,
WebSocket action posts, REST emergency cancels, typed account subscriptions,
unknown-outcome reconciliation, account-aware sizing, actual fees, rate
reserves, ALO batches, and reduce-only IOC market close. Network and persistence
work stay off the hot thread. Lifecycle persistence is delta-written on a
dedicated single-writer thread (only changed orders and dedup keys touch redb);
a newly reserved nonce range is fsynced before use, and the next range is
prefetched on a background thread before the current one runs out. Durable
history is bounded: the fill/funding replay horizon advances with each
authoritative reconcile (24h retention, checkpoint fsynced before pruning) and
aged terminal orders are dropped, so per-event cost stays flat over a session.
Superseded market quote revisions coalesce until the configured minimum order
lifetime, while fill/risk cancellations bypass that delay; a resting order
within the requote hold window (`replace_threshold_ticks` /
`replace_threshold_bps`, evidence in `docs/requote_hysteresis_sweep.md`) is
kept to preserve queue position. The connector tracks its local contribution
to the venue address-action budget, preserves a separate WebSocket allowance
for cancels and emergency reduction, and config validation rejects budgets the
worst-case requote rate plus pings and dead-man refreshes cannot fit.

Normal shutdown, latency refusal, and risk kills cancel orders but retain known
inventory. The live loop's teardown — cancel resting orders, reconcile, clear
the dead-man — runs on **every** exit path, error exits included, and the
release profile unwinds on panic so a panicking task cannot bypass it.
Consecutive-loss and daily-loss kill inputs are tracked durably from closing
fills. `live-flatten` is the explicit CASHCAT-only reduce-only maintenance
path and escalates IOC limits from 25 to 250 bps without allowing a position
flip. Because it can only reduce, it opens the durable store even when the
configuration has drifted while orders are still working — the state a quoting
session refuses to inherit is exactly the one flatten exists to clear. The
stored fingerprint is adopted only once the store is flat, so an interrupted
flatten still blocks the next quoting session.

The current protocol research, exact signing specification, repository-wide
connector audit, CASHCAT constraints, and staged release gates are consolidated
in [`HYPERLIQUID_LIVE_CONNECTOR.md`](HYPERLIQUID_LIVE_CONNECTOR.md).

For connector/live use, copy the tracked dummy credential template and edit only
the ignored copy:

```powershell
Copy-Item rust_live/hyperliquid.env.example rust_live/hyperliquid.env
```

The dotenv template has the same four fields used by Passivbot:
`exchange`, `wallet_address`, `private_key`, and `is_vault`. It deliberately
contains an invalid all-zero API/agent key. Disabled `live` refuses before
reading it. Enabled live and the explicit connector/maintenance commands load
the ignored real copy.

The shared field names do not create a runtime dependency: both dry-run and the
live connector are pure Rust. Python SDK, Passivbot, and CCXT are allowed
only as offline test-oracle references.

## Latency monitoring

The hot thread samples one in every 16 decision cycles. The timestamp is taken
only after the quote has already been published, and four samples are batched in
thread-local storage before one lock-free queue publication. A separate observer
thread calculates rolling 10-second and 60-second `last/min/p50/p95/p99/p99.9/max`
distributions and sample age every five seconds. It performs all sorting, JSON
serialization, file I/O, and fsync work off the hot and execution threads.

Current levels are written atomically to:

```powershell
Get-Content rust_live/run/cashcat-latency.json
```

The same snapshot is embedded in the final session report. It distinguishes:

- local market-event dispatch delay;
- separate public/account WebSocket application-ping round-trip time;
- `/info` request round-trip time;
- sampled Cartea–Jaimungal hot-decision compute;
- market-data receive to backend start for every actual quote publication;
- decision start to backend start;
- backend reconciliation processing;
- decision start to backend completion;
- execution-queue wait, action preparation, signing, socket write, and complete
  decision-to-wire timing;
- submit/cancel acknowledgements, fill-to-close-send, close-to-fill, and
  fill-to-flat distributions.

The central `[latency]` configuration enables the production gate and sets
its default maximum acceptable rolling p95 to 150 ms. Monitoring always runs,
but enforcement is bypassed for validation, API/WebSocket probes, replay,
dry-run, and the feature-gated acceptance runner. Production `live` always
enables enforcement and requires 20 fresh samples from both sockets plus three
healthy observer windows before reopening. Warm-up, stale/dropped samples,
observer failures, or a breached rolling p95 withdraw both quotes with reason
`latency_limit`. Dropped-sample and observer-error blocks apply to the
evaluation window they occurred in — the gate judges deltas, not lifetime
totals, so one transient snapshot-write failure or queue burst no longer halts
trading for the rest of the session (snapshot-write failures count only after
three consecutive misses). The tokio runtime is bounded and, when
`runtime.hot_path_cpu` is set (as the shipped config now does), workers and
blocking threads are pinned away from the hot core; pair with OS-level
isolation for a genuinely quiet core.

`window_samples` must be inspected before interpreting tail percentiles. The
dry-run backend's configured 250 ms decision/ack/cancel delays are simulation
assumptions; the latency file measures actual software timing. The live backend
publishes real venue acknowledgement and close timings into the same observer.

## Commands

Run from the repository root:

```powershell
cargo run --locked --release --manifest-path rust_live/Cargo.toml -- `
  build-info

cargo run --locked --release --manifest-path rust_live/Cargo.toml -- `
  --config rust_live/config/cashcat.toml validate

cargo run --locked --release --manifest-path rust_live/Cargo.toml -- `
  --config rust_live/config/cashcat.toml calibrate

cargo run --locked --release --manifest-path rust_live/Cargo.toml -- `
  --config rust_live/config/cashcat.toml replay

cargo run --locked --release --manifest-path rust_live/Cargo.toml -- `
  --config rust_live/config/cashcat.toml dry-run --no-write-parquet

cargo run --locked --release --manifest-path rust_live/Cargo.toml -- `
  --config rust_live/config/cashcat.toml connector-check `
  --credentials rust_live/hyperliquid.env --duration-seconds 65

# Requires [live].enabled=true; production latency cannot be bypassed.
cargo run --locked --release --manifest-path rust_live/Cargo.toml -- `
  --config rust_live/config/cashcat.toml live

# Explicit reduce-only maintenance close.
cargo run --locked --release --manifest-path rust_live/Cargo.toml -- `
  --config rust_live/config/cashcat.toml live-flatten
```

Linux shells use the same arguments without PowerShell backticks.
Real-account acceptance code is excluded from the production binary. It is
available only through the separately feature-gated `mm-live-acceptance` binary.
Build it explicitly only for an authorized dedicated-account campaign:

```powershell
cargo build --locked --release --manifest-path rust_live/Cargo.toml `
  --features live-acceptance --bin mm-live-acceptance
```

## Validation

```powershell
cargo fmt --manifest-path rust_live/Cargo.toml --all -- --check
cargo clippy --locked --manifest-path rust_live/Cargo.toml --workspace --all-targets --all-features -- -D warnings
cargo test --locked --manifest-path rust_live/Cargo.toml --workspace --all-targets --all-features
cargo build --locked --manifest-path rust_live/Cargo.toml --release --bins
cargo run --locked --manifest-path rust_live/Cargo.toml --release --bin hot-path-bench
cargo fmt --manifest-path rust_live/fuzz/Cargo.toml --all -- --check
cargo check --locked --manifest-path rust_live/fuzz/Cargo.toml --all-targets
cargo deny --manifest-path rust_live/Cargo.toml check advisories bans sources
```

The portable Docker image uses O3/fat LTO. Build a host-specific image only on
the intended production CPU:

```powershell
docker build --build-arg "MM_RUSTFLAGS=-C target-cpu=native" `
  --build-arg MM_BUILD_FLAVOR=native-o3 `
  --build-arg MM_GIT_REVISION=<commit-or-build-id> `
  -f rust_live/Dockerfile -t cashcat-cj-rust:native .
```

`tests/python_parity.rs` pins deterministic outputs from the Python reference for
schema-v4 unsmoothed parameter estimation, HJB solving, time/inventory interpolation, fee
assembly, and final tick-rounded quotes. Parameters and ordinary HJB values use
a `1e-8` tolerance, high-sensitivity HJB points use `1e-7`, and rounded prices
must be exactly equal.

## Important units

- price and HJB depth: USDC per base asset;
- kappa: `1 / USDC`;
- lambda: market orders per second per side;
- epsilon: USDC per base asset;
- sigma squared: `USDC^2 / second`;
- inventory `q`: physical base position divided by the flat-state inventory
  unit.

The CASHCAT profile dynamically verifies venue metadata and currently expects
integer base sizes, up to six price decimals, five significant figures, and at
most 3x venue leverage.
