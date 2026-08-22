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
- `live-canary` is a separately guarded connector test, not a trading mode. It
  requires explicit real-money confirmation, a dedicated flat/order-free
  subaccount, CASHCAT, and a hard maximum of 12 USDC.
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
fixed-point actions, crash-safe nonce ranges, ACID order/fill/funding state,
WebSocket action posts, REST emergency cancels, typed account subscriptions,
unknown-outcome reconciliation, account-aware sizing, actual fees, rate
reserves, ALO batches, and reduce-only IOC market close. Network and persistence
work stay off the hot thread.

Normal shutdown, latency refusal, and risk kills cancel orders but retain known
inventory. `live-flatten` is the explicit CASHCAT-only reduce-only maintenance
path and escalates IOC limits from 25 to 250 bps without allowing a position
flip.

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
thread calculates rolling five-minute `last/min/p50/p95/p99/p99.9/max`
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
- submit/cancel acknowledgements, fill-to-close-send, close-to-fill, and
  fill-to-flat distributions.

The central `[latency]` configuration enables the production gate and sets
its default maximum acceptable rolling p99 to 150 ms. Monitoring always runs,
but enforcement is bypassed for validation, API/WebSocket probes, replay,
dry-run, `live-canary`, and `acceptance_test`. Production `live` always enables
enforcement and requires 20 fresh samples from both sockets. Warm-up,
stale/dropped samples, observer failures, or a breached p99 withdraw both quotes
with reason `latency_limit`.

`window_samples` must be inspected before interpreting tail percentiles. The
dry-run backend's configured 250 ms decision/ack/cancel delays are simulation
assumptions; the latency file measures actual software timing. The live backend
publishes real venue acknowledgement and close timings into the same observer.

## Commands

Run from the repository root:

```powershell
cargo run --manifest-path rust_live/Cargo.toml -- `
  --config rust_live/config/cashcat.toml validate

cargo run --manifest-path rust_live/Cargo.toml -- `
  --config rust_live/config/cashcat.toml calibrate

cargo run --manifest-path rust_live/Cargo.toml -- `
  --config rust_live/config/cashcat.toml replay

cargo run --manifest-path rust_live/Cargo.toml -- `
  --config rust_live/config/cashcat.toml dry-run --no-write-parquet

cargo run --manifest-path rust_live/Cargo.toml -- `
  --config rust_live/config/cashcat.toml connector-check `
  --credentials rust_live/hyperliquid.env --duration-seconds 65

# Requires [live].enabled=true; production latency cannot be bypassed.
cargo run --manifest-path rust_live/Cargo.toml -- `
  --config rust_live/config/cashcat.toml live

# Explicit reduce-only maintenance close.
cargo run --manifest-path rust_live/Cargo.toml -- `
  --config rust_live/config/cashcat.toml live-flatten
```

Linux shells use the same arguments without PowerShell backticks.
`live-canary` is intentionally omitted from routine commands because it sends
real orders and is only for an expressly authorized, bounded dedicated-account
test.

## Validation

```powershell
cargo fmt --manifest-path rust_live/Cargo.toml --all -- --check
cargo clippy --manifest-path rust_live/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path rust_live/Cargo.toml --all-targets
cargo build --manifest-path rust_live/Cargo.toml --release --bins
cargo run --manifest-path rust_live/Cargo.toml --release --bin hot-path-bench
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
