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
- `live` always exits with an error. There is no private key, signer, or order
  submission implementation in this release.
- Any causal event loss or reconnect invalidates the dry-run evidence and
  withdraws quotes.
- An active legacy collector is detected before Rust becomes a Parquet writer.
  Use `--no-write-parquet` when intentionally running beside the reference
  collector.

Only the `cashcat` instrument profile is scientifically validated. Shared model,
calibration, execution, accounting, and storage code accepts an `InstrumentSpec`;
future markets require a new validated profile and parity fixtures, not a model
rewrite.

## Future live connector boundary

The model and hot path publish venue-neutral `DesiredQuotes` containing integer
price/size `OrderIntent`s. Execution is behind `ExecutionBackend`; account reads
are behind `AccountStateProvider`; replay market data is behind
`MarketDataSource`, and the public Hyperliquid stream produces the same canonical
`MarketEvent` values. The current implementations are the deterministic
`DryRunBackend` and the fail-closed `LiveExecutionUnavailable` stub.

A later authenticated Hyperliquid adapter therefore owns signing, private
WebSocket/REST transport, client/exchange order identity, acknowledgements,
cancels, rejects, partial fills, fee/funding reconciliation, and authoritative
account state. It should not contain calibration, HJB, quote, sizing, or risk
formulae. Adding that adapter must be separately tested against venue responses;
merely replacing the stub is not treated as live-trading validation.

The current protocol research, exact signing specification, repository-wide
connector audit, CASHCAT constraints, and staged release gates are consolidated
in [`HYPERLIQUID_LIVE_CONNECTOR.md`](HYPERLIQUID_LIVE_CONNECTOR.md).

For future connector development, copy the tracked dummy credential template and
edit only the ignored copy:

```powershell
Copy-Item rust_live/hyperliquid.env.example rust_live/hyperliquid.env
```

The template deliberately contains an invalid all-zero agent key. The current
`live` command still refuses before reading this file.

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
```

Linux shells use the same arguments without PowerShell backticks.

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
