# Validation status

Validated locally on 2026-08-22. No real order path exists and no real order was
submitted.

## Deterministic gates

- `cargo fmt --check`: passed.
- strict Clippy over all targets with warnings denied: passed.
- Rust tests: 42 passed across unit and integration targets.
- Python/Freqtrade reference tests: 722 passed.
- locked optimized build of both binaries: passed.
- Docker image build with the bounded context: passed; containerized metadata
  validation resolved CASHCAT correctly, and the containerized `live` command
  retained the intentional refusal boundary.
- the 30-page mathematical walkthrough rebuilt successfully after removing the
  obsolete smoothing description.
- source-hygiene gate: passed; the Rust model contains only the selected
  Cartea–Jaimungal policy.
- live-command safety gate: passed; it exits before credentials or order
  transport.
- mock Hyperliquid WebSocket tests: public parsing, application ping/pong,
  protocol ping response, idle timeout, reconnect/resubscription, and mixed-age
  initial trade snapshots all passed.
- causal-ring saturation test: passed and invalidates the session.
- independent offline connector evidence: 17 Rust signing/wire tests, nine Rust
  execution-envelope/response tests, and 12 JavaScript conformance tests passed
  with throwaway keys and mocked transports. None submitted an action.
- runtime purity gate: the release image contains no Python, CCXT, credential
  file, or key material. Python and JavaScript were used only as test oracles.

## Python parity

The deterministic oracle test covers all three scientific layers:

1. schema-v4 direct-window kappa, lambda, epsilon, variance, counts, fit points and R-squared;
2. asymmetric backward-Euler HJB solving plus time/inventory interpolation;
3. maker-fee spread assembly, bps clamps and final venue-rounded prices.

Ordinary parameter/HJB comparisons use `1e-8`, high-sensitivity HJB points use
`1e-7`, and rounded quotes must be exactly equal. All passed.

## Production-path observations

- Current Hyperliquid metadata dynamically resolved CASHCAT as a perp with
  integer size units, six maximum price decimals, 3x maximum leverage,
  `onlyIsolated=true`, and margin table 3.
- The EMA-free bounded public-feed canary
  `dry_run-1787346470124.json` connected successfully and received 13 BBO, six
  trade, and 12 L2 events. It made 573 quote decisions with zero invalid
  messages, reconnects, causal drops, or risk refusals. The market-event ring
  high-water mark was 6/65,536, maximum observed hot-path latency was 30.421 us,
  and the report was scientifically valid.
- Docker `SIGTERM` handling was exercised separately. The unbounded process
  withdrew virtual orders, wrote `dry_run-1787346558316.json`, and exited 0
  without an OOM kill; the earlier forced exit-137 behavior is fixed.
- During the recorded persistent observation, the dry-run published 19
  consecutive schema-v4 `direct_window_v1` calibrations at the 30-second cadence
  while processing 656 BBO, 186 trade, and 110 L2 events and logging 442 quote
  publications. The public-feed log contained no reconnect or warning. Parameter
  movement between revisions is the direct response to the moving 120-minute
  observation window, not an update filter.
- A later 292.1-second corrected public-feed observation
  (`dry_run-1787353320541.json`) produced nine calibrations, 324 BBO updates, 182
  live trades, 54 L2 books, 2,953 quote decisions, and four virtual fills (three
  partial), with zero reconnects, invalid messages, or causal drops. It shut down
  cleanly and was scientifically valid.
- The first heartbeat canary exposed a false reconnect caused by a mixed-age
  initial `trades` snapshot: its 30 rows spanned up to about eight seconds and
  were not ordered in the way the row-level transition assumed. A frame-level
  startup transition plus regression test fixed it. The final bounded canary
  (`dry_run-1787353638368.json`) then recorded three application pings and three
  pongs, no idle timeout, reconnect, invalid message, or causal drop, and was
  scientifically valid.
- Read-only authenticated-account validation used the ignored credential file
  without emitting its key: Hyperliquid identified the wallet as a subaccount,
  the derived signer as its approved agent, and both mapped to the same master.
  All required `/info` schemas returned successfully. A 65-second account
  WebSocket observation acknowledged all six subscriptions and answered all
  four application heartbeats, with no parse errors and a maximum inbound gap
  of 6.076 seconds. No signed action, order, modify, cancel, leverage change, or
  dead-man action was sent.
- The already-running Python estimator and both Freqtrade dry-run consumers were
  restarted so their in-memory schema guards also use v4. The estimator now
  copies and publishes every cycle successfully; both consumers accepted the
  refreshed direct parameters and rebuilt their episodic HJB surfaces. Primary
  kappa, lambda, and epsilon values equal their retained `*_raw` diagnostic
  aliases exactly.
- Starting Rust as a Parquet writer while the reference collector was active was
  refused during the observation preflight, before any competing shard was
  written.
- The fresh direct-window 120-minute CASHCAT replay
  `replay-1787353244623.json` completed with 11,060 mids, 4,638 trades, 112 fills
  (74 partial), and no liquidation-buffer breach. It finished at -17.4315 USDC
  mark-to-market PnL. Markout was +2.2175 USDC at 100 ms but -10.7904, -17.9604,
  and -18.9831 USDC at 1 s, 5 s, and 30 s: direct adverse-selection evidence.
  The loss is retained; validation does not require profitability.
- The quote kernel measured 152.34 ns per decision and a 600-step HJB solve
  measured 0.322 ms in the current Linux verification container. See
  `PERFORMANCE.md` for benchmark scope.

## Remaining boundary

`live` is an intentional unavailable backend. Public metadata/market data,
instrument-neutral order intents, execution events, account-state interfaces,
and post-only semantics are present so a future authenticated adapter can be
added and validated independently. This release is for replay and dry-run only.
