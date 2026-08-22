# Validation status

Validated locally on 2026-08-22. The persistent `live` strategy backend remains
disabled. A separately guarded pure-Rust connector canary was explicitly
authorized and exercised minimum-notional real actions on a dedicated CASHCAT
subaccount; all canaries ended flat with zero open orders.

## Deterministic gates

- `cargo fmt --check`: passed.
- strict Clippy over all targets with warnings denied: passed.
- Rust tests: 62 passed across unit and integration targets.
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
- mock Hyperliquid WebSocket tests: public parsing, the eight account
  subscriptions, application ping/pong with measured RTT, protocol ping
  response, idle timeout, reconnect/resubscription, order/fill delivery, and
  mixed-age initial trade snapshots all passed.
- causal-ring saturation test: passed and invalidates the session.
- exact connector tests cover secret-safe four-key dotenv parsing, fixed-point
  wire numbers, CLOIDs, monotonic nonces, MessagePack bytes, vault/expiry action
  hashes, both-network EIP-712 digests, secp256k1 signatures, and known versus
  unknown outcomes. Golden values match the independent Python oracle fixtures.
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
- A 192.6-second containerized public-feed dry run
  (`dry_run-1787384508441.json`) remained scientifically valid through five
  background calibrations. It received 407 messages (261 BBO, 146 trade, 35 L2),
  completed 5/5 application heartbeats, and had zero reconnects, invalid
  messages, or causal drops. Moving the roughly 11-second desktop calibration
  job off the async event loop reduced market-event dispatch p99 to 0.21 ms and
  decision-to-backend-completion p99 to 0.71 ms during the observation.
- The final minimal release-image smoke (`dry_run-1787387382194.json`) ran 78.7
  seconds, processed 176 messages (117 BBO, 69 trade, 14 L2), completed three
  heartbeat RTT measurements and two calibrations, and exited 0 scientifically
  valid. It had zero reconnects, causal/latency drops, calibration failures, or
  observer errors. Dispatch p99 was 0.103 ms, hot-decision p99 24.019 us,
  decision-to-backend p99 0.487 ms, and WebSocket RTT p99 574.286 ms. The gate
  remained visible but non-enforcing (`not_enforced_in_mode`) as required for
  dry-run.
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
- The current pure-Rust read-only connector check used the ignored credential
  file without emitting its key. Hyperliquid reported a dedicated subaccount
  with 299.8 USDC, zero position, zero open orders, isolated CASHCAT leverage,
  and maker/taker rates 0.00015/0.00045. All seven `/info` calls succeeded. A
  65-second account WebSocket observation acknowledged all eight subscriptions,
  received 56 messages, completed 3/3 application heartbeats, and had no
  reconnect, timeout, parse error, or dropped event. The latency gate reported
  `enforced=false` as required for probes.
- The authorized passive canary placed 89 CASHCAT at 0.114 (10.146 USDC) as
  ALO, received resting OID 522935896688, canceled it by its own CLOID, and
  finished with no fill, unchanged equity, zero position, and zero open orders.
- The first IOC canary exposed an account-decimal compatibility bug: the venue
  reports integer CASHCAT size as `88.0`. The entry bought 88 at 0.11507; after
  fixing the exact parser, a reduce-only recovery sold all 88 at 0.11523. Fees
  were 0.009119 USDC, closed PnL was +0.01408 USDC, and net equity rose
  0.004961 USDC. The account was independently confirmed flat and empty. A
  regression test now pins redundant trailing-zero parsing.
- The corrected integrated round-trip then bought 88 at 0.11451 and closed all
  88 reduce-only at 0.11436 about 1.82 seconds later. It lost 0.022262 USDC
  including fees, received order and fill stream events, and again finished
  flat with zero open orders. This is connector validation, not evidence of
  strategy profitability.
- On this development machine, the read-only probe measured `/info` p99 around
  324 ms and WebSocket ping p99 around 561 ms; the final action canary measured
  submit acknowledgements around 959-976 ms. These exceed the configured 150 ms
  production p99 threshold, so an enforced production gate would refuse to
  quote. Probe, dry-run, and canary enforcement was intentionally bypassed.
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

`live` is still an intentional unavailable backend. Low-level authenticated
reads, private/account WebSocket listening, signing, ALO/IOC placement, and
CLOID/OID cancellation are now pure Rust and empirically exercised. Production
activation still requires durable nonce and order state, fill/funding
deduplication, unknown-outcome restart reconciliation, rate limiting, and a
dead-man policy. Replay and dry-run remain the only continuous strategy modes.
