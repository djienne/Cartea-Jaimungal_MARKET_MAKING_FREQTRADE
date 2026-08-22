# Validation status

Validated locally on 2026-08-22. The stateful continuous `live` backend is
implemented but the tracked CASHCAT profile ships with `live.enabled=false`.
An explicitly authorized acceptance campaign exercised minimum-notional real
actions on a dedicated CASHCAT subaccount and ended flat with zero open orders.

## Hot/cold-path hardening

- The optimized workspace and all-feature suite cover allocation, Loom
  interleavings, contention, properties, mock WebSockets, persistence migration,
  lifecycle reordering, and Python parity.
- `cj-core` HJB and quote-policy line coverage is above 92%; calibration is
  above 91%, dry-run execution above 88%, latency aggregation above 92%, live
  state above 87%, signing above 89%, and public wire decoding above 90%.
- The all-feature workspace line total is about 52%. This lower aggregate is
  reported rather than hidden: real-account acceptance, CLI orchestration, and
  much of the network-backed live backend are intentionally not executed by
  normal CI. CI hard-gates the pure core, dry-run execution, and runtime crates
  while publishing the complete connector report for review.
- An in-process live lifecycle exercises fill-before-ack, duplicate fill, stale
  acknowledgement, partial-fill cancellation, foreign-fill refusal, action
  refusal, and ambiguous-write reconciliation without touching a real account.
- Final portable and host-native O3/fat-LTO Docker images built successfully.
  A 15-second credential-free public-feed dry-run in the portable image exited
  successfully with `scientifically_valid=true`; no order channel was opened.

## Current data and replay status

Checked on 2026-08-22 around 17:30 UTC:

- `hl-cashcat-collector` had run for about 28 hours and was still writing
  non-empty price, trade, and 20-level order-book Parquet shards every ten
  seconds. Its running counters were about 180,314 BBO updates, 38,931 trades,
  and 18,596 order-book updates. The multi-day CASHCAT store reaches back to
  2026-08-16/17 and remains the replay source.
- The collector experienced repeated DNS/WebSocket interruptions but recovered.
  The current 120-minute estimator correctly excluded about 331 seconds of
  outage and still returned `status=ok` from 11,409 mids and 4,046 trade prints,
  with zero unreadable shards, R-squared about 0.951/0.953, and toxicity about
  0.20/0.33. Collection is usable, but the network interruptions remain visible
  evidence rather than being hidden.
- The latest staged Python sweep is a completed 95.23-hour pinned-tape result,
  not a continuously updating job. Its selected configuration earned +7.55 USDC
  on train but lost 10.24 USDC held out; only 5/16 six-hour windows were
  positive. The infrastructure ladder was +23.07 at 50 ms/100 ms refresh,
  -10.24 at 100/250 ms, -16.49 at 200/500 ms, -49.19 at 500/1000 ms, and
  -148.60 at 500 ms/30 s. Latency and refresh move together, so these are
  machine scenarios rather than a pure latency-only causal sweep.
- That sweep tape ends at 2026-08-20T21:11:41Z and therefore does not contain
  the newest August 22 volatility. Two abandoned August 17 sweep processes and
  their scratch trees were removed after confirming no CPU or file progress;
  completed reports were preserved.
- The newer Rust two-hour replay (`replay-1787353244623.json`) is scientifically
  valid but also negative: 112 fills, -17.43 USDC mark-to-market, and markout
  moving from +2.22 USDC at 100 ms to -10.79/-17.96/-18.98 at 1/5/30 seconds.
- The concurrently running Freqtrade processes are dry-run forward tests, not
  historical backtests. Since their current restart, the long database records
  seven closed trades for -241.09 USDC; the short database records 30 closed
  trades for -66.27 USDC plus an open simulated short carrying -230.85 USDC of
  realized partial-exit loss. These dry-run fills include an extreme CASHCAT
  price dislocation and are operational warnings, not substitutes for the queue
  replay.

## Deterministic gates

- `cargo fmt --check`: passed.
- strict Clippy over all targets with warnings denied: passed.
- Rust tests: 110 passed across optimized unit and integration targets.
- Python/Freqtrade reference tests: 722 passed.
- locked optimized builds of the production and feature-gated acceptance
  binaries: passed.
- Docker image build with the bounded context: passed; containerized metadata
  validation resolved CASHCAT correctly, and disabled `live` refused before
  credential or network access.
- the 30-page mathematical walkthrough rebuilt successfully after removing the
  obsolete smoothing description.
- source-hygiene gate: passed; the Rust model contains only the selected
  Cartea–Jaimungal policy.
- live-command safety gate: passed; disabled live exits before credentials or
  order transport, while production-mode smoke submitted zero actions during
  mandatory latency warm-up.
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
- The post-live-backend release-image smoke (`dry_run-1787406166955.json`) also
  exited 0 scientifically valid: 98 messages, 59 BBO, eight trades, 14 L2
  books, 14/14 heartbeat replies, two calibrations, and zero reconnects,
  causal drops, or calibration failures. Dispatch p99 was 0.078 ms,
  hot-decision p99 25.861 us, and public WebSocket RTT p99 529.733 ms.
- The final rebuilt portable image repeated a bounded dry-run in
  `dry_run-1787407198386.json`: scientifically valid with clean shutdown and no
  reconnect or causal-drop warning; it processed 36 messages and completed 6/6
  heartbeat RTTs.
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
- The stateful live-backend campaign began at 299.782699 USDC and ended flat at
  299.480038 USDC with zero open or unresolved orders. Filled turnover was
  41.0306 USDC and campaign realized PnL after fees was -0.302661 USDC, inside
  the authorized 60-USDC/0.5-USDC limits.
- The real WebSocket action path set and verified strict-isolated leverage 2x,
  rested a two-sided 20.32891-USDC ALO batch, canceled one leg by CLOID and one
  by OID, and proved a crossing ALO is rejected without a fill.
- A deliberately discarded post response forced a socket reconnect. Durable
  state plus CLOID/open-order/historical-order reconciliation recovered and
  canceled the one original order without submitting a duplicate. A separate
  hard process exit left one resting ALO; the next process recovered the same
  CLOID/OID and canceled it.
- A genuine maker bid filled 86 CASHCAT at 0.11791 with `crossed=false` and a
  0.001521-USDC fee. The reduce-only IOC close sold all 86 at 0.11843; once the
  close command was running, authoritative flat confirmation took 1.58 seconds.
- A separate IOC sold 86 at 0.11845, the process exited with the short persisted,
  and a new process used the explicit market-close path to buy all 86
  reduce-only at 0.12231. Authoritative flat confirmation took 1.79 seconds.
  The adverse move caused most of the campaign loss and is retained as real
  evidence rather than hidden.
- `scheduleCancel` was signed correctly but Hyperliquid refused it before any
  test order because the subaccount has about 1,930 USDC cumulative volume and
  the venue currently requires 1,000,000 USDC. Production with
  `deadman_enabled=true` therefore fails closed on this account; the option can
  be disabled explicitly, but no dead-man trigger is claimed as validated.
- A bounded continuous-production smoke ran the actual Cartea-Jaimungal hot
  loop, public/account sockets, background calibration, and live backend with
  the latency gate enforced. It submitted zero orders/cancels/dead-man actions,
  recorded three public and five account RTT samples, and remained in
  `warming_up` as designed.
- On this development machine, the read-only probe measured `/info` p99 around
  324 ms and WebSocket ping p99 around 561 ms; the final action canary measured
  submit acknowledgements around 959-976 ms. These exceed the configured 150 ms
  production p95 threshold, so an enforced production gate would refuse to
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
- The pre-hardening Linux verification measured 152.34 ns per decision and a
  0.322 ms 600-step HJB solve. The post-hardening unpinned Windows run measured
  an 89.22 ns paired quote-loop median; see `PERFORMANCE.md` for the complete
  distributions and benchmark scope.

## Remaining boundary

Continuous live trading is implemented and selected by one central TOML flag,
which remains off in the tracked profile. This development machine is not a
production host: measured API/action latency exceeds the enforced 150-ms p95
limit. The current low-volume subaccount also cannot use Hyperliquid's dead-man
feature. Production activation therefore requires suitable infrastructure and
either an eligible account with dead-man enabled or an explicit decision to run
without it.
