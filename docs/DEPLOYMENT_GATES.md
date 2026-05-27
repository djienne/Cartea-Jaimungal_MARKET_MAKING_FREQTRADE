# Deployment Gates

This project must stay fail-closed until every automated and manual gate below
has evidence. The local checks can be run with:

```bash
python scripts/run_safety_gates.py --markdown-output docs/LAST_SAFETY_GATES.md
```

When Docker is available, include non-trading runtime load checks:

```bash
python scripts/run_safety_gates.py --include-runtime --markdown-output docs/LAST_SAFETY_GATES.md
```

For promotion evidence from retained dry-run, testnet, or canary audit logs,
pass the log explicitly:

```bash
python scripts/run_safety_gates.py --include-runtime --audit-log-input docs/testnet_mm_debug.jsonl --markdown-output docs/LAST_SAFETY_GATES.md
```

Promotion runs can also tighten the accepted evidence window from the default
86400 seconds:

```bash
python scripts/run_safety_gates.py --include-runtime --audit-log-input docs/testnet_mm_debug.jsonl --max-evidence-age-seconds 3600 --max-canary-event-age-seconds 86400 --markdown-output docs/LAST_SAFETY_GATES.md
```

Non-dry-run strategy enablement is also guarded at runtime. Setting
`market_making.trading_enabled=true` in live mode is rejected unless
`market_making.deployment_stage` is explicitly set:

- `canary`: requires `post_only_evidence_report`, `fee_evidence_report`, and
  `replay_acceptance_report` to be `ok=true`, plus
  `market_making.manual_monitoring_ack=true`.
- `production`: requires all canary prerequisites and
  `live_canary_report` to be `ok=true`.

Every required report must include a fresh `generated_at` timestamp within
`market_making.max_deployment_report_age_seconds` seconds, defaulting to 86400.
The same deployment-gate state is rechecked during quote validation so a runtime
toggle cannot bypass startup gating.

## Automated Local Gates

These gates do not require a live exchange connection:

- `compileall`: Python syntax/import bytecode checks for `scripts` and
  `user_data/strategies`.
- `pytest_core`: HJB math, strategy guard, parameter writer, and replay smoke
  tests.
- `config_safety_report`: writes `docs/config_safety_report.json` and fails if
  the checked-in config drifts away from Phase 0 safety defaults: dry-run on,
  force-entry disabled, tiny fixed stake, capped tradable balance, aligned
  maker fee, limit orders, and no checked-in market-making live enablement.
- `strategy_safety_report`: writes `docs/strategy_safety_report.json` and fails
  if checked-in strategy defaults drift away from fail-closed behavior:
  `trading_enabled=false`, no `minimal_roi=-1` force exit, long-only research
  mode, positive inventory-risk parameters, limit passive orders, kill switches
  enabled, no internal estimator calls from strategy callbacks, and current
  callback surface present.
- `compute_spreads_boundary_smoke`: verifies disabled HJB boundary sides render
  as disabled instead of finite quotes.
- `replay_smoke`: verifies the replay CLI runs without candle-fill assumptions.
- `post_only_probe_plan`: documents the Hyperliquid `Alo` evidence required
  without placing orders.
- `post_only_evidence_report`: writes
  `docs/post_only_evidence_report.json`. Without submit artifacts this command
  is expected to return nonzero and the gate runner treats that as an artifact
  check, not as exchange proof. The report must become `ok=true` before live
  trading can be considered. The crossing/passive submit artifacts must also
  carry fresh `generated_at` timestamps, so regenerating a fresh report from
  stale exchange probes does not satisfy Gate 4. The safety runner passes
  `--max-evidence-age-seconds` through to the checker, accepts explicit
  `--post-only-crossing-result` and `--post-only-passive-result` artifact paths,
  and auto-detects `docs/post_only_crossing_result.json` plus
  `docs/post_only_passive_result.json` when those conventional files exist.
  Submit-mode CCXT probes enforce a default 25 USDC notional cap unless the
  operator explicitly raises `--max-notional-usdc`.
- `direct_alo_adapter_plan`: writes `docs/direct_alo_adapter_plan.json`,
  documenting the no-network direct Hyperliquid SDK fallback path that submits
  `order_type={"limit": {"tif": "Alo"}}` only after local BBO maker-safety.
  The direct adapter also applies a default 25 USDC submit-notional cap for
  evidence probes unless the operator explicitly raises `--max-notional-usdc`.
  Direct SDK orders can include quote-linked `cloid` evidence generated from
  `session_id`, `quote_id`, side, and HJB generation.

Optional Docker runtime gates:

- `docker_compose_config`: validates Compose configuration.
- `freqtrade_runtime_load`: runs `freqtrade list-strategies` inside the
  configured container and requires `Market_Making` to load with status `OK`.
- `freqtrade_callback_surface`: imports `Market_Making` inside the same
  Freqtrade container and writes `docs/freqtrade_callback_surface_report.json`.
  It verifies the runtime-visible callback signatures for custom pricing,
  custom stake sizing, confirmation, repricing, leverage, and fill callbacks,
  including `**kwargs` support for Freqtrade callback evolution.
- `freqtrade_tif_runtime`: writes `docs/freqtrade_tif_runtime_report.json`
  after probing the exact Freqtrade container with generated `GTC`, `PO`, and
  `Alo` configs using `freqtrade list-strategies`. This is a startup/config
  compatibility artifact only. It can show whether a post-only TIF config loads,
  but it still does not prove Hyperliquid native `Alo` order submission,
  resting behavior, maker fills, or fee evidence.
- `dry_run_disabled_smoke`: starts the configured bot briefly in dry-run with
  the default safety lock, stops it, stores logs, and fails if any order creation
  line appears.
- `dry_run_enabled_smoke`: starts the public collector, writes temporary
  schema-v2 OK parameter snapshots under `user_data/logs`, starts Freqtrade with
  `dry_run=true` and `trading_enabled=true` through a temporary config for a
  10-minute enabled window, and
  requires fresh collector data, fresh params, fresh HJB, an accepted quote
  decision, and dry-run order evidence. Checked-in config and checked-in params
  remain fail-closed. Strategy fill callbacks also schedule delayed
  `fill_markout` audit events at 100 ms, 1 s, 5 s, and 30 s after fills when
  fill price, size, and side are available. Quote, health, and fill logs include
  fee snapshots that compare the strategy maker fee with config, exchange
  metadata, and actual fill fee rates when available. Trading enablement and
  quote confirmation reject config fee mismatches; quote confirmation also
  rejects exchange maker-fee mismatches when exchange metadata is available.
  When post-only support is marked verified, final entry/exit confirmation also
  rejects non-post-only time-in-force values before submission, and any fill
  reported with a non-post-only or missing TIF triggers a kill switch. Verified
  post-only mode also requires fills to carry maker/taker liquidity; unknown
  liquidity triggers `unknown_fill_liquidity`. Repeated non-post-only TIF
  confirmation rejects count toward the post-only reject-rate kill switch.
  Realized-PnL fills emit `risk_update` audit events with daily drawdown and
  consecutive-loss counters before any drawdown or consecutive-loss kill switch
  fires.
- `dry_run_quality_report`: parses the enabled dry-run audit log and writes
  `docs/dry_run_quality_report.json`. It is stricter than the smoke gate: it
  requires at least 9 minutes of runtime, at least 5 minutes of audit-log span,
  at least two accepted model-valid quotes and quote-linked order attempts, or
  at least one accepted quote/order when that order gets a dry-run fill and the
  PnL/health checks remain bounded. It also requires quote distance/depth caps,
  order amount and notional caps, fresh health fields, no kill/error events, and
  bounded loss/final PnL.
  This is the automated answer to "did the dry run make reasonable quotes,
  reasonable trade amount, and avoid losing too much too quickly?" Passing this
  gate is still promotion evidence, not proof that live fills will remain maker
  or match dry-run behavior.
- Strategy risk guard defaults now include `max_notional_exposure_usdc`,
  `max_margin_used_usdc`, and `min_liquidation_buffer_usdc`. Quote validation
  and final order confirmation reject risk-increasing orders that would exceed
  those caps; live quoting fails closed if liquidation-buffer evidence is
  missing or too low. Risk-reducing asks are allowed so an over-limit long can
  still be unwound.
- `replay_latest_data_smoke`: after runtime collector gates have produced fresh
  shards, replays the newest local data window and writes
  `docs/replay_latest_smoke.json` with input coverage, maker/taker counts,
  post-only rejects, stale cancels, inventory, PnL, and markout samples where
  fills occur.
- `replay_acceptance_report_artifact`: runs baseline, fee, latency, and
  parameter-perturbation replay variants and writes
  `docs/replay_acceptance_report.json` plus a markdown summary. The gate command
  uses `--allow-incomplete` so the artifact is always produced; the report's own
  `ok` field remains false until the multi-day acceptance criteria are met. By
  default the safety runner caps this artifact to a short local data window; for
  a promotion run, pass `--replay-acceptance-newest-per-stream 0`,
  `--replay-acceptance-max-price-events 0`, tune
  `--replay-acceptance-min-price-events-per-day` and
  `--replay-acceptance-max-price-gap-seconds` if the source cadence differs,
  and optionally `--replay-acceptance-require-pass` so the same runner evaluates
  the full multi-day dataset instead of overwriting it with the smoke-sized
  report. The report includes price-event density and maximum inter-event gap
  checks so isolated start/end timestamps cannot satisfy multi-day coverage. It
  also includes a directional-drift ratio so mark-to-market PnL that is not
  explained by net realized spread cannot masquerade as maker edge. Replay
  quotes and fill accounting both use the configured maker fee, so fee
  sensitivity changes quote distances as well as realized fees. The replay
  queue model includes queue-ahead traded volume plus conservative cancellation
  decay, reported as `queue_decay_base`. The report also includes refusal checks
  for bad parameters and stale collector data, proving those scenarios reject
  quoting instead of silently producing orders. Replay metrics also track
  starting equity, leverage, notional exposure, margin used, maintenance margin,
  equity, liquidation buffer, and maintenance-margin breach counts; acceptance
  fails any variant with a maintenance-margin breach.
- `replay_log_calibration_artifact`: parses the JSONL audit log from dry-run or
  testnet, matches fills back to accepted quotes by quote id when available and
  otherwise by pair/side/price/time window, and writes
  `docs/replay_log_calibration.json` with observed fill
  probabilities by side and depth bucket, maker ratio, fee-rate summary, and
  markout summary. The artifact may be `usable_for_calibration=false` when the
  sample is too small; that is expected until enough dry-run/testnet fills
  exist. Use the safety runner's `--audit-log-input` option to evaluate a
  retained testnet or canary log instead of the default
  `user_data/logs/mm_debug.jsonl`.
- `fee_evidence_report`: parses the same audit log and writes
  `docs/fee_evidence_report.json`. The report requires strategy/config fee
  agreement, an exchange/account maker-fee snapshot, and at least one maker fill
  whose actual fee rate matches the configured maker fee. Fee snapshots and
  actual maker-fill fee records must carry fresh event timestamps; regenerating
  a report from stale fee-tier or fill logs does not satisfy the gate. It is
  expected to be `ok=false` until testnet/tiny integration produces real
  fee-tier and fill-fee evidence. Use `--audit-log-input` when those records are
  in a retained integration log rather than the default debug log, and
  `--max-evidence-age-seconds` to tighten the accepted event/report age.
- `live_canary_evidence_report`: parses the audit log and prior gate artifacts,
  then writes `docs/live_canary_report.json`. It is expected to be `ok=false`
  until post-only, fee-tier, and multi-day replay gates are already `ok=true`
  and several tiny live sessions provide non-dry-run health, fresh accepted
  quotes, final accepted post-only order attempts with quote IDs, maker-only
  fill evidence with quote IDs, no parameter/HJB/collector errors, no kill
  switches, and explicit manual-monitoring acknowledgement evidence. The
  acknowledgement requires both the `--manual-monitoring-ack` operator flag and
  a fresh audit-log event named `manual_monitoring_ack` or
  `canary_manual_monitoring_ack` with `acknowledged=true`; the CLI flag alone
  cannot satisfy the gate. Prior gate reports must carry fresh `generated_at`
  timestamps, and the canary session events themselves must be recent, so a
  newly generated report cannot reuse old live evidence. Future-dated canary
  events are rejected instead of being treated as fresh. When
  `run_safety_gates.py --include-runtime` is used, this gate
  runs after fee and replay artifacts are regenerated so it evaluates the
  current dependency reports. Use `--audit-log-input` together with
  `--manual-monitoring-ack` after actual monitored live-canary sessions, plus
  `--max-evidence-age-seconds` to tighten dependency report freshness and
  `--max-canary-event-age-seconds` to tighten canary audit event freshness.
- `hl_data_validation_report`: reads the newest collector Parquet shards and
  validates required streams/columns plus the actual `timestamp` values inside
  the files. Freshness is based on row timestamps, not file modification time,
  so copied stale files cannot satisfy the collector-fresh gate.

Parameter sidecar:

- `param-estimator` is a separate Docker Compose service that runs
  `periodic_test_runner.py --loop` against the mounted collector data and
  atomically updates the strategy snapshots. The strategy itself only consumes
  snapshots; checked-in config keeps `run_estimators_in_strategy=false`, and the
  strategy safety report rejects any reintroduced `schedule_tests()` call.
- `param_update.lock` remains a fail-closed guard. The sidecar may replace a
  stale lock after `PARAM_UPDATE_LOCK_STALE_SECONDS`; the strategy does not
  remove locks and rejects active, stale, unreadable, or future-dated locks
  until a valid status/snapshot set is present.

Data freshness/replay-readiness report:

```bash
python scripts/validate_hl_data.py --symbol ETH --newest-per-stream 25 --max-age-seconds 30 --output docs/hl_data_validation.json --fail-on-bad-data
```

The freshness report is intentionally separate from the default pass/fail gate:
historical replay can use older data, but live/dry-run quoting must refuse stale
collector data.

Passing these gates means the code is locally coherent. It does not mean the bot
is safe to trade.

## Manual Runtime Gates

These gates require the real Freqtrade/Hyperliquid runtime:

- `deterministic_dry_run_trading_disabled`: covered by
  `dry_run_disabled_smoke` when Docker is available. It proves the bot starts,
  sees fresh collector data, writes health logs, and creates zero orders while
  the default lock is active.
- `dry_run_order_creation`: covered by `dry_run_enabled_smoke` when Docker is
  available. It proves a temporary unlocked dry-run can accept a valid passive
  quote and create a dry-run order, while checked-in config/params remain
  locked.
- `hyperliquid_post_only_mapping`: Freqtrade `PO` is currently not verified for
  Hyperliquid live execution. Current local runtime evidence shows that
  Freqtrade 2025.4 accepts generated `GTC` and `PO` configs at startup, while
  native `Alo` is rejected by Freqtrade config validation. The checked-in
  dry-run harness still uses `GTC` for research startup compatibility and live
  trading remains blocked. Even if `post_only_verified=true` is supplied, live
  strategy enablement also rejects unless configured entry and exit TIF
  canonicalize to post-only/Alo. See `docs/POST_ONLY_VERIFICATION.md`.
  The `freqtrade_tif_runtime` artifact records current container-level config
  acceptance for `GTC`, `PO`, and `Alo`, but exchange submit artifacts are still
  required before this gate can pass. Startup acceptance of `PO` is not proof
  that Hyperliquid receives native `Alo`.
  Prove a native `Alo` path before live use; intentionally crossing `Alo` orders
  must reject or cancel without filling, passive `Alo` orders must rest or fill
  maker-only, and submit artifacts must confirm the actual order TIF as `Alo`
  separately from the requested params. A zero-filled passive cancel without
  resting proof does not satisfy the gate.
  Start with the no-network probe plan:

```bash
python scripts/verify_post_only_mapping.py --mode plan
```

  Any submit-mode use must be run only on testnet or tiny canary size with
  explicit acknowledgement flags and retained exchange logs. The canonical
  checker is:

```bash
python scripts/verify_post_only_mapping.py --mode evaluate-evidence --crossing-result docs/post_only_crossing_result.json --passive-result docs/post_only_passive_result.json --output docs/post_only_evidence_report.json
```

  The default maximum age for those submit artifacts is 86400 seconds. Use
  `--max-evidence-age-seconds` to make the evidence window stricter.

  If Freqtrade remains unable to submit native `Alo`, the direct SDK fallback
  scaffold is:

```bash
python scripts/hyperliquid_alo_executor.py --mode plan
```

  Direct SDK submit artifacts from `hyperliquid_alo_executor.py` can be fed to
  the same `verify_post_only_mapping.py --mode evaluate-evidence` checker. The
  checker normalizes SDK `order_type={"limit": {"tif": "Alo"}}` and evaluates
  the adapter's resting/rejected/fill classification. Use
  `hyperliquid_alo_executor.py --mode submit-crossing-alo` for a direct SDK
  rejection probe and `--mode submit-passive-alo` for passive resting/maker
  evidence with automatic cancellation of any resting order ids returned by the
  SDK. These submit modes enforce a default 25 USDC notional cap. Plain
  `--mode submit-alo` is reserved for the future direct execution layer and may
  leave a passive order working. For evidence tied to strategy decisions, pass
  `--quote-id`, `--session-id`, and `--hjb-generation`; the artifact stores the
  readable client order id and the submitted Hyperliquid `cloid`.
- `hyperliquid_risk_flatten`: maker quoting and risk reduction are separated.
  The no-network plan is:

```bash
python scripts/hyperliquid_risk_executor.py --mode plan
```

  The executor builds only reduce-only IOC flatten orders. Positive signed base
  exposure maps to a sell reduce-only IOC; negative signed base exposure maps to
  a buy reduce-only IOC. Submit mode requires
  `HYPERLIQUID_RISK_FLATTEN_ALLOW=1`, `--acknowledge-risk-reducing-taker`, a
  non-empty `--reason`, a non-zero `--signed-position-base`, a positive
  `--reference-price`, and testnet unless `--allow-mainnet-flatten` is supplied.
  This path is for emergency inventory reduction after quoting is disabled and
  open maker orders have been cancelled; it is not maker-PnL evidence.
  Strategy kill switches emit `risk_flatten_requested` audit events when a
  non-zero signed base position remains visible, including the reduce-only IOC
  side/size, reference price, risk action id, readable client order id, and
  deterministic Hyperliquid `cloid` to use with this executor.
- `multi_day_event_replay`: the automated latest-data smoke proves the replay
  parser and conservative fill loop work on real shards, and the acceptance
  report records exactly which criteria are still failing. The remaining manual
  gate is a several-day replay with acceptable realized spread, markouts, maker
  ratio, inventory, latency, fee sensitivity, parameter perturbation, and
  directional-drift attribution.
- `hyperliquid_fee_tier`: provide exchange/account fee snapshots and maker fill
  logs, then run `python scripts/verify_fee_evidence.py --input user_data/logs/mm_debug.jsonl --output docs/fee_evidence_report.json`.
  This report must be `ok=true` before canary: config/strategy fee agreement is
  not enough without exchange/account fee and actual maker fill-fee evidence.
  The default maximum event age for fee snapshots and actual maker-fill fee
  records is 86400 seconds; use `--max-evidence-age-seconds` to make the
  evidence window stricter. The full safety runner can evaluate a retained
  integration log with `--audit-log-input`.
- `live_canary`: only after all previous gates pass, with tiny fixed stake,
  one symbol, hard loss limits, post-only required, kill-on-taker-fill enabled,
  and manual monitoring. After the sessions, run:

```bash
python scripts/verify_live_canary.py --input user_data/logs/mm_debug.jsonl --manual-monitoring-ack --output docs/live_canary_report.json
```

  Or run the full gate bundle against a retained canary log:

```bash
python scripts/run_safety_gates.py --include-runtime --audit-log-input docs/live_canary_mm_debug.jsonl --manual-monitoring-ack --markdown-output docs/LAST_SAFETY_GATES.md
```

  The report must be `ok=true` before any larger deployment. It checks the prior
  post-only, fee, and replay artifacts, then rejects taker fills, unknown fill
  liquidity, kill switches, stale accepted quotes, accepted order attempts that
  are not live/post-only/quote-linked, fills missing quote IDs, fills that do
  not reconcile to a prior accepted order attempt, parameter/HJB/collector
  error events, excessive stake, excessive symbols, missing live health, and
  missing or stale manual-monitoring acknowledgement evidence in the audit log.
  Record the required acknowledgement event with:

```bash
python scripts/record_manual_monitoring_ack.py --output user_data/logs/mm_debug.jsonl --session-id <CANARY_SESSION_ID> --operator <NAME> --acknowledge-risk
```

  The default dependency-report age is 86400 seconds, and the default live
  canary event age is 604800 seconds; tighten those windows with
  `--max-dependency-report-age-seconds` and `--max-canary-event-age-seconds`
  when promoting a canary run.

Full `run_safety_gates.py --include-runtime` runs automatically verify the
human-readable PLAN status against the latest machine-readable gate evidence and
write `docs/plan_status_audit.json`. To run that audit directly:

```bash
python scripts/verify_plan_status.py --status docs/PLAN_IMPLEMENTATION_STATUS.md --gates docs/last_safety_gates.json --output docs/plan_status_audit.json
```

## Current Safety Posture

- The strategy default is `trading_enabled = False`.
- Checked-in parameter snapshots are schema v2 but have
  `status = "seeded_unverified"`, so the strategy rejects them.
- A real estimator run must atomically replace the snapshots with
  `status = "ok"`, current timestamps, and sufficient diagnostics before any
  quote can pass the guards.
- Live trading is still blocked unless post-only support is verified outside
  dry-run.
- In live mode, the strategy also refuses enablement unless the declared
  deployment stage has passing gate artifact reports. Canary mode requires
  post-only, fee, replay, and manual-monitoring evidence; production additionally
  requires passing live canary evidence. Passing reports must be fresh; stale or
  timestamp-less reports do not unlock live trading.
