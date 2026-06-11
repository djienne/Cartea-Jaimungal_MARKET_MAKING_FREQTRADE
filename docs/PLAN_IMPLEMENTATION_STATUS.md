# PLAN Implementation Status

Generated from the current local worktree after the latest safety-gate run.

## Automated Evidence

- Unit/integration tests: `python -m pytest tests` passed with 424 tests.
- Runtime gate runner:
  `python scripts/run_safety_gates.py --include-runtime --markdown-output docs/LAST_SAFETY_GATES.md --json-output docs/last_safety_gates.json`
  passed all automated checks. The JSON payload now separates
  `all_automated_passed` from `deployment_ready`, with
  `deployment_ready=false` and named `deployment_blockers` until the external
  Hyperliquid, multi-day replay, fee, and live-canary gates have fresh evidence.
  The audit requires the runtime load gate `freqtrade_runtime_load` plus locked
  and enabled dry-run smoke gates, so Gates 2-3 cannot disappear silently.
  The remaining manual gates are now evidence-aware: each external gate stays
  in `deployment_blockers` unless its corresponding report artifact is `ok=true`
  and carries a fresh `generated_at` timestamp within the deployment report-age
  limit.
- PLAN status audit:
  `scripts/verify_plan_status.py` writes `docs/plan_status_audit.json` and
  fails if this status document's test count, required local gate list, or
  remaining external-gate text drifts away from the latest safety-gate evidence.
  Full `run_safety_gates.py --include-runtime` runs execute this audit as a
  post-run check.
- Checked-in config safety:
  `scripts/verify_config_safety.py` writes `docs/config_safety_report.json` and
  fails if the repository config reintroduces live mode, force-entry, unlimited
  stake, oversized test exposure, fee mismatch, non-limit passive order types,
  or checked-in market-making live enablement.
- Checked-in strategy safety:
  `scripts/verify_strategy_safety.py` writes `docs/strategy_safety_report.json`
  and fails if strategy defaults reintroduce live enablement, `minimal_roi=-1`,
  disabled inventory-risk penalties, shorting, non-limit passive order types,
  limit-only emergency exits, missing kill switches, missing final tick-price
  guards, or callback surface regressions. The callback surface now includes
  the Freqtrade-stable `custom_stake_amount(..., leverage, entry_tag, side, ...)`
  signature plus an explicit 1x `leverage()` callback. Final tick-price guard
  detection is AST-based, so comments or unrelated source text cannot satisfy
  the check.
- Runtime callback surface:
  `freqtrade_callback_surface` runs
  `scripts/verify_freqtrade_callback_surface.py` and writes
  `docs/freqtrade_callback_surface_report.json` from inside the
  `freqtradeorg/freqtrade:2025.4` Docker service. It imports the actual mounted
  strategy module and verifies the callback signatures Freqtrade sees for
  `custom_entry_price`, `custom_exit_price`, `custom_stake_amount`,
  `confirm_trade_entry`, `confirm_trade_exit`, `adjust_entry_price`,
  `adjust_exit_price`, `leverage`, and `order_filled`, including `**kwargs`
  support for forward-compatible callback changes.
- Runtime TIF compatibility:
  `freqtrade_tif_runtime` runs `scripts/verify_freqtrade_tif_runtime.py` and
  writes `docs/freqtrade_tif_runtime_report.json` after probing generated
  `GTC`, `PO`, and `Alo` configs with `freqtrade list-strategies` in the exact
  Docker runtime. This artifact records whether the container accepts post-only
  config values, but it is deliberately not deployment proof: even an accepted
  config still requires exchange submit evidence that Hyperliquid receives
  native `Alo`, crossing orders reject/cancel without fills, passive orders rest
  or fill maker-only, and actual fill fees/liquidity are logged.
- Post-only evidence harness:
  `scripts/verify_post_only_mapping.py` now writes a plan artifact and evaluates
  crossing/passive `Alo` submit artifacts. The current
  `docs/post_only_evidence_report.json` is expected to be `ok=false` until
  real exchange/testnet artifacts are supplied; the local safety gate treats
  both incomplete and passing reports as artifact checks, while deployment
  readiness still depends on the report's `ok` field. The checker now also
  requires fresh `generated_at` timestamps on the underlying crossing/passive
  submit artifacts so stale probes cannot unlock Gate 4. It also requires
  actual exchange/order TIF confirmation as `Alo`, separate from submitted
  params, and passive probes must prove a resting order or maker fill rather
  than only a zero-fill cancel. CCXT submit-mode probes also enforce a default
  25 USDC notional cap unless the operator explicitly raises the limit.
- Direct `Alo` fallback scaffold:
  `scripts/hyperliquid_alo_executor.py` builds Hyperliquid SDK orders with
  `order_type={"limit": {"tif": "Alo"}}`, applies local maker-safety from BBO,
  applies a default 25 USDC submit-notional cap for evidence probes, classifies
  SDK responses, and refuses submit mode without explicit environment/CLI
  acknowledgements. Direct SDK submit artifacts are normalized by
  `scripts/verify_post_only_mapping.py --mode evaluate-evidence`, so direct
  execution evidence and CCXT/Freqtrade probe evidence use the same Gate 4
  checker. The direct adapter now has a separate guarded
  `submit-crossing-alo` mode for ALO rejection evidence and a guarded
  `submit-passive-alo` evidence mode that cancels any resting order ids after
  classification. Regular `submit-alo` remains locally maker-safe for the future
  direct execution layer. The adapter accepts `--quote-id`, `--session-id`, and
  `--hjb-generation`, stores a readable client order id in the artifact, and
  submits a deterministic Hyperliquid `cloid` derived from those fields for
  future quote-to-fill reconciliation. Direct submit artifacts now also carry
  `actual_time_in_force="Alo"` with source
  `hyperliquid_sdk_order_type`, so the post-only evidence checker can evaluate
  generated SDK artifacts without manual edits. The direct adapter now accepts
  `--price-tick-size` and `--amount-step-size`; regular/passive maker orders
  round bids down and asks up before the local maker-safety check, while
  intentionally crossing ALO probes round bids up and asks down so the rejection
  probe remains crossing after exchange-style tick rounding. The safety-gate runner can consume post-only
  crossing/passive evidence through explicit artifact path flags, and it also
  auto-detects the conventional `docs/post_only_crossing_result.json` and
  `docs/post_only_passive_result.json` paths when they exist.
- Hyperliquid risk flatten scaffold:
  `scripts/hyperliquid_risk_executor.py` writes
  `docs/hyperliquid_risk_flatten_plan.json` without network access and builds
  only reduce-only IOC flatten orders for emergency inventory reduction after
  maker quoting has been disabled and open orders have been cancelled. Positive
  signed base exposure maps to a sell reduce-only IOC, negative signed base
  exposure maps to a buy reduce-only IOC, submit mode requires explicit
  environment and CLI acknowledgements, and the generated order carries a
  readable risk client order id plus deterministic Hyperliquid `cloid`.
- Fresh collector data validation:
  `docs/hl_data_validation.json` validates required streams/columns and bases
  freshness on the Parquet row `timestamp` values, not file modification time,
  so copied stale files cannot satisfy the collector-fresh gate.
- Locked dry-run evidence:
  `docs/dry_run_disabled_gate.json` uses temporary ignored params/config and
  shows `trading_enabled=false`, `params_fresh=true`,
  `collector_fresh=true`, health logging, HJB freshness, and zero order
  creation.
- Enabled dry-run evidence:
  `docs/dry_run_enabled_gate.json` uses temporary ignored params/config and
  runs a 10-minute enabled dry-run window. It shows
  `trading_enabled=true`, `params_fresh=true`, `collector_fresh=true`,
  accepted passive bid quotes, dry-run order creation lines, and quote audit
  fields for parameter age, collector age, HJB age, orderbook age, and
  strategy/config/exchange fee agreement snapshots. Quote admission now
  rejects config or exchange maker-fee mismatches instead of merely logging
  them, and verified post-only mode rejects non-post-only TIF values before
  order submission. Live enablement also rejects verified post-only mode unless
  the configured entry and exit TIFs are actually post-only/Alo. If a fill is
  still reported with a non-post-only or missing TIF while post-only mode is
  asserted, the strategy triggers a kill switch.
  Verified post-only mode also requires fills to report maker/taker liquidity;
  unknown liquidity triggers `unknown_fill_liquidity` so live/testnet evidence
  cannot silently omit the field. The enabled dry-run gate now classifies
  Hyperliquid HTTP 429 startup failures as `exchange_rate_limited`, exits early
  when the container dies during startup, and retries before declaring the
  smoke failed. Repeated non-post-only TIF confirmation rejects also count
  toward the post-only reject-rate kill switch.
- Dry-run quality report:
  `scripts/verify_dry_run_quality.py` writes `docs/dry_run_quality_report.json`
  from the enabled dry-run gate and JSONL audit log. It requires at least 9
  minutes of runtime, at least 5 minutes of audit-log span, at least two
  accepted model-valid quotes and quote-linked order attempts, or at least one
  accepted quote/order when a dry-run fill occurs and the PnL/health checks stay
  bounded. It also requires quote distance and depth caps, order amount and
  notional caps, fresh health fields, no kill/error events, and bounded
  drawdown/final PnL plus bounded loss velocity in USDC/hour. This is a
  stricter dry-run promotion gate for checking whether quotes, trade amount,
  and losses look reasonable over time; it is not live post-only/fill proof by
  itself. `scripts/run_safety_gates.py` now accepts enabled dry-run duration and
  quality-threshold overrides, including an optional final-PnL floor. The report
  now carries a `quality_verdict` and conclusion that distinguish small-profit
  or break-even evidence with fills from bounded-loss/no-fill dry-run evidence.
  The promotion manifest includes a guarded 30-minute dry-run command for
  collecting stronger dry-run evidence with final total PnL required to be at
  least zero.
- Latest-data replay smoke:
  `docs/replay_latest_smoke.json` replays the newest local shards and records
  input coverage, post-only rejects, stale cancels, maker/taker counts,
  inventory, PnL, and markout samples where fills occur.
- Replay acceptance report:
  `docs/replay_acceptance_report.json` and
  `docs/replay_acceptance_report.md` run baseline, fee, latency, parameter
  perturbation, and widened-tick variants, including directional-drift
  attribution so PnL from one-way price movement is reported separately from net
  realized spread.
  Replay queueing includes traded-volume queue-ahead and conservative
  cancellation decay, with `queue_decay_base` reported in metrics.
  Replay quote generation and fill accounting both use the configured
  maker/taker fee schedule, so fee-sensitivity variants change quoted depth as
  well as fees paid. Replay now tracks starting equity, leverage, notional
  exposure, margin used, maintenance margin, equity, liquidation buffer, and
  maintenance-margin breach counts. Replay also records quote attempts by
  depth, fills by depth, and computed fill ratios by depth; the acceptance
  checker rejects inconsistent depth-count/ratio evidence. Replay side-level
  PnL now records side spread capture net of maker fees, and the acceptance
  checker rejects `pnl_by_side` totals that do not reconcile to net realized
  spread. Replay parameter and toxicity time series are timestamped with
  source/unit metadata, and the acceptance report rejects missing, untimestamped,
  or non-finite parameter/toxicity evidence. Replays with maker fills must
  include the PLAN-required markout horizons of 100 ms, 1 s, 5 s, and 30 s. The
  report rejects post-only reject ratios above the strategy kill-switch
  threshold and stale-quote cancel ratios above the replay acceptance threshold,
  reconciles those ratios back to raw quote counts, and rejects sparse or gappy
  price-event coverage so a multi-day replay cannot pass from isolated start/end
  timestamps. Replay quote placement now uses an explicit
  `quote_refresh_interval_ms` cadence instead of implicitly cancelling on every
  BBO event, records `quote_decision_events`, and consumes matched trade events
  once so overlapping simulated quote windows cannot count the same historical
  trade as multiple fills. Replay now supports exchange-style
  `price_tick_size` and `amount_step_size`: bids round down, asks round up, the
  maker-safety check runs after tick rounding, and fills use the rounded-down
  base amount so lot-size constraints cannot be silently ignored. Replay
  acceptance can now consume
  `docs/replay_log_calibration.json`, bucket simulated quotes with the same
  depth-bucket convention as the calibration artifact, and conservatively
  throttle potential fills to the observed side/depth fill probability. A
  promotion run can require this with
  `--replay-acceptance-require-fill-calibration`; unusable or missing
  calibration remains a blocking reason instead of silently becoming optimistic.
  The report also includes
  refusal checks proving bad
  parameters, stale collector data, and missing required collector streams
  reject quoting, and fails any replay variant that breaches maintenance
  margin. The safety-gate runner
  defaults to a short capped replay artifact, but exposes replay acceptance
  symbol, mid, shard cap, event cap, price-density threshold, max-gap threshold,
  and require-pass knobs so a promotion run can regenerate the same report from
  the full multi-day dataset. The current
  report correctly fails because the local data window is much shorter than the
  required multi-day coverage and has no maker fills.
- Replay log calibration:
  `scripts/calibrate_replay_from_logs.py` reads JSONL quote/fill/markout audit
  logs and writes `docs/replay_log_calibration.json` with accepted quote counts,
  fill probabilities by side/depth bucket, maker ratio, fee-rate summary, and
  markout summary. The runtime safety gate generates this artifact after the
  enabled dry-run smoke. The safety-gate runner accepts `--audit-log-input` so
  dry-run, testnet, or live-canary audit logs can be evaluated without copying
  them over the default debug log. Quote decisions now carry stable `quote_id`
  values, and the calibration report prefers exact quote-id matches when fills
  include them before falling back to time/side/price matching for older logs.
  It can be `usable_for_calibration=false` until enough real dry-run/testnet
  fills exist. When usable calibration evidence is supplied to replay
  acceptance, simulated fills are limited by the observed fill probability
  instead of assuming every queue-eligible historical trade would have filled.
- Quote admission and sizing hardening from `MEGA_PLAN.md`:
  `custom_entry_price()` and `custom_exit_price()` now log `quote_decision`
  `accept` only after `_quote_state_valid()`, custom-price distance, and local
  maker-safety checks pass. Rejections that return `proposed_rate` are cached so
  `confirm_trade_entry()` / `confirm_trade_exit()` still abort the order rather
  than allowing a silent proposed-rate fallback. `custom_stake_amount()` refuses
  to raise exposure to satisfy an exchange minimum when that minimum exceeds one
  inventory unit, refuses amounts that round below lot/min-stake constraints,
  and logs `stake_rejected` instead of silently increasing inventory risk.
  Parameter snapshots and deployment gate reports with future timestamps beyond
  the configured clock-skew window are rejected.
- Live exposure and liquidation-buffer guards from `MEGA_PLAN.md`:
  quote validation and final order confirmation now project the resulting base
  exposure and reject risk-increasing bids that would exceed
  `max_notional_exposure_usdc` or `max_margin_used_usdc`. Non-dry-run quoting
  also requires liquidation-buffer evidence and rejects missing or too-low
  buffers before order submission. Risk-reducing asks remain allowed so an
  over-limit position can still be unwound. Health logs include notional,
  margin, maintenance-margin, equity, and liquidation-buffer fields.
- Parameter sidecar posture from `MEGA_PLAN.md`:
  the strategy no longer imports or calls `schedule_tests()` from
  `bot_loop_start()`. It only reloads already-written atomic parameter
  snapshots on a configurable interval, validates them, and refreshes HJB from
  accepted snapshots. Estimator execution is moved to the separate
  `param-estimator` Docker Compose service, which runs
  `periodic_test_runner.py --loop`. `param_update.lock` now has a TTL for the
  sidecar, while the strategy treats active, stale, unreadable, or future-dated
  locks as fail-closed parameter states.
- Fee evidence report:
  `scripts/verify_fee_evidence.py` writes `docs/fee_evidence_report.json` from
  JSONL audit logs. It requires strategy/config fee agreement, exchange/account
  maker-fee evidence, and at least one maker fill with an actual fee rate that
  matches the configured maker fee. Maker-fill proof must also include quote
  side, limit order type, post-only/Alo TIF, expected fee rate, and actual fee
  paid. The checker now also requires fill price and amount, and reconciles
  `actual_fee_paid` against `price * amount * actual_fee_rate`. The underlying
  fee snapshot and actual maker-fill fee records must also have fresh event
  timestamps, so stale logs cannot satisfy the gate by regenerating a new
  report. The same
  `--audit-log-input` runner option feeds this report, so promotion evidence can
  come from retained testnet/tiny integration logs. The current report can be
  `ok=false` until testnet/tiny integration supplies real account fee and
  fill-fee evidence.
- Hyperliquid fee evidence capture:
  `scripts/capture_hyperliquid_fee_evidence.py` writes a read-only plan artifact
  at `docs/hyperliquid_fee_capture_plan.json` and can normalize saved
  Hyperliquid `userFees` / `userFills` payloads into the JSONL audit events
  consumed by `scripts/verify_fee_evidence.py`. Fill events only carry
  post-only TIF and limit-order proof when matching order-id evidence is
  supplied, so raw fills alone cannot satisfy the fee gate. When the matching
  direct ALO order artifact contains `quote_link`, the normalized fill now also
  preserves `quote_id`, `session_id`, `hjb_generation`, `client_order_id`, and
  Hyperliquid `cloid`, giving the live-canary verifier a direct path to
  reconcile exchange fills back to accepted quote decisions and order attempts.
- Direct ALO probe preparation:
  `scripts/hyperliquid_alo_executor.py --mode prepare-probes` converts an
  observed BBO into exact guarded crossing/passive ALO submit commands without
  placing an order. It computes the crossing/passive prices, notional checks,
  quote-linked client order id / `cloid`, and the downstream
  `verify_post_only_mapping.py --mode evaluate-evidence` command. Optional
  public BBO fetching requires `--fetch-bbo --acknowledge-public-market-read`.
  When tick and amount-step constraints are supplied, the preparation artifact
  records both raw and rounded price/size evidence and emits submit commands
  carrying the same constraints.
- Promotion evidence manifest:
  `scripts/build_promotion_evidence_manifest.py` writes
  `docs/promotion_evidence_manifest.json` from the current dry-run quality, TIF
  runtime, post-only, fee, replay, and canary reports. It gives the operator one
  machine-readable checklist for the remaining external gates: the current
  dry-run quote/size/PnL assessment, the current Freqtrade `PO`/`Alo` runtime
  status, which deployment blockers remain, and the guarded commands/artifact
  paths needed to gather post-only, fee, multi-day replay, and canary evidence.
  The manifest is intentionally not a shortcut to live trading: `ok=true` means
  the manifest was generated, while `deployment_ready` stays false until all
  four external gate reports are `ok=true`.
- Risk audit trail:
  realized-PnL fills now emit `risk_update` events with daily realized PnL,
  daily loss limits, consecutive-loss counts, and consecutive-loss limits.
  Kill switches now try `cancel_all_orders` when available and fall back to
  per-order cancellation from exchange open-order sources when possible, while
  logging the cancel method, order ids, request count, and cancel errors. If a
  non-zero signed base position remains visible after the kill switch, the
  strategy emits a `risk_flatten_requested` audit event with a deterministic
  risk action id, reduce-only IOC side/size, reference price, readable client
  order id, Hyperliquid `cloid`, and explicit external-executor
  acknowledgement requirements. Tests cover drawdown kills, consecutive-loss
  kills, duplicate fill-PnL deduplication, flatten request logging, and
  cancel-order fallback behavior.
- HJB parameter audit trail:
  HJB refresh events and quote decisions now carry a compact parameter snapshot
  plus `hjb_param_fingerprint`, so a quote can be tied back to the exact kappa,
  epsilon, lambda, schema, timestamp, and diagnostic metadata used to create its
  HJB generation. Each quote decision also carries a monotonic `quote_id`, so
  future order/fill evidence can point back to the exact accepted or rejected
  quote decision that produced it. Accepted order attempts log the matched
  quote ID when the submitted side/price corresponds to a recent accepted quote,
  and fill logs infer that quote ID from the accepted-order attempt cache when
  the exchange/Freqtrade order object does not preserve an explicit client or
  quote id. The accepted-order log also carries live/dry-run and TIF context,
  and the canary verifier requires live maker fills to include a quote ID and
  reconcile to both a prior accepted quote decision and a prior live, post-only,
  quote-linked accepted order attempt.
- Live canary evidence report:
  `scripts/verify_live_canary.py` writes `docs/live_canary_report.json` from
  audit logs and the prior post-only, fee, and replay gate artifacts. It
  requires several live, non-dry-run sessions, tiny stake, one symbol, manual
  monitoring acknowledgement, post-only verification, fee agreement, fresh
  accepted quotes, zero taker fills, no unknown fill liquidity, no kill
  switches, and no parameter/HJB/collector error events. It also requires fresh
  `generated_at` timestamps on dependency gate reports and recent timestamps on
  canary session events, so stale canary evidence cannot pass by regenerating a
  new report. One-symbol canary validation is enforced across health, quote, and
  fill events rather than health events alone. The safety-gate runner now
  generates this artifact after
  regenerating fee and replay reports in runtime mode, so it summarizes the
  current dependency artifacts. The runner withholds the manual monitoring
  acknowledgement by default and only passes it through with the explicit
  `--manual-monitoring-ack` flag after actual monitored canary sessions. It
  also requires the audit log to contain a fresh `manual_monitoring_ack` or
  `canary_manual_monitoring_ack` event with `acknowledged=true`; the CLI flag
  alone is not enough to satisfy the gate.
  `scripts/record_manual_monitoring_ack.py` appends that event to a retained
  JSONL audit log and refuses to write it unless `--acknowledge-risk` is passed.
  Canary evidence with future-dated timestamps is rejected rather than treated
  as age-zero evidence.
  The runner accepts `--audit-log-input` so retained canary logs can be
  evaluated by the same full gate runner. Accepted quote and fill evidence must
  now carry
  `trading_enabled=true` and `dry_run=false`, so live health events cannot be
  combined with unrelated research/dry-run quotes to satisfy canary session
  eligibility. Live maker fills must carry an order id plus side/price/amount
  fields and reconcile to both a prior live accepted quote and a prior live
  post-only `order_attempt_accepted` event with a quote ID in the same canary
  session by symbol, side, and price, so unrelated exchange fills cannot satisfy
  the canary. Maker fills must carry a `quote_id`; the canary verifier treats
  that ID as authoritative and will not accept a same-price fill linked to a
  different quote decision or order attempt. The current report is expected to
  be `ok=false` until Gates 4-5 pass and real canary sessions are supplied.
- Promotion evidence freshness controls:
  `scripts/run_safety_gates.py` now exposes `--max-evidence-age-seconds` for
  external report freshness plus post-only/fee/dependency evidence age, and
  `--max-canary-event-age-seconds` for live canary audit events. This allows a
  single promotion gate run to require fresher external proof than the default
  86400-second report/evidence window.
- Strategy-level deployment gate enforcement:
  non-dry-run `trading_enabled=true` now requires a declared
  `market_making.deployment_stage`. `canary` mode requires post-only, fee, and
  replay reports to be `ok=true` plus `manual_monitoring_ack=true`; `production`
  mode additionally requires `docs/live_canary_report.json` to be `ok=true`.
  Those reports must carry fresh `generated_at` timestamps within
  `max_deployment_report_age_seconds`, and quote validation rechecks the same
  gate state so runtime toggles cannot bypass the startup guard.

## Phase Status

| Phase | Status | Evidence |
| --- | --- | --- |
| Phase 0 - fail closed | Automated pass | Default strategy/config remain locked; locked dry-run gate proves zero orders. |
| Phase 1 - HJB math | Automated pass | `tests/test_hjb.py`; `compute_spreads_boundary_smoke`. |
| Phase 2 - inventory/units/risk | Automated pass for long-only research mode | Strategy guard tests, `docs/UNITS.md`, quote logs include signed base and q. |
| Phase 3 - Freqtrade fail-closed wiring | Automated pass | Confirm gates, source and runtime callback signature tests, proposed-rate fallback rejection tests, inventory-limit custom-pricing rejection tests, disabled and enabled dry-run smokes. |
| Phase 4 - maker safety | Partial | Local maker guards, fee alignment, fee agreement fail-closed guards, fee evidence evaluator and capture normalizer, price and amount rounding guards, final confirm-time tick/lot safety guards, post-only TIF confirmation/fill kill-switch guards, kill-on-taker-fill tests, post-only probe plan, Alo evidence evaluator, and direct SDK Alo adapter scaffold exist. Exchange-level `Alo` and account fee-tier evidence are not verified. |
| Phase 5 - parameter/data pipeline | Automated pass for local pipeline | Atomic writers, atomic strategy-facing snapshot copies, schema v3 tests, status locking, process-level estimator locking, deterministic kappa -> epsilon -> raw-lambda updater order, snapshot validation for timestamped windows/fit diagnostics/toxicity diagnostics, `mo_survival_fit` enforcement for HJB lambda (survival-based per-side MO arrival rates; mid-relative survival-fit kappa; per-MO 5s-horizon epsilon; EMA-smoothed primaries with raw companions; sigma2 volatility channel), row-timestamp-based collector freshness validation, and no hardcoded symbol fallback when the strategy has no active pair. |
| Phase 6 - replay | Partial | Event replay exists, runs on latest local shards, models latency, tick-safe price rounding, amount-step rounding, widened-tick stress, queue-ahead volume, conservative queue decay, fees/funding, margin/equity exposure, dry-run/testnet fill calibration artifacts, optional calibration-throttled fills by side/depth bucket, quote-quality ratio gates, missing-stream refusal checks, price-density/max-gap coverage gates, and has a multi-variant acceptance report. Multi-day replay acceptance is still not complete. |
| Phase 7 - observability/kill switches | Automated pass for local fields | Health, quote decisions, accepted-order quote linkage, fill-to-order-attempt reconciliation, freshness-age fields, stable quote IDs, HJB parameter fingerprints, fee agreement snapshots, canary-relevant health fields, source-labeled exchange/Trade/accepted-confirmation open-order counts, mark-to-mid unrealized PnL, fill accounting, realized-PnL risk updates, delayed fill markouts, post-only reject-rate enforcement, kill-switch cancellation fallback, strategy-side `risk_flatten_requested` audit events, guarded reduce-only IOC flatten scaffold, and kill-switch tests/log artifacts exist. |
| Phase 8 - deployment gates | Partial | Gates 1-3 are automated and passing. Gate 6 now has a log/artifact verifier with live fill-to-quote reconciliation, and live strategy enablement is gated on deployment-stage artifacts. Gates 4-6 still require external/manual evidence. |

## Remaining Required Gates

- Hyperliquid post-only/Alo integration:
  prove passive `Alo` orders rest, crossing `Alo` orders reject/cancel without
  filling, actual order TIF is `Alo`, and liquidity/fee fields are logged.
- Hyperliquid account fee tier:
  provide exchange/account fee snapshots and maker fills whose actual fee rate
  matches the configured maker fee so `docs/fee_evidence_report.json` becomes
  `ok=true`.
- Multi-day event replay:
  run several days of collected data with fee, latency, parameter
  perturbation, widened tick, stale data, and missing-stream sensitivity; review
  realized spread, markouts, maker ratio, inventory boundaries, and
  directional-drift attribution.
- Live canary:
  only after every prior gate passes, using tiny fixed stake, hard loss limits,
  post-only required, kill-on-taker-fill enabled, manual monitoring acknowledged
  by both CLI flag and fresh audit-log acknowledgement event, and
  `docs/live_canary_report.json` becomes `ok=true`.

Until those remaining gates have evidence, the project is still a fail-closed
research/dry-run implementation, not a production-ready live market maker.
