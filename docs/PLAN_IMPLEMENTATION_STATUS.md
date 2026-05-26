# PLAN Implementation Status

Generated from the current local worktree after the latest safety-gate run.

## Automated Evidence

- Unit/integration tests: `python -m pytest tests` passed with 197 tests.
- Runtime gate runner:
  `python scripts/run_safety_gates.py --include-runtime --markdown-output docs/LAST_SAFETY_GATES.md --json-output docs/last_safety_gates.json`
  passed all automated checks. The JSON payload now separates
  `all_automated_passed` from `deployment_ready`, with
  `deployment_ready=false` and named `deployment_blockers` until the external
  Hyperliquid, multi-day replay, fee, and live-canary gates have fresh evidence.
  The audit requires the runtime load gate `freqtrade_runtime_load` plus locked
  and enabled dry-run smoke gates, so Gates 2-3 cannot disappear silently.
  The remaining manual gates are now evidence-aware: each external gate stays
  in `deployment_blockers` only while its corresponding report artifact is not
  `ok=true`.
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
  missing kill switches, or callback surface regressions.
- Post-only evidence harness:
  `scripts/verify_post_only_mapping.py` now writes a plan artifact and evaluates
  crossing/passive `Alo` submit artifacts. The current
  `docs/post_only_evidence_report.json` is expected to be `ok=false` until
  real exchange/testnet artifacts are supplied; the local safety gate treats
  both incomplete and passing reports as artifact checks, while deployment
  readiness still depends on the report's `ok` field. The checker now also
  requires fresh `generated_at` timestamps on the underlying crossing/passive
  submit artifacts so stale probes cannot unlock Gate 4.
- Direct `Alo` fallback scaffold:
  `scripts/hyperliquid_alo_executor.py` builds Hyperliquid SDK orders with
  `order_type={"limit": {"tif": "Alo"}}`, applies local maker-safety from BBO,
  classifies SDK responses, and refuses submit mode without explicit
  environment/CLI acknowledgements. Direct SDK submit artifacts are normalized
  by `scripts/verify_post_only_mapping.py --mode evaluate-evidence`, so direct
  execution evidence and CCXT/Freqtrade probe evidence use the same Gate 4
  checker. The direct adapter now has a separate guarded
  `submit-crossing-alo` mode for ALO rejection evidence and a guarded
  `submit-passive-alo` evidence mode that cancels any resting order ids after
  classification. Regular `submit-alo` remains locally maker-safe for the future
  direct execution layer. The safety-gate runner can consume post-only
  crossing/passive evidence through explicit artifact path flags, and it also
  auto-detects the conventional `docs/post_only_crossing_result.json` and
  `docs/post_only_passive_result.json` paths when they exist.
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
  shows `trading_enabled=true`, `params_fresh=true`, `collector_fresh=true`,
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
- Latest-data replay smoke:
  `docs/replay_latest_smoke.json` replays the newest local shards and records
  input coverage, post-only rejects, stale cancels, maker/taker counts,
  inventory, PnL, and markout samples where fills occur.
- Replay acceptance report:
  `docs/replay_acceptance_report.json` and
  `docs/replay_acceptance_report.md` run baseline, fee, latency, and parameter
  perturbation variants, including directional-drift attribution so PnL from
  one-way price movement is reported separately from net realized spread.
  Replay queueing includes traded-volume queue-ahead and conservative
  cancellation decay, with `queue_decay_base` reported in metrics.
  Replay quote generation and fill accounting both use the configured
  maker/taker fee schedule, so fee-sensitivity variants change quoted depth as
  well as fees paid. Replay now tracks starting equity, leverage, notional
  exposure, margin used, maintenance margin, equity, liquidation buffer, and
  maintenance-margin breach counts. Replay parameter and toxicity time series
  are timestamped with source/unit metadata, and the acceptance report rejects
  missing, untimestamped, or non-finite parameter/toxicity evidence. The report
  also includes refusal checks proving bad parameters and stale collector data
  reject quoting, and fails any replay variant that breaches maintenance
  margin. The safety-gate runner
  defaults to a short capped replay artifact, but exposes replay acceptance
  symbol, mid, shard cap, event cap, and require-pass knobs so a promotion run
  can regenerate the same report from the full multi-day dataset. The current
  report correctly fails because the local data window is much shorter than the
  required multi-day coverage and has no maker fills.
- Replay log calibration:
  `scripts/calibrate_replay_from_logs.py` reads JSONL quote/fill/markout audit
  logs and writes `docs/replay_log_calibration.json` with accepted quote counts,
  fill probabilities by side/depth bucket, maker ratio, fee-rate summary, and
  markout summary. The runtime safety gate generates this artifact after the
  enabled dry-run smoke. The safety-gate runner accepts `--audit-log-input` so
  dry-run, testnet, or live-canary audit logs can be evaluated without copying
  them over the default debug log. It can be `usable_for_calibration=false`
  until enough real dry-run/testnet fills exist.
- Fee evidence report:
  `scripts/verify_fee_evidence.py` writes `docs/fee_evidence_report.json` from
  JSONL audit logs. It requires strategy/config fee agreement, exchange/account
  maker-fee evidence, and at least one maker fill with an actual fee rate that
  matches the configured maker fee. The underlying fee snapshot and actual
  maker-fill fee records must also have fresh event timestamps, so stale logs
  cannot satisfy the gate by regenerating a new report. The same
  `--audit-log-input` runner option feeds this report, so promotion evidence can
  come from retained testnet/tiny integration logs. The current report can be
  `ok=false` until testnet/tiny integration supplies real account fee and
  fill-fee evidence.
- Risk audit trail:
  realized-PnL fills now emit `risk_update` events with daily realized PnL,
  daily loss limits, consecutive-loss counts, and consecutive-loss limits.
  Kill switches now try `cancel_all_orders` when available and fall back to
  per-order cancellation from exchange open-order sources when possible, while
  logging the cancel method, order ids, request count, and cancel errors. Tests
  cover drawdown kills, consecutive-loss kills, duplicate fill-PnL
  deduplication, and cancel-order fallback behavior.
- Live canary evidence report:
  `scripts/verify_live_canary.py` writes `docs/live_canary_report.json` from
  audit logs and the prior post-only, fee, and replay gate artifacts. It
  requires several live, non-dry-run sessions, tiny stake, one symbol, manual
  monitoring acknowledgement, post-only verification, fee agreement, fresh
  accepted quotes, zero taker fills, no unknown fill liquidity, no kill
  switches, and no parameter/HJB/collector error events. It also requires fresh
  `generated_at` timestamps on dependency gate reports and recent timestamps on
  canary session events, so stale canary evidence cannot pass by regenerating a
  new report. The safety-gate runner now generates this artifact after
  regenerating fee and replay reports in runtime mode, so it summarizes the
  current dependency artifacts. The runner withholds the manual monitoring
  acknowledgement by default and only passes it through with the explicit
  `--manual-monitoring-ack` flag after actual monitored canary sessions. It
  also accepts `--audit-log-input` so retained canary logs can be evaluated by
  the same full gate runner. The current report is expected to be `ok=false`
  until Gates 4-5 pass and real canary sessions are supplied.
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
| Phase 3 - Freqtrade fail-closed wiring | Automated pass | Confirm gates, callback signature tests, disabled and enabled dry-run smokes. |
| Phase 4 - maker safety | Partial | Local maker guards, fee alignment, fee agreement fail-closed guards, fee evidence evaluator, post-only TIF confirmation/fill kill-switch guards, kill-on-taker-fill tests, post-only probe plan, Alo evidence evaluator, and direct SDK Alo adapter scaffold exist. Exchange-level `Alo` and account fee-tier evidence are not verified. |
| Phase 5 - parameter/data pipeline | Automated pass for local pipeline | Atomic writers, schema v2 tests, status locking, and row-timestamp-based collector freshness validation. |
| Phase 6 - replay | Partial | Event replay exists, runs on latest local shards, models latency, queue-ahead volume, conservative queue decay, fees/funding, margin/equity exposure, dry-run/testnet fill calibration artifacts, and has a multi-variant acceptance report. Multi-day replay acceptance is still not complete. |
| Phase 7 - observability/kill switches | Automated pass for local fields | Health, quote decisions, freshness-age fields, fee agreement snapshots, canary-relevant health fields, source-labeled exchange/Trade/accepted-confirmation open-order counts, mark-to-mid unrealized PnL, fill accounting, realized-PnL risk updates, delayed fill markouts, post-only reject-rate enforcement, kill-switch cancellation fallback, and kill-switch tests/log artifacts exist. |
| Phase 8 - deployment gates | Partial | Gates 1-3 are automated and passing. Gate 6 now has a log/artifact verifier, and live strategy enablement is gated on deployment-stage artifacts. Gates 4-6 still require external/manual evidence. |

## Remaining Required Gates

- Hyperliquid post-only/Alo integration:
  prove passive `Alo` orders rest, crossing `Alo` orders reject/cancel without
  filling, actual order TIF is `Alo`, and liquidity/fee fields are logged.
- Hyperliquid account fee tier:
  provide exchange/account fee snapshots and maker fills whose actual fee rate
  matches the configured maker fee so `docs/fee_evidence_report.json` becomes
  `ok=true`.
- Multi-day event replay:
  run several days of collected data with fee, latency, and parameter
  perturbation sensitivity; review realized spread, markouts, maker ratio,
  inventory boundaries, and directional-drift attribution.
- Live canary:
  only after every prior gate passes, using tiny fixed stake, hard loss limits,
  post-only required, kill-on-taker-fill enabled, manual monitoring acknowledged,
  and `docs/live_canary_report.json` becomes `ok=true`.

Until those remaining gates have evidence, the project is still a fail-closed
research/dry-run implementation, not a production-ready live market maker.
