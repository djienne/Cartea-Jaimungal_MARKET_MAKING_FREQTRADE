# PLAN Implementation Status

Generated from the current local worktree after the latest safety-gate run.

## Automated Evidence

- Unit/integration tests: `python -m pytest tests` passed with 87 tests.
- Runtime gate runner:
  `python scripts/run_safety_gates.py --include-runtime --markdown-output docs/LAST_SAFETY_GATES.md --json-output docs/last_safety_gates.json`
  passed all automated checks.
- Post-only evidence harness:
  `scripts/verify_post_only_mapping.py` now writes a plan artifact and evaluates
  crossing/passive `Alo` submit artifacts. The current
  `docs/post_only_evidence_report.json` is expected to be `ok=false` until
  real exchange/testnet artifacts are supplied; the local safety gate treats
  that incomplete report as an artifact check, not as post-only proof.
- Direct `Alo` fallback scaffold:
  `scripts/hyperliquid_alo_executor.py` builds Hyperliquid SDK orders with
  `order_type={"limit": {"tif": "Alo"}}`, applies local maker-safety from BBO,
  classifies SDK responses, and refuses submit mode without explicit
  environment/CLI acknowledgements. Direct SDK submit artifacts are normalized
  by `scripts/verify_post_only_mapping.py --mode evaluate-evidence`, so direct
  execution evidence and CCXT/Freqtrade probe evidence use the same Gate 4
  checker. The direct adapter now has a separate guarded
  `submit-crossing-alo` mode for ALO rejection evidence and keeps regular
  `submit-alo` locally maker-safe for passive evidence.
- Fresh collector data validation:
  `docs/hl_data_validation.json` shows fresh ETH shards, 75 checked files, and
  0 bad files.
- Locked dry-run evidence:
  `docs/dry_run_disabled_gate.json` shows `trading_enabled=false`,
  `collector_fresh=true`, health logging, and zero order creation.
- Enabled dry-run evidence:
  `docs/dry_run_enabled_gate.json` uses temporary ignored params/config and
  shows `trading_enabled=true`, `params_fresh=true`, `collector_fresh=true`,
  one accepted passive bid quote, and a dry-run order creation line.
- Latest-data replay smoke:
  `docs/replay_latest_smoke.json` replays the newest local shards and records
  input coverage, post-only rejects, stale cancels, maker/taker counts,
  inventory, PnL, and markout samples where fills occur.
- Replay acceptance report:
  `docs/replay_acceptance_report.json` and
  `docs/replay_acceptance_report.md` run baseline, fee, latency, and parameter
  perturbation variants, including directional-drift attribution so PnL from
  one-way price movement is reported separately from net realized spread.
  Replay quote generation and fill accounting both use the configured
  maker/taker fee schedule, so fee-sensitivity variants change quoted depth as
  well as fees paid. The current report correctly fails because the local data
  window is much shorter than the required multi-day coverage and has no maker
  fills.

## Phase Status

| Phase | Status | Evidence |
| --- | --- | --- |
| Phase 0 - fail closed | Automated pass | Default strategy/config remain locked; locked dry-run gate proves zero orders. |
| Phase 1 - HJB math | Automated pass | `tests/test_hjb.py`; `compute_spreads_boundary_smoke`. |
| Phase 2 - inventory/units/risk | Automated pass for long-only research mode | Strategy guard tests, `docs/UNITS.md`, quote logs include signed base and q. |
| Phase 3 - Freqtrade fail-closed wiring | Automated pass | Confirm gates, callback signature tests, disabled and enabled dry-run smokes. |
| Phase 4 - maker safety | Partial | Local maker guards, fee alignment, kill-on-taker-fill tests, post-only probe plan, Alo evidence evaluator, and direct SDK Alo adapter scaffold exist. Exchange-level `Alo` is not verified. |
| Phase 5 - parameter/data pipeline | Automated pass for local pipeline | Atomic writers, schema v2 tests, status locking, freshness validation. |
| Phase 6 - replay | Partial | Event replay exists, runs on latest local shards, and has a multi-variant acceptance report. Multi-day replay acceptance is still not complete. |
| Phase 7 - observability/kill switches | Automated pass for local fields | Health, quote decisions, fill accounting, and kill-switch tests/log artifacts exist. |
| Phase 8 - deployment gates | Partial | Gates 1-3 are automated and passing. Gates 4-6 require external/manual evidence. |

## Remaining Required Gates

- Hyperliquid post-only/Alo integration:
  prove passive `Alo` orders rest, crossing `Alo` orders reject/cancel without
  filling, actual order TIF is `Alo`, and liquidity/fee fields are logged.
- Multi-day event replay:
  run several days of collected data with fee, latency, and parameter
  perturbation sensitivity; review realized spread, markouts, maker ratio,
  inventory boundaries, and directional-drift attribution.
- Live canary:
  only after every prior gate passes, using tiny fixed stake, hard loss limits,
  post-only required, and kill-on-taker-fill enabled.

Until those remaining gates have evidence, the project is still a fail-closed
research/dry-run implementation, not a production-ready live market maker.
