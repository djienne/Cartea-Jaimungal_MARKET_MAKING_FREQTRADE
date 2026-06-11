# Safety Gate Results

Automated gates: PASS
Deployment ready: NO
Manual gates remaining: 4
Smoke artifacts: REUSED from previous battery (freshness-validated)

- PASS `compileall` (0.146s)
- PASS `pytest_core` (57.605s)
- PASS `config_safety_report` (0.093s)
- PASS `strategy_safety_report` (0.139s)
- PASS `compute_spreads_boundary_smoke` (1.81s)
- PASS `replay_smoke` (1.986s)
- PASS `adapter_plans` (0.507s)
- PASS `post_only_evidence_report` (0.073s)
- PASS `docker_compose_config` (0.162s)
- PASS `freqtrade_runtime_load` (7.12s)
- PASS `freqtrade_callback_surface` (7.624s)
- PASS `freqtrade_tif_runtime` (23.352s)
- PASS `dry_run_quality_report` (0.108s)
- PASS `replay_log_calibration_artifact` (0.779s)
- PASS `fee_evidence_report` (0.098s)
- PASS `hl_data_validation_report` (0.833s)
- PASS `replay_latest_data_smoke` (2.219s)
- PASS `replay_acceptance_report_artifact` (10.629s)
- PASS `live_canary_evidence_report` (0.149s)
- PASS `promotion_evidence_manifest` (0.119s)
- PASS `dry_run_disabled_smoke` (0.0s) (reused)
- PASS `dry_run_enabled_smoke` (0.0s) (reused)

Post-run audits:
- PASS `plan_status_audit` (0.108s)

Manual/external gate evidence:
- WAIT `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that either Freqtrade/CCXT PO maps to Hyperliquid Alo or the direct SDK Alo fallback submits native post-only orders safely.
  - reason: `ok_not_true`
- WAIT `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
  - reason: `ok_not_true`
- WAIT `hyperliquid_fee_tier`: Requires exchange/account maker-fee evidence and actual maker fill fee rates.
  - reason: `ok_not_true`
- WAIT `live_canary`: Requires docs/live_canary_report.json to pass after several tiny live sessions with post-only, fee, replay, zero-taker, freshness, and kill-switch evidence.
  - reason: `ok_not_true`

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that either Freqtrade/CCXT PO maps to Hyperliquid Alo or the direct SDK Alo fallback submits native post-only orders safely.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `hyperliquid_fee_tier`: Requires exchange/account maker-fee evidence and actual maker fill fee rates.
- `live_canary`: Requires docs/live_canary_report.json to pass after several tiny live sessions with post-only, fee, replay, zero-taker, freshness, and kill-switch evidence.