# Safety Gate Results

Automated gates: FAIL
Deployment ready: NO
Manual gates remaining: 4

- PASS `compileall` (1.74s)
- PASS `pytest_core` (13.399s)
- PASS `config_safety_report` (0.054s)
- PASS `strategy_safety_report` (0.107s)
- PASS `strategy_attribute_report` (0.133s)
- PASS `compute_spreads_boundary_smoke` (0.751s)
- PASS `replay_smoke` (0.892s)
- PASS `adapter_plans` (0.575s)
- PASS `post_only_evidence_report` (0.067s)
- PASS `docker_compose_config` (0.285s)
- PASS `freqtrade_runtime_load` (34.721s)
- PASS `freqtrade_callback_surface` (27.694s)
- PASS `freqtrade_tif_runtime` (76.376s)
- PASS `dry_run_disabled_smoke` (154.343s)
- PASS `dry_run_enabled_smoke` (672.124s)
- FAIL `dry_run_quality_report` (0.136s)
  - returncode: `1`
- PASS `replay_log_calibration_artifact` (0.553s)
- PASS `fee_evidence_report` (0.079s)
- PASS `hl_data_validation_report` (0.892s)
- PASS `replay_latest_data_smoke` (1.299s)
- PASS `replay_acceptance_report_artifact` (2.444s)
- PASS `live_canary_evidence_report` (0.119s)
- PASS `promotion_evidence_manifest` (0.081s)

Post-run audits:
- FAIL `plan_status_audit` (0.075s)
  - returncode: `1`

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