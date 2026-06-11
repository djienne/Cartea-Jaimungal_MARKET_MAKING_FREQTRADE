# Safety Gate Results

Automated gates: FAIL
Deployment ready: NO
Manual gates remaining: 4

- PASS `compileall` (0.12s)
- PASS `pytest_core` (60.625s)
- PASS `config_safety_report` (0.095s)
- PASS `strategy_safety_report` (0.154s)
- PASS `compute_spreads_boundary_smoke` (2.242s)
- PASS `replay_smoke` (2.186s)
- PASS `post_only_probe_plan` (0.085s)
- PASS `post_only_evidence_report` (0.074s)
- PASS `direct_alo_adapter_plan` (0.118s)
- PASS `direct_alo_probe_preparation_plan` (0.122s)
- PASS `direct_risk_flatten_plan` (0.122s)
- PASS `hyperliquid_fee_capture_plan` (0.109s)
- PASS `docker_compose_config` (0.186s)
- PASS `freqtrade_runtime_load` (8.8s)
- PASS `freqtrade_callback_surface` (8.737s)
- PASS `freqtrade_tif_runtime` (26.251s)
- PASS `dry_run_disabled_smoke` (224.634s)
- PASS `dry_run_enabled_smoke` (673.471s)
- PASS `dry_run_quality_report` (0.108s)
- PASS `replay_log_calibration_artifact` (0.895s)
- PASS `fee_evidence_report` (0.084s)
- PASS `hl_data_validation_report` (0.975s)
- PASS `replay_latest_data_smoke` (2.597s)
- PASS `replay_acceptance_report_artifact` (11.095s)
- PASS `live_canary_evidence_report` (0.138s)
- PASS `promotion_evidence_manifest` (0.107s)

Post-run audits:
- FAIL `plan_status_audit` (0.1s)
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