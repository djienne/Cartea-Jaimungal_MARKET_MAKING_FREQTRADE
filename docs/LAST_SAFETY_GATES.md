# Safety Gate Results

Automated gates: PASS
Deployment ready: NO
Manual gates remaining: 4

- PASS `compileall` (1.531s)
- PASS `pytest_core` (22.73s)
- PASS `config_safety_report` (0.063s)
- PASS `strategy_safety_report` (0.109s)
- PASS `compute_spreads_boundary_smoke` (1.279s)
- PASS `replay_smoke` (1.639s)
- PASS `post_only_probe_plan` (0.062s)
- PASS `post_only_evidence_report` (0.065s)
- PASS `direct_alo_adapter_plan` (0.084s)
- PASS `direct_alo_probe_preparation_plan` (0.078s)
- PASS `direct_risk_flatten_plan` (0.074s)
- PASS `hyperliquid_fee_capture_plan` (0.059s)
- PASS `docker_compose_config` (0.109s)
- PASS `freqtrade_runtime_load` (5.839s)
- PASS `freqtrade_callback_surface` (5.895s)
- PASS `freqtrade_tif_runtime` (17.376s)
- PASS `dry_run_disabled_smoke` (224.033s)
- PASS `dry_run_enabled_smoke` (1873.044s)
- PASS `dry_run_quality_report` (0.08s)
- PASS `replay_log_calibration_artifact` (1.363s)
- PASS `fee_evidence_report` (0.071s)
- PASS `hl_data_validation_report` (1.968s)
- PASS `replay_latest_data_smoke` (5.127s)
- PASS `replay_acceptance_report_artifact` (17.702s)
- PASS `live_canary_evidence_report` (0.092s)
- PASS `promotion_evidence_manifest` (0.077s)

Post-run audits:
- PASS `plan_status_audit` (0.08s)

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