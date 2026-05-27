# Safety Gate Results

Automated gates: PASS
Deployment ready: NO
Manual gates remaining: 4

- PASS `compileall` (1.384s)
- PASS `pytest_core` (22.59s)
- PASS `config_safety_report` (0.063s)
- PASS `strategy_safety_report` (0.095s)
- PASS `compute_spreads_boundary_smoke` (1.284s)
- PASS `replay_smoke` (1.627s)
- PASS `post_only_probe_plan` (0.062s)
- PASS `post_only_evidence_report` (0.063s)
- PASS `direct_alo_adapter_plan` (0.078s)
- PASS `direct_risk_flatten_plan` (0.08s)
- PASS `hyperliquid_fee_capture_plan` (0.063s)
- PASS `docker_compose_config` (0.105s)
- PASS `freqtrade_runtime_load` (6.143s)
- PASS `freqtrade_callback_surface` (6.7s)
- PASS `freqtrade_tif_runtime` (16.999s)
- PASS `dry_run_disabled_smoke` (224.054s)
- PASS `dry_run_enabled_smoke` (672.863s)
- PASS `dry_run_quality_report` (0.094s)
- PASS `replay_log_calibration_artifact` (0.854s)
- PASS `fee_evidence_report` (0.078s)
- PASS `hl_data_validation_report` (1.903s)
- PASS `replay_latest_data_smoke` (4.971s)
- PASS `replay_acceptance_report_artifact` (17.37s)
- PASS `live_canary_evidence_report` (0.099s)
- PASS `promotion_evidence_manifest` (0.086s)

Post-run audits:
- PASS `plan_status_audit` (0.071s)

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