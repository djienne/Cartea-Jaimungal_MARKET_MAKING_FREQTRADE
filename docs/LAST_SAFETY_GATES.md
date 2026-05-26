# Safety Gate Results

Automated gates: PASS
Deployment ready: NO
Manual gates remaining: 4

- PASS `compileall` (1.112s)
- PASS `pytest_core` (22.418s)
- PASS `config_safety_report` (0.113s)
- PASS `strategy_safety_report` (0.158s)
- PASS `compute_spreads_boundary_smoke` (1.443s)
- PASS `replay_smoke` (2.061s)
- PASS `post_only_probe_plan` (0.069s)
- PASS `post_only_evidence_report` (0.066s)
- PASS `direct_alo_adapter_plan` (0.074s)
- PASS `hyperliquid_fee_capture_plan` (0.063s)
- PASS `docker_compose_config` (0.165s)
- PASS `freqtrade_runtime_load` (7.68s)
- PASS `dry_run_disabled_smoke` (225.071s)
- PASS `dry_run_enabled_smoke` (313.638s)
- PASS `replay_log_calibration_artifact` (0.764s)
- PASS `fee_evidence_report` (0.14s)
- PASS `hl_data_validation_report` (1.561s)
- PASS `replay_latest_data_smoke` (2.998s)
- PASS `replay_acceptance_report_artifact` (15.48s)
- PASS `live_canary_evidence_report` (0.187s)

Post-run audits:
- PASS `plan_status_audit` (0.157s)

Manual/external gate evidence:
- WAIT `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
  - reason: `ok_not_true`
- WAIT `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
  - reason: `ok_not_true`
- WAIT `hyperliquid_fee_tier`: Requires exchange/account maker-fee evidence and actual maker fill fee rates.
  - reason: `ok_not_true`
- WAIT `live_canary`: Requires docs/live_canary_report.json to pass after several tiny live sessions with post-only, fee, replay, zero-taker, freshness, and kill-switch evidence.
  - reason: `ok_not_true`

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `hyperliquid_fee_tier`: Requires exchange/account maker-fee evidence and actual maker fill fee rates.
- `live_canary`: Requires docs/live_canary_report.json to pass after several tiny live sessions with post-only, fee, replay, zero-taker, freshness, and kill-switch evidence.