# Safety Gate Results

Automated gates: PASS
Deployment ready: NO
Manual gates remaining: 4

- PASS `compileall` (0.978s)
- PASS `pytest_core` (25.243s)
- PASS `config_safety_report` (0.084s)
- PASS `strategy_safety_report` (0.085s)
- PASS `compute_spreads_boundary_smoke` (1.377s)
- PASS `replay_smoke` (1.95s)
- PASS `post_only_probe_plan` (0.1s)
- PASS `post_only_evidence_report` (0.127s)
- PASS `direct_alo_adapter_plan` (0.126s)
- PASS `hyperliquid_fee_capture_plan` (0.092s)
- PASS `docker_compose_config` (0.204s)
- PASS `freqtrade_runtime_load` (7.69s)
- PASS `dry_run_disabled_smoke` (224.656s)
- PASS `dry_run_enabled_smoke` (313.603s)
- PASS `replay_log_calibration_artifact` (0.838s)
- PASS `fee_evidence_report` (0.099s)
- PASS `hl_data_validation_report` (1.817s)
- PASS `replay_latest_data_smoke` (3.169s)
- PASS `replay_acceptance_report_artifact` (15.695s)
- PASS `live_canary_evidence_report` (0.171s)

Post-run audits:
- PASS `plan_status_audit` (0.081s)

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