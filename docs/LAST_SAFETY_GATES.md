# Safety Gate Results

Automated gates: PASS
Deployment ready: NO
Manual gates remaining: 4

- PASS `compileall` (1.299s)
- PASS `pytest_core` (25.414s)
- PASS `config_safety_report` (0.077s)
- PASS `strategy_safety_report` (0.099s)
- PASS `compute_spreads_boundary_smoke` (1.527s)
- PASS `replay_smoke` (2.331s)
- PASS `post_only_probe_plan` (0.095s)
- PASS `post_only_evidence_report` (0.132s)
- PASS `direct_alo_adapter_plan` (0.13s)
- PASS `docker_compose_config` (0.226s)
- PASS `freqtrade_runtime_load` (8.309s)
- PASS `dry_run_disabled_smoke` (224.778s)
- PASS `dry_run_enabled_smoke` (313.345s)
- PASS `replay_log_calibration_artifact` (0.748s)
- PASS `fee_evidence_report` (0.079s)
- PASS `hl_data_validation_report` (1.626s)
- PASS `replay_latest_data_smoke` (2.859s)
- PASS `replay_acceptance_report_artifact` (16.4s)
- PASS `live_canary_evidence_report` (0.156s)

Post-run audits:
- PASS `plan_status_audit` (0.097s)

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