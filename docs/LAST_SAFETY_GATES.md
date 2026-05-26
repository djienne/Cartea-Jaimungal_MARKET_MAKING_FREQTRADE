# Safety Gate Results

Automated gates: PASS
Deployment ready: NO
Manual gates remaining: 4

- PASS `compileall` (1.105s)
- PASS `pytest_core` (23.078s)
- PASS `config_safety_report` (0.07s)
- PASS `strategy_safety_report` (0.083s)
- PASS `compute_spreads_boundary_smoke` (1.52s)
- PASS `replay_smoke` (1.946s)
- PASS `post_only_probe_plan` (0.124s)
- PASS `post_only_evidence_report` (0.12s)
- PASS `direct_alo_adapter_plan` (0.116s)
- PASS `docker_compose_config` (0.222s)
- PASS `freqtrade_runtime_load` (7.691s)
- PASS `dry_run_disabled_smoke` (224.845s)
- PASS `dry_run_enabled_smoke` (313.566s)
- PASS `replay_log_calibration_artifact` (0.643s)
- PASS `fee_evidence_report` (0.117s)
- PASS `hl_data_validation_report` (1.495s)
- PASS `replay_latest_data_smoke` (2.862s)
- PASS `replay_acceptance_report_artifact` (15.994s)
- PASS `live_canary_evidence_report` (0.097s)

Post-run audits:
- PASS `plan_status_audit` (0.067s)

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `hyperliquid_fee_tier`: Requires exchange/account maker-fee evidence and actual maker fill fee rates.
- `live_canary`: Requires docs/live_canary_report.json to pass after several tiny live sessions with post-only, fee, replay, zero-taker, freshness, and kill-switch evidence.