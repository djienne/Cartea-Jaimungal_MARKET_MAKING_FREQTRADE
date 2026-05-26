# Safety Gate Results

- PASS `compileall` (1.02s)
- PASS `pytest_core` (23.451s)
- PASS `config_safety_report` (0.116s)
- PASS `strategy_safety_report` (0.162s)
- PASS `compute_spreads_boundary_smoke` (1.543s)
- PASS `replay_smoke` (2.072s)
- PASS `post_only_probe_plan` (0.076s)
- PASS `post_only_evidence_report` (0.076s)
- PASS `direct_alo_adapter_plan` (0.082s)
- PASS `docker_compose_config` (0.207s)
- PASS `freqtrade_runtime_load` (7.64s)
- PASS `dry_run_disabled_smoke` (224.788s)
- PASS `dry_run_enabled_smoke` (313.276s)
- PASS `replay_log_calibration_artifact` (0.522s)
- PASS `fee_evidence_report` (0.073s)
- PASS `hl_data_validation_report` (1.502s)
- PASS `replay_latest_data_smoke` (3.147s)
- PASS `replay_acceptance_report_artifact` (16.213s)
- PASS `live_canary_evidence_report` (0.141s)

Post-run audits:
- PASS `plan_status_audit` (0.072s)

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `hyperliquid_fee_tier`: Requires exchange/account maker-fee evidence and actual maker fill fee rates.
- `live_canary`: Requires docs/live_canary_report.json to pass after several tiny live sessions with post-only, fee, replay, zero-taker, freshness, and kill-switch evidence.