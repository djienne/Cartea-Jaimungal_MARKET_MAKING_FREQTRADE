# Safety Gate Results

Automated gates: PASS
Deployment ready: NO
Manual gates remaining: 4

- PASS `compileall` (1.252s)
- PASS `pytest_core` (23.876s)
- PASS `config_safety_report` (0.088s)
- PASS `strategy_safety_report` (0.079s)
- PASS `compute_spreads_boundary_smoke` (1.423s)
- PASS `replay_smoke` (2.213s)
- PASS `post_only_probe_plan` (0.103s)
- PASS `post_only_evidence_report` (0.109s)
- PASS `direct_alo_adapter_plan` (0.125s)
- PASS `docker_compose_config` (0.284s)
- PASS `freqtrade_runtime_load` (8.133s)
- PASS `dry_run_disabled_smoke` (224.906s)
- PASS `dry_run_enabled_smoke` (313.958s)
- PASS `replay_log_calibration_artifact` (0.931s)
- PASS `fee_evidence_report` (0.096s)
- PASS `hl_data_validation_report` (1.27s)
- PASS `replay_latest_data_smoke` (3.338s)
- PASS `replay_acceptance_report_artifact` (16.466s)
- PASS `live_canary_evidence_report` (0.118s)

Post-run audits:
- PASS `plan_status_audit` (0.079s)

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `hyperliquid_fee_tier`: Requires exchange/account maker-fee evidence and actual maker fill fee rates.
- `live_canary`: Requires docs/live_canary_report.json to pass after several tiny live sessions with post-only, fee, replay, zero-taker, freshness, and kill-switch evidence.