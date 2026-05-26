# Safety Gate Results

- PASS `compileall` (1.081s)
- PASS `pytest_core` (22.867s)
- PASS `compute_spreads_boundary_smoke` (1.377s)
- PASS `replay_smoke` (2.087s)
- PASS `post_only_probe_plan` (0.108s)
- PASS `post_only_evidence_report` (0.093s)
- PASS `direct_alo_adapter_plan` (0.111s)
- PASS `live_canary_evidence_report` (0.14s)
- PASS `docker_compose_config` (0.295s)
- PASS `freqtrade_runtime_load` (7.34s)
- PASS `dry_run_disabled_smoke` (224.843s)
- PASS `dry_run_enabled_smoke` (313.736s)
- PASS `replay_log_calibration_artifact` (0.88s)
- PASS `fee_evidence_report` (0.111s)
- PASS `hl_data_validation_report` (1.633s)
- PASS `replay_latest_data_smoke` (3.244s)
- PASS `replay_acceptance_report_artifact` (16.728s)

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `hyperliquid_fee_tier`: Requires exchange/account maker-fee evidence and actual maker fill fee rates.
- `live_canary`: Requires docs/live_canary_report.json to pass after several tiny live sessions with post-only, fee, replay, zero-taker, freshness, and kill-switch evidence.