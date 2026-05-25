# Safety Gate Results

- PASS `compileall` (1.071s)
- PASS `pytest_core` (22.596s)
- PASS `compute_spreads_boundary_smoke` (1.459s)
- PASS `replay_smoke` (1.855s)
- PASS `post_only_probe_plan` (0.061s)
- PASS `post_only_evidence_report` (0.059s)
- PASS `direct_alo_adapter_plan` (0.076s)
- PASS `live_canary_evidence_report` (0.077s)
- PASS `docker_compose_config` (0.162s)
- PASS `freqtrade_runtime_load` (7.745s)
- PASS `dry_run_disabled_smoke` (225.324s)
- PASS `dry_run_enabled_smoke` (313.89s)
- PASS `replay_log_calibration_artifact` (0.534s)
- PASS `fee_evidence_report` (0.069s)
- PASS `hl_data_validation_report` (1.534s)
- PASS `replay_latest_data_smoke` (3.188s)
- PASS `replay_acceptance_report_artifact` (16.309s)

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `hyperliquid_fee_tier`: Requires exchange/account maker-fee evidence and actual maker fill fee rates.
- `live_canary`: Requires docs/live_canary_report.json to pass after several tiny live sessions with post-only, fee, replay, zero-taker, freshness, and kill-switch evidence.