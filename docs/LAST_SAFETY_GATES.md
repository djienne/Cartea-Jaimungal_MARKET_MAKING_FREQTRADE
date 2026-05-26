# Safety Gate Results

- PASS `compileall` (1.095s)
- PASS `pytest_core` (24.58s)
- PASS `config_safety_report` (0.091s)
- PASS `strategy_safety_report` (0.094s)
- PASS `compute_spreads_boundary_smoke` (1.524s)
- PASS `replay_smoke` (2.087s)
- PASS `post_only_probe_plan` (0.096s)
- PASS `post_only_evidence_report` (0.115s)
- PASS `direct_alo_adapter_plan` (0.136s)
- PASS `docker_compose_config` (0.265s)
- PASS `freqtrade_runtime_load` (8.066s)
- PASS `dry_run_disabled_smoke` (224.7s)
- PASS `dry_run_enabled_smoke` (313.744s)
- PASS `replay_log_calibration_artifact` (0.91s)
- PASS `fee_evidence_report` (0.081s)
- PASS `hl_data_validation_report` (1.965s)
- PASS `replay_latest_data_smoke` (3.522s)
- PASS `replay_acceptance_report_artifact` (16.882s)
- PASS `live_canary_evidence_report` (0.101s)

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `hyperliquid_fee_tier`: Requires exchange/account maker-fee evidence and actual maker fill fee rates.
- `live_canary`: Requires docs/live_canary_report.json to pass after several tiny live sessions with post-only, fee, replay, zero-taker, freshness, and kill-switch evidence.