# Safety Gate Results

- PASS `compileall` (0.909s)
- PASS `pytest_core` (24.307s)
- PASS `config_safety_report` (0.103s)
- PASS `strategy_safety_report` (0.115s)
- PASS `compute_spreads_boundary_smoke` (1.512s)
- PASS `replay_smoke` (1.788s)
- PASS `post_only_probe_plan` (0.058s)
- PASS `post_only_evidence_report` (0.056s)
- PASS `direct_alo_adapter_plan` (0.073s)
- PASS `docker_compose_config` (0.176s)
- PASS `freqtrade_runtime_load` (7.091s)
- PASS `dry_run_disabled_smoke` (226.069s)
- PASS `dry_run_enabled_smoke` (313.577s)
- PASS `replay_log_calibration_artifact` (0.718s)
- PASS `fee_evidence_report` (0.078s)
- PASS `hl_data_validation_report` (1.526s)
- PASS `replay_latest_data_smoke` (3.515s)
- PASS `replay_acceptance_report_artifact` (15.914s)
- PASS `live_canary_evidence_report` (0.092s)

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `hyperliquid_fee_tier`: Requires exchange/account maker-fee evidence and actual maker fill fee rates.
- `live_canary`: Requires docs/live_canary_report.json to pass after several tiny live sessions with post-only, fee, replay, zero-taker, freshness, and kill-switch evidence.