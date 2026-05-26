# Safety Gate Results

- PASS `compileall` (1.085s)
- PASS `pytest_core` (23.615s)
- PASS `compute_spreads_boundary_smoke` (1.434s)
- PASS `replay_smoke` (1.969s)
- PASS `post_only_probe_plan` (0.066s)
- PASS `post_only_evidence_report` (0.061s)
- PASS `direct_alo_adapter_plan` (0.078s)
- PASS `live_canary_evidence_report` (0.107s)
- PASS `docker_compose_config` (0.25s)
- PASS `freqtrade_runtime_load` (6.989s)
- PASS `dry_run_disabled_smoke` (225.037s)
- PASS `dry_run_enabled_smoke` (313.588s)
- PASS `replay_log_calibration_artifact` (0.588s)
- PASS `fee_evidence_report` (0.093s)
- PASS `hl_data_validation_report` (1.647s)
- PASS `replay_latest_data_smoke` (3.423s)
- PASS `replay_acceptance_report_artifact` (16.377s)

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `hyperliquid_fee_tier`: Requires exchange/account maker-fee evidence and actual maker fill fee rates.
- `live_canary`: Requires docs/live_canary_report.json to pass after several tiny live sessions with post-only, fee, replay, zero-taker, freshness, and kill-switch evidence.