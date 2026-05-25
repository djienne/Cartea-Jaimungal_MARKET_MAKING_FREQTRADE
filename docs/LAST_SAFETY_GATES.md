# Safety Gate Results

- PASS `compileall` (1.187s)
- PASS `pytest_core` (22.265s)
- PASS `compute_spreads_boundary_smoke` (1.563s)
- PASS `replay_smoke` (2.419s)
- PASS `post_only_probe_plan` (0.117s)
- PASS `post_only_evidence_report` (0.105s)
- PASS `direct_alo_adapter_plan` (0.155s)
- PASS `docker_compose_config` (0.291s)
- PASS `freqtrade_runtime_load` (9.194s)
- PASS `dry_run_disabled_smoke` (225.226s)
- PASS `dry_run_enabled_smoke` (313.681s)
- PASS `replay_log_calibration_artifact` (0.698s)
- PASS `hl_data_validation_report` (1.73s)
- PASS `replay_latest_data_smoke` (2.824s)
- PASS `replay_acceptance_report_artifact` (15.624s)

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `live_canary`: Only allowed after every prior gate passes with post-only and kill-on-taker-fill verified.