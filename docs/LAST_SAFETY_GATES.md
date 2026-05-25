# Safety Gate Results

- PASS `compileall` (1.125s)
- PASS `pytest_core` (23.296s)
- PASS `compute_spreads_boundary_smoke` (1.286s)
- PASS `replay_smoke` (1.743s)
- PASS `post_only_probe_plan` (0.086s)
- PASS `post_only_evidence_report` (0.078s)
- PASS `direct_alo_adapter_plan` (0.12s)
- PASS `docker_compose_config` (0.253s)
- PASS `freqtrade_runtime_load` (7.597s)
- PASS `dry_run_disabled_smoke` (224.921s)
- PASS `dry_run_enabled_smoke` (313.65s)
- PASS `replay_log_calibration_artifact` (0.729s)
- PASS `hl_data_validation_report` (1.727s)
- PASS `replay_latest_data_smoke` (3.264s)
- PASS `replay_acceptance_report_artifact` (17.666s)

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `live_canary`: Only allowed after every prior gate passes with post-only and kill-on-taker-fill verified.