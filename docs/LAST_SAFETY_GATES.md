# Safety Gate Results

- PASS `compileall` (0.985s)
- PASS `pytest_core` (27.788s)
- PASS `compute_spreads_boundary_smoke` (1.471s)
- PASS `replay_smoke` (2.097s)
- PASS `post_only_probe_plan` (0.094s)
- PASS `post_only_evidence_report` (0.093s)
- PASS `direct_alo_adapter_plan` (0.104s)
- PASS `docker_compose_config` (0.275s)
- PASS `freqtrade_runtime_load` (7.766s)
- PASS `dry_run_disabled_smoke` (224.593s)
- PASS `dry_run_enabled_smoke` (313.626s)
- PASS `replay_log_calibration_artifact` (1.171s)
- PASS `replay_latest_data_smoke` (3.558s)
- PASS `replay_acceptance_report_artifact` (16.238s)

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `live_canary`: Only allowed after every prior gate passes with post-only and kill-on-taker-fill verified.