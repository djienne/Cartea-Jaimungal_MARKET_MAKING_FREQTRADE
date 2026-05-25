# Safety Gate Results

- PASS `compileall` (1.115s)
- PASS `pytest_core` (25.949s)
- PASS `compute_spreads_boundary_smoke` (1.354s)
- PASS `replay_smoke` (2.045s)
- PASS `post_only_probe_plan` (0.095s)
- PASS `post_only_evidence_report` (0.091s)
- PASS `direct_alo_adapter_plan` (0.12s)
- PASS `docker_compose_config` (0.267s)
- PASS `freqtrade_runtime_load` (7.806s)
- PASS `dry_run_disabled_smoke` (224.703s)
- PASS `dry_run_enabled_smoke` (314.853s)
- PASS `replay_log_calibration_artifact` (1.242s)
- PASS `replay_latest_data_smoke` (4.188s)
- PASS `replay_acceptance_report_artifact` (20.344s)

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `live_canary`: Only allowed after every prior gate passes with post-only and kill-on-taker-fill verified.