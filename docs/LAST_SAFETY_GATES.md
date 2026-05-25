# Safety Gate Results

- PASS `compileall` (0.961s)
- PASS `pytest_core` (25.118s)
- PASS `compute_spreads_boundary_smoke` (1.773s)
- PASS `replay_smoke` (2.15s)
- PASS `post_only_probe_plan` (0.09s)
- PASS `post_only_evidence_report` (0.096s)
- PASS `direct_alo_adapter_plan` (0.077s)
- PASS `docker_compose_config` (0.157s)
- PASS `freqtrade_runtime_load` (8.211s)
- PASS `dry_run_disabled_smoke` (224.921s)
- PASS `dry_run_enabled_smoke` (313.958s)
- PASS `replay_latest_data_smoke` (3.431s)
- PASS `replay_acceptance_report_artifact` (15.665s)

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `live_canary`: Only allowed after every prior gate passes with post-only and kill-on-taker-fill verified.