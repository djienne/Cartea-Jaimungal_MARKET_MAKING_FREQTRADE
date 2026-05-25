# Safety Gate Results

- PASS `compileall` (1.005s)
- PASS `pytest_core` (20.833s)
- PASS `compute_spreads_boundary_smoke` (1.376s)
- PASS `replay_smoke` (2.137s)
- PASS `post_only_probe_plan` (0.102s)
- PASS `post_only_evidence_report` (0.1s)
- PASS `direct_alo_adapter_plan` (0.112s)
- PASS `docker_compose_config` (0.267s)
- PASS `freqtrade_runtime_load` (8.139s)
- PASS `dry_run_disabled_smoke` (225.023s)
- PASS `dry_run_enabled_smoke` (313.704s)
- PASS `replay_latest_data_smoke` (3.578s)
- PASS `replay_acceptance_report_artifact` (17.035s)

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `live_canary`: Only allowed after every prior gate passes with post-only and kill-on-taker-fill verified.