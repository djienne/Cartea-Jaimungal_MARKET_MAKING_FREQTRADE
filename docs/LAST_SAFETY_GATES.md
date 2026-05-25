# Safety Gate Results

- PASS `compileall` (1.142s)
- PASS `pytest_core` (25.861s)
- PASS `compute_spreads_boundary_smoke` (1.403s)
- PASS `replay_smoke` (2.224s)
- PASS `post_only_probe_plan` (0.069s)
- PASS `post_only_evidence_report` (0.071s)
- PASS `direct_alo_adapter_plan` (0.086s)
- PASS `docker_compose_config` (0.17s)
- PASS `freqtrade_runtime_load` (8.418s)
- PASS `dry_run_disabled_smoke` (225.092s)
- PASS `dry_run_enabled_smoke` (314.347s)
- PASS `replay_latest_data_smoke` (3.392s)
- PASS `replay_acceptance_report_artifact` (16.043s)

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `live_canary`: Only allowed after every prior gate passes with post-only and kill-on-taker-fill verified.