# Safety Gate Results

- PASS `compileall` (0.948s)
- PASS `pytest_core` (25.795s)
- PASS `compute_spreads_boundary_smoke` (1.606s)
- PASS `replay_smoke` (1.98s)
- PASS `post_only_probe_plan` (0.132s)
- PASS `post_only_evidence_report` (0.096s)
- PASS `direct_alo_adapter_plan` (0.122s)
- PASS `docker_compose_config` (0.176s)
- PASS `freqtrade_runtime_load` (8.611s)
- PASS `dry_run_disabled_smoke` (224.8s)
- PASS `dry_run_enabled_smoke` (314.0s)
- PASS `replay_latest_data_smoke` (4.306s)
- PASS `replay_acceptance_report_artifact` (17.1s)

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `live_canary`: Only allowed after every prior gate passes with post-only and kill-on-taker-fill verified.