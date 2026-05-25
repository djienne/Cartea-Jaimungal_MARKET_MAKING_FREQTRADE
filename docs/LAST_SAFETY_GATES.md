# Safety Gate Results

- PASS `compileall` (1.248s)
- PASS `pytest_core` (25.22s)
- PASS `compute_spreads_boundary_smoke` (1.656s)
- PASS `replay_smoke` (1.965s)
- PASS `post_only_probe_plan` (0.109s)
- PASS `post_only_evidence_report` (0.1s)
- PASS `direct_alo_adapter_plan` (0.086s)
- PASS `docker_compose_config` (0.188s)
- PASS `freqtrade_runtime_load` (7.786s)
- PASS `dry_run_disabled_smoke` (225.352s)
- PASS `dry_run_enabled_smoke` (314.355s)
- PASS `replay_latest_data_smoke` (3.52s)
- PASS `replay_acceptance_report_artifact` (17.083s)

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `live_canary`: Only allowed after every prior gate passes with post-only and kill-on-taker-fill verified.