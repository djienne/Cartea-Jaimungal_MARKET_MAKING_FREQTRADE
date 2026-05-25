# Safety Gate Results

- PASS `compileall` (1.093s)
- PASS `pytest_core` (22.256s)
- PASS `compute_spreads_boundary_smoke` (1.492s)
- PASS `replay_smoke` (1.969s)
- PASS `post_only_probe_plan` (0.067s)
- PASS `post_only_evidence_report` (0.063s)
- PASS `direct_alo_adapter_plan` (0.076s)
- PASS `docker_compose_config` (0.168s)
- PASS `freqtrade_runtime_load` (7.503s)
- PASS `dry_run_disabled_smoke` (225.147s)
- PASS `dry_run_enabled_smoke` (313.79s)
- PASS `replay_latest_data_smoke` (2.803s)
- PASS `replay_acceptance_report_artifact` (15.276s)

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `live_canary`: Only allowed after every prior gate passes with post-only and kill-on-taker-fill verified.