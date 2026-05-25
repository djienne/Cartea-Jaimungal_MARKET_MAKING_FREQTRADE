# Safety Gate Results

- PASS `compileall` (1.056s)
- PASS `pytest_core` (20.942s)
- PASS `compute_spreads_boundary_smoke` (1.649s)
- PASS `replay_smoke` (2.203s)
- PASS `post_only_probe_plan` (0.099s)
- PASS `post_only_evidence_report` (0.099s)
- PASS `direct_alo_adapter_plan` (0.128s)
- PASS `docker_compose_config` (0.27s)
- PASS `freqtrade_runtime_load` (8.06s)
- PASS `dry_run_disabled_smoke` (225.005s)
- PASS `dry_run_enabled_smoke` (314.035s)
- PASS `replay_latest_data_smoke` (3.768s)
- PASS `replay_acceptance_report_artifact` (19.612s)

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `live_canary`: Only allowed after every prior gate passes with post-only and kill-on-taker-fill verified.