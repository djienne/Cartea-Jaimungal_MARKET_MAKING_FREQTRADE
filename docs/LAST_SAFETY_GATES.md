# Safety Gate Results

- PASS `compileall` (1.15s)
- PASS `pytest_core` (21.088s)
- PASS `compute_spreads_boundary_smoke` (1.577s)
- PASS `replay_smoke` (2.284s)
- PASS `post_only_probe_plan` (0.101s)
- PASS `post_only_evidence_report` (0.095s)
- PASS `direct_alo_adapter_plan` (0.149s)
- PASS `docker_compose_config` (0.274s)
- PASS `freqtrade_runtime_load` (7.947s)
- PASS `dry_run_disabled_smoke` (225.164s)
- PASS `dry_run_enabled_smoke` (313.88s)
- PASS `replay_latest_data_smoke` (3.683s)
- PASS `replay_acceptance_report_artifact` (17.706s)

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `live_canary`: Only allowed after every prior gate passes with post-only and kill-on-taker-fill verified.