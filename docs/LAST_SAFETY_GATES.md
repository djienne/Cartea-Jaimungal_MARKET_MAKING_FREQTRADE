# Safety Gate Results

- PASS `compileall` (0.938s)
- PASS `pytest_core` (28.081s)
- PASS `compute_spreads_boundary_smoke` (1.471s)
- PASS `replay_smoke` (1.951s)
- PASS `post_only_probe_plan` (0.093s)
- PASS `post_only_evidence_report` (0.085s)
- PASS `direct_alo_adapter_plan` (0.109s)
- PASS `docker_compose_config` (0.271s)
- PASS `freqtrade_runtime_load` (8.033s)
- PASS `dry_run_disabled_smoke` (225.309s)
- PASS `dry_run_enabled_smoke` (313.839s)
- PASS `replay_latest_data_smoke` (3.248s)
- PASS `replay_acceptance_report_artifact` (16.286s)

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `live_canary`: Only allowed after every prior gate passes with post-only and kill-on-taker-fill verified.