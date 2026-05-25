# Safety Gate Results

- PASS `compileall` (1.216s)
- PASS `pytest_core` (18.595s)
- PASS `compute_spreads_boundary_smoke` (1.539s)
- PASS `replay_smoke` (2.138s)
- PASS `post_only_probe_plan` (0.093s)
- PASS `post_only_evidence_report` (0.096s)
- PASS `direct_alo_adapter_plan` (0.117s)
- PASS `docker_compose_config` (0.279s)
- PASS `freqtrade_runtime_load` (9.372s)
- PASS `dry_run_disabled_smoke` (225.456s)
- PASS `dry_run_enabled_smoke` (254.459s)
- PASS `replay_latest_data_smoke` (4.269s)
- PASS `replay_acceptance_report_artifact` (18.052s)

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `live_canary`: Only allowed after every prior gate passes with post-only and kill-on-taker-fill verified.