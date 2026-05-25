# Safety Gate Results

- PASS `compileall` (1.08s)
- PASS `pytest_core` (21.878s)
- PASS `compute_spreads_boundary_smoke` (1.45s)
- PASS `replay_smoke` (2.171s)
- PASS `post_only_probe_plan` (0.174s)
- PASS `post_only_evidence_report` (0.082s)
- PASS `direct_alo_adapter_plan` (0.116s)
- PASS `docker_compose_config` (0.325s)
- PASS `freqtrade_runtime_load` (8.105s)
- PASS `dry_run_disabled_smoke` (225.028s)
- PASS `dry_run_enabled_smoke` (313.896s)
- PASS `replay_latest_data_smoke` (3.963s)
- PASS `replay_acceptance_report_artifact` (16.231s)

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `live_canary`: Only allowed after every prior gate passes with post-only and kill-on-taker-fill verified.