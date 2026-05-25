# Safety Gate Results

- PASS `compileall` (1.167s)
- PASS `pytest_core` (27.448s)
- PASS `compute_spreads_boundary_smoke` (1.32s)
- PASS `replay_smoke` (1.814s)
- PASS `post_only_probe_plan` (0.108s)
- PASS `post_only_evidence_report` (0.091s)
- PASS `direct_alo_adapter_plan` (0.125s)
- PASS `docker_compose_config` (0.266s)
- PASS `freqtrade_runtime_load` (7.604s)
- PASS `dry_run_disabled_smoke` (224.629s)
- PASS `dry_run_enabled_smoke` (313.152s)
- PASS `replay_latest_data_smoke` (3.126s)
- PASS `replay_acceptance_report_artifact` (15.806s)

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `live_canary`: Only allowed after every prior gate passes with post-only and kill-on-taker-fill verified.