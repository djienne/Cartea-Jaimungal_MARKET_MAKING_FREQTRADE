# Safety Gate Results

- PASS `compileall` (1.225s)
- PASS `pytest_core` (22.889s)
- PASS `compute_spreads_boundary_smoke` (1.574s)
- PASS `replay_smoke` (1.95s)
- PASS `post_only_probe_plan` (0.11s)
- PASS `post_only_evidence_report` (0.115s)
- PASS `direct_alo_adapter_plan` (0.132s)
- PASS `docker_compose_config` (0.241s)
- PASS `freqtrade_runtime_load` (7.743s)
- PASS `dry_run_disabled_smoke` (224.81s)
- PASS `dry_run_enabled_smoke` (313.663s)
- PASS `replay_latest_data_smoke` (3.292s)
- PASS `replay_acceptance_report_artifact` (16.989s)

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `live_canary`: Only allowed after every prior gate passes with post-only and kill-on-taker-fill verified.