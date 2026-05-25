# Safety Gate Results

- PASS `compileall` (1.14s)
- PASS `pytest_core` (20.321s)
- PASS `compute_spreads_boundary_smoke` (1.553s)
- PASS `replay_smoke` (1.82s)
- PASS `post_only_probe_plan` (0.12s)
- PASS `post_only_evidence_report` (0.084s)
- PASS `direct_alo_adapter_plan` (0.135s)
- PASS `docker_compose_config` (0.254s)
- PASS `freqtrade_runtime_load` (8.155s)
- PASS `dry_run_disabled_smoke` (224.644s)
- PASS `dry_run_enabled_smoke` (314.143s)
- PASS `replay_latest_data_smoke` (3.348s)
- PASS `replay_acceptance_report_artifact` (16.993s)

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `live_canary`: Only allowed after every prior gate passes with post-only and kill-on-taker-fill verified.