# Safety Gate Results

- PASS `compileall` (1.148s)
- PASS `pytest_core` (20.792s)
- PASS `compute_spreads_boundary_smoke` (1.536s)
- PASS `replay_smoke` (2.138s)
- PASS `post_only_probe_plan` (0.092s)
- PASS `post_only_evidence_report` (0.08s)
- PASS `direct_alo_adapter_plan` (0.116s)
- PASS `docker_compose_config` (0.32s)
- PASS `freqtrade_runtime_load` (11.471s)
- PASS `dry_run_disabled_smoke` (224.822s)
- PASS `dry_run_enabled_smoke` (315.275s)
- PASS `replay_latest_data_smoke` (3.587s)
- PASS `replay_acceptance_report_artifact` (17.159s)

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `live_canary`: Only allowed after every prior gate passes with post-only and kill-on-taker-fill verified.