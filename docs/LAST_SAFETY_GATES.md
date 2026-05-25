# Safety Gate Results

- PASS `compileall` (1.154s)
- PASS `pytest_core` (24.614s)
- PASS `compute_spreads_boundary_smoke` (1.475s)
- PASS `replay_smoke` (2.12s)
- PASS `post_only_probe_plan` (0.104s)
- PASS `post_only_evidence_report` (0.098s)
- PASS `direct_alo_adapter_plan` (0.118s)
- PASS `docker_compose_config` (0.269s)
- PASS `freqtrade_runtime_load` (8.205s)
- PASS `dry_run_disabled_smoke` (225.16s)
- PASS `dry_run_enabled_smoke` (253.954s)
- PASS `replay_latest_data_smoke` (4.488s)
- PASS `replay_acceptance_report_artifact` (17.192s)

Manual gates still required:
- `hyperliquid_post_only_mapping`: Requires testnet/tiny integration evidence that Freqtrade/CCXT PO maps to Hyperliquid Alo.
- `multi_day_event_replay`: Requires several days of fresh HL_data and review of markout/latency/fee sensitivity.
- `live_canary`: Only allowed after every prior gate passes with post-only and kill-on-taker-fill verified.