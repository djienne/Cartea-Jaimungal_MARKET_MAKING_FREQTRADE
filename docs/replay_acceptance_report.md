# Replay Acceptance Report

- status: FAIL
- symbol: `ETH`
- generated_at: `2026-05-25T20:32:24Z`

## Variant Summary

| Variant | Status | Coverage days | Quotes | Maker fills | Taker fills | Net spread | Directional ratio | Reasons |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | FAIL | 0.005553 | 3365 | 1 | 0 | 0.00852888 | 1.461742 | insufficient_coverage_days:0.005553<min_3.000000, directional_drift_dominates_pnl:1.461742>max_0.750000 |
| fee_2x | FAIL | 0.005553 | 2000 | 0 | 0 | 0.00000000 | 0.000000 | insufficient_coverage_days:0.005553<min_3.000000, no_maker_fills |
| latency_2x | FAIL | 0.005553 | 3365 | 1 | 0 | 0.00852888 | 1.461742 | insufficient_coverage_days:0.005553<min_3.000000, directional_drift_dominates_pnl:1.461742>max_0.750000 |
| params_soft | FAIL | 0.005553 | 3365 | 1 | 0 | 0.00990436 | 1.579350 | insufficient_coverage_days:0.005553<min_3.000000, directional_drift_dominates_pnl:1.579350>max_0.750000 |
| params_hard | FAIL | 0.005553 | 3365 | 1 | 0 | 0.00785762 | 1.410483 | insufficient_coverage_days:0.005553<min_3.000000, directional_drift_dominates_pnl:1.410483>max_0.750000 |

## Refusal Checks

| Check | Status | Expected | Decision | Reason |
| --- | --- | --- | --- | --- |
| bad_params_nonpositive_kappa | PASS | reject | reject | invalid_kappa |
| bad_params_toxicity | PASS | reject | reject | toxicity_too_high |
| stale_collector_data | PASS | reject | reject | stale_collector_data |

## Blocking Reasons

- `baseline:insufficient_coverage_days:0.005553<min_3.000000`
- `baseline:directional_drift_dominates_pnl:1.461742>max_0.750000`
- `fee_2x:insufficient_coverage_days:0.005553<min_3.000000`
- `fee_2x:no_maker_fills`
- `latency_2x:insufficient_coverage_days:0.005553<min_3.000000`
- `latency_2x:directional_drift_dominates_pnl:1.461742>max_0.750000`
- `params_soft:insufficient_coverage_days:0.005553<min_3.000000`
- `params_soft:directional_drift_dominates_pnl:1.579350>max_0.750000`
- `params_hard:insufficient_coverage_days:0.005553<min_3.000000`
- `params_hard:directional_drift_dominates_pnl:1.410483>max_0.750000`