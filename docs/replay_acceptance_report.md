# Replay Acceptance Report

- status: FAIL
- symbol: `ETH`
- generated_at: `2026-05-26T06:55:13Z`

## Variant Summary

| Variant | Status | Coverage days | Quotes | Maker fills | Taker fills | Net spread | Directional ratio | Reasons |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | FAIL | 0.005295 | 3458 | 1 | 0 | 0.00807707 | 0.352851 | insufficient_coverage_days:0.005295<min_3.000000 |
| fee_2x | FAIL | 0.005295 | 2000 | 0 | 0 | 0.00000000 | 0.000000 | insufficient_coverage_days:0.005295<min_3.000000, no_maker_fills |
| latency_2x | FAIL | 0.005295 | 3458 | 1 | 0 | 0.00807707 | 0.352851 | insufficient_coverage_days:0.005295<min_3.000000 |
| params_soft | FAIL | 0.005295 | 3458 | 1 | 0 | 0.00987765 | 0.303716 | insufficient_coverage_days:0.005295<min_3.000000 |
| params_hard | FAIL | 0.005295 | 3458 | 1 | 0 | 0.00743937 | 0.383097 | insufficient_coverage_days:0.005295<min_3.000000 |

## Refusal Checks

| Check | Status | Expected | Decision | Reason |
| --- | --- | --- | --- | --- |
| bad_params_nonpositive_kappa | PASS | reject | reject | invalid_kappa |
| bad_params_toxicity | PASS | reject | reject | toxicity_too_high |
| stale_collector_data | PASS | reject | reject | stale_collector_data |

## Blocking Reasons

- `baseline:insufficient_coverage_days:0.005295<min_3.000000`
- `fee_2x:insufficient_coverage_days:0.005295<min_3.000000`
- `fee_2x:no_maker_fills`
- `latency_2x:insufficient_coverage_days:0.005295<min_3.000000`
- `params_soft:insufficient_coverage_days:0.005295<min_3.000000`
- `params_hard:insufficient_coverage_days:0.005295<min_3.000000`