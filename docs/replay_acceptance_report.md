# Replay Acceptance Report

- status: FAIL
- symbol: `ETH`
- generated_at: `2026-05-25T15:33:52Z`

## Variant Summary

| Variant | Status | Coverage days | Quotes | Maker fills | Taker fills | Net spread | Directional ratio | Reasons |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | FAIL | 0.003800 | 2000 | 0 | 0 | 0.00000000 | 0.000000 | insufficient_coverage_days:0.003800<min_3.000000, no_maker_fills |
| fee_2x | FAIL | 0.003800 | 2000 | 0 | 0 | 0.00000000 | 0.000000 | insufficient_coverage_days:0.003800<min_3.000000, no_maker_fills |
| latency_2x | FAIL | 0.003800 | 2000 | 0 | 0 | 0.00000000 | 0.000000 | insufficient_coverage_days:0.003800<min_3.000000, no_maker_fills |
| params_soft | FAIL | 0.003800 | 2000 | 0 | 0 | 0.00000000 | 0.000000 | insufficient_coverage_days:0.003800<min_3.000000, no_maker_fills |
| params_hard | FAIL | 0.003800 | 3736 | 1 | 0 | 0.00786722 | 1.599052 | insufficient_coverage_days:0.003800<min_3.000000, directional_drift_dominates_pnl:1.599052>max_0.750000 |

## Blocking Reasons

- `baseline:insufficient_coverage_days:0.003800<min_3.000000`
- `baseline:no_maker_fills`
- `fee_2x:insufficient_coverage_days:0.003800<min_3.000000`
- `fee_2x:no_maker_fills`
- `latency_2x:insufficient_coverage_days:0.003800<min_3.000000`
- `latency_2x:no_maker_fills`
- `params_soft:insufficient_coverage_days:0.003800<min_3.000000`
- `params_soft:no_maker_fills`
- `params_hard:insufficient_coverage_days:0.003800<min_3.000000`
- `params_hard:directional_drift_dominates_pnl:1.599052>max_0.750000`