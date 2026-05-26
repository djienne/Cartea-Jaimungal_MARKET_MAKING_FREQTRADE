# Post-Only Verification Status

Current status: **not verified for Freqtrade live execution**.

Runtime evidence from the Docker/Freqtrade dry-run gate showed that Freqtrade
2025.4 rejects `order_time_in_force = {"entry": "PO", "exit": "PO"}` for
Hyperliquid with:

```text
Configuration error: Time in force policies are not supported for Hyperliquid yet.
```

Therefore the repository uses `GTC` in `user_data/config.json` and
`Market_Making.order_time_in_force` only so the Freqtrade research/dry-run
harness can start. This does **not** satisfy maker-safe live execution.

Live trading remains blocked by strategy state:

```python
trading_enabled = False
post_only_verified = False
```

The only acceptable live maker path is:

1. Prove Freqtrade/CCXT can submit Hyperliquid native `Alo` orders, or
2. Move live execution to a direct Hyperliquid SDK bot that submits `Alo`
   explicitly.

Use the no-network probe to document required evidence:

```bash
python scripts/verify_post_only_mapping.py --mode plan --output docs/post_only_probe_plan.json
```

Submit-mode probes require explicit acknowledgement and should be run only on
testnet or tiny canary size:

```bash
$env:HYPERLIQUID_POST_ONLY_PROBE_ALLOW = "1"
python scripts/verify_post_only_mapping.py --mode submit-crossing-alo --sandbox --amount <min_size> --acknowledge-real-orders --output docs/post_only_crossing_result.json
python scripts/verify_post_only_mapping.py --mode submit-passive-alo --sandbox --amount <min_size> --acknowledge-real-orders --output docs/post_only_passive_result.json
python scripts/verify_post_only_mapping.py --mode evaluate-evidence --crossing-result docs/post_only_crossing_result.json --passive-result docs/post_only_passive_result.json --output docs/post_only_evidence_report.json
```

The evidence report passes only when:

- submitted params contain `timeInForce=Alo` and `postOnly=true`
- actual exchange/order TIF is confirmed as `Alo`; submitted params alone are
  not enough
- the intentionally crossing ALO order has zero fill and rejects/cancels/expires
- the passive ALO order rests or fills maker-only; a zero-filled cancel without
  resting proof is not enough
- no result contains taker liquidity
- both submit artifacts include fresh `generated_at` timestamps; stale or
  timestamp-less submit artifacts are rejected even if the report itself is
  regenerated later

The automated safety gate writes an incomplete
`docs/post_only_evidence_report.json` when submit artifacts are absent. That
file should remain `ok=false` until real testnet/tiny exchange evidence is
provided.

After the crossing/passive artifacts exist at the conventional paths above,
the full safety runner will pass them through automatically. Custom artifact
paths can be supplied explicitly:

```bash
python scripts/run_safety_gates.py --include-runtime --post-only-crossing-result docs/post_only_crossing_result.json --post-only-passive-result docs/post_only_passive_result.json --markdown-output docs/LAST_SAFETY_GATES.md
```

## Direct SDK Fallback

Because Freqtrade 2025.4 rejects Hyperliquid `PO`, the repo also includes a
guarded direct SDK adapter:

```bash
python scripts/hyperliquid_alo_executor.py --mode plan --output docs/direct_alo_adapter_plan.json
```

The adapter builds SDK orders with:

```python
order_type = {"limit": {"tif": "Alo"}}
```

and requires local BBO maker-safety before submit. Submit mode is intentionally
hard to invoke:

```bash
$env:HYPERLIQUID_DIRECT_ALO_ALLOW = "1"
python scripts/hyperliquid_alo_executor.py --mode submit-alo --testnet --symbol ETH/USDC:USDC --side bid --size <min_size> --price <passive_bid> --best-bid <best_bid> --best-ask <best_ask> --acknowledge-real-orders --output docs/direct_alo_submit_result.json
```

For evidence generation, prefer the passive probe mode. It submits the same
native `Alo` order, classifies the response, and cancels any resting order ids
reported by the SDK:

```bash
$env:HYPERLIQUID_DIRECT_ALO_ALLOW = "1"
python scripts/hyperliquid_alo_executor.py --mode submit-passive-alo --testnet --symbol ETH/USDC:USDC --side bid --size <min_size> --price <passive_bid> --best-bid <best_bid> --best-ask <best_ask> --allow-passive-probe --acknowledge-real-orders --output docs/direct_alo_passive_result.json
```

The canonical evidence checker also accepts direct SDK submit artifacts. It
normalizes `sdk_order_args.order_type.limit.tif = Alo` into the same post-only
evidence shape used for CCXT/Freqtrade probes and reads the direct adapter's
`classification` block for resting, maker-fill, taker-fill, or ALO-rejection
outcomes:

```bash
python scripts/verify_post_only_mapping.py --mode evaluate-evidence --crossing-result docs/direct_alo_reject_result.json --passive-result docs/direct_alo_passive_result.json --output docs/post_only_evidence_report.json
```

The direct adapter requires local maker-safety before normal submit mode, so an
intentional crossing-order evidence uses a separate guarded mode:

```bash
$env:HYPERLIQUID_DIRECT_ALO_ALLOW = "1"
python scripts/hyperliquid_alo_executor.py --mode submit-crossing-alo --testnet --symbol ETH/USDC:USDC --side bid --size <min_size> --best-bid <best_bid> --best-ask <best_ask> --allow-crossing-probe --acknowledge-real-orders --output docs/direct_alo_reject_result.json
```

For a bid probe, the adapter submits at the observed best ask; for an ask probe,
it submits at the observed best bid. The order type remains native SDK
`{"limit": {"tif": "Alo"}}`, so the acceptable outcome is zero fill with an ALO
post-only rejection/cancel. Non-testnet crossing probes require the additional
`--allow-mainnet-crossing-probe` flag and should only use the smallest possible
order size. A direct SDK ALO rejection caused by market movement is still valid
rejection evidence if the retained result shows zero fill and the classifier
marks `alo_rejected=true`.

This adapter is not wired into the Freqtrade strategy. It is the implementation
scaffold for a future direct Hyperliquid execution layer if Freqtrade cannot be
made maker-safe.
