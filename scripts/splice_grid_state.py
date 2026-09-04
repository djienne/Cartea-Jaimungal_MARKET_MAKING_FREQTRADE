#!/usr/bin/env python3
"""Carry a grid run forward across a base-config change, restarting named rows.

One-shot. Raising `max_consecutive_losses` rewrites every variant's
`config_fingerprint`, so the resume check refuses the checkpoint and the whole
grid restarts from zero. For the 19 rows that never reached the old cap the
change is a no-op on behaviour, so their accounting is still theirs. The row
that DID reach it is the one whose history the change invalidates, and that row
is restarted rather than spliced.

Takes the FRESH checkpoint as the template -- so the new fingerprints are the
ones the binary actually computed, never ones written by hand -- and copies the
old accounting into it.
"""
import argparse
import json
import sys

ACCOUNTING = (
    "account",
    "diagnostics",
    "fills",
    "inventory_unit",
    "current_day",
    "daily_realized_pnl_usdc",
    "max_drawdown_usdc",
    "peak_equity_usdc",
    "failure",
)
RUN_IDENTITY = (
    "run_id",
    "started_at_ms",
    "checkpoint_ms",
    "resumes",
    "resumed_downtime_ms",
    "feed_health",
    "trade_prints",
    "replayed_trades_ignored",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--fresh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--restart", nargs="*", default=[])
    args = ap.parse_args()

    old = json.load(open(args.old, encoding="utf-8"))
    fresh = json.load(open(args.fresh, encoding="utf-8"))

    if old["symbol"] != fresh["symbol"]:
        sys.exit("symbol mismatch")
    if old["schema_version"] != fresh["schema_version"]:
        sys.exit("state schema changed; splice by hand or start fresh")
    if old["grid_fingerprint"] == fresh["grid_fingerprint"]:
        sys.exit("fingerprints already agree -- nothing to splice, just restart")

    by_name = {v["name"]: v for v in old["variants"]}
    missing = [v["name"] for v in fresh["variants"] if v["name"] not in by_name]
    if missing:
        sys.exit(f"the old checkpoint has no {missing}; the grid spec changed too")

    out = json.loads(json.dumps(fresh))
    for key in RUN_IDENTITY:
        out[key] = old[key]

    for variant in out["variants"]:
        name = variant["name"]
        if name in args.restart:
            print(f"  {name:<16} RESTART (kept the fresh zeroed account)")
            continue
        source = by_name[name]
        for key in ACCOUNTING:
            variant[key] = json.loads(json.dumps(source[key]))
        print(
            f"  {name:<16} kept  pnl={source['account']['mark_to_market_pnl_usdc']:+8.2f} "
            f"fills={source['fills']:>5} cl={source['account']['consecutive_losses']}"
        )

    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {args.out}")
    print(f"  grid_fingerprint: {out['grid_fingerprint'][:60]}...")
    print(f"  run_id {out['run_id']}  checkpoint_ms {out['checkpoint_ms']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
