"""Replay one grid variant over a range of the collected tape at assumed latencies.

    python scripts/replay_latency.py --variant sweep1_flat300 \
        --from 2026-09-02T00:00:00Z --to 2026-09-06T20:00:00Z --latency 0 50 150 300

Each rung runs `mm-live replay --from --to --latency-ms` in a throwaway
container of the grid image, so the running grid is untouched. The first
train_fraction of the range only seeds sizing; the rest is scored. Reports
land in --out. Note the variant's exit deadline is flatten_after_ms plus twice
the latency, so a rung moves that too.

    python scripts/replay_latency.py --variant sweep1_flat300 --against-live

replays the running grid's own window at the configured latency and prints
the leaderboard row above it: the fidelity check between replay and dry run.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "rust_live" / "config"
GRID = REPO / "rust_live" / "reports" / "grid_live"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True)
    p.add_argument("--from", dest="start", help="RFC 3339 or epoch ms")
    p.add_argument("--to", dest="end", help="RFC 3339 or epoch ms")
    p.add_argument("--latency", type=int, nargs="+", help="ms, one rung each")
    p.add_argument("--against-live", action="store_true", help="the grid's own window at the config latency")
    p.add_argument("--train-fraction", type=float, default=0.05)
    p.add_argument("--data-dir", type=Path, default=REPO / "scripts" / "HL_data")
    p.add_argument("--out", type=Path, default=REPO / "rust_live" / "reports" / "replay_latency")
    p.add_argument("--image", default="mm-live:local")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    config = (CONFIG / "cashcat_dryrun_realistic.toml").read_text()
    starting_equity = float(re.search(r"^starting_equity_usdc\s*=\s*([\d.]+)", config, re.M).group(1))

    if args.against_live:
        board = json.loads((GRID / "leaderboard.json").read_text())
        row = next(r for r in board["rows"] if r["name"] == args.variant)
        args.start, args.end = str(board["started_at_ms"]), str(board["generated_at_ms"])
        args.latency = [int(re.search(r"^decision_latency_ms\s*=\s*(\d+)", config, re.M).group(1))]
        print(f"live  {row['net_pnl_usdc']:8.2f} {row['realized_pnl_usdc']:8.2f} {row['fees_usdc']:6.2f} "
              f"{row['inventory_units']:5d} {row['fills']:6d}   resumes={board['resumes']} "
              f"downtime_s={board['resumed_downtime_ms'] / 1000:.0f} feed_failures={board['feed_failures']}")
    elif not (args.start and args.end and args.latency):
        p.error("--from, --to and --latency are required unless --against-live")

    print(f"{'lat':>5} {'net':>8} {'real':>8} {'fees':>6} {'inv':>5} {'fills':>6} {'rejects':>8} "
          f"{'unkQ':>6} {'orders':>7} {'flat':>5} {'score_h':>8} valid", flush=True)
    for lat in args.latency:
        report = f"{args.variant}_lat{lat}.json"
        subprocess.run([
            "docker", "run", "--rm", "--cpus", "3",
            "-v", f"{CONFIG}:/opt/mm/config:ro",
            "-v", f"{args.out.resolve()}:/opt/mm/reports",
            "-v", f"{args.data_dir.resolve()}:/opt/scripts/HL_data:ro",
            "--entrypoint", "/usr/local/bin/mm-live", args.image,
            "--config", "/opt/mm/config/cashcat_dryrun_realistic.toml", "replay",
            "--grid", "/opt/mm/config/grid_cashcat.toml", "--variant", args.variant,
            "--from", args.start, "--to", args.end, "--latency-ms", str(lat),
            "--train-fraction", str(args.train_fraction), "--report", f"/opt/mm/reports/{report}",
        ], check=True)
        j = json.loads((args.out / report).read_text())
        a, e, r = j["account"], j["execution"], j["replay"]
        hours = (r["scored_until_ms"] - r["scoring_start_ms"]) / 3_600_000
        print(f"{lat:5d} {a['equity_usdc'] - starting_equity:8.2f} {a['realized_pnl_usdc']:8.2f} "
              f"{a['fees_usdc']:6.2f} {a['inventory_units']:5d} {e['fills']:6d} {e['post_only_rejects']:8d} "
              f"{e['unknown_queue_activations']:6d} {e['virtual_orders_created']:7d} {e['flatten_events']:5d} "
              f"{hours:8.2f} {j['scientifically_valid']}", flush=True)


if __name__ == "__main__":
    main()
