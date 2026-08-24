"""P&L curve over time for dry-run-grid variants.

Two sources, preferred in this order:

1. **`equity_history.csv`** written by the grid itself (one row per variant
   per `--history-seconds`, default 60). This is each variant's own view of
   its equity, including funding and drawdown, and it needs nothing else on
   disk. Always prefer it.
2. **Reconstruction from the fill logs**, for runs that predate the history
   file (or if it was disabled with `--history-seconds 0`):

       pnl(t) = cash(t) + inventory(t) * mid(t) - fees(t)

   where cash/inventory/fees come from the fill stream and `mid(t)` from the
   collector's `prices` tape -- not the log's `quote_decision` lines, because
   `DesiredQuotes::empty` hardcodes `mid: 0.0`, so the logged mid is unusable
   exactly when a variant is gated. This path is slower, needs the tape to
   still be within retention, and cannot recover funding.

Usage:
    python scripts/grid_pnl_curve.py                       # every variant
    python scripts/grid_pnl_curve.py wide8 baseline        # a subset
    python scripts/grid_pnl_curve.py --out curve.png --minutes 5
    python scripts/grid_pnl_curve.py --from-fills          # force source 2
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "guard_study"))
import tape as T  # noqa: E402

REPO = SCRIPTS.parent
GRID_DIR = REPO / "rust_live" / "reports" / "grid_live"
DATA_DIR = REPO / "scripts" / "HL_data"

# Integer price units per 1.0 of quote currency, from the wire encoding
# (a fill logged px=131510 is a mid of 0.13151).
PX_SCALE = 1e6


def open_log(path: Path):
    """Open a grid event log, transparently decompressing `.jsonl.zst`.

    NEVER use `ZstdDecompressor.decompress()` on these files: the log appends a
    new zstd frame on every restart, and the one-shot API stops at the first
    frame and reports success -- silently returning a fraction of the data.
    `stream_reader` reads across frames, and tolerates the final frame being
    mid-write while the grid is still running.
    """
    if path.suffix != ".zst":
        return open(path, encoding="utf-8")
    import io

    import zstandard

    reader = zstandard.ZstdDecompressor().stream_reader(
        path.open("rb"), read_across_frames=True
    )
    return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")


def variant_files(names: list[str] | None) -> dict[str, Path]:
    """Newest log per variant.

    Two naming schemes coexist: the current stable `grid-<variant>.jsonl.zst`,
    which appends across restarts, and the historical timestamped
    `grid-<variant>-<ms>.jsonl` from runs before compression.
    """
    found: dict[str, Path] = {}
    for path in sorted(GRID_DIR.glob("grid-*.jsonl*")):
        name = path.name
        for suffix in (".jsonl.zst", ".jsonl"):
            if name.endswith(suffix):
                stem = name[: -len(suffix)]
                break
        else:
            continue
        # Strip the legacy trailing -<13-digit ms> if present.
        variant = re.sub(r"-\d{13}$", "", stem[len("grid-") :])
        if names and variant not in names:
            continue
        if variant not in found or path.stat().st_mtime > found[variant].stat().st_mtime:
            found[variant] = path
    return found


def read_fills(path: Path) -> pd.DataFrame:
    """Fill stream for one variant. Only fill lines are parsed; the last line
    can be a partial write because the grid is still running."""
    rows = []
    with open_log(path) as handle:
        for line in handle:
            if '"kind":"fill"' not in line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn final line
            payload = record["payload"]
            when = record.get("exchange_ms") or payload.get("exchange_ms")
            if when is None:
                continue
            signed = payload["qty_units"] if payload["side"].lower() == "buy" else -payload["qty_units"]
            rows.append((float(when), float(signed), float(payload["px"]) / PX_SCALE, float(payload["fee_usdc"])))
    return pd.DataFrame(rows, columns=["ts_ms", "signed_qty", "px", "fee"])


def equity_curve(fills: pd.DataFrame, grid_ms: np.ndarray, mid: np.ndarray) -> np.ndarray:
    """Mark the fill-derived position to `mid` on a regular time grid."""
    if fills.empty:
        return np.zeros(len(grid_ms))
    # Cash is the signed notional paid out; inventory the signed units held.
    cash = np.cumsum(-fills["signed_qty"].to_numpy() * fills["px"].to_numpy())
    inventory = np.cumsum(fills["signed_qty"].to_numpy())
    fees = np.cumsum(fills["fee"].to_numpy())
    # State as of each grid point: the last fill at or before it (none => flat).
    idx = np.searchsorted(fills["ts_ms"].to_numpy(), grid_ms, side="right") - 1
    valid = idx >= 0
    safe = np.clip(idx, 0, None)
    out = np.where(valid, cash[safe] + inventory[safe] * mid - fees[safe], 0.0)
    return out


def curves_from_history(path: Path, names: list[str] | None):
    """Read the grid's own equity history. Returns (hours, mid, curves, fills)."""
    frame = pd.read_csv(path)
    if names:
        frame = frame[frame["variant"].isin(names)]
    if frame.empty:
        raise SystemExit(f"{path} has no rows for the requested variants")
    # A restart appends to the same file; keep only the newest run so two
    # series are never spliced into one misleading curve.
    latest_run = frame["run_started_ms"].max()
    runs = frame["run_started_ms"].nunique()
    frame = frame[frame["run_started_ms"] == latest_run]
    if runs > 1:
        print(f"note: {path.name} holds {runs} runs; plotting the newest only")
    grid_ms = np.sort(frame["ts_ms"].unique())
    hours = (grid_ms - grid_ms[0]) / 3_600_000.0
    mid_by_ts = frame.groupby("ts_ms")["mid"].last()
    mid = mid_by_ts.reindex(grid_ms).to_numpy(dtype=float)
    curves, fills = {}, {}
    for name, group in frame.groupby("variant"):
        series = group.set_index("ts_ms")["net_pnl_usdc"].reindex(grid_ms)
        curves[str(name)] = series.ffill().fillna(0.0).to_numpy(dtype=float)
        fills[str(name)] = int(group["fills"].iloc[-1])
    return hours, mid, curves, fills


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="*", help="variant names (default: all)")
    parser.add_argument("--out", default=str(GRID_DIR / "pnl_curve.png"))
    parser.add_argument("--minutes", type=float, default=2.0, help="grid resolution (fill reconstruction only)")
    parser.add_argument("--csv", help="also write the curves as CSV")
    parser.add_argument("--history", default=str(GRID_DIR / "equity_history.csv"))
    parser.add_argument("--from-fills", action="store_true", help="ignore the history file")
    args = parser.parse_args()

    history_path = Path(args.history)
    if history_path.exists() and not args.from_fills:
        hours, mid, curves, fill_counts = curves_from_history(history_path, args.variants or None)
        print(f"source: {history_path.name} ({len(hours)} samples)")
        render(hours, mid, curves, fill_counts, args)
        return

    files = variant_files(args.variants or None)
    if not files:
        raise SystemExit(f"no variant logs found under {GRID_DIR}")

    fills = {name: read_fills(path) for name, path in files.items()}
    non_empty = [f for f in fills.values() if not f.empty]
    if not non_empty:
        raise SystemExit("no fills recorded yet")
    start_ms = min(f["ts_ms"].iloc[0] for f in non_empty)
    end_ms = max(f["ts_ms"].iloc[-1] for f in non_empty)

    prices = T.load_stream(DATA_DIR, "prices", start_ms - 600_000, end_ms + 600_000)
    bbo = T.build_bbo_full(prices)
    if bbo.empty:
        raise SystemExit("no price tape covering the run window")

    step_ms = args.minutes * 60_000.0
    grid_ms = np.arange(start_ms, end_ms + step_ms, step_ms)
    mid = np.interp(grid_ms, bbo["ts_ms"].to_numpy(), bbo["mid"].to_numpy())

    curves = {name: equity_curve(frame, grid_ms, mid) for name, frame in fills.items()}
    hours = (grid_ms - grid_ms[0]) / 3_600_000.0
    print("source: fill logs (no equity_history.csv)")
    render(hours, mid, curves, {n: len(f) for n, f in fills.items()}, args)


def render(hours, mid, curves, fill_counts, args) -> None:
    ordered = sorted(curves, key=lambda n: -curves[n][-1])
    print(f"{'variant':11s}{'final':>10s}{'peak':>10s}{'trough':>10s}{'fills':>8s}")
    for name in ordered:
        curve = curves[name]
        print(
            f"{name:11s}{curve[-1]:+10.2f}{curve.max():+10.2f}"
            f"{curve.min():+10.2f}{fill_counts.get(name, 0):8d}"
        )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax_px) = plt.subplots(
        2, 1, figsize=(13, 8.5), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}
    )
    palette = plt.get_cmap("turbo")(np.linspace(0.06, 0.95, len(ordered)))
    for colour, name in zip(palette, ordered):
        curve = curves[name]
        emphasis = name in ("wide8", "baseline")
        ax.plot(
            hours, curve, label=f"{name}  {curve[-1]:+.2f}",
            color=colour, linewidth=2.4 if emphasis else 1.3, alpha=1.0 if emphasis else 0.75,
            zorder=3 if emphasis else 2,
        )
    ax.axhline(0, color="#444", linewidth=1.0, linestyle="--", zorder=1)
    ax.set_ylabel("net P&L (USDC)")
    ax.set_title(
        # Resolution comes from the data, not the flag: the history file has
        # its own interval and --minutes does not apply to it.
        f"CASHCAT dry-run grid — net P&L vs time "
        f"({hours[-1]:.1f} h, {np.median(np.diff(hours)) * 60:.0f} min resolution)",
        fontsize=13, pad=12,
    )
    ax.legend(loc="lower left", fontsize=9, ncol=2, framealpha=0.9)
    ax.grid(alpha=0.25)

    ax_px.plot(hours, mid, color="#222", linewidth=1.4)
    ax_px.set_ylabel("CASHCAT mid")
    ax_px.set_xlabel("hours since first fill")
    ax_px.grid(alpha=0.25)

    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nwrote {args.out}")

    if args.csv:
        frame = pd.DataFrame({"hours": hours, "mid": mid, **curves})
        frame.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
