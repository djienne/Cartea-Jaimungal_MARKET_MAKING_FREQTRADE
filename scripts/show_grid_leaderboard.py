"""Show the current CASHCAT dry-run grid leaderboard.

The live grid rewrites ``leaderboard.json`` every few seconds.  This command
reads that snapshot, joins it to the variant overrides in ``grid_cashcat.toml``,
and sorts by the board's own ranking metric, ``promotion_pnl_usdc``, from best
to worst by default.

Usage:
    python scripts/show_grid_leaderboard.py
    python scripts/show_grid_leaderboard.py --watch 5
    python scripts/show_grid_leaderboard.py --markdown
    python scripts/show_grid_leaderboard.py --sort net-pnl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
DEFAULT_LEADERBOARD = REPO / "rust_live" / "reports" / "grid_live" / "leaderboard.json"
DEFAULT_GRID = REPO / "rust_live" / "config" / "grid_cashcat.toml"

ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
OVERRIDE_LABELS = {
    "q_max": "q_max",
    "horizon_seconds": "T_s",
    "parameter_profile": "params",
    "phi_kappa_t": "phiKT",
    "phi_kappa_t_max": "phiKTmax",
    "min_half_spread_bps": "halfspread_bps",
    "min_order_lifetime_ms": "lifetime_ms",
    "replace_threshold_bps": "replace_bps",
    "flow_guard_enabled": "guard",
}


def parse_variant_overrides(path: Path) -> dict[str, str]:
    """Read the simple ``[[variant]]`` tables without adding a TOML dependency.

    The project supports Python 3.10, before ``tomllib`` entered the standard
    library.  This file only needs scalar assignments from one repeated table,
    so a deliberately narrow parser is preferable to making an operational
    status command depend on the scientific Python environment.
    """

    variants: dict[str, str] = {}
    current: dict[str, str] | None = None

    def finish() -> None:
        if current is None:
            return
        name = current.get("name")
        if not name:
            raise ValueError(f"variant without a name in {path}")
        changes = [
            f"{OVERRIDE_LABELS.get(key, key)}={value}"
            for key, value in current.items()
            if key != "name"
        ]
        variants[name] = ", ".join(changes) or "none (base control)"

    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line == "[[variant]]":
            finish()
            current = {}
            continue
        if current is None:
            continue
        match = ASSIGNMENT.match(line)
        if not match:
            raise ValueError(f"unsupported variant syntax at {path}:{line_number}: {raw}")
        key, value = match.groups()
        if len(value) >= 2 and value[0] == value[-1] == '"':
            value = value[1:-1]
        current[key] = value

    finish()
    return variants


def read_leaderboard(path: Path) -> dict[str, Any]:
    try:
        board = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SystemExit(f"leaderboard not found: {path}") from error
    except json.JSONDecodeError as error:
        raise SystemExit(f"cannot parse {path}: {error}") from error
    if not isinstance(board.get("rows"), list):
        raise SystemExit(f"{path} has no leaderboard rows")
    return board


def signed_money(value: Any) -> str:
    return f"{float(value):+.2f}"


def money(value: Any) -> str:
    return f"{float(value):.2f}"


def sort_rows(rows: list[dict[str, Any]], field: str, ascending: bool) -> list[dict[str, Any]]:
    keys = {
        "promotion-pnl": "promotion_pnl_usdc",
        "net-pnl": "net_pnl_usdc",
        "realized-pnl": "realized_pnl_usdc",
        "fills": "fills",
        "max-drawdown": "max_drawdown_usdc",
    }
    if field == "name":
        return sorted(rows, key=lambda row: str(row["name"]).lower(), reverse=not ascending)
    key = keys[field]

    def rank(row: dict[str, Any]) -> tuple[int, float, str]:
        raw = row.get(key)
        name = str(row["name"]).lower()
        if raw is None:
            # Only ``promotion_pnl_usdc`` is optional, and it is null exactly
            # when the row was disqualified: there is no executable exit price
            # for a variant that stopped trading. An unscored row cannot be
            # ranked against scored ones, so it sits last in either direction.
            return (1, 0.0, name)
        value = float(raw)
        # Negate only the scientific metric. Ties remain alphabetic instead of
        # reversing their names as ``reverse=True`` would do.
        return (0, value if ascending else -value, name)

    return sorted(rows, key=rank)


def verdict(row: dict[str, Any]) -> str:
    """Say why, not just whether.

    ``scientifically_valid: false`` alone cannot tell a variant that blew
    through its liquidation buffer from one that merely printed a bad number,
    and those are different claims about the parameters.
    """

    if row["scientifically_valid"]:
        return "yes"
    reason = row.get("invalid_reason")
    return f"NO: {reason}" if reason else "NO"


def result_rows(
    board: dict[str, Any], overrides: dict[str, str], sort: str, ascending: bool
) -> tuple[list[str], list[list[str]]]:
    ordered = sort_rows(board["rows"], sort, ascending)
    headers = [
        "#",
        "Variant",
        "Overrides from base",
        "Promo",
        "Net",
        "Realized",
        "Fills",
        "Fees",
        "Inv",
        "Max DD",
        "Valid",
    ]
    rows = []
    for rank, row in enumerate(ordered, 1):
        name = str(row["name"])
        rows.append(
            [
                str(rank),
                name,
                overrides.get(name, "not found in grid spec"),
                "n/a" if row.get("promotion_pnl_usdc") is None else signed_money(row["promotion_pnl_usdc"]),
                signed_money(row["net_pnl_usdc"]),
                signed_money(row["realized_pnl_usdc"]),
                f"{int(row['fills']):,}",
                money(row["fees_usdc"]),
                f"{int(row['inventory_units']):+,}",
                money(row["max_drawdown_usdc"]),
                verdict(row),
            ]
        )
    return headers, rows


def status_lines(board: dict[str, Any]) -> list[str]:
    generated_ms = int(board["generated_at_ms"])
    generated = datetime.fromtimestamp(generated_ms / 1_000).astimezone()
    age_seconds = max(0.0, time.time() - generated_ms / 1_000)
    valid = sum(bool(row.get("scientifically_valid")) for row in board["rows"])
    if window := board.get("replay"):
        # A replay board is the live schema on purpose, so the rows compare
        # field for field -- which is exactly what makes it easy to read one as
        # the other. Say which this is, and drop the feed counters: a replay
        # consumes a tape slice and cannot see a gap inside it, so printing
        # "gaps 0, downtime 0.000%" would assert something nobody measured.
        scored_hours = (window["scoring_end_ms"] - window["scoring_start_ms"]) / 3_600_000.0
        scored_from = datetime.fromtimestamp(window["scoring_start_ms"] / 1_000).astimezone()
        return [
            (
                f"{board.get('symbol', '?')} REPLAY (not a live run) | "
                f"{generated:%Y-%m-%d %H:%M:%S %z} ({age_seconds:.1f}s old)"
            ),
            (
                f"scored {scored_hours:.2f}h from {scored_from:%Y-%m-%d %H:%M} | "
                f"latency {window['latency_ms']} ms assumed | "
                f"train {100.0 * float(window['train_fraction']):.0f}% | "
                f"valid {valid}/{len(board['rows'])}"
            ),
        ]
    feed = board.get("feed_health", {})
    down_ms = int(board.get("feed_down_for_ms", 0))
    downtime_pct = 100.0 * float(feed.get("downtime_fraction", 0.0))
    gaps = int(feed.get("gaps", 0))
    event_loss = "YES" if feed.get("event_loss", False) else "no"
    feed_state = f"DOWN for {down_ms / 1_000:.1f}s" if down_ms else "up"
    return [
        (
            f"{board.get('symbol', '?')} dry-run grid | "
            f"{generated:%Y-%m-%d %H:%M:%S %z} ({age_seconds:.1f}s old)"
        ),
        (
            f"elapsed {float(board.get('elapsed_seconds', 0)) / 3_600:.2f}h | "
            f"feed {feed_state} | gaps {gaps} | event loss {event_loss} | "
            f"downtime {downtime_pct:.3f}% | "
            f"resumes {int(board.get('resumes', 0))} | valid {valid}/{len(board['rows'])}"
            + (
                f" | reconfigured {reconfigured} (history continued)"
                if (reconfigured := sum(1 for r in board["rows"] if r.get("config_changes", 0)))
                else ""
            )
        ),
    ]


def terminal_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    numeric = {0, 3, 4, 5, 6, 7, 8, 9}

    def render(row: list[str]) -> str:
        cells = []
        for index, value in enumerate(row):
            cells.append(value.rjust(widths[index]) if index in numeric else value.ljust(widths[index]))
        return "  ".join(cells)

    rule = "  ".join("-" * width for width in widths)
    return "\n".join([render(headers), rule, *(render(row) for row in rows)])


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    def line(values: list[str]) -> str:
        return "| " + " | ".join(value.replace("|", "\\|") for value in values) + " |"

    alignment = ["---:", "---", "---", *["---:"] * 7, "---"]
    return "\n".join([line(headers), line(alignment), *(line(row) for row in rows)])


def render(args: argparse.Namespace) -> str:
    board = read_leaderboard(args.leaderboard)
    try:
        overrides = parse_variant_overrides(args.grid)
    except (OSError, ValueError) as error:
        raise SystemExit(f"cannot read grid specification: {error}") from error
    headers, rows = result_rows(board, overrides, args.sort, args.ascending)
    status = status_lines(board)
    table = markdown_table(headers, rows) if args.markdown else terminal_table(headers, rows)
    return "\n".join([*status, "", table])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--leaderboard", type=Path, default=DEFAULT_LEADERBOARD)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument(
        "--sort",
        choices=("promotion-pnl", "net-pnl", "realized-pnl", "fills", "max-drawdown", "name"),
        default="promotion-pnl",
        help=(
            "ranking field (default: promotion-pnl, the board's own metric: "
            "executable-side, fee-adjusted flatten P&L, which stops a large "
            "directional inventory winning because the market moved with it)"
        ),
    )
    parser.add_argument("--ascending", action="store_true", help="put the lowest value first")
    parser.add_argument("--markdown", action="store_true", help="print a copyable Markdown table")
    parser.add_argument(
        "--watch",
        type=float,
        metavar="SECONDS",
        help="refresh continuously at this interval",
    )
    args = parser.parse_args()
    if args.watch is not None and args.watch <= 0:
        parser.error("--watch must be greater than zero")
    if args.watch is not None and args.markdown:
        parser.error("--watch and --markdown cannot be used together")
    return args


def main() -> None:
    args = parse_args()
    if args.watch is None:
        print(render(args))
        return

    try:
        while True:
            # ANSI clear works in Windows Terminal and keeps one live table on screen.
            if sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            print(render(args), flush=True)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
