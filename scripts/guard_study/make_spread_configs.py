"""Generate replay configs for a min_half_spread_bps ladder on the frozen tape.

The live grid's spread ladder was still improving at its widest rung after
minutes; this runs the same ladder over the whole frozen tape so the turning
point (if there is one) is measured on days rather than guessed from a live
leaderboard that has barely started.

Everything except `quoting.min_half_spread_bps` is held at the shipped
configuration, so the ladder isolates one lever. The flow guard is left ON,
matching what the live grid runs.
"""

from __future__ import annotations

import math
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = REPO / "rust_live" / "config" / "cashcat_dryrun_realistic.toml"
VALIDATION = REPO / "rust_live" / "config" / "cashcat.validation.json"
TAPE = REPO / "scripts" / "guard_study_tapes" / "full_tape_185h"
CONFIGS = HERE / "spread_configs"
RUNS = REPO / "scripts" / "guard_study_tapes" / "spread_runs"

# The live grid's rungs, plus the shipped baseline and one beyond the widest
# live rung to bracket the turning point. max_half_spread_bps is 80.
LADDER = [1.5, 4.0, 8.0, 16.0, 24.0, 40.0, 60.0]


def tape_span_minutes(tape: Path) -> int:
    stamps = [
        int(match.group(1))
        for path in (tape / "CASHCAT" / "prices").glob("*.parquet")
        if (match := re.search(r"(\d{13})", path.name))
    ]
    if not stamps:
        raise SystemExit(f"no shards under {tape}")
    return int(math.ceil((max(stamps) - min(stamps)) / 60_000.0)) + 30


def derive(text: str, data_dir: Path, report_dir: Path, minutes: int, half_spread: float) -> str:
    out: list[str] = []
    section = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            section = stripped
        if section == "[storage]" and stripped.startswith("data_dir"):
            line = f'data_dir = "{data_dir.as_posix()}"'
        elif section == "[storage]" and stripped.startswith("report_dir"):
            line = f'report_dir = "{report_dir.as_posix()}"'
        elif section == "[storage]" and stripped.startswith("retention_minutes"):
            # Validation requires retention > window + 30. Replay never prunes.
            line = f"retention_minutes = {minutes + 60}"
        elif section == "[storage]" and stripped.startswith("write_parquet"):
            # A replay must never take the collector's writer lock.
            line = "write_parquet = false"
        elif section == "[calibration]" and stripped.startswith("window_minutes"):
            line = f"window_minutes = {minutes}"
        elif section == "[quoting]" and stripped.startswith("min_half_spread_bps"):
            line = f"min_half_spread_bps = {half_spread}"
        out.append(line)
    return "\n".join(out) + "\n"


def main() -> None:
    base = BASE.read_text(encoding="utf-8")
    CONFIGS.mkdir(exist_ok=True)
    shutil.copy2(VALIDATION, CONFIGS / VALIDATION.name)
    minutes = tape_span_minutes(TAPE)
    print(f"tape span {minutes / 60.0:.1f} h -> window_minutes={minutes}")
    for half_spread in LADDER:
        name = f"spread{half_spread:g}".replace(".", "_")
        report_dir = RUNS / name
        report_dir.mkdir(parents=True, exist_ok=True)
        path = CONFIGS / f"{name}.toml"
        path.write_text(derive(base, TAPE, report_dir, minutes, half_spread), encoding="utf-8")
        print(f"  {path.name}: min_half_spread_bps={half_spread}")


if __name__ == "__main__":
    main()
