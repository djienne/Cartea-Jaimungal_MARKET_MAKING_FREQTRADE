"""Generate the six replay configs for the guard-candidate study.

{crash, calm, full} x {guardoff, guardon}, each derived from
rust_live/config/cashcat.toml by section-aware line replacement:
  storage.data_dir       -> the frozen tape
  storage.report_dir     -> scripts/guard_study_tapes/runs/<name>/
  calibration.window_minutes -> the tape's span (full) or 960 (crash/calm)
  flow_guard.enabled     -> per leg  (the [live] enabled key is NOT touched)

Configs land in scripts/guard_study/configs/ with cashcat.validation.json
copied alongside (the instrument-evidence preflight resolves it relative to
the config file).
"""

from __future__ import annotations

import math
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = REPO / "rust_live" / "config" / "cashcat.toml"
VALIDATION = REPO / "rust_live" / "config" / "cashcat.validation.json"
TAPES = REPO / "scripts" / "guard_study_tapes"
CONFIGS = HERE / "configs"
RUNS = TAPES / "runs"

CRASH_CALM_MINUTES = 960


def tape_span_minutes(tape: Path) -> int:
    stamps = []
    for path in (tape / "CASHCAT" / "prices").glob("*.parquet"):
        match = re.search(r"(\d{13})", path.name)
        if match:
            stamps.append(int(match.group(1)))
    if not stamps:
        raise SystemExit(f"no shards under {tape}")
    span_min = (max(stamps) - min(stamps)) / 60_000.0
    return int(math.ceil(span_min)) + 30  # margin so window_start precedes the data


def derive(text: str, data_dir: Path, report_dir: Path, window_minutes: int, enabled: bool) -> str:
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
            # Validation requires retention > window + 30 min. Replay never
            # prunes (only dry-run records), so this is a validation
            # constraint, not a behavior change.
            line = f"retention_minutes = {window_minutes + 60}"
        elif section == "[calibration]" and stripped.startswith("window_minutes"):
            line = f"window_minutes = {window_minutes}"
        elif section == "[flow_guard]" and stripped.startswith("enabled"):
            line = f"enabled = {str(enabled).lower()}"
        out.append(line)
    return "\n".join(out) + "\n"


def main() -> None:
    base = BASE.read_text(encoding="utf-8")
    CONFIGS.mkdir(exist_ok=True)
    shutil.copy2(VALIDATION, CONFIGS / VALIDATION.name)
    windows = {
        "crash": CRASH_CALM_MINUTES,
        "calm": CRASH_CALM_MINUTES,
        "full": tape_span_minutes(TAPES / "full_tape"),
    }
    for tape_name, minutes in windows.items():
        for leg, enabled in (("guardoff", False), ("guardon", True)):
            name = f"{tape_name}_{leg}"
            report_dir = RUNS / name
            report_dir.mkdir(parents=True, exist_ok=True)
            text = derive(base, TAPES / f"{tape_name}_tape", report_dir, minutes, enabled)
            path = CONFIGS / f"{name}.toml"
            path.write_text(text, encoding="utf-8")
            print(f"{path.name}: window_minutes={minutes} flow_guard.enabled={enabled}")


if __name__ == "__main__":
    main()
