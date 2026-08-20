#!/usr/bin/env python3
"""Zip every git-tracked file (current working-tree content) for handoff to another agent.

Uses `git ls-files` for the path list -- which is exactly the .gitignore-filtered
set (no secrets, no keys, no bulk parquet/log data, no venvs) -- but reads each
file straight off disk, so uncommitted edits are included rather than the stale
last-commit version. Run from anywhere; it resolves the repo root itself.

    python scripts/make_agent_bundle.py [-o OUTPUT.zip]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True,
    )
    return Path(out.stdout.strip())


def tracked_files(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        capture_output=True, text=True, check=True, cwd=root,
    )
    return [p for p in out.stdout.split("\0") if p]


def build_manifest(root: Path, paths: list[str]) -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=root
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--short"], capture_output=True, text=True, cwd=root
    ).stdout
    dirty_paths = sorted(
        line[3:] for line in dirty.splitlines() if not line.startswith("??")
    )
    lines = [
        f"# Agent bundle -- {root.name}",
        f"generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"git HEAD: {head}",
        f"tracked files: {len(paths)}",
        "",
        "Every path below is a git-tracked file, read from the CURRENT working",
        "tree -- uncommitted edits are included, not just the last commit.",
        f"{len(dirty_paths)} file(s) differ from HEAD:",
    ]
    lines += [f"  M  {p}" for p in dirty_paths] or ["  (none -- tree is clean)"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output zip path (default: <repo>_bundle_<timestamp>.zip next to the repo)",
    )
    args = parser.parse_args()

    root = repo_root()
    paths = tracked_files(root)
    if not paths:
        print("No git-tracked files found -- is this a git repo?", file=sys.stderr)
        return 1

    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = root.parent / f"{root.name}_bundle_{stamp}.zip"
    output.parent.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(root, paths)

    missing: list[str] = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("BUNDLE_MANIFEST.txt", manifest)
        for rel in paths:
            src = root / rel
            if not src.is_file():
                # Deleted-but-still-staged, or a submodule gitlink entry.
                missing.append(rel)
                continue
            zf.write(src, arcname=f"{root.name}/{rel}")

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"wrote {output}  ({size_mb:.1f} MB, {len(paths) - len(missing)} files)")
    if missing:
        print(f"skipped {len(missing)} tracked path(s) not present on disk:")
        for rel in missing:
            print(f"  - {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
