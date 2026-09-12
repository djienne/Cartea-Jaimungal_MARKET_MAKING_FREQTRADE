#!/usr/bin/env python3
"""Zip current source, review prompt and a read-only paper-history snapshot.

Includes tracked edits and non-ignored new files. Excludes credential files,
builds and bulk data; git tracking alone is not a guarantee against secrets.

    python scripts/make_agent_bundle.py [-o OUTPUT.zip]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

SKIP_DIRS = {".git", ".claude", ".codex", ".agents", ".venv", "venv", "__pycache__", "target", "node_modules", "hl_data", "run", "reports"}
SKIP_SUFFIXES = {".env", ".key", ".pem", ".p12", ".pfx", ".zip", ".7z", ".parquet", ".zst", ".exe", ".dll", ".pyc", ".db", ".sqlite"}
REFERENCE_BOOK = "docs/[Mathematics, Finance and Risk] Álvaro Cartea, Sebastian Jaimungal, José Penalva - Algorithmic and High-Frequency Trading (2015, Cambridge University Press) - libgen.li.pdf"


def repo_root() -> Path:
    out = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "--show-toplevel"],
        capture_output=True, encoding="utf-8", check=True,
    )
    return Path(out.stdout.strip())


def project_files(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        capture_output=True, encoding="utf-8", check=True, cwd=root,
    )
    if not (root / REFERENCE_BOOK).is_file():
        raise FileNotFoundError(f"Requested reference book is missing: {REFERENCE_BOOK}")
    paths = []
    for name in sorted((set(out.stdout.split("\0")) | {REFERENCE_BOOK}) - {""}):
        path = root / name
        if (set(part.lower() for part in path.relative_to(root).parts[:-1]) & SKIP_DIRS
                or path.suffix.lower() in SKIP_SUFFIXES or ".env." in path.name.lower()
                or path.name.lower() == ".env" or path.is_symlink()
                or not path.resolve().is_relative_to(root.resolve()) or not path.is_file()):
            continue
        paths.append(name)
    return paths


def paper_files(root: Path) -> list[str]:
    """Only paper artifacts; never traverse live account state or event logs."""
    base = root / "rust_live/reports/grid_live"
    checkpoint = base / "grid_state.json"
    if not checkpoint.exists():
        return []
    run_id = json.loads(checkpoint.read_text(encoding="utf-8"))["run_id"]
    if not run_id.startswith("run-") or not run_id[4:].isdigit():
        raise ValueError("invalid paper run ID")
    run = base / "runs" / run_id
    paths = [checkpoint, base / "leaderboard.json", run / "equity_history.csv", *run.glob("fix-*.md")]
    return [p.relative_to(root).as_posix() for p in paths
            if p.is_file() and p.resolve().is_relative_to(root.resolve())]


def build_manifest(root: Path, paths: list[str]) -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, encoding="utf-8", cwd=root
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--short"], capture_output=True, encoding="utf-8", cwd=root
    ).stdout
    lines = [
        f"# Agent bundle -- {root.name}",
        f"generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"git HEAD: {head}",
        f"included files: {len(paths)}",
        "",
        "Start with project/docs/AGENT_REVIEW_PROMPT.md.",
        "Source comes from the CURRENT working tree, including non-ignored new files.",
        "Paper snapshots are copied without stopping the run; compare their timestamps.",
        "Includes the Cartea/Jaimungal/Penalva book and docs/market_making_introduction.ipynb.",
        "Raw market tape, event logs, live account state, credentials and builds are excluded.",
        "The paper history spans configuration changes; it is not a fixed-configuration benchmark.",
        "Working-tree status:",
    ]
    lines += dirty.splitlines() or ["  (clean)"]
    return "\n".join(lines) + "\n"


def create_bundle(root: Path, output: Path) -> int:
    paths = project_files(root) + paper_files(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite a previous handoff. Each invocation is a new snapshot.
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("BUNDLE_MANIFEST.txt", build_manifest(root, paths))
        for rel in paths:
            data = (root / rel).read_bytes()
            if rel.endswith("/equity_history.csv"):
                # The writer stays running; omit only an unfinished trailing row.
                data = data[:data.rfind(b"\n") + 1]
            zf.writestr(f"project/{rel}", data)
    return len(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output zip path (default: <repo>_bundle_<timestamp>.zip next to the repo)",
    )
    args = parser.parse_args()

    root = repo_root()
    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = root.parent / f"{root.name}_bundle_{stamp}.zip"
    count = create_bundle(root, output)
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"wrote {output}  ({size_mb:.1f} MB, {count} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
