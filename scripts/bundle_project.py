#!/usr/bin/env python3
"""Create a Markdown bundle of review-relevant project files.

The default profile is meant for sending this repository to another AI for
review: include source/config/text files, skip generated data and binaries, and
redact common secret fields.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_OUTPUT = "project_bundle.md"
DEFAULT_MAX_BYTES = 300_000
MAX_SKIPPED_EXAMPLES = 40

INCLUDE_EXTENSIONS = {
    ".cfg",
    ".conf",
    ".dockerignore",
    ".env",
    ".gitignore",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".requirements",
    ".sh",
    ".toml",
    ".txt",
    ".tex",
    ".yaml",
    ".yml",
}

INCLUDE_FILENAMES = {
    ".gitignore",
    "Dockerfile",
    "Dockerfile.technical",
    "requirements.txt",
}

SKIP_DIR_PARTS = {
    ".git",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    ".vscode",
    "__pycache__",
    "backtest_results",
    "build",
    "dist",
    "env",
    "hl_data",
    "logs",
    "node_modules",
    "venv",
}

BINARY_OR_BULKY_EXTENSIONS = {
    ".7z",
    ".db",
    ".dll",
    ".exe",
    ".gif",
    ".gz",
    ".jpeg",
    ".jpg",
    ".parquet",
    ".pdf",
    ".png",
    ".pyc",
    ".sqlite",
    ".sqlite3",
    ".tar",
    ".webp",
    ".zip",
}

LANG_BY_EXTENSION = {
    ".cfg": "ini",
    ".conf": "ini",
    ".dockerignore": "gitignore",
    ".env": "dotenv",
    ".gitignore": "gitignore",
    ".ini": "ini",
    ".ipynb": "json",
    ".json": "json",
    ".md": "markdown",
    ".py": "python",
    ".sh": "bash",
    ".toml": "toml",
    ".txt": "text",
    ".tex": "tex",
    ".yaml": "yaml",
    ".yml": "yaml",
}

SENSITIVE_KEY_NAMES = (
    "api_key",
    "apikey",
    "api_secret",
    "chat_id",
    "jwt_secret_key",
    "key",
    "mnemonic",
    "passphrase",
    "password",
    "private_key",
    "secret",
    "token",
    "wallet_key",
)

SENSITIVE_JSON_RE = re.compile(
    r"(?i)([\"']?(?:"
    + "|".join(re.escape(key) for key in SENSITIVE_KEY_NAMES)
    + r")[\"']?\s*:\s*)([\"'])(.*?)(\2)"
)

SENSITIVE_ASSIGN_RE = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:API_KEY|API_SECRET|JWT_SECRET|PASSWORD|PASSphrase|PRIVATE_KEY|SECRET|TOKEN|MNEMONIC)[A-Z0-9_]*\s*=\s*)([\"']).*?(\2)"
)

SENSITIVE_ENV_RE = re.compile(
    r"(?im)^([A-Z0-9_]*(?:API_KEY|API_SECRET|JWT_SECRET|PASSWORD|PASSPHRASE|PRIVATE_KEY|SECRET|TOKEN|MNEMONIC)[A-Z0-9_]*=).*$"
)


@dataclass(frozen=True)
class IncludedFile:
    rel_path: str
    size: int
    language: str
    content: str


@dataclass(frozen=True)
class SkippedFile:
    rel_path: str
    reason: str


def run_git(root: Path, args: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, OSError):
        return None


def find_repo_root(start: Path) -> Path:
    result = run_git(start, ["rev-parse", "--show-toplevel"])
    if result and result.returncode == 0:
        return Path(result.stdout.strip()).resolve()
    return start.resolve()


def rel_from_root(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def candidate_paths(root: Path, include_ignored: bool) -> list[Path]:
    if not include_ignored:
        result = run_git(root, ["ls-files", "-co", "--exclude-standard", "-z"])
        if result and result.returncode == 0:
            names = [name for name in result.stdout.split("\0") if name]
            return [root / name for name in names]

    paths: list[Path] = []
    for current_root, dirnames, filenames in os.walk(root):
        current = Path(current_root)
        dirnames[:] = [
            name for name in dirnames if name.lower() not in SKIP_DIR_PARTS
        ]
        for filename in filenames:
            paths.append(current / filename)
    return paths


def has_skipped_dir(path: Path, root: Path) -> bool:
    rel_parts = path.resolve().relative_to(root).parts[:-1]
    return any(part.lower() in SKIP_DIR_PARTS for part in rel_parts)


def is_bundle_output(path: Path, output_path: Path, root: Path) -> bool:
    if path.resolve() == output_path.resolve():
        return True
    rel_name = path.resolve().relative_to(root).name.lower()
    return rel_name.startswith("project_bundle") and rel_name.endswith(".md")


def should_include(
    path: Path,
    root: Path,
    output_path: Path,
    max_bytes: int,
    include_notebooks: bool,
) -> tuple[bool, str]:
    if not path.is_file():
        return False, "not a file"
    if is_bundle_output(path, output_path, root):
        return False, "generated bundle"
    if has_skipped_dir(path, root):
        return False, "generated/cache/data directory"

    suffix = path.suffix.lower()
    name = path.name

    if suffix == ".ipynb":
        if not include_notebooks:
            return False, "notebook skipped by default"
        return True, ""
    if suffix in BINARY_OR_BULKY_EXTENSIONS:
        return False, "binary or bulky format"

    size = path.stat().st_size
    if size > max_bytes:
        return False, f"larger than --max-bytes ({size} bytes)"

    if suffix in INCLUDE_EXTENSIONS or name in INCLUDE_FILENAMES:
        return True, ""

    return False, "unsupported extension"


def decode_text(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def clean_notebook(text: str) -> str:
    notebook = json.loads(text)
    for cell in notebook.get("cells", []):
        cell.pop("outputs", None)
        cell["execution_count"] = None
    return json.dumps(notebook, indent=2, ensure_ascii=False)


def redact(text: str) -> str:
    text = SENSITIVE_JSON_RE.sub(r"\1\2<REDACTED>\4", text)
    text = SENSITIVE_ASSIGN_RE.sub(r"\1\2<REDACTED>\3", text)
    text = SENSITIVE_ENV_RE.sub(r"\1<REDACTED>", text)
    return text


def language_for(path: Path) -> str:
    if path.name.startswith("Dockerfile"):
        return "dockerfile"
    return LANG_BY_EXTENSION.get(path.suffix.lower(), "text")


def code_fence(content: str, language: str) -> str:
    longest_tick_run = max((len(match.group(0)) for match in re.finditer(r"`+", content)), default=0)
    fence = "`" * max(3, longest_tick_run + 1)
    return f"{fence}{language}\n{content.rstrip()}\n{fence}"


def git_status(root: Path) -> str:
    result = run_git(root, ["status", "--short"])
    if not result or result.returncode != 0:
        return "Git status unavailable."
    status = result.stdout.strip()
    return status if status else "Clean working tree."


def build_bundle(
    root: Path,
    output_path: Path,
    max_bytes: int,
    include_ignored: bool,
    include_notebooks: bool,
    no_redact: bool,
) -> tuple[str, list[IncludedFile], list[SkippedFile]]:
    included: list[IncludedFile] = []
    skipped: list[SkippedFile] = []

    for path in sorted(set(candidate_paths(root, include_ignored))):
        try:
            rel_path = rel_from_root(path, root)
            include, reason = should_include(
                path=path,
                root=root,
                output_path=output_path,
                max_bytes=max_bytes,
                include_notebooks=include_notebooks,
            )
            if not include:
                skipped.append(SkippedFile(rel_path, reason))
                continue

            content = decode_text(path)
            if path.suffix.lower() == ".ipynb":
                content = clean_notebook(content)
            if not no_redact:
                content = redact(content)

            included.append(
                IncludedFile(
                    rel_path=rel_path,
                    size=path.stat().st_size,
                    language=language_for(path),
                    content=content,
                )
            )
        except Exception as exc:  # Keep bundling even if one file is odd.
            rel_path = rel_from_root(path, root) if path.exists() else str(path)
            skipped.append(SkippedFile(rel_path, f"read error: {exc}"))

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    total_bytes = sum(item.size for item in included)

    lines: list[str] = [
        "# Project Bundle for AI Review",
        "",
        f"- Repository: `{root.name}`",
        f"- Generated at: `{generated_at}`",
        f"- Included files: `{len(included)}`",
        f"- Included source bytes: `{total_bytes}`",
        f"- Max file size: `{max_bytes}` bytes",
        f"- Redaction: `{'off' if no_redact else 'on'}`",
        "",
        "## Reviewer Notes",
        "",
        "- This bundle is generated from the local working tree, so it can include uncommitted changes.",
        "- Obvious secret-like values are replaced with `<REDACTED>` unless generated with `--no-redact`.",
        "- Binary files, generated datasets, caches, logs, and oversized files are skipped by default.",
        "",
        "## Git Status",
        "",
        code_fence(git_status(root), "text"),
        "",
        "## Included Files",
        "",
    ]

    for item in included:
        lines.append(f"- `{item.rel_path}` ({item.size} bytes)")

    if skipped:
        grouped: dict[str, list[str]] = defaultdict(list)
        for item in skipped:
            grouped[item.reason].append(item.rel_path)

        lines.extend(["", "## Skipped Files", ""])
        for reason, paths in sorted(grouped.items()):
            lines.append(f"- {reason}: {len(paths)} file(s)")
            for rel_path in paths[:MAX_SKIPPED_EXAMPLES]:
                lines.append(f"  - `{rel_path}`")
            remaining = len(paths) - MAX_SKIPPED_EXAMPLES
            if remaining > 0:
                lines.append(f"  - ... {remaining} more")

    lines.extend(["", "## File Contents", ""])

    for item in included:
        lines.extend(
            [
                f"### `{item.rel_path}`",
                "",
                f"Size: `{item.size}` bytes",
                "",
                code_fence(item.content, item.language),
                "",
            ]
        )

    return "\n".join(lines), included, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bundle review-relevant project files into one Markdown file."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root. Defaults to the current git repository root.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help=f"Output Markdown path. Defaults to {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help=f"Skip individual files larger than this. Defaults to {DEFAULT_MAX_BYTES}.",
    )
    parser.add_argument(
        "--include-ignored",
        action="store_true",
        help="Include git-ignored files that still pass the other filters.",
    )
    parser.add_argument(
        "--include-notebooks",
        action="store_true",
        help="Include .ipynb files after stripping cell outputs.",
    )
    parser.add_argument(
        "--no-redact",
        action="store_true",
        help="Disable secret redaction. Use only for private/local bundles.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the include/skip summary without writing the bundle.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = find_repo_root((args.root or Path.cwd()).resolve())
    output_path = args.output
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path = output_path.resolve()

    markdown, included, skipped = build_bundle(
        root=root,
        output_path=output_path,
        max_bytes=args.max_bytes,
        include_ignored=args.include_ignored,
        include_notebooks=args.include_notebooks,
        no_redact=args.no_redact,
    )

    if args.dry_run:
        print(f"Would include {len(included)} files and skip {len(skipped)} files.")
        for item in included:
            print(f"INCLUDE {item.rel_path}")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8", newline="\n")
    print(f"Wrote {output_path}")
    print(f"Included {len(included)} files; skipped {len(skipped)} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
