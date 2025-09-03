"""
Periodic asynchronous runner for test_kappa.py and test_epsilon.py.

Key features:
- Locates test files by walking up from this script's directory.
- Runs once by default; use --loop to repeat at intervals (default 20s).
- Works regardless of current working directory when invoked.

CLI examples:
  python periodic_test_runner.py --once
  python periodic_test_runner.py --loop --interval 15

Programmatic:
  from periodic_test_runner import schedule_tests
  schedule_tests(run_once=True)            # one cycle (default)
  schedule_tests(run_once=False, interval_seconds=20)  # loop
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


TEST_FILES = ("test_kappa.py", "test_epsilon.py")
CONFIG_FILES = ("epsilon.json", "kappa.json")


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _env_override(name: str) -> Optional[Path]:
    env_name = f"TEST_{name.rsplit('.', 1)[0].upper()}_PATH"
    val = os.getenv(env_name)
    if val:
        p = Path(val).expanduser().resolve()
        if p.exists():
            return p
    return None


def _find_upwards(
    filename: str,
    *,
    start_dir: Optional[Path] = None,
    max_up: int = 10,
    include_start: bool = True,
) -> Optional[Path]:
    """Search for `filename` in current or parent dirs (and common subdirs).

    - Starts from this script's directory by default (not the CWD).
    - Checks each ancestor directory for the file directly and in common subdirs.
    - Stops after `max_up` levels to avoid scanning the whole disk.
    - When `include_start` is False, the initial `start_dir` is skipped and the
      search begins at its parent.
    """

    if start_dir is None:
        start_dir = Path(__file__).resolve().parent

    # Allow environment variable override first
    override = _env_override(filename)
    if override is not None:
        return override

    common_subdirs = ("", "tests", "test", "scripts")

    cur = start_dir.resolve()
    if not include_start:
        parent = cur.parent
        if parent == cur:
            return None
        cur = parent
    for _ in range(max_up + 1):
        for sub in common_subdirs:
            candidate = (cur / sub / filename) if sub else (cur / filename)
            if candidate.exists():
                return candidate.resolve()
        parent = cur.parent
        if parent == cur:
            break
        cur = parent

    return None


def locate_all(
    *,
    start_dir: Optional[Path] = None,
    max_up: int = 10,
) -> Dict[str, Optional[Path]]:
    return {name: _find_upwards(name, start_dir=start_dir, max_up=max_up) for name in TEST_FILES}


def locate_configs(
    *,
    start_dir: Optional[Path] = None,
    max_up: int = 10,
    ignore_start_dir: bool = True,
) -> Dict[str, Optional[Path]]:
    return {
        name: _find_upwards(
            name,
            start_dir=start_dir,
            max_up=max_up,
            include_start=not ignore_start_dir,
        )
        for name in CONFIG_FILES
    }


async def _stream_process(name: str, cmd: list[str], cwd: Path) -> int:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(cwd),
    )

    assert proc.stdout is not None
    prefix = f"[{name}]"
    async for raw in proc.stdout:
        try:
            line = raw.decode(errors="replace").rstrip()
        except Exception:
            line = str(raw).rstrip()
        print(f"{_ts()} {prefix} {line}")
    return await proc.wait()


async def _run_once(found: Dict[str, Optional[Path]]) -> None:
    tasks = []
    for name in TEST_FILES:
        path = found.get(name)
        if not path:
            print(f"{_ts()} [finder] WARNING: Could not find {name}; skipping this cycle.")
            continue
        cmd = [sys.executable, str(path)]
        tasks.append(asyncio.create_task(_stream_process(name, cmd, cwd=path.parent)))

    if not tasks:
        # Nothing to run this cycle
        return

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for name, res in zip([t for t in TEST_FILES if found.get(t)], results):
        if isinstance(res, Exception):
            print(f"{_ts()} [{name}] ERROR: {res}")
        else:
            print(f"{_ts()} [{name}] Exit code: {res}")


def _copy_if_needed(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(src, dst)
    return True


def _copy_configs_to_cwd(config_paths: Dict[str, Optional[Path]], cwd: Path) -> None:
    for name, src in config_paths.items():
        if not src:
            print(f"{_ts()} [copy] WARNING: Could not find {name} during search.")
            continue
        dst = cwd / name
        changed = _copy_if_needed(src, dst)
        print(f"{_ts()} [copy] Copied {name} to {dst}")


async def _periodic_worker(
    interval_seconds: float,
    start_dir: Optional[Path],
    max_up: int,
    copy_configs: bool,
    run_once: bool,
) -> None:
    while True:
        found = locate_all(start_dir=start_dir, max_up=max_up)
        # print(found)
        if copy_configs:
            configs = locate_configs(start_dir=start_dir, max_up=max_up)
            print(configs)
            _copy_configs_to_cwd(configs, Path(__file__).resolve().parent)
        missing = [k for k, v in found.items() if v is None]
        if missing:
            print(
                f"{_ts()} [finder] Searching... missing: {', '.join(missing)}. "
                f"Start dir: {(start_dir or Path(__file__).resolve().parent)}"
            )
        await _run_once(found)
        if run_once:
            break
        await asyncio.sleep(interval_seconds)


def schedule_tests(
    *,
    interval_seconds: float = 20.0,
    start_dir: Optional[Path] = None,
    max_up: int = 10,
    copy_configs: bool = True,
    run_once: bool = True,
) -> None:
    """Run test_kappa.py and test_epsilon.py.

    - run_once: run a single cycle and exit (default). When False, loops.
    - interval_seconds: seconds to wait after each cycle completes (looping only).
    - start_dir: where to start upward search (default: this file's directory).
    - max_up: how many directory levels to traverse upward.

    The runner sets each test's working directory to the directory containing
    the test file, so relative imports work even when launched from elsewhere.
    """

    if start_dir is not None:
        start_dir = Path(start_dir).resolve()

    try:
        asyncio.run(_periodic_worker(interval_seconds, start_dir, max_up, copy_configs, run_once))
    except KeyboardInterrupt:
        print(f"{_ts()} [runner] Stopped by user.")


def _parse_args(argv: list[str]):
    import argparse

    p = argparse.ArgumentParser(description="Periodic async runner for kappa/epsilon tests")
    p.add_argument(
        "--interval",
        type=float,
        default=20.0,
        help="Seconds between cycles (default: 20.0)",
    )
    p.add_argument(
        "--start-dir",
        type=str,
        default=None,
        help="Directory to start upward search (default: script directory)",
    )
    p.add_argument(
        "--max-up",
        type=int,
        default=10,
        help="Max number of parent levels to search (default: 10)",
    )
    p.add_argument(
        "--no-copy-configs",
        action="store_true",
        help="Disable copying epsilon.json/kappa.json into the current directory",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Run a single cycle and exit (default if --loop not provided)",
    )
    p.add_argument(
        "--loop",
        action="store_true",
        help="Run repeatedly at the specified --interval",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    if args.once and args.loop:
        import sys as _sys
        print("Cannot use --once and --loop together", file=_sys.stderr)
        raise SystemExit(2)
    run_once = True if args.once or not args.loop else False
    schedule_tests(
        interval_seconds=args.interval,
        start_dir=Path(args.start_dir).resolve() if args.start_dir else None,
        max_up=args.max_up,
        copy_configs=not args.no_copy_configs,
        run_once=run_once,
    )
