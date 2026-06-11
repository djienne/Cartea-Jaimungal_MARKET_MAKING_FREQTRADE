"""
Periodic deterministic runner for get_kappa.py, get_epsilon.py, and get_lambda.py.

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
import json
import os
import sys
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


TEST_FILES = ("get_kappa.py", "get_epsilon.py", "get_lambda.py")
# lambda.json holds baseline λ₀ from get_kappa.py; lambda_trades.json is optional monitoring output.
CONFIG_FILES = ("epsilon.json", "kappa.json", "lambda.json", "lambda_trades.json")
RUNNER_STATUS_FILE = "param_update_status.json"
RUNNER_LOCK_FILE = "param_update.lock"
_RUNNER_LOCK = threading.Lock()


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=4, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _write_status(status: str, payload: dict | None = None) -> None:
    body = {"status": status, "generated_at": _utc_now_iso(), **(payload or {})}
    _atomic_write_json(Path(__file__).resolve().parent / RUNNER_STATUS_FILE, body)


def _runner_lock_path() -> Path:
    return Path(__file__).resolve().parent / RUNNER_LOCK_FILE


def _parse_utc_timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if not text:
            return None
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _lock_age_seconds(path: Path) -> float | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    started_at = _parse_utc_timestamp(payload.get("started_at"))
    if started_at is None:
        return None
    return (datetime.now(timezone.utc) - started_at).total_seconds()


def _stale_lock_can_be_replaced(path: Path, stale_after_seconds: float) -> bool:
    if stale_after_seconds <= 0:
        return False
    age = _lock_age_seconds(path)
    if age is None:
        try:
            age = datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
        except Exception:
            return False
    return age > float(stale_after_seconds)


def _acquire_process_lock(path: Path, crypto: str, stale_after_seconds: float = 3600.0) -> str | None:
    """Acquire the cross-process lock; return an ownership token, or None.

    The token (random per acquisition) is embedded in the lock payload and must
    be presented to _release_process_lock. This prevents a runner that never
    acquired the lock (or that had its stale lock replaced) from deleting a lock
    another process currently holds. PIDs are not sufficient for ownership:
    containers have separate PID namespaces and routinely collide.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(str(path), flags)
    except FileExistsError:
        if not _stale_lock_can_be_replaced(path, stale_after_seconds):
            return None
        try:
            path.unlink()
        except OSError:
            return None
        try:
            fd = os.open(str(path), flags)
        except FileExistsError:
            return None
    token = uuid.uuid4().hex
    payload = {
        "pid": os.getpid(),
        "crypto": crypto,
        "started_at": _utc_now_iso(),
        "token": token,
    }
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=4, sort_keys=True)
            handle.write("\n")
    except OSError:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return token


def _release_process_lock(path: Path, token: str | None) -> None:
    """Release the lock only if we own it (payload token matches).

    A non-matching or unreadable payload means another process holds the lock
    (or it is corrupt); leave it alone — the stale-lock handling in
    _acquire_process_lock recovers abandoned locks.
    """
    if not token:
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return
    except Exception:
        print(f"{_ts()} [runner] WARNING: Lock payload unreadable; leaving {path} for stale handling.")
        return
    if not isinstance(payload, dict) or payload.get("token") != token:
        print(f"{_ts()} [runner] WARNING: Lock {path} is held by another process; not releasing.")
        return
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        print(f"{_ts()} [runner] WARNING: Could not remove process lock {path}: {exc}")


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


async def _run_once(found: Dict[str, Optional[Path]], crypto: str) -> None:
    ran_any = False
    for name in TEST_FILES:
        path = found.get(name)
        if not path:
            print(f"{_ts()} [finder] WARNING: Could not find {name}; skipping this cycle.")
            continue
        cmd = [sys.executable, str(path), "--crypto", crypto]
        ran_any = True
        try:
            res = await _stream_process(name, cmd, cwd=path.parent)
        except Exception as exc:
            print(f"{_ts()} [{name}] ERROR: {exc}")
            raise
        print(f"{_ts()} [{name}] Exit code: {res}")
        if int(res) != 0:
            raise RuntimeError(f"{name} failed with exit code {res}")

    if not ran_any:
        return


def _copy_if_needed(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    try:
        tmp.write_bytes(src.read_bytes())
        tmp.replace(dst)
        return True
    except OSError as e:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        print(f"{_ts()} [copy] ERROR copying {src} -> {dst}: {e}")
        return False


def _copy_configs_to_cwd(config_paths: Dict[str, Optional[Path]], cwd: Path) -> None:
    for name, src in config_paths.items():
        if not src:
            print(f"{_ts()} [copy] WARNING: Could not find {name} during search.")
            continue
        dst = cwd / name
        changed = _copy_if_needed(src, dst)
        if changed:
            print(f"{_ts()} [copy] Copied {name} to {dst}")
        else:
            print(f"{_ts()} [copy] WARNING: Failed to copy {name} to {dst}")


def _load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"{path.name} is unreadable JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return data


def _valid_timestamp(value: object) -> bool:
    if not value:
        return False
    try:
        datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return False
    return True


def _validate_symbol_snapshot(config_paths: Dict[str, Optional[Path]], crypto: str) -> tuple[bool, str]:
    required_files = ("kappa.json", "lambda.json", "epsilon.json")
    required_keys = {
        "kappa.json": (
            "kappa+",
            "kappa-",
            "lambda+",
            "lambda-",
            "window_start",
            "window_end",
            "n_quotes",
            "n_trades",
            "n_points_plus",
            "n_points_minus",
            "r2_plus",
            "r2_minus",
        ),
        "lambda.json": ("lambda+", "lambda-", "lambda_source", "window_start", "window_end", "n_trades"),
        "epsilon.json": (
            "epsilon+",
            "epsilon-",
            "window_start",
            "window_end",
            "window_ms",
            "estimator",
            "unit",
            "n_buy_events",
            "n_sell_events",
            "toxicity_plus",
            "toxicity_minus",
        ),
    }
    timestamp_keys = {"generated_at", "window_start", "window_end"}
    string_keys = {"lambda_source", "estimator", "unit"}
    for name in required_files:
        path = config_paths.get(name)
        if path is None:
            return False, f"missing_{name}"
        try:
            payload = _load_json(path)
        except ValueError as exc:
            return False, str(exc)
        entry = payload.get(crypto)
        if not isinstance(entry, dict):
            return False, f"missing_symbol_{crypto}_in_{name}"
        if int(entry.get("schema_version", 0) or 0) != 2:
            return False, f"unsupported_schema_{name}"
        if str(entry.get("status", "")).lower() != "ok":
            return False, f"status_not_ok_{name}:{entry.get('status')}"
        if not entry.get("generated_at"):
            return False, f"missing_generated_at_{name}"
        if not _valid_timestamp(entry.get("generated_at")):
            return False, f"invalid_generated_at_{name}"
        for key in required_keys[name]:
            if key not in entry:
                return False, f"missing_{key}_in_{name}"
            if key == "lambda_source":
                if name == "lambda.json" and entry[key] != "lambda0_fit":
                    return False, "lambda_json_must_be_lambda0_fit"
                continue
            if key in timestamp_keys:
                if not _valid_timestamp(entry.get(key)):
                    return False, f"invalid_{key}_in_{name}"
                continue
            if key in string_keys:
                if not str(entry.get(key) or "").strip():
                    return False, f"missing_{key}_in_{name}"
                continue
            try:
                value = float(entry[key])
            except Exception:
                return False, f"nonfinite_{key}_in_{name}"
            if value != value or value in (float("inf"), float("-inf")):
                return False, f"nonfinite_{key}_in_{name}"
    return True, "ok"


def _load_param_store():
    """Load the optional param_store helper (from scripts/) by name or by path."""
    try:
        import param_store  # type: ignore

        return param_store
    except Exception:
        pass
    path = _find_upwards("param_store.py")
    if path is None:
        return None
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("param_store", str(path))
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None


def _publish_params_to_redis(config_paths: Dict[str, Optional[Path]], crypto: str) -> None:
    """Best-effort publish of the validated snapshot to Redis as one atomic blob."""
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        return
    store = _load_param_store()
    if store is None:
        print(f"{_ts()} [redis] param_store unavailable; skipping publish.")
        return
    try:
        kappa = _load_json(config_paths["kappa.json"]).get(crypto)
        epsilon = _load_json(config_paths["epsilon.json"]).get(crypto)
        lam = _load_json(config_paths["lambda.json"]).get(crypto)
    except Exception as exc:
        print(f"{_ts()} [redis] could not read snapshot for publish: {exc}")
        return
    if not all(isinstance(x, dict) for x in (kappa, epsilon, lam)):
        print(f"{_ts()} [redis] snapshot incomplete for {crypto}; skipping publish.")
        return
    ok = store.publish_params(
        redis_url,
        crypto,
        kappa=kappa,
        epsilon=epsilon,
        lambda_=lam,
        published_at=_utc_now_iso(),
    )
    print(
        f"{_ts()} [redis] {'published' if ok else 'publish skipped (redis unavailable)'} "
        f"params for {crypto}."
    )


async def _run_cycle_body(
    start_dir: Optional[Path],
    max_up: int,
    copy_configs: bool,
    crypto: str,
) -> None:
    """Run estimators once and (optionally) validate+publish the snapshots."""
    found = locate_all(start_dir=start_dir, max_up=max_up)
    missing = [k for k, v in found.items() if v is None]
    if missing:
        print(
            f"{_ts()} [finder] Searching... missing: {', '.join(missing)}. "
            f"Start dir: {(start_dir or Path(__file__).resolve().parent)}"
        )
    await _run_once(found, crypto)
    if copy_configs:
        configs = locate_configs(start_dir=start_dir, max_up=max_up)
        ok, reason = _validate_symbol_snapshot(configs, crypto)
        if not ok:
            raise RuntimeError(f"parameter snapshot validation failed: {reason}")
        _copy_configs_to_cwd(configs, Path(__file__).resolve().parent)
        _publish_params_to_redis(configs, crypto)


async def _periodic_worker(
    interval_seconds: float,
    start_dir: Optional[Path],
    max_up: int,
    copy_configs: bool,
    run_once: bool,
    crypto: str,
    lock_path: Path,
    lock_stale_seconds: float,
    lock_holder: dict,
) -> None:
    """Run estimator cycles, holding the process lock only for the duration of
    each cycle.

    The market-making strategy treats the presence of param_update.lock (and a
    "running" status) as "params are mid-update, do not quote". Holding the lock
    for the whole lifetime of a looping runner would therefore block quoting
    forever, so the lock is acquired at the start of each cycle and released
    before sleeping between cycles.

    lock_holder["token"] mirrors the currently held ownership token so the
    caller can release the lock if this coroutine is interrupted mid-cycle.
    """
    while True:
        token = _acquire_process_lock(lock_path, crypto, lock_stale_seconds)
        if not token:
            print(f"{_ts()} [runner] Skipping cycle; process lock exists at {lock_path}.")
        else:
            lock_holder["token"] = token
            _write_status("running", {"crypto": crypto, "lock_path": str(lock_path)})
            try:
                await _run_cycle_body(start_dir, max_up, copy_configs, crypto)
                _write_status("success", {"crypto": crypto})
            except Exception as exc:
                _write_status("failed", {"crypto": crypto, "error": str(exc)})
                if run_once:
                    raise
                print(f"{_ts()} [runner] Cycle failed: {exc}; retrying after {interval_seconds:.0f}s.")
            finally:
                _release_process_lock(lock_path, token)
                lock_holder["token"] = None
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
    crypto: str = "ETH",
    lock_stale_seconds: float = 3600.0,
) -> None:
    """Run get_kappa.py, get_epsilon.py, and get_lambda.py.

    - run_once: run a single cycle and exit (default). When False, loops.
    - interval_seconds: seconds to wait after each cycle completes (looping only).
    - start_dir: where to start upward search (default: this file's directory).
    - max_up: how many directory levels to traverse upward.

    The runner sets each test's working directory to the directory containing
    the test file, so relative imports work even when launched from elsewhere.
    """

    if start_dir is not None:
        start_dir = Path(start_dir).resolve()

    acquired_thread_lock = _RUNNER_LOCK.acquire(blocking=False)
    if not acquired_thread_lock:
        print(f"{_ts()} [runner] Skipping; parameter update already running.")
        return

    lock_path = _runner_lock_path()
    lock_holder: dict = {"token": None}
    try:
        asyncio.run(
            _periodic_worker(
                interval_seconds,
                start_dir,
                max_up,
                copy_configs,
                run_once,
                crypto,
                lock_path,
                lock_stale_seconds,
                lock_holder,
            )
        )
    except KeyboardInterrupt:
        _write_status("interrupted", {"crypto": crypto})
        print(f"{_ts()} [runner] Stopped by user.")
    finally:
        # Defensive: the worker releases the lock per cycle; this only covers an
        # interrupt that lands mid-cycle (token still held). Release is
        # ownership-checked, so a lock held by another process is never deleted.
        _release_process_lock(lock_path, lock_holder.get("token"))
        _RUNNER_LOCK.release()


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
    p.add_argument(
        "--crypto",
        type=str,
        default=os.getenv("CRYPTO_NAME", "ETH"),
        help="Symbol to pass to estimator scripts (default: ETH)",
    )
    p.add_argument(
        "--lock-stale-seconds",
        type=float,
        default=float(os.getenv("PARAM_UPDATE_LOCK_STALE_SECONDS", "3600")),
        help="Replace param_update.lock after this age in seconds (default: 3600)",
    )
    return p.parse_args(argv)


def _sigterm_to_keyboard_interrupt(signum, frame):
    """Translate SIGTERM into KeyboardInterrupt so lock release/status cleanup runs.

    Docker stops containers with SIGTERM; python's default action terminates
    immediately, skipping every finally block — which abandons param_update.lock
    mid-cycle and stalls both the next runner (skips cycles until the stale
    window, 1h) and the strategy (fail-closed on the lock). Raising
    KeyboardInterrupt reuses the existing interrupted-shutdown path instead.
    """
    raise KeyboardInterrupt


def _install_sigterm_handler() -> None:
    try:
        import signal

        signal.signal(signal.SIGTERM, _sigterm_to_keyboard_interrupt)
    except Exception:
        # Non-main thread or platform without SIGTERM: keep going without it.
        pass


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    if args.once and args.loop:
        import sys as _sys
        print("Cannot use --once and --loop together", file=_sys.stderr)
        raise SystemExit(2)
    _install_sigterm_handler()
    run_once = True if args.once or not args.loop else False
    schedule_tests(
        interval_seconds=args.interval,
        start_dir=Path(args.start_dir).resolve() if args.start_dir else None,
        max_up=args.max_up,
        copy_configs=not args.no_copy_configs,
        run_once=run_once,
        crypto=args.crypto,
        lock_stale_seconds=float(args.lock_stale_seconds),
    )
