from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# v4: lambda± is the direct per-side market-order arrival rate (survival-based,
# not the binned-density intercept), kappa± is fitted on mid-relative depths from
# the BBO stream, and epsilon± is the direct validated arrival-jump estimate.
# No temporal smoothing is applied to model parameters. Consumers fail closed on
# older schemas because the primary-value semantics changed from v3.
#
# v5 (2026-09-02): lambda± is that raw rate scaled by the survival fit's
# intercept A. The fit says P(depth >= delta) = A * exp(-kappa * delta) over its
# support while the HJB models fill intensity as lambda * exp(-kappa * delta),
# so v4 was off by A at every depth (1.04 / 0.99 on CASHCAT). The raw rate is
# still published as lambda±_raw. Consumers fail closed on v4.
PARAM_SCHEMA_VERSION = 5


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def timestamp_to_iso(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if hasattr(value, "to_pydatetime"):
            value = value.to_pydatetime()
        if isinstance(value, datetime):
            dt = value
        else:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    except Exception:
        return str(value)


def load_json_object(path: str | Path) -> dict:
    json_path = Path(path)
    if not json_path.exists():
        return {}
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def atomic_write_json(path: str | Path, payload: dict) -> None:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=4, sort_keys=True), encoding="utf-8")
    tmp.replace(json_path)


def finite_or_none(value: Any) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if val != val or val in (float("inf"), float("-inf")):
        return None
    return val
