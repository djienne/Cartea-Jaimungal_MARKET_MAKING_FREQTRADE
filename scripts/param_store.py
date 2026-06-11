"""Optional Redis-backed transport for the kappa/epsilon/lambda parameter snapshot.

The estimator publishes all three snapshots for a symbol as ONE atomic blob, and
the strategy reads that single blob — so freqtrade can never observe a torn /
partially-updated set of params. This removes the need for the file lock on the
read path (the lock only ever existed to guard the multi-file read).

Everything degrades gracefully: if the ``redis`` package is missing or the server
is unreachable, the functions no-op / return ``None`` and callers fall back to the
JSON files on disk, so the file-based pipeline keeps working unchanged.
"""

from __future__ import annotations

import json
from typing import Any

PARAM_KEY_PREFIX = "mm:params:"
PARAM_CHANNEL_PREFIX = "mm:params:updated:"
BLOB_SCHEMA_VERSION = 1


def _client(redis_url: str | None):
    if not redis_url:
        return None
    try:
        import redis  # type: ignore
    except Exception:
        return None
    try:
        client = redis.Redis.from_url(
            redis_url,
            socket_timeout=2.0,
            socket_connect_timeout=2.0,
            decode_responses=True,
        )
        client.ping()
        return client
    except Exception:
        return None


def publish_params(
    redis_url: str | None,
    crypto: str,
    *,
    kappa: dict[str, Any],
    epsilon: dict[str, Any],
    lambda_: dict[str, Any],
    published_at: str,
) -> bool:
    """Atomically publish the per-symbol kappa/epsilon/lambda snapshot.

    Returns True on success, False if Redis is unavailable (caller can ignore;
    the files on disk remain the fallback transport).
    """
    client = _client(redis_url)
    if client is None:
        return False
    blob = json.dumps(
        {
            "blob_schema_version": BLOB_SCHEMA_VERSION,
            "crypto": crypto,
            "published_at": published_at,
            "kappa": kappa,
            "epsilon": epsilon,
            "lambda": lambda_,
        },
        sort_keys=True,
    )
    try:
        client.set(PARAM_KEY_PREFIX + crypto, blob)
        client.publish(PARAM_CHANNEL_PREFIX + crypto, published_at)
        return True
    except Exception:
        return False


def fetch_params(redis_url: str | None, crypto: str) -> dict[str, Any] | None:
    """Return the published blob {"kappa":..., "epsilon":..., "lambda":..., ...}
    for ``crypto``, or None if Redis is unavailable / no blob / malformed."""
    client = _client(redis_url)
    if client is None:
        return None
    try:
        raw = client.get(PARAM_KEY_PREFIX + crypto)
    except Exception:
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("blob_schema_version") != BLOB_SCHEMA_VERSION:
        return None
    if not all(isinstance(data.get(k), dict) for k in ("kappa", "epsilon", "lambda")):
        return None
    return data
