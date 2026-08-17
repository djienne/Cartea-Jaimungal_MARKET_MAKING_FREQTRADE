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

# Inventory heartbeat for the two-sided pair. The param key above is namespaced
# by SYMBOL only, so two instances quoting the same symbol would clobber it;
# the heartbeat is namespaced by symbol AND role.
INVENTORY_KEY_PREFIX = "mm:inv:"
INVENTORY_SCHEMA_VERSION = 1
ROLES = ("long", "short")


def peer_role(role: str) -> str:
    """The other leg of the pair."""
    return "short" if str(role).strip().lower() == "long" else "long"


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


def publish_inventory(
    redis_url: str | None,
    crypto: str,
    role: str,
    *,
    q_own: int,
    param_fingerprint: str | None,
    published_at: str,
    ttl_seconds: int = 30,
) -> bool:
    """Publish this instance's own inventory so its peer can price off the net.

    ``param_fingerprint`` is the CONTENT hash of the parameter snapshot the
    quote was built from -- deliberately not the strategy's ``hjb_generation``,
    which is a per-process counter that two instances could never agree on.

    A TTL is set so a dead instance's inventory expires rather than lingering as
    a plausible-looking stale value. Publishing must never be gated on the
    instance's own willingness to quote: if a stalled instance stopped
    publishing, staleness would become mutual and permanent.
    """
    client = _client(redis_url)
    if client is None:
        return False
    blob = json.dumps(
        {
            "inventory_schema_version": INVENTORY_SCHEMA_VERSION,
            "crypto": crypto,
            "role": role,
            "q_own": int(q_own),
            "param_fingerprint": param_fingerprint,
            "published_at": published_at,
        },
        sort_keys=True,
    )
    try:
        client.set(
            INVENTORY_KEY_PREFIX + f"{crypto}:{role}",
            blob,
            ex=max(int(ttl_seconds), 1),
        )
        return True
    except Exception:
        return False


def fetch_inventory(redis_url: str | None, crypto: str, role: str) -> dict[str, Any] | None:
    """Read one leg's published inventory, or None if absent/expired/malformed.

    None is the fail-closed signal: the caller must not assume a missing peer is
    flat, because "flat" is a perfectly plausible value that would silently
    mis-price the net inventory.
    """
    client = _client(redis_url)
    if client is None:
        return None
    try:
        raw = client.get(INVENTORY_KEY_PREFIX + f"{crypto}:{role}")
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
    if data.get("inventory_schema_version") != INVENTORY_SCHEMA_VERSION:
        return None
    if data.get("role") != role or data.get("crypto") != crypto:
        return None
    try:
        data["q_own"] = int(data["q_own"])
    except (KeyError, TypeError, ValueError):
        return None
    return data
