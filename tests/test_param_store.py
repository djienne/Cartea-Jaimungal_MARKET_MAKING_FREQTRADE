from __future__ import annotations

import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


class _FakeRedis:
    store: dict[str, str] = {}
    published: list = []

    @classmethod
    def from_url(cls, url, **kwargs):
        return cls()

    def ping(self):
        return True

    def set(self, key, value):
        _FakeRedis.store[key] = value

    def get(self, key):
        return _FakeRedis.store.get(key)

    def publish(self, channel, message):
        _FakeRedis.published.append((channel, message))


def _install_fake_redis():
    module = types.ModuleType("redis")
    module.Redis = _FakeRedis
    sys.modules["redis"] = module


_install_fake_redis()
import param_store  # noqa: E402


def setup_function():
    _FakeRedis.store = {}
    _FakeRedis.published = []


def test_publish_then_fetch_roundtrip():
    ok = param_store.publish_params(
        "redis://x:6379/0",
        "ETH",
        kappa={"kappa+": 1.0, "status": "ok"},
        epsilon={"epsilon+": 0.0, "status": "ok"},
        lambda_={"lambda+": 0.1, "lambda_source": "lambda0_fit"},
        published_at="2026-05-28T00:00:00Z",
    )
    assert ok is True
    assert _FakeRedis.published  # pub/sub notification emitted

    blob = param_store.fetch_params("redis://x:6379/0", "ETH")
    assert blob is not None
    assert blob["crypto"] == "ETH"
    assert blob["kappa"] == {"kappa+": 1.0, "status": "ok"}
    assert blob["epsilon"] == {"epsilon+": 0.0, "status": "ok"}
    assert blob["lambda"] == {"lambda+": 0.1, "lambda_source": "lambda0_fit"}


def test_fetch_returns_none_when_missing():
    assert param_store.fetch_params("redis://x:6379/0", "BTC") is None


def test_fetch_returns_none_for_malformed_blob():
    _FakeRedis.store[param_store.PARAM_KEY_PREFIX + "ETH"] = "{\"kappa\": 5}"  # not a dict
    assert param_store.fetch_params("redis://x:6379/0", "ETH") is None


def test_fetch_returns_none_for_unknown_blob_schema_version():
    import json

    blob = {
        "blob_schema_version": param_store.BLOB_SCHEMA_VERSION + 1,
        "crypto": "ETH",
        "published_at": "2026-05-28T00:00:00Z",
        "kappa": {"kappa+": 1.0},
        "epsilon": {"epsilon+": 0.0},
        "lambda": {"lambda+": 0.1},
    }
    _FakeRedis.store[param_store.PARAM_KEY_PREFIX + "ETH"] = json.dumps(blob)
    assert param_store.fetch_params("redis://x:6379/0", "ETH") is None


def test_no_redis_url_is_noop():
    assert param_store.publish_params(
        None, "ETH", kappa={}, epsilon={}, lambda_={}, published_at="x"
    ) is False
    assert param_store.fetch_params(None, "ETH") is None
