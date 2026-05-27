from __future__ import annotations

import json
import sys
from uuid import uuid4
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from verify_freqtrade_tif_runtime import (  # noqa: E402
    build_report,
    build_variant_config,
    container_config_path,
    freqtrade_list_strategies_command,
    write_variant_configs,
)


def test_build_variant_config_keeps_probe_fail_closed():
    config = build_variant_config(
        {
            "dry_run": False,
            "force_entry_enable": True,
            "order_time_in_force": {"entry": "GTC", "exit": "GTC"},
            "market_making": {"trading_enabled": True, "post_only_verified": True},
            "api_server": {"forcebuy_enable": True, "force_entry_enable": True},
        },
        "Alo",
    )

    assert config["dry_run"] is True
    assert config["force_entry_enable"] is False
    assert config["order_time_in_force"] == {"entry": "Alo", "exit": "Alo"}
    assert config["market_making"]["trading_enabled"] is False
    assert config["market_making"]["post_only_verified"] is False
    assert config["api_server"]["forcebuy_enable"] is False


def test_write_variant_configs_writes_container_mounted_configs(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"api_server": {}}), encoding="utf-8")
    work_dir = ROOT / "user_data" / "logs" / f"test_tif_runtime_{uuid4().hex}"

    paths = write_variant_configs(base, work_dir, ["GTC"])

    try:
        assert paths["GTC"].exists()
        assert container_config_path(paths["GTC"]).startswith("/freqtrade/user_data/logs/test_tif_runtime_")
    finally:
        for path in paths.values():
            path.unlink(missing_ok=True)
        work_dir.rmdir()


def test_freqtrade_list_strategies_command_uses_runtime_config():
    command = freqtrade_list_strategies_command("/freqtrade/user_data/logs/config_tif_alo.json")

    assert command[:5] == ["docker", "compose", "run", "--rm", "--no-deps"]
    assert "list-strategies" in command
    assert "/freqtrade/user_data/logs/config_tif_alo.json" in command


def test_build_report_records_unsupported_post_only_without_failing_artifact():
    report = build_report(
        [
            {"time_in_force": "GTC", "returncode": 0},
            {"time_in_force": "PO", "returncode": 2},
            {"time_in_force": "Alo", "returncode": 2},
        ]
    )

    assert report["ok"] is True
    assert report["research_gtc_supported"] is True
    assert report["freqtrade_post_only_config_supported"] is False
    assert report["freqtrade_post_only_unsupported_tifs"] == ["PO", "Alo"]
    assert report["live_safe_via_freqtrade"] is False
    assert report["requires_exchange_submit_evidence"] is True


def test_build_report_fails_when_research_gtc_config_rejected():
    report = build_report([{"time_in_force": "GTC", "returncode": 2}])

    assert report["ok"] is False
    assert "gtc_research_config_rejected" in report["reasons"]
