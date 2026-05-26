from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_safety_gates import local_gates  # noqa: E402


def gate_names(*, include_runtime: bool) -> list[str]:
    return [name for name, _, _ in local_gates(include_runtime=include_runtime)]


def test_live_canary_gate_runs_after_dependency_artifacts_with_runtime_gates():
    names = gate_names(include_runtime=True)

    assert names.index("post_only_evidence_report") < names.index("live_canary_evidence_report")
    assert names.index("fee_evidence_report") < names.index("live_canary_evidence_report")
    assert names.index("replay_acceptance_report_artifact") < names.index("live_canary_evidence_report")


def test_live_canary_gate_is_still_available_without_runtime_gates():
    names = gate_names(include_runtime=False)

    assert "live_canary_evidence_report" in names
