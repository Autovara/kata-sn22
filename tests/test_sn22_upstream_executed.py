"""The executed half of the SN22-5 exit gate: run the REAL pinned upstream and diff it live.

`test_sn22_parity.py` checks the adapter against *recorded* upstream outputs. That is the right
thing for CI and for the installed lane, but it has one hole a reviewer should not have to take on
trust: the recording itself. This module closes it by importing the vendored upstream under
`tools/upstream_shim.py`, running it over the same cases, and comparing the numbers directly.

Skipped when the parity extra is absent (`uv sync --extra parity`), because a lane runtime has no
business carrying pydantic and numpy. Skipping is safe precisely because the recorded evidence is
still checked without them — what is lost is the live re-derivation, not the comparison.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from kata_sn22 import parity
from kata_sn22.upstream_snapshot import snapshot_root

TOOLS = Path(__file__).resolve().parent.parent / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import upstream_shim  # noqa: E402

pytestmark = pytest.mark.skipif(
    not upstream_shim.available(),
    reason="needs the parity extra (uv sync --extra parity) to import the pinned upstream")


@pytest.fixture(scope="module")
def recorded():
    """A FRESH recording, produced by executing the pinned upstream right now."""
    import record_parity

    return record_parity.record(snapshot_root())


def test_the_pinned_upstream_actually_imports_and_runs(recorded):
    """If the shim ever stopped reaching real code, every comparison below would be vacuous."""
    assert recorded["cases"], "no cases were recorded"
    # A stub leaks through as a non-numeric value; a real penalty is a float in [0, 1].
    for case_id, values in recorded["cases"].items():
        for name, value in values.items():
            if name.endswith("_penalty"):
                assert isinstance(value, (int, float)), f"{case_id}.{name} is {value!r}"
                assert 0.0 <= value <= 1.0, f"{case_id}.{name} = {value}"


def test_adapter_matches_a_live_upstream_run(recorded):
    findings = parity.compare_against_expectations(recorded)
    assert not findings, "\n".join(findings)


def test_stored_evidence_equals_a_live_recording(recorded):
    """The committed evidence must be exactly what the pinned upstream produces today.

    This is what makes the recorded file trustworthy without re-reading the recorder: if anyone
    hand-edited a number into agreement, this fails.
    """
    stored = json.loads(parity.expectations_path().read_text(encoding="utf-8"))
    for key in ("upstream_commit", "upstream_tree_sha256", "constants", "cases", "scalars",
                "source_pins"):
        assert stored[key] == recorded[key], f"stored {key} differs from a live recording"


def test_upstream_weight_tables_are_what_the_plan_states(recorded):
    """Plan §5.3 quotes specific numbers; they are read back out of the executed upstream."""
    constants = recorded["constants"]
    assert constants["SEARCH_TYPE_WEIGHTS"] == {"ai_search": 0.90, "x_search": 0.10}
    assert constants["AI_MODE_WEIGHTS"] == {"fast": 0.60, "balanced": 0.20, "deep": 0.20}
    assert constants["AI_CONTENT_WEIGHT"] == 0.60
    assert constants["AI_SUMMARY_WEIGHT"] == 0.40


def test_shim_refuses_a_tree_that_is_not_the_upstream(tmp_path):
    """A wrong root must fail loudly rather than import whatever is on sys.path."""
    with pytest.raises(upstream_shim.ShimUnavailable):
        upstream_shim.install(tmp_path)
