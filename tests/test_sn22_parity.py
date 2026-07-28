"""The always-on half of the SN22-5 exit gate: snapshot integrity and adapter/upstream parity.

These run offline with no third-party dependency. They do not execute the upstream — they check the
adapter against the evidence a reviewer recorded from it (`tools/record_parity.py`), and they check
that the evidence still describes the tree on disk. The executed half lives in
`test_sn22_upstream_executed.py` and needs the parity extra.

The division is what makes the gate meaningful. If both halves could run in the same command, a
build could regenerate its own expectations and every adapter would pass.
"""
from __future__ import annotations

import json

import pytest

from kata_sn22 import parity
from kata_sn22 import upstream_adapter as adapter
from kata_sn22 import upstream_snapshot as snapshot

# ---- the pinned snapshot ------------------------------------------------------------------------

def test_pinned_snapshot_matches_its_manifest():
    verification = snapshot.verify_snapshot()
    assert verification.ok, "\n".join(verification.findings)
    assert verification.observed_tree_sha256 == verification.expected_tree_sha256


def test_manifest_pins_the_audited_commit():
    manifest = snapshot.load_manifest()
    assert manifest["upstream_commit"] == snapshot.UPSTREAM_COMMIT
    assert manifest["upstream_repo"] == snapshot.UPSTREAM_REPO
    # A snapshot with a handful of files is a subset someone trimmed, not the pinned tree.
    assert manifest["file_count"] > 150


def test_snapshot_contains_the_scoring_components_the_adapter_claims():
    root = snapshot.snapshot_root()
    for component in parity.COMPONENTS:
        assert (root / component.upstream_path).is_file(), component.upstream_path


def test_require_intact_raises_on_a_removed_file(tmp_path):
    """Verification is fail-closed: a missing file is a finding, not a shrug."""
    import shutil

    copy = tmp_path / "upstream"
    shutil.copytree(snapshot.snapshot_root(), copy)
    (copy / "desearch" / "utils.py").unlink()
    verification = snapshot.verify_snapshot(copy)
    assert not verification.ok
    assert any("desearch/utils.py" in finding for finding in verification.findings)
    with pytest.raises(snapshot.SnapshotError):
        snapshot.require_intact(copy)


def test_an_unlisted_file_is_a_finding(tmp_path):
    """An EXTRA file matters as much as a changed one: the adapter imports from this tree."""
    import shutil

    copy = tmp_path / "upstream"
    shutil.copytree(snapshot.snapshot_root(), copy)
    (copy / "sitecustomize.py").write_text("import os\n", encoding="utf-8")
    verification = snapshot.verify_snapshot(copy)
    assert not verification.ok
    assert any("not listed in the manifest" in finding for finding in verification.findings)


def test_one_changed_byte_moves_the_tree_digest(tmp_path):
    """The exit gate's core claim, exercised rather than asserted."""
    import shutil

    copy = tmp_path / "upstream"
    shutil.copytree(snapshot.snapshot_root(), copy)
    target = copy / "neurons" / "validators" / "scoring" / "constants.py"
    target.write_text(target.read_text(encoding="utf-8").replace("0.90", "0.91"), encoding="utf-8")
    verification = snapshot.verify_snapshot(copy)
    assert not verification.ok
    assert verification.observed_tree_sha256 != verification.expected_tree_sha256
    assert any("digest drift" in finding for finding in verification.findings)


def test_a_symlink_in_the_snapshot_is_rejected(tmp_path):
    import shutil

    copy = tmp_path / "upstream"
    shutil.copytree(snapshot.snapshot_root(), copy)
    (copy / "leak.py").symlink_to("/etc/passwd")
    verification = snapshot.verify_snapshot(copy)
    assert not verification.ok
    assert any("symlink" in finding for finding in verification.findings)


# ---- the recorded evidence ----------------------------------------------------------------------

def test_recorded_evidence_describes_the_installed_tree():
    findings = parity.evidence_is_current()
    assert not findings, "\n".join(findings)


def test_adapter_agrees_with_the_recorded_upstream_outputs():
    findings = parity.compare_against_expectations()
    assert not findings, "\n".join(findings)


def test_every_registered_component_has_a_source_pin():
    pins = parity.source_pins()
    assert set(pins) == {component.name for component in parity.COMPONENTS}
    for name, pin in pins.items():
        assert "error" not in pin, f"{name}: {pin.get('error')}"
        assert len(pin["file_sha256"]) == 64
        assert len(pin["symbol_sha256"]) == 64


def test_evidence_records_the_upstream_identity():
    expectations = parity.load_expectations()
    assert expectations["upstream_commit"] == snapshot.UPSTREAM_COMMIT
    assert expectations["upstream_repo"] == snapshot.UPSTREAM_REPO
    assert expectations["upstream_tree_sha256"] == snapshot.load_manifest()["tree_sha256"]
    assert expectations["recorded_by"] == "tools/record_parity.py"


def test_stale_evidence_is_reported_not_ignored(monkeypatch):
    """Evidence recorded against a different tree must fail, not silently pass."""
    stale = json.loads(parity.expectations_path().read_text(encoding="utf-8"))
    stale["upstream_tree_sha256"] = "0" * 64
    monkeypatch.setattr(parity, "load_expectations", lambda: stale)
    findings = parity.evidence_is_current()
    assert any("different upstream tree" in finding for finding in findings)


def test_evidence_for_a_different_commit_is_rejected(monkeypatch):
    stale = json.loads(parity.expectations_path().read_text(encoding="utf-8"))
    stale["upstream_commit"] = "f" * 40
    monkeypatch.setattr(parity, "load_expectations", lambda: stale)
    assert any("upstream commit" in finding for finding in parity.evidence_is_current())


def test_a_wrong_adapter_constant_is_caught():
    """The comparison has teeth: perturb one weight and the diff must find it."""
    expectations = json.loads(parity.expectations_path().read_text(encoding="utf-8"))
    expectations["constants"]["AI_CONTENT_WEIGHT"] = 0.65
    findings = parity.compare_against_expectations(expectations)
    assert any("AI_CONTENT_WEIGHT" in finding for finding in findings)


def test_a_wrong_penalty_value_is_caught():
    expectations = json.loads(parity.expectations_path().read_text(encoding="utf-8"))
    expectations["cases"]["ai-fast-count-shortfall"]["count_penalty"] = 0.5
    findings = parity.compare_against_expectations(expectations)
    assert any("count_penalty" in finding for finding in findings)


def test_float_tolerance_is_far_below_any_promotion_margin():
    """A tolerance loose enough to hide a real difference would make parity decorative."""
    from kata_sn22.plugin import PROMOTION_MARGINS

    smallest = min(value for value in PROMOTION_MARGINS.values() if value > 0)
    assert parity.FLOAT_TOLERANCE < smallest / 1000


# ---- the parity report --------------------------------------------------------------------------

def test_parity_report_is_clean_and_names_its_boundary():
    report = parity.parity_report()
    assert report["ok"], json.dumps(report, indent=2)
    assert report["upstream_commit"] == snapshot.UPSTREAM_COMMIT
    assert report["case_count"] == len(parity.PARITY_CASES)
    # Components that are pinned but NOT executed must say so, by name and with a reason. A report
    # that quietly listed one as executed would be the exact overclaim this gate exists to prevent,
    # so the set is asserted exactly: adding to it has to be a deliberate edit here.
    #
    # Both entries are live validator STEPS rather than pure functions -- one logs to W&B and writes
    # a metagraph-sized array, the other walks pydantic synapses inside a reward model. Running
    # either would mean reconstructing a validator. Every INPUT they combine is executed above, and
    # each is pinned by the digest of its own source text, so an upstream edit still invalidates the
    # evidence.
    pinned_only = report["pinned_only_components"]
    assert sorted(component["name"] for component in pinned_only) == [
        "MIN_MINER_TWEETS", "score_response"]
    assert all(component["note"] for component in pinned_only)
    assert len(report["executed_components"]) == len(parity.COMPONENTS) - len(pinned_only)


def test_report_is_json_serializable():
    json.dumps(parity.parity_report())


# ---- coverage of the adapted surface -------------------------------------------------------------

def test_every_penalty_fires_in_at_least_one_case_and_stays_quiet_in_another():
    """A penalty only ever observed at 0.0 is untested; so is one that never rests."""
    outputs = {case["id"]: parity.adapter_case_outputs(case) for case in parity.PARITY_CASES}
    for name in adapter.PENALTY_FUNCTIONS:
        values = [case[name] for case in outputs.values()]
        assert any(value > 0 for value in values), f"{name} never fires"
        assert any(value == 0 for value in values), f"{name} always fires"


def test_both_search_types_and_every_ai_mode_are_covered():
    kinds = {case["response"]["kind"] for case in parity.PARITY_CASES}
    assert kinds == {"ai_search", "x_search"}
    modes = {case["response"].get("mode") for case in parity.PARITY_CASES
             if case["response"]["kind"] == "ai_search"}
    assert modes == set(adapter.AI_MODE_WEIGHTS)


def test_every_case_explains_itself():
    for case in parity.PARITY_CASES:
        assert case["why"].strip(), case["id"]


def test_case_ids_are_unique():
    ids = [case["id"] for case in parity.PARITY_CASES]
    assert len(ids) == len(set(ids))
