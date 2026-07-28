"""The SN22 validator must accept only quote-bound remote execution."""

from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from kata_sn22.execution import policy as execution_policy
from kata_sn22.execution import tee_room
from kata_sn22.plugin import ROOM_ENDPOINT_ENV, Sn22DesearchPlugin

MEASUREMENT = "ab" * 32
NONCE = b"0123456789abcdef0123"
BUNDLE_DIGEST = "cd" * 32
PROJECT_KEY = '{"task_id":"t000"}'
PROVENANCE = {"profile": "sn22", "job_id": NONCE.hex()}
REPORT = {"task_id": "t000", "answer": "{}", "timed_out": False}


class _Verifier:
    def __init__(self, *, report=REPORT, measurement=MEASUREMENT):
        binding = {
            "report": report,
            "bundle_sha256": BUNDLE_DIGEST,
            "provenance": PROVENANCE,
        }
        binding_hash = hashlib.sha256(tee_room.canonical(binding)).digest()
        self.report_data = hashlib.sha256(
            NONCE + PROJECT_KEY.encode() + binding_hash
        ).digest()
        self.measurement = measurement

    def verify(self, _quote):
        return tee_room.VerifiedQuote(True, self.report_data, self.measurement, "OK")


def _result(report=REPORT):
    return tee_room.RoomResult(report, "quote", BUNDLE_DIGEST, PROVENANCE)


def test_valid_room_result_is_bound_to_task_bundle_answer_and_measurement():
    assert tee_room.verify_room_run(
        result=_result(),
        nonce=NONCE,
        project_key=PROJECT_KEY,
        expected_bundle_sha256=BUNDLE_DIGEST,
        policy=tee_room.RoomPolicy(frozenset({MEASUREMENT})),
        verifier=_Verifier(),
    ) is None


def test_answer_swap_and_unapproved_measurement_are_rejected():
    swapped = {"task_id": "t000", "answer": '{"swapped":true}', "timed_out": False}
    assert "bind" in tee_room.verify_room_run(
        result=_result(swapped),
        nonce=NONCE,
        project_key=PROJECT_KEY,
        expected_bundle_sha256=BUNDLE_DIGEST,
        policy=tee_room.RoomPolicy(frozenset({MEASUREMENT})),
        verifier=_Verifier(),
    )
    assert "not approved" in tee_room.verify_room_run(
        result=_result(),
        nonce=NONCE,
        project_key=PROJECT_KEY,
        expected_bundle_sha256=BUNDLE_DIGEST,
        policy=tee_room.RoomPolicy(frozenset({"ef" * 32})),
        verifier=_Verifier(),
    )


@pytest.mark.parametrize(
    ("status", "accepted"),
    [
        ("OK", True),
        ("SW_HARDENING_NEEDED", True),
        ("CONFIGURATION_NEEDED", False),
        ("OUT_OF_DATE", False),
    ],
)
def test_dcap_verifier_uses_the_pinned_status_contract(monkeypatch, status, accepted):
    report = SimpleNamespace(
        report_data=b"\x11" * 64,
        mr_config_id=b"\x01" + bytes.fromhex(MEASUREMENT),
    )
    parsed = SimpleNamespace(report=report, is_tdx=lambda: True)
    fake_qvl = SimpleNamespace(
        PHALA_PCCS_URL="https://pccs.example",
        parse_quote=lambda _raw: parsed,
        get_collateral=lambda _url, _raw: object(),
        verify=lambda _raw, _collateral, _now: SimpleNamespace(status=status),
    )
    monkeypatch.setitem(sys.modules, "dcap_qvl", fake_qvl)

    result = tee_room.DcapQvlVerifier().verify("00")

    assert result.ok is accepted
    assert result.measurement == MEASUREMENT


@pytest.mark.parametrize(
    "url",
    [
        "https://:password@room.example",
        "https://room.example?redirect=https://evil.example",
        "https://room.example#fragment",
    ],
)
def test_room_url_rejects_hidden_credentials_query_and_fragment(url):
    with pytest.raises(RuntimeError, match="KATA_SN22_ROOM_URL"):
        tee_room.validate_room_url(url)


def test_room_http_client_refuses_redirects():
    assert (
        tee_room._RejectRedirects().redirect_request(
            None, None, 307, "redirect", {}, "https://evil.example"
        )
        is None
    )


def test_bundle_transfer_refuses_symlinks(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "agent.py").write_text("print('{}')\n", encoding="utf-8")
    (bundle / "escape").symlink_to("/etc/passwd")
    with pytest.raises(RuntimeError, match="symlink"):
        tee_room.hash_bundle(bundle)


def test_bundle_transfer_refuses_payloads_larger_than_the_room_policy(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "agent.py").write_bytes(b"x" * (tee_room.MAX_BUNDLE_BYTES + 1))

    with pytest.raises(RuntimeError, match="byte room policy"):
        tee_room.hash_bundle(bundle)


def test_bundle_hash_matches_room_binding_and_excludes_public_ciphertext(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    source = b"def agent_main(): return {'ok': True}\n"
    (bundle / "agent.py").write_bytes(source)
    sealed = bundle / tee_room.SEALED_CREDENTIAL_FILENAME
    sealed.write_text("aa" * 64, encoding="utf-8")

    digest = hashlib.sha256(tee_room._BUNDLE_BINDING_DOMAIN)
    name = b"agent.py"
    digest.update(len(name).to_bytes(4, "big"))
    digest.update(name)
    digest.update(len(source).to_bytes(8, "big"))
    digest.update(source)
    expected = digest.hexdigest()

    assert tee_room.hash_bundle(bundle) == expected
    sealed.write_text("bb" * 64, encoding="utf-8")
    assert tee_room.hash_bundle(bundle) == expected
    (bundle / "agent.py").write_bytes(source + b"# changed\n")
    assert tee_room.hash_bundle(bundle) != expected


def test_plugin_uses_attested_usage_not_candidate_self_report(monkeypatch, tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "agent.py").write_text(
        "raise AssertionError('must run remotely')\n", encoding="utf-8")
    (bundle / "sealed_inference_key").write_text("aa" * 64, encoding="utf-8")

    plugin = Sn22DesearchPlugin()
    problems = plugin.sample_problems(seed="tee-result", config={"task_count": 1})
    task = problems.tasks[0]
    answer = json.dumps({
        "protocol_version": 1,
        "task_id": task.task_id,
        "summary": "",
        "results": [],
        "tweets": [],
        "citations": [],
        # Deliberately false. The attested gateway summary below is authoritative.
        "usage": {"provider_calls": 0, "tokens": 0, "elapsed_seconds": 0},
    })

    monkeypatch.setenv(execution_policy.EXECUTION_BACKEND_ENV, "tee")
    monkeypatch.setenv(ROOM_ENDPOINT_ENV, "https://room.example")
    monkeypatch.setenv(tee_room.ROOM_MEASUREMENTS_ENV, MEASUREMENT)
    monkeypatch.setenv(tee_room.ROOM_AUTH_SECRET_ENV, "shared-secret")
    monkeypatch.setattr(
        tee_room,
        "evaluate_candidate_in_room",
        lambda **_kwargs: tee_room.CandidateOutcome(
            True,
            {
                "task_id": task.task_id,
                "answer": answer,
                "timed_out": False,
                "returncode": 0,
                "truncated": False,
            },
            "ok",
            {"inference_summary": {"requests": 2, "tokens": 123}},
        ),
    )

    context = type(
        "Context",
        (),
        {"label": "candidate", "output_root": str(tmp_path), "progress": None},
    )()
    raw = plugin.run_candidate(
        agent_path=str(bundle),
        problems=problems,
        context=context,
    )

    assert raw.isolated is True
    assert raw.usage.totals("candidate") == {
        "provider_calls": 2.0,
        "tokens": 123.0,
        "spend_usd": 0.0,
    }
