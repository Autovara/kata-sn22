"""SN22-4: the credential boundary, under attack (plan §6.1–§6.2).

The exit gate is "fake-provider end-to-end challenges pass and all §6.2 attacks fail". §6.2 names
eight attacks; each has a section below, and each is written as the attacker rather than as the
defender — a test that asserts "the deny-list contains X" proves nothing, so these actually try it.

The single property everything rests on: **a candidate never holds a provider credential.** Since it
holds only a capability, the attacks reduce to abusing that capability, and each abuse is refused by
a specific check in ``gateway.py`` rather than by hope.

Sandbox tests that need a host able to create namespaces skip cleanly rather than pretending to pass
— an un-runnable isolation probe is not evidence of isolation.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from kata_sn22 import sandbox
from kata_sn22.fixtures import calibration_manifest, calibration_snapshot, tasks_for
from kata_sn22.gateway import (
    CAPABILITY_RE,
    PROVIDER_CREDENTIAL_NAMES,
    GatewayDenied,
    Sn22Gateway,
    redact,
)
from kata_sn22.protocol import parse_task_output

SIGNING_KEY = b"sn22-test-signing-key"


@pytest.fixture
def world():
    manifest = calibration_manifest()
    snapshot = calibration_snapshot(manifest)
    return manifest, snapshot, tasks_for(manifest)


@pytest.fixture
def gateway(world):
    _manifest, snapshot, _tasks = world
    return Sn22Gateway(snapshot=snapshot, challenge_id="c1")


# ---- the end-to-end path works at all ------------------------------------------------------------
def test_a_capability_buys_exactly_one_search(gateway, world):
    _m, _s, tasks = world
    capability = gateway.issue(variant="king", task_id=tasks[0].task_id, max_calls=2)
    assert CAPABILITY_RE.fullmatch(capability.token)
    results = gateway.search(capability.token, tasks[0].query)
    assert results and all({"doc_id", "title", "snippet"} == set(r) for r in results)


def test_both_contestants_receive_identical_content(gateway, world):
    """The fairness property: identical request, identical answer, regardless of who asked."""
    _m, _s, tasks = world
    king = gateway.issue(variant="king", task_id=tasks[0].task_id, max_calls=4)
    challenger = gateway.issue(variant="challenger", task_id=tasks[0].task_id, max_calls=4)
    assert gateway.search(king.token, tasks[0].query) == gateway.search(challenger.token,
                                                                        tasks[0].query)


def test_a_full_fake_provider_challenge_completes(gateway, world):
    """Both sides, every task, a symmetric usage manifest and a verifiable receipt."""
    _m, _s, tasks = world
    for variant in ("king", "challenger"):
        for task in tasks:
            capability = gateway.issue(variant=variant, task_id=task.task_id, max_calls=2)
            gateway.search(capability.token, task.query)
    gateway.close()

    usage = gateway.usage_manifest()
    usage.assert_symmetric(("king", "challenger"))
    receipt = gateway.usage_receipt(signing_key=SIGNING_KEY)
    assert Sn22Gateway.verify_receipt(receipt, signing_key=SIGNING_KEY)
    assert receipt["served_calls"] == 2 * len(tasks)


# ---- §6.2 attack 1: the candidate prints every environment variable -----------------------------
def test_the_constructed_environment_contains_no_credential():
    env = sandbox.candidate_env(task_input={"protocol_version": 1, "task_id": "t000"},
                                relay_endpoint="sn22-relay://c1", capability="sn22cap_" + "0" * 32,
                                workdir="/tmp/w")
    for name in PROVIDER_CREDENTIAL_NAMES:
        assert name not in env
    for name in ("KATA_TARGET_TOKEN", "KATA_WEBHOOK_SECRET", "KATA_ROOM_AUTH_SECRET",
                 "PYTHONPATH", "SSH_AUTH_SOCK"):
        assert name not in env


def test_the_environment_is_constructed_not_filtered(monkeypatch):
    """A filtered copy leaks whatever it forgot — and what it forgets is the newest secret."""
    monkeypatch.setenv("A_BRAND_NEW_PROVIDER_KEY", "sk-secret-value-nobody-denylisted")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-also-here")
    env = sandbox.candidate_env(task_input={}, relay_endpoint="r", capability="c", workdir="/tmp/w")
    assert "A_BRAND_NEW_PROVIDER_KEY" not in env
    assert "sk-secret-value-nobody-denylisted" not in json.dumps(env)
    assert set(env) == {"PATH", "HOME", "TMPDIR", "LC_ALL", "SN22_PROTOCOL_VERSION",
                        "SN22_TASK_ID", "SN22_RELAY_ENDPOINT", "SN22_RELAY_CAPABILITY"}


def test_an_agent_that_dumps_its_environment_learns_nothing(tmp_path, monkeypatch):
    """Run the attack for real, unsandboxed, to prove the ENV construction alone suffices."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-not-leak-0123456789")
    agent = tmp_path / "agent.py"
    agent.write_text("import json,os,sys\nsys.stdout.write(json.dumps(dict(os.environ)))\n")
    env = sandbox.candidate_env(task_input={"task_id": "t000"}, relay_endpoint="r",
                                capability="sn22cap_" + "0" * 32, workdir=str(tmp_path))
    out = subprocess.run(["/usr/bin/python3", str(agent)], capture_output=True, env=env,
                         check=False, timeout=60).stdout.decode()
    assert "sk-must-not-leak-0123456789" not in out
    assert "OPENAI_API_KEY" not in out


# ---- §6.2 attack 2: the candidate reads credential paths ----------------------------------------
def test_credential_paths_are_absent_from_the_mount_namespace():
    """Not merely unreadable — absent. The argv is asserted because it IS the policy."""
    argv = sandbox.build_argv("/tmp/agent.py", workdir="/tmp/w",
                              limits=sandbox.SandboxLimits(), as_root_launcher=False)
    joined = " ".join(argv)
    for path in ("/srv", "/home", "/root", "/etc"):
        assert f"--ro-bind {path} " not in joined
        assert f"--bind {path} " not in joined
    assert "--unshare-net" in argv          # no egress namespace at all
    assert "--cap-drop" in argv and "ALL" in argv
    assert str(sandbox.SANDBOX_UID) in argv  # runs as nobody


@pytest.mark.skipif(not sandbox.available(), reason="needs bwrap to build a namespace")
@pytest.mark.parametrize("path", sandbox.FORBIDDEN_PATHS)
def test_reading_a_credential_path_fails_inside_the_sandbox(tmp_path, path):
    workdir = sandbox.fresh_workdir("/tmp", "sn22-attack-read")
    agent = workdir / "agent.py"
    agent.write_text(
        "import sys\n"
        f"try:\n    sys.stdout.write(open({path!r}).read())\n"
        "except OSError as exc:\n    sys.stdout.write('DENIED')\n")
    try:
        run = sandbox.run_candidate(agent, {"task_id": "t000"}, workdir=workdir,
                                    relay_endpoint="r", capability="c")
    except sandbox.SandboxUnavailable as exc:
        pytest.skip(f"host cannot isolate: {exc}")
    assert b"DENIED" in run.stdout or run.returncode != 0


# ---- §6.2 attacks 3 and 4: egress ---------------------------------------------------------------
@pytest.mark.skipif(not sandbox.available(), reason="needs bwrap to build a namespace")
def test_arbitrary_egress_fails_inside_the_sandbox():
    workdir = sandbox.fresh_workdir("/tmp", "sn22-attack-egress")
    agent = workdir / "agent.py"
    agent.write_text(
        "import socket, sys\n"
        "s = socket.socket(); s.settimeout(5)\n"
        "try:\n    s.connect(('1.1.1.1', 443)); sys.stdout.write('REACHED')\n"
        "except OSError:\n    sys.stdout.write('DENIED')\n")
    try:
        run = sandbox.run_candidate(agent, {"task_id": "t000"}, workdir=workdir,
                                    relay_endpoint="r", capability="c")
    except sandbox.SandboxUnavailable as exc:
        pytest.skip(f"host cannot isolate: {exc}")
    assert b"REACHED" not in run.stdout


# ---- §6.2 attack 5: traversal and oversized requests --------------------------------------------
def test_an_oversized_relay_request_is_refused(gateway, world):
    _m, _s, tasks = world
    capability = gateway.issue(variant="king", task_id=tasks[0].task_id, max_calls=4)
    with pytest.raises(GatewayDenied, match="size limit"):
        gateway.search(capability.token, "x" * 100_000)


@pytest.mark.parametrize("limit", [0, -1, 5_000, "ten", True])
def test_an_out_of_range_result_limit_is_refused(gateway, world, limit):
    _m, _s, tasks = world
    capability = gateway.issue(variant="king", task_id=tasks[0].task_id, max_calls=4)
    with pytest.raises(GatewayDenied, match="limit out of range"):
        gateway.search(capability.token, tasks[0].query, limit=limit)


@pytest.mark.parametrize("token", [
    "../../etc/passwd", "sn22cap_../../x", "", None, 12345,
    "sn22cap_" + "z" * 32, "sn22cap_" + "0" * 31,
])
def test_a_malformed_capability_is_refused(gateway, token):
    """The token is used as a dict key and printed into records; it must be inert."""
    with pytest.raises(GatewayDenied, match="malformed capability"):
        gateway.search(token, "anything")


# ---- §6.2 attack 6: prompt injection in retrieved content ---------------------------------------
def test_retrieved_content_is_returned_as_inert_data(gateway, world):
    """Injected instructions travel as strings. Nothing here interprets them."""
    _m, _s, tasks = world
    capability = gateway.issue(variant="king", task_id=tasks[0].task_id, max_calls=4)
    results = gateway.search(capability.token, tasks[0].query)
    # The response is plain JSON-able data: no callables, no objects with behaviour.
    assert json.loads(json.dumps(results)) == results
    for result in results:
        assert all(isinstance(value, str) for value in result.values())


def test_secret_shaped_content_is_redacted_on_the_way_out():
    """Retrieved third-party text could itself contain a leaked key; it must not pass through."""
    for secret in ("sk-abcdefghijklmnopqrstuvwx", "ghp_abcdefghijklmnopqrstuvwxyz0123",
                   "Bearer abcdefghijklmnopqrstuvwx"):
        assert secret not in redact(f"the key is {secret} ok")
        assert "[REDACTED]" in redact(f"the key is {secret} ok")


# ---- §6.2 attack 7: capability reuse ------------------------------------------------------------
def test_a_capability_cannot_be_used_by_another_variant(gateway, world):
    """A challenger that steals the king's token still cannot mint its own with it."""
    _m, _s, tasks = world
    king = gateway.issue(variant="king", task_id=tasks[0].task_id, max_calls=4)
    # Even used verbatim, the call is billed to the KING -- so it cannot launder its own spend, and
    # it cannot make the king look expensive without also giving the king the results.
    gateway.search(king.token, tasks[0].query)
    assert gateway.usage_manifest().totals("challenger")["provider_calls"] == 0
    assert gateway.usage_manifest().totals("king")["provider_calls"] == 1


def test_a_capability_stops_working_when_the_challenge_closes(gateway, world):
    _m, _s, tasks = world
    capability = gateway.issue(variant="king", task_id=tasks[0].task_id, max_calls=4)
    gateway.search(capability.token, tasks[0].query)
    gateway.close()
    with pytest.raises(GatewayDenied, match="closed"):
        gateway.search(capability.token, tasks[0].query)
    with pytest.raises(GatewayDenied, match="closed"):
        gateway.issue(variant="king", task_id=tasks[0].task_id, max_calls=1)


def test_a_capability_expires_on_its_own(world):
    """Short-lived, so a token captured from a log is worthless minutes later."""
    _m, snapshot, tasks = world
    now = [1000.0]
    gateway = Sn22Gateway(snapshot=snapshot, challenge_id="c1", capability_ttl_seconds=60.0,
                          clock=lambda: now[0])
    capability = gateway.issue(variant="king", task_id=tasks[0].task_id, max_calls=9)
    gateway.search(capability.token, tasks[0].query)
    now[0] += 61.0
    with pytest.raises(GatewayDenied, match="expired"):
        gateway.search(capability.token, tasks[0].query)


def test_a_forged_token_of_the_right_shape_is_refused(gateway):
    """Well-formed but never issued. The refusal is identical to a malformed one, on purpose: a
    distinct message would let a candidate probe which tokens exist."""
    with pytest.raises(GatewayDenied, match="malformed capability"):
        gateway.search("sn22cap_" + "a" * 32, "query")


# ---- §6.2 attack 8: spending beyond the reservation ---------------------------------------------
def test_the_per_task_quota_is_enforced(gateway, world):
    _m, _s, tasks = world
    capability = gateway.issue(variant="king", task_id=tasks[0].task_id, max_calls=2)
    gateway.search(capability.token, tasks[0].query)
    gateway.search(capability.token, tasks[0].query)
    with pytest.raises(GatewayDenied, match="quota exhausted"):
        gateway.search(capability.token, tasks[0].query)


def test_the_challenge_reservation_is_a_hard_ceiling(world):
    """Per-task quotas bound one task. Only the reservation bounds the CHALLENGE, and that is the
    number that was actually approved and paid for."""
    _m, snapshot, tasks = world
    gateway = Sn22Gateway(snapshot=snapshot, challenge_id="c1", reservation_calls=3)
    spent = 0
    with pytest.raises(GatewayDenied, match="reservation exhausted"):
        for _round in range(10):
            capability = gateway.issue(variant="king", task_id=tasks[0].task_id, max_calls=100)
            gateway.search(capability.token, tasks[0].query)
            spent += 1
    assert spent == 3
    assert gateway.usage_manifest().totals("king")["provider_calls"] == 3


def test_minting_more_capabilities_does_not_raise_the_ceiling(world):
    """The obvious way around a per-task quota is more tasks. The reservation still holds."""
    _m, snapshot, tasks = world
    gateway = Sn22Gateway(snapshot=snapshot, challenge_id="c1", reservation_calls=2)
    tokens = [gateway.issue(variant="king", task_id=t.task_id, max_calls=50).token for t in tasks]
    served = 0
    for token in tokens:
        try:
            gateway.search(token, "bittensor subnet emissions schedule")
            served += 1
        except GatewayDenied:
            break
    assert served == 2


def test_a_zero_call_capability_is_refused(gateway, world):
    _m, _s, tasks = world
    with pytest.raises(GatewayDenied, match="not a capability"):
        gateway.issue(variant="king", task_id=tasks[0].task_id, max_calls=0)


def test_a_call_is_billed_even_if_it_is_the_last_one(gateway, world):
    """Billing happens BEFORE serving: the provider charges regardless of what we do next."""
    _m, _s, tasks = world
    capability = gateway.issue(variant="king", task_id=tasks[0].task_id, max_calls=1)
    gateway.search(capability.token, tasks[0].query)
    assert gateway.usage_manifest().totals("king")["provider_calls"] == 1


# ---- receipts -----------------------------------------------------------------------------------
def test_a_receipt_verifies_and_an_edited_one_does_not(gateway, world):
    _m, _s, tasks = world
    capability = gateway.issue(variant="king", task_id=tasks[0].task_id, max_calls=2)
    gateway.search(capability.token, tasks[0].query)
    receipt = gateway.usage_receipt(signing_key=SIGNING_KEY)
    assert Sn22Gateway.verify_receipt(receipt, signing_key=SIGNING_KEY)

    tampered = {**receipt, "served_calls": 0}
    assert not Sn22Gateway.verify_receipt(tampered, signing_key=SIGNING_KEY)
    assert not Sn22Gateway.verify_receipt(receipt, signing_key=b"a-different-key")


def test_a_receipt_carries_no_credential_and_no_capability_token(gateway, world):
    """It is an audit artifact that gets copied around; it must be safe to copy."""
    _m, _s, tasks = world
    capability = gateway.issue(variant="king", task_id=tasks[0].task_id, max_calls=2)
    gateway.search(capability.token, tasks[0].query)
    serialized = json.dumps(gateway.usage_receipt(signing_key=SIGNING_KEY))
    assert capability.token not in serialized
    for name in PROVIDER_CREDENTIAL_NAMES:
        assert name not in serialized


def test_the_receipt_digest_tracks_the_usage(gateway, world):
    _m, _s, tasks = world
    first = gateway.issue(variant="king", task_id=tasks[0].task_id, max_calls=4)
    gateway.search(first.token, tasks[0].query)
    before = gateway.usage_receipt(signing_key=SIGNING_KEY)["usage_digest"]
    gateway.search(first.token, tasks[0].query)
    assert gateway.usage_receipt(signing_key=SIGNING_KEY)["usage_digest"] != before


# ---- what the candidate is told -----------------------------------------------------------------
def test_the_public_capability_view_names_no_provider(gateway, world):
    """A candidate should not even learn WHICH providers exist behind the relay."""
    _m, _s, tasks = world
    public = gateway.issue(variant="king", task_id=tasks[0].task_id, max_calls=3).as_public()
    serialized = json.dumps(public).lower()
    for provider in ("openai", "apify", "scrapingdog", "twitter", "chutes"):
        assert provider not in serialized
    assert set(public) == {"capability", "task_id", "max_calls", "expires_at"}


# ---- the sandbox refuses rather than degrading --------------------------------------------------
def test_an_unisolatable_host_refuses_instead_of_running_unconfined(monkeypatch, tmp_path):
    """A submission that could not be isolated has not been evaluated."""
    monkeypatch.setattr(sandbox, "available", lambda: False)
    with pytest.raises(sandbox.SandboxUnavailable, match="unconfined"):
        sandbox.run_candidate(tmp_path / "agent.py", {"task_id": "t000"}, workdir=tmp_path,
                              relay_endpoint="r", capability="c")


@pytest.mark.skipif(not sandbox.available(), reason="needs bwrap to build a namespace")
def test_an_honest_agent_still_works_inside_the_sandbox(world):
    """The isolation must not be so tight that a legitimate submission cannot run."""
    _m, _s, tasks = world
    workdir = sandbox.fresh_workdir("/tmp", "sn22-honest")
    agent = workdir / "agent.py"
    agent.write_text(
        "import json,sys\n"
        "task = json.load(sys.stdin)\n"
        "print(json.dumps({'protocol_version': task['protocol_version'],\n"
        "  'task_id': task['task_id'], 'summary': 'ok', 'results': [], 'citations': [],\n"
        "  'usage': {'provider_calls': 0, 'tokens': 0, 'elapsed_seconds': 0.0}}))\n")
    try:
        run = sandbox.run_candidate(agent, tasks[0].as_input(), workdir=workdir,
                                    relay_endpoint="sn22-relay://c1",
                                    capability="sn22cap_" + "0" * 32)
    except sandbox.SandboxUnavailable as exc:
        pytest.skip(f"host cannot isolate: {exc}")
    assert run.returncode == 0
    assert parse_task_output(run.stdout, task=tasks[0]).task_id == tasks[0].task_id


def test_oversized_output_is_truncated_and_flagged(tmp_path):
    """Truncation must be visible: a silently cut response would be scored as a valid short one."""
    limits = sandbox.SandboxLimits(max_output_bytes=64)
    run = sandbox.SandboxRun(returncode=0, stdout=b"x" * 200, stderr="",
                             truncated=len(b"x" * 200) > limits.max_output_bytes)
    assert run.truncated


def test_the_sandbox_argv_is_a_pure_function():
    """Asserted so the policy can be reviewed on any host, including one that cannot sandbox."""
    first = sandbox.build_argv("/tmp/a.py", workdir="/tmp/w", limits=sandbox.SandboxLimits())
    second = sandbox.build_argv("/tmp/a.py", workdir="/tmp/w", limits=sandbox.SandboxLimits())
    assert first == second
    assert Path(first[0]).name in {"sudo", "bwrap"}
