"""The SN22 room profile, driven through its REAL path rather than its fixture stub.

Every test here exists because its absence hid a defect. The profile had three, and all three would
have fired on the first production round while looking like something else:

1. it called ``inference_gateway_url(job_id=…, credential=…)``; the signature is
   ``(job_id, provider)`` — a **TypeError**, which would have read as "the room is broken";
2. it never passed the miner's decrypted API key to the agent, so the gateway would have answered
   **401** — which would have read as "the miner's credential is bad";
3. the room mounts the bundle read-only and provided no ``sn22_relay`` module, so every agent would
   have died on **ImportError** — which would have read as "the miner wrote a broken agent".

None was caught because the room tests all used ``fixture_project = "fixture-task"``, the no-docker
stub, and the stub returns before reaching any of it. So these tests deliberately do NOT take that
path: they drive ``run()`` with a real task and intercept at the ``docker`` boundary, which is the
last point before something outside this repository would be needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "deploy" / "sn22-runner"
ROOM = Path(__file__).resolve().parents[2] / "kata-tee-runner"

pytestmark = pytest.mark.skipif(
    not (ROOM / "room").is_dir(), reason="kata-tee-runner is not checked out beside this repo")


@pytest.fixture
def profile(monkeypatch, tmp_path):
    """The real ``Sn22TeeProfile``, with the room importable and docker intercepted."""
    for path in (str(ROOM), str(RUNNER)):
        if path not in sys.path:
            sys.path.insert(0, path)
    monkeypatch.setenv(
        "KATA_SN22_TEE_AGENT_IMAGE",
        "docker.io/example/kata-sn22-agent@sha256:" + "a" * 64)
    # The room refuses to sign a gateway route without one, and fails CLOSED when it is absent --
    # which is right, and means a test driving the real path has to supply one.
    monkeypatch.setenv("KATA_ROOM_AUTH_SECRET", "test-room-secret")

    import tee_profile

    calls: list[list[str]] = []

    class _Completed:
        returncode = 0
        stdout = json.dumps({"protocol_version": 1, "task_id": "t000", "summary": "an answer",
                             "results": [], "tweets": [], "citations": [],
                             "usage": {"provider_calls": 1, "tokens": 10}}).encode()
        stderr = b""

    monkeypatch.setattr(tee_profile, "docker", lambda: "docker")
    monkeypatch.setattr(tee_profile, "ensure_inference_network_once", lambda: None)
    monkeypatch.setattr(tee_profile, "start_inference_gateway_once", lambda: None)
    monkeypatch.setattr(tee_profile.subprocess, "run",
                        lambda argv, **kw: calls.append(list(argv)) or _Completed())
    return tee_profile.Sn22TeeProfile(), calls


@pytest.fixture
def credential():
    from room.profile import MinerInferenceCredential

    return MinerInferenceCredential(provider="openai", api_key="sk-miner-key",
                                    bundle_binding="b" * 64)


#: Job ids are 16..64 lowercase hex — the room validates that before signing a gateway route, so a
#: readable placeholder like "job-1" is refused. Using a real-shaped one keeps this a test of the
#: PROFILE rather than of the room's id validation.
JOB_ID = "0123456789abcdef"

TASK = json.dumps({"protocol_version": 1, "task_id": "t000", "query": "what is the figure?",
                   "search_type": "ai_search", "ai_mode": "fast",
                   "limits": {"max_results": 5, "max_wall_seconds": 60}})


def _env_of(argv: list[str]) -> dict:
    """The ``--env K=V`` pairs from a docker argv."""
    env = {}
    for index, item in enumerate(argv):
        if item == "--env" and index + 1 < len(argv):
            key, _, value = argv[index + 1].partition("=")
            env[key] = value
    return env


# ---- the three defects -------------------------------------------------------------------------

def test_a_real_task_reaches_the_agent_at_all(profile, credential, tmp_path):
    """Defect 1. This is the whole test: `run()` with a real task used to raise TypeError before it
    got anywhere near starting an agent."""
    plugin, calls = profile
    (tmp_path / "agent.py").write_text("print('{}')", encoding="utf-8")

    result = plugin.run(project_key=TASK, credential=credential, bundle_root=str(tmp_path),
                        job_id=JOB_ID, bundle_sha256="c" * 64)

    assert calls, "the profile never started an agent container"
    assert result.report["task_id"] == "t000"


def test_the_miners_own_key_travels_to_the_agent(profile, credential, tmp_path):
    """Defect 2. Without it the in-room gateway answers 401 for every call, and the agent cannot
    tell that apart from a bad query."""
    plugin, calls = profile
    (tmp_path / "agent.py").write_text("print('{}')", encoding="utf-8")

    plugin.run(project_key=TASK, credential=credential, bundle_root=str(tmp_path),
               job_id=JOB_ID, bundle_sha256="c" * 64)

    env = _env_of(calls[0])
    assert env["SN22_INFERENCE_API_KEY"] == "sk-miner-key"
    assert env["SN22_INFERENCE_GATEWAY"].startswith("http://")


def test_the_gateway_route_is_bound_to_the_credentials_provider(profile, credential, tmp_path):
    """The route is derived from the PROVIDER, which is what stops an untrusted agent pointing the
    miner's key at a destination of its own choosing."""
    plugin, calls = profile
    (tmp_path / "agent.py").write_text("print('{}')", encoding="utf-8")

    plugin.run(project_key=TASK, credential=credential, bundle_root=str(tmp_path),
               job_id=JOB_ID, bundle_sha256="c" * 64)

    from room.inference_network import inference_gateway_url

    assert _env_of(calls[0])["SN22_INFERENCE_GATEWAY"] == inference_gateway_url(JOB_ID, "openai")


def test_an_agent_can_import_the_relay_module_the_image_ships(profile):
    """Defect 3. The room mounts the bundle READ-ONLY, so nothing can be added at run time -- the
    agent image has to carry the module. This asserts the image is built to do that, and that the
    file it copies is the one the sandbox serves, so the two paths cannot drift."""
    dockerfile = (RUNNER.parent / "sn22-agent" / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY sn22_relay.py /opt/kata/sn22_relay.py" in dockerfile
    assert "PYTHONPATH=/opt/kata" in dockerfile

    build = (RUNNER.parent / "sn22-agent" / "build.sh").read_text(encoding="utf-8")
    assert "kata_sn22/relay_client.py" in build, (
        "the agent image must be built from the SAME client the sandbox copies in, or a submission "
        "calibrated in the sandbox behaves differently in the room")


# ---- no operator-funded fallback ---------------------------------------------------------------

def test_a_submission_with_no_sealed_credential_is_refused_before_execution(profile, tmp_path):
    """A keyless job can only burn room capacity and score zero, so reject it before Docker."""
    plugin, calls = profile
    (tmp_path / "agent.py").write_text("print('{}')", encoding="utf-8")

    with pytest.raises(RuntimeError, match="sealed inference credential"):
        plugin.run(project_key=TASK, credential=None, bundle_root=str(tmp_path),
                   job_id=JOB_ID, bundle_sha256="c" * 64)
    assert calls == []


# ---- the agent container's confinement ----------------------------------------------------------

def test_the_agent_runs_confined(profile, credential, tmp_path):
    """The agent is a stranger's code with a decrypted credential in its environment. Each is one
    thing it cannot do; asserted together because dropping any one silently is the failure."""
    plugin, calls = profile
    (tmp_path / "agent.py").write_text("print('{}')", encoding="utf-8")

    plugin.run(project_key=TASK, credential=credential, bundle_root=str(tmp_path),
               job_id=JOB_ID, bundle_sha256="c" * 64)

    argv = calls[0]
    assert "--read-only" in argv
    assert ["--cap-drop", "ALL"] == argv[argv.index("--cap-drop"):argv.index("--cap-drop") + 2]
    assert "no-new-privileges" in argv
    assert argv[argv.index("--user") + 1] == "65532:65532"
    assert argv[argv.index("--pids-limit") + 1] == "64"
    assert any(item.startswith("/tmp:rw,noexec,nosuid") for item in argv)
    assert "--memory" in argv and "--cpus" in argv
    # The sealed inference network, never the host's.
    from room.inference_network import INF_NET

    assert argv[argv.index("--network") + 1] == INF_NET
    # The bundle is READ-ONLY: a submission cannot rewrite its own agent between tasks.
    assert any("target=/bundle,readonly" in item for item in argv)


def test_the_agent_image_must_be_an_immutable_digest(profile, monkeypatch):
    """A tag is a pointer somebody else can move, and this image executes code from a stranger."""
    plugin, _calls = profile
    monkeypatch.setenv("KATA_SN22_TEE_AGENT_IMAGE", "docker.io/example/kata-sn22-agent:latest")
    with pytest.raises(RuntimeError, match="immutable digest"):
        plugin.agent_image()


def test_an_unset_agent_image_is_refused(profile, monkeypatch):
    plugin, _calls = profile
    monkeypatch.delenv("KATA_SN22_TEE_AGENT_IMAGE", raising=False)
    with pytest.raises(RuntimeError, match="unspecified image"):
        plugin.agent_image()


# ---- the room's own image identity --------------------------------------------------------------

def test_the_runner_image_loads_the_sn22_profile_and_nothing_else():
    """Separate images per subnet is what keeps the attested measurement meaningful: one image
    carrying both profiles would attest to code that can run the other subnet's containers, and the
    measurement would stop identifying which room a validator is talking to."""
    dockerfile = (RUNNER / "Dockerfile").read_text(encoding="utf-8")
    assert "ENV KATA_TEE_PROFILE=tee_profile:Sn22TeeProfile" in dockerfile
    assert "Sn60" not in dockerfile
    # Pinned to the base by digest, with a fail-closed placeholder so a careless direct build
    # cannot silently select a mutable one.
    assert "ARG BASE=example.invalid/kata-tee-runner@sha256:" in dockerfile


def test_the_build_scripts_refuse_a_mutable_base_and_a_foreign_platform():
    for script in (RUNNER / "build.sh", RUNNER.parent / "sn22-agent" / "build.sh"):
        body = script.read_text(encoding="utf-8")
        assert "@sha256:" in body and "must be an immutable image digest" in body, script.name
        assert "linux/amd64" in body, script.name
