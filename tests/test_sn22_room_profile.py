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
    answer = json.dumps(
        {
            "protocol_version": 1,
            "task_id": "t000",
            "summary": "an answer",
            "results": [],
            "tweets": [],
            "citations": [],
            "usage": {"provider_calls": 1, "tokens": 10},
        }
    ).encode()

    class _Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_docker(args, **kwargs):
        calls.append(["docker", *args])
        if args[:3] == ["start", "--attach", "--interactive"]:
            kwargs["stdout"].write(answer)
        return _Completed()

    monkeypatch.setattr(tee_profile, "docker", fake_docker)
    monkeypatch.setattr(tee_profile, "ensure_broker_network_once", lambda _broker: None)
    return tee_profile.Sn22TeeProfile(), calls


#: The one string that must never appear anywhere the agent can reach.
MINER_KEY = "sk-miner-secret-key-0123456789"


@pytest.fixture
def credential():
    from room.profile import MinerCredentialSet

    from kata_sn22.credentials_v2 import CREDENTIAL_PROFILE, REQUIRED_PROVIDERS

    return MinerCredentialSet(
        credentials={name: f"{MINER_KEY}-{name}" for name in REQUIRED_PROVIDERS},
        bundle_binding="b" * 64,
        credential_profile=CREDENTIAL_PROFILE,
    )


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


def _agent_create_calls(calls: list[list[str]]) -> list[list[str]]:
    return [
        argv
        for argv in calls
        if argv[:2] == ["docker", "create"] and "--read-only" in argv
    ]


# ---- the three defects -------------------------------------------------------------------------

def test_a_real_task_reaches_the_agent_at_all(profile, credential, tmp_path):
    """Defect 1. This is the whole test: `run()` with a real task used to raise TypeError before it
    got anywhere near starting an agent."""
    plugin, calls = profile
    (tmp_path / "agent.py").write_text("print('{}')", encoding="utf-8")

    result = plugin.run(project_key=TASK, credential=credential, bundle_root=str(tmp_path),
                        job_id=JOB_ID, bundle_sha256="c" * 64)

    assert calls, "the profile never started an agent container"
    # Since Phase F a job is a POOL. A bare task descriptor is still accepted as a pool of one.
    assert [task["task_id"] for task in result.report["tasks"]] == ["t000"]


def test_no_provider_key_reaches_the_agent_container(profile, credential, tmp_path):
    """The Phase C exit gate, at the exact boundary where it used to fail.

    This test previously asserted the OPPOSITE -- that ``SN22_INFERENCE_API_KEY`` carried the
    miner's decrypted key into the container -- and it passed, because that is what the profile did.
    That put a real credential in the environment of code written by a stranger, where
    ``os.environ``, ``/proc/self/environ``, a crash dump or a stray ``print`` would have sufficed.

    The whole docker argv is searched, not just the ``--env`` pairs: a key smuggled into a mount
    path, a label or an image reference would be just as readable from inside.
    """
    plugin, calls = profile
    (tmp_path / "agent.py").write_text("print('{}')", encoding="utf-8")

    plugin.run(project_key=TASK, credential=credential, bundle_root=str(tmp_path),
               job_id=JOB_ID, bundle_sha256="c" * 64)

    argv = " ".join(_agent_create_calls(calls)[0])
    assert MINER_KEY not in argv
    for provider, key in credential.credentials.items():
        assert key not in argv, f"the {provider} key reached the agent container"
    assert "SN22_INFERENCE_API_KEY" not in argv


def test_the_agent_gets_a_capability_and_a_broker_url(profile, credential, tmp_path):
    """What replaces the key. A capability is worth its remaining calls and nothing else, and it is
    dead the moment the job closes."""
    from room.broker import CAPABILITY_RE

    plugin, calls = profile
    (tmp_path / "agent.py").write_text("print('{}')", encoding="utf-8")

    plugin.run(project_key=TASK, credential=credential, bundle_root=str(tmp_path),
               job_id=JOB_ID, bundle_sha256="c" * 64)

    env = _env_of(_agent_create_calls(calls)[0])
    assert env["SN22_BROKER_URL"].startswith("http://")
    assert CAPABILITY_RE.fullmatch(env["SN22_BROKER_CAPABILITY"])


def test_the_capability_is_dead_once_the_job_ends(profile, credential, tmp_path):
    """A capability that outlived its job would let a slow agent keep spending the miner's money
    after it had already been scored."""
    from room.broker import BrokerDenied

    plugin, calls = profile
    (tmp_path / "agent.py").write_text("print('{}')", encoding="utf-8")

    plugin.run(project_key=TASK, credential=credential, bundle_root=str(tmp_path),
               job_id=JOB_ID, bundle_sha256="c" * 64)

    token = _env_of(_agent_create_calls(calls)[0])["SN22_BROKER_CAPABILITY"]
    with pytest.raises(BrokerDenied):
        plugin.broker.dispatch(token, "web-search", {"query": "anything"}, over_http=True)


def test_the_broker_records_every_provider_call_in_the_attested_provenance(
    profile, credential, tmp_path
):
    """The broker's own record, not the agent's self-report -- and it rides in the provenance, so
    it is bound into the quote."""
    plugin, _calls = profile
    (tmp_path / "agent.py").write_text("print('{}')", encoding="utf-8")

    result = plugin.run(project_key=TASK, credential=credential, bundle_root=str(tmp_path),
                        job_id=JOB_ID, bundle_sha256="c" * 64)

    assert "provider_calls" in result.provenance
    assert MINER_KEY not in json.dumps(result.provenance)


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

    argv = _agent_create_calls(calls)[0]
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
    # No daemon-host bind path exists. A daemon-managed volume is mounted read-only, and the root
    # filesystem is read-only too.
    assert not any("type=bind" in item for item in argv)
    bundle_mount = argv[argv.index("--mount") + 1]
    assert bundle_mount.startswith("type=volume,source=kata-sn22-")
    assert bundle_mount.endswith(",target=/bundle,readonly")
    assert any(item.startswith("/work:rw,noexec,nosuid") for item in argv)
    assert ["--log-driver", "none"] == (
        argv[argv.index("--log-driver"):argv.index("--log-driver") + 2]
    )


def test_the_bundle_is_copied_into_a_named_container_before_start(
    profile, credential, tmp_path
):
    plugin, calls = profile
    (tmp_path / "agent.py").write_text("print('{}')", encoding="utf-8")

    plugin.run(
        project_key=TASK,
        credential=credential,
        bundle_root=str(tmp_path),
        job_id=JOB_ID,
        bundle_sha256="c" * 64,
    )

    assert [argv[1] for argv in calls] == [
        "rm",
        "rm",
        "volume",
        "volume",
        "create",
        "cp",
        "rm",
        "create",
        "start",
        "rm",
        "rm",
        "volume",
    ]
    create = _agent_create_calls(calls)[0]
    container = create[create.index("--name") + 1]
    staging = container + "-stage"
    volume = container + "-bundle"
    assert container.startswith("kata-sn22-")
    assert calls[5] == ["docker", "cp", f"{tmp_path}/.", f"{staging}:/bundle"]
    assert calls[8] == ["docker", "start", "--attach", "--interactive", container]
    assert ["docker", "rm", "--force", container] in calls
    assert ["docker", "rm", "--force", staging] in calls
    assert ["docker", "volume", "rm", "--force", volume] in calls


def test_a_timed_out_agent_is_force_removed(profile, tmp_path, monkeypatch):
    import tee_profile

    plugin, calls = profile
    workdir = tmp_path / "work"
    workdir.mkdir()
    (tmp_path / "agent.py").write_text("print('{}')", encoding="utf-8")

    class _Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def timeout_on_start(args, **kwargs):
        calls.append(["docker", *args])
        if args[:3] == ["start", "--attach", "--interactive"]:
            raise tee_profile.subprocess.TimeoutExpired(args, kwargs["timeout"])
        return _Completed()

    monkeypatch.setattr(tee_profile, "docker", timeout_on_start)
    result = plugin._run_agent(
        bundle_root=str(tmp_path),
        workdir=str(workdir),
        task=json.loads(TASK),
        broker="http://broker:8100",
        capability="kcap_" + "a" * 32,
        job_id=JOB_ID,
        task_index=0,
    )

    assert result[2:4] == (True, 124)
    create = _agent_create_calls(calls)[0]
    container = create[create.index("--name") + 1]
    assert ["docker", "rm", "--force", container] in calls[calls.index(create):]
    assert calls[-1] == ["docker", "volume", "rm", "--force", container + "-bundle"]


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


# ---- per-pool jobs ---
#
# Sixty tasks behind one HTTP request is one timeout away from losing every answer already paid
# for. Four bounded pool jobs per contestant lose at most a quarter, and each is separately
# attested.

POOL_JOB = json.dumps({
    "pool": "ai_search:fast",
    "tasks": [
        {"protocol_version": 2, "task_id": f"t{index:03d}", "search_type": "ai_search",
         "prompt": "what were 2024 emissions?", "mode": "fast",
         "result_type": "LINKS_WITH_FINAL_SUMMARY", "tools": ["Web Search"], "count": 10,
         "limits": {"max_execution_time": 15}}
        for index in range(15)
    ],
})


@pytest.fixture
def upstream_scoring(monkeypatch):
    """Replace the one upstream-dependent call these orchestration tests reach.

    ``plugin.run`` scores the pool inside the room, and ``production_scorer.score_pool`` loads the
    pinned upstream, which needs the real pydantic, numpy, pytz and tiktoken. Those live in the
    ``upstream`` extra; CI installs ``dev``. So these five failed on every CI run with
    ``UpstreamUnavailable`` while passing locally -- a developer venv that had once been synced with
    ``--extra parity`` keeps those packages, and nothing re-checks that they are still declared.

    Skipping them was the obvious fix and the wrong one. They pin room orchestration -- that every
    task in the pool runs, that results come back in task order, that agent concurrency is bounded,
    that the room measures ``process_time`` itself, and that one capability covers the whole pool.
    None of that involves upstream, and the concurrency bound in particular is a property worth
    checking on every push. Neither CI job could run them either way: the ``parity`` extra omits
    tiktoken, so an upstream skipif would have skipped them everywhere.

    So the stub is deliberately narrow. Everything else in ``_score`` still runs for real: the typed
    tasks are built by ``build_task_from_input``, the evaluator capability is minted, and the judge,
    fetch and rescrape closures are wired to the live broker. Only the scoring call itself is
    replaced. Scoring has its own coverage in ``test_sn22_production_scorer.py``.

    Yields the recorded calls so a test can assert scoring happened, and happened in the room.
    """
    from kata_sn22 import production_scorer

    calls = []

    class _Dict:
        def __init__(self, payload):
            self._payload = payload

        def as_dict(self):
            return dict(self._payload)

    class _Score:
        king = _Dict({"combined_score": 0.5})
        credentials = _Dict({"scrapingdog": "ok"})

    async def _score_pool(**kwargs):
        calls.append(kwargs)
        return _Score()

    monkeypatch.setattr(production_scorer, "score_pool", _score_pool)
    return calls


def test_a_pool_job_runs_every_task(profile, credential, tmp_path, upstream_scoring):
    plugin, calls = profile
    (tmp_path / "agent.py").write_text("print('{}')", encoding="utf-8")

    result = plugin.run(project_key=POOL_JOB, credential=credential,
                        bundle_root=str(tmp_path), job_id=JOB_ID, bundle_sha256="c" * 64)

    assert len(_agent_create_calls(calls)) == 15, "the pool did not run every task"
    assert result.report["pool"] == "ai_search:fast"
    assert [task["task_id"] for task in result.report["tasks"]] == [
        f"t{index:03d}" for index in range(15)]


def test_pool_results_come_back_in_task_order_not_completion_order(
        profile, credential, tmp_path, upstream_scoring):
    """Two contestants' reports have to line up task for task, and a thread pool does not promise
    completion order."""
    import random
    import time as _time

    plugin, _calls = profile
    (tmp_path / "agent.py").write_text("print('{}')", encoding="utf-8")

    original = plugin._run_agent

    def _jittered(**kwargs):
        _time.sleep(random.uniform(0, 0.01))
        return original(**kwargs)

    plugin._run_agent = _jittered
    result = plugin.run(project_key=POOL_JOB, credential=credential,
                        bundle_root=str(tmp_path), job_id=JOB_ID, bundle_sha256="c" * 64)

    assert [task["task_id"] for task in result.report["tasks"]] == [
        f"t{index:03d}" for index in range(15)]


def test_task_concurrency_is_bounded(profile, credential, tmp_path, upstream_scoring):
    """Fifteen agent containers at once would contend for the room's memory and CPU, and a
    contestant whose tasks contended with each other would post a worse process_time for a reason
    unrelated to its answers."""
    import threading

    import tee_profile as profile_module

    plugin, _calls = profile
    (tmp_path / "agent.py").write_text("print('{}')", encoding="utf-8")

    live = 0
    peak = 0
    lock = threading.Lock()
    original = plugin._run_agent

    def _counted(**kwargs):
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        try:
            return original(**kwargs)
        finally:
            with lock:
                live -= 1

    plugin._run_agent = _counted
    plugin.run(project_key=POOL_JOB, credential=credential, bundle_root=str(tmp_path),
               job_id=JOB_ID, bundle_sha256="c" * 64)

    assert peak <= profile_module.TASK_CONCURRENCY, f"{peak} agents ran at once"


def test_the_room_measures_each_task_rather_than_trusting_the_agent(
        profile, credential, tmp_path, upstream_scoring):
    """``process_time`` drives upstream's performance reward and timeout penalty. An agent that
    reported its own would be grading its own speed."""
    plugin, _calls = profile
    (tmp_path / "agent.py").write_text("print('{}')", encoding="utf-8")

    result = plugin.run(project_key=POOL_JOB, credential=credential,
                        bundle_root=str(tmp_path), job_id=JOB_ID, bundle_sha256="c" * 64)

    for task in result.report["tasks"]:
        assert isinstance(task["process_time"], float)
        assert task["process_time"] >= 0.0


def test_one_capability_covers_the_whole_pool_and_dies_with_it(
        profile, credential, tmp_path, upstream_scoring):
    """Minting one per task would give a contestant fifteen times the allowance."""
    from room.broker import BrokerDenied

    plugin, calls = profile
    (tmp_path / "agent.py").write_text("print('{}')", encoding="utf-8")

    plugin.run(project_key=POOL_JOB, credential=credential, bundle_root=str(tmp_path),
               job_id=JOB_ID, bundle_sha256="c" * 64)

    tokens = {
        _env_of(argv)["SN22_BROKER_CAPABILITY"] for argv in _agent_create_calls(calls)
    }
    assert len(tokens) == 1, "a capability was minted per task"
    with pytest.raises(BrokerDenied):
        plugin.broker.dispatch(tokens.pop(), "web-search", {"query": "x"}, over_http=True)


@pytest.mark.parametrize("bad", [
    '{"pool": "ai_search:fast"}',
    '{"pool": "ai_search:fast", "tasks": []}',
    '{"tasks": [{"task_id": "t0"}]}',
    '{"pool": "ai_search:fast", "tasks": [{"prompt": "no task_id"}]}',
    "not json",
    '["a list"]',
])
def test_a_malformed_pool_job_is_refused(profile, credential, tmp_path, bad):
    """Refused before an agent starts, so a malformed job costs the contestant nothing."""
    plugin, calls = profile
    (tmp_path / "agent.py").write_text("print('{}')", encoding="utf-8")

    with pytest.raises(RuntimeError):
        plugin.run(project_key=bad, credential=credential, bundle_root=str(tmp_path),
                   job_id=JOB_ID, bundle_sha256="c" * 64)
    assert not calls, "an agent was started for a malformed pool job"


def test_the_pool_is_scored_inside_the_room_while_the_capability_lives(
        profile, credential, tmp_path, upstream_scoring):
    """``run`` scores between ``open_job`` and ``close_job`` deliberately: the attestation binds the
    pool tuple, so a tuple computed on the host afterwards would be a number the quote does not
    cover. Nothing asserted that ordering until the scoring call was made observable.
    """
    plugin, _calls = profile
    (tmp_path / "agent.py").write_text("print('{}')", encoding="utf-8")

    closed = []
    original_close = plugin._broker.close_job

    def _recording_close(job_id):
        closed.append(len(upstream_scoring))
        return original_close(job_id)

    plugin._broker.close_job = _recording_close
    plugin.run(project_key=POOL_JOB, credential=credential, bundle_root=str(tmp_path),
               job_id=JOB_ID, bundle_sha256="c" * 64)

    assert len(upstream_scoring) == 1, "the pool was not scored"
    assert closed == [1], "the job was closed before the pool was scored"
    # And it scored THIS pool, not an empty one.
    assert upstream_scoring[0]["pool"] == "ai_search:fast"
    assert len(upstream_scoring[0]["tasks"]) == 15
