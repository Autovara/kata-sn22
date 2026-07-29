"""SN22 (Desearch) TEE job profile: how to run one SN22 search task inside the sealed room.

Implements the generic ``room.profile.TeeJobProfile`` seam. Sealing, the miner-funded inference
gateway, the sealed network, attestation and HTTP are the generic room's responsibilities; this file
is the only SN22-specific part, and the room itself still names no subnet.

**What changes versus running SN22 locally.** Under the local sandbox the LANE holds provider
credentials and serves search from its own relay. In the room that inverts: the miner's credential
is sealed to its exact bundle, decrypted only inside the room, and the agent funds its own search
and inference through the in-room gateway. The validator handles ciphertext only and pays for
nothing. That is the same economics SN60 already runs on, which is why the room needed no change to
support it — a search provider is simply another allowlisted gateway route.

**What does NOT change.** The task is still the lane's: the query set is secret, identical for both
contestants, and hashed into the benchmark identity. The miner pays for *how well it answers*, never
for *what it is asked*. Keeping the question lane-owned is what stops a paired challenge becoming
two different challenges run at two different moments.
"""

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

from room.broker import ROLE_AGENT, ROLE_EVALUATOR, Broker, OperationSpec
from room.inference_network import (
    INF_NET,
    broker_url,
    docker,
    ensure_broker_network_once,
)
from room.profile import (
    MinerCredentialSet,
    TeeJobResult,
    resolve_agent_execution_timeout_seconds,
)

from kata_sn22.broker_ops import OPERATIONS as SN22_OPERATIONS
from kata_sn22.credentials_v2 import CREDENTIAL_PROFILE, REQUIRED_PROVIDERS

#: The agent image SN22 runs a submission in. An immutable digest, supplied by the deployer — never
#: a tag: a tag is a pointer someone else can move, and this one executes untrusted code.
AGENT_IMAGE_ENV = "KATA_SN22_TEE_AGENT_IMAGE"
#: Memory / CPU ceilings for the agent container. A search agent that allocates without bound would
#: take the room down for the contestant scheduled after it.
AGENT_MEMORY_ENV = "KATA_SN22_TEE_AGENT_MEMORY"
AGENT_CPUS_ENV = "KATA_SN22_TEE_AGENT_CPUS"
DEFAULT_AGENT_MEMORY = "2g"
DEFAULT_AGENT_CPUS = "2"
MAX_AGENT_OUTPUT_BYTES = 512 * 1024
MAX_AGENT_STDERR_BYTES = 16 * 1024

#: Agent containers running at once inside one pool job. Fixed rather than tuned: it bounds the
#: room's memory and CPU without changing task contents or score arithmetic, and a contestant whose
#: tasks happened to contend with each other would post a worse process_time for a reason unrelated
#: to its answers.
TASK_CONCURRENCY = 3


def _agent_container_name(job_id: str, task_index: int, task_id: object) -> str:
    identity = f"{job_id}\0{task_index}\0{task_id}".encode("utf-8")
    return "kata-sn22-" + hashlib.sha256(identity).hexdigest()[:20]


def _bundle_volume_name(container: str) -> str:
    return container + "-bundle"


def _docker_error(completed) -> str:
    stderr = completed.stderr or ""
    stdout = completed.stdout or ""
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="replace")
    return (stderr.strip() or stdout.strip() or f"docker exited {completed.returncode}")[:500]


def _remove_agent_container(container: str, *, allow_missing: bool) -> None:
    try:
        removed = docker(["rm", "--force", container], timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"could not remove SN22 agent container {container}: {exc}") from exc
    if removed.returncode == 0:
        return
    detail = _docker_error(removed)
    if allow_missing and "no such container" in detail.lower():
        return
    raise RuntimeError(f"could not remove SN22 agent container {container}: {detail}")


def _remove_bundle_volume(volume: str, *, allow_missing: bool) -> None:
    try:
        removed = docker(["volume", "rm", "--force", volume], timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"could not remove SN22 bundle volume {volume}: {exc}") from exc
    if removed.returncode == 0:
        return
    detail = _docker_error(removed)
    if allow_missing and "no such volume" in detail.lower():
        return
    raise RuntimeError(f"could not remove SN22 bundle volume {volume}: {detail}")


def _remove_agent_resources(container: str, staging: str, volume: str) -> None:
    errors = []
    for remove, name in (
        (_remove_agent_container, container),
        (_remove_agent_container, staging),
        (_remove_bundle_volume, volume),
    ):
        try:
            remove(name, allow_missing=True)
        except RuntimeError as exc:
            errors.append(str(exc))
    if errors:
        raise RuntimeError("; ".join(errors))


def build_broker() -> Broker:
    """The room's trusted broker, carrying SN22's six reviewed operations.

    The operation table lives in ``kata_sn22.broker_ops`` and is turned into the room's own
    ``OperationSpec`` here. That is the seam: the base image enforces capabilities, roles and quotas
    while knowing none of these provider names, and this file supplies them without reimplementing
    any of the enforcement.
    """
    return Broker([
        OperationSpec(
            name=name, role=role, provider=provider, handler=handler, max_calls=max_calls,
            # Evaluator operations are never on the network. The untrusted agent reaches the broker
            # only over HTTP, so an operation that is not exposed there is one it cannot invoke even
            # if it somehow obtained an evaluator token.
            http_exposed=(role == ROLE_AGENT),
        )
        for name, role, provider, handler, max_calls in SN22_OPERATIONS
    ])


class Sn22TeeProfile:
    """Runs one SN22 task: hand the agent its sealed task, let it search, collect its answer."""

    #: Selects the no-docker plumbing stub the generic room uses for local tests.
    fixture_project = "fixture-task"

    #: The Phase B credential contract. Four keys, all required, sealed as one set: an epoch covers
    #: four pools and a run that discovers the fourth key missing has already spent the miner's
    #: money on the first three.
    credential_version = 2
    required_providers = REQUIRED_PROVIDERS
    credential_profile = CREDENTIAL_PROFILE

    def __init__(self) -> None:
        self._broker = build_broker()

    @property
    def broker(self) -> Broker:
        """The trusted broker. The ONLY holder of a contestant's decrypted keys."""
        return self._broker

    def evaluator_capability(self, job_id: str) -> str:
        """Mint the trusted evaluator's capability for a job.

        Separate from the agent's, with its own quota, so an agent that burned its whole allowance
        cannot starve the verification that is about to check its work.
        """
        return self._broker.issue(job_id, role=ROLE_EVALUATOR).token

    # ---- deployment inputs -------------------------------------------------------------------
    def agent_image(self) -> str:
        digest = os.environ.get(AGENT_IMAGE_ENV, "").strip()
        if not digest:
            raise RuntimeError(
                f"{AGENT_IMAGE_ENV} is not set; the room will not run an untrusted agent in an "
                f"unspecified image")
        if "@sha256:" not in digest:
            raise RuntimeError(
                f"{AGENT_IMAGE_ENV} must pin an immutable digest (image@sha256:...), got "
                f"{digest!r}. A tag is a pointer somebody else can move, and this image executes "
                f"code from a stranger")
        return digest

    # ---- the seam ----------------------------------------------------------------------------
    def run(self, *, project_key: str, credential: MinerCredentialSet | None,
            bundle_root: str | None, job_id: str, bundle_sha256: str) -> TeeJobResult:
        """Run one contestant's POOL: fifteen tasks, then score them with the real upstream.

        ``project_key`` carries the pool job -- the pool name and its fifteen task descriptors --
        so the questions come from the validator and the answers' cost comes from the miner.

        **One pool per request, not a whole epoch.** Sixty tasks behind one HTTP call is one
        timeout away from losing every answer already paid for; four bounded jobs per contestant
        lose at most a quarter, and each one is separately attested.

        A single task descriptor is still accepted, and is treated as a pool of one. That keeps the
        earlier single-task callers working while the epoch path is wired up.
        """
        if project_key == self.fixture_project:
            return self._fixture_result(job_id=job_id, bundle_sha256=bundle_sha256)

        if not bundle_root:
            raise RuntimeError("no extracted submission bundle to run")
        if credential is None:
            raise RuntimeError(
                "SN22 TEE execution requires a miner-owned sealed inference credential")
        pool, tasks = self._pool_from(project_key)

        ensure_broker_network_once(self._broker)
        # The decrypted keys go into the BROKER, not into the agent. This is the whole of Phase C:
        # what the container receives is a capability, which is worth its remaining calls and
        # nothing else, and which is dead the moment close_job runs below.
        self._broker.open_job(job_id, dict(credential.credentials), contestant=bundle_sha256[:12])
        try:
            capability = self._broker.issue(job_id, role=ROLE_AGENT).token
            attempts = self._run_pool(
                bundle_root=bundle_root,
                tasks=tasks,
                broker=broker_url(),
                capability=capability,
                job_id=job_id,
            )
            # SCORING happens here, inside the room, while the evaluator capability is still live.
            # It has to: the pool tuple is what the attestation binds, and a tuple computed on the
            # host afterwards would be a number the quote does not cover.
            pool_result, credentials = self._score(
                pool=pool, tasks=tasks, attempts=attempts, job_id=job_id)
        finally:
            # Always, including on an exception: a capability that outlived its job would let a
            # slow agent keep spending the miner's money after it had been scored.
            provider_records = self._broker.close_job(job_id)["records"]

        report = {
            "schema_version": 1,
            "pool": pool,
            "tasks": [
                {
                    "task_id": attempt["task_id"],
                    "answer": attempt["answer"],
                    "timed_out": attempt["timed_out"],
                    "returncode": attempt["returncode"],
                    "truncated": attempt["truncated"],
                    "process_time": attempt["process_time"],
                }
                for attempt in attempts
            ],
            # Truncated and kept for the operator only. It is the miner's own process output and is
            # never scored -- a candidate that could influence scoring by what it printed to stderr
            # would be scoring itself.
            "stderr_tail": "\n".join(
                attempt["stderr"][-2000:] for attempt in attempts if attempt["stderr"])[-4000:],
            # Upstream's own four numbers for this contestant in this pool. Bound into the quote,
            # so the host verifies a score rather than recomputing one it would have to trust.
            "pool_result": pool_result,
            "credential_status": credentials,
        }
        provenance = {
            "profile": "sn22",
            "job_id": job_id,
            "bundle_sha256": bundle_sha256,
            "pool": pool,
            "task_ids": [attempt["task_id"] for attempt in attempts],
            "task_concurrency": TASK_CONCURRENCY,
            # The MINER funded this run. Recorded in the attestation so a later cost review can see
            # that no validator credential was in play, which is the whole point of the room.
            "inference_funded_by": "miner",
            "agent_image": self.agent_image(),
            # Provider, phase, task and status per call -- the trusted broker's own record, not the
            # agent's self-report. It rides in the provenance, so it is bound into the quote.
            "provider_calls": provider_records,
        }
        return TeeJobResult(report=report, provenance=provenance)

    # ---- internals ---------------------------------------------------------------------------
    @staticmethod
    def _pool_from(project_key: str) -> tuple:
        """``(pool_name, tasks)`` from the pool job, or from a single task descriptor.

        Both shapes are accepted on purpose. A bare task is a pool of one, which is what the
        pre-epoch callers send; refusing it would break them for no gain while the epoch path is
        still being wired.
        """
        try:
            document = json.loads(project_key)
        except ValueError as exc:
            raise RuntimeError(f"SN22 project_key must be the pool job JSON: {exc}") from exc
        if not isinstance(document, dict):
            raise RuntimeError("SN22 pool job must be a JSON object")

        if document.get("task_id"):
            return "", [document]

        pool = str(document.get("pool") or "")
        tasks = document.get("tasks")
        if not pool or not isinstance(tasks, list) or not tasks:
            raise RuntimeError(
                "SN22 pool job must carry a pool name and a non-empty tasks list")
        for task in tasks:
            if not isinstance(task, dict) or not task.get("task_id"):
                raise RuntimeError("every SN22 task must be an object carrying a task_id")
        return pool, tasks

    def _score(self, *, pool: str, tasks: list, attempts: list, job_id: str) -> tuple:
        """Score this contestant's pool with the REAL upstream validator, in the room.

        The evaluator's three provider operations are invoked in-process with an evaluator
        capability -- never over HTTP, and never reachable from the agent's. Verification is paid
        for by the contestant's own credential, which is the funding rule: a contestant that cannot
        fund the check of its own answers scores zero rather than going unchecked.

        A pool with no name is a pre-epoch single task; it is run but not scored, because a pool
        tuple for a pool of one would be a number nobody should aggregate.
        """
        if not pool:
            return None, {}

        from kata_sn22 import production_scorer
        from kata_sn22.epoch_manifest import build_task_from_input

        capability = self.evaluator_capability(job_id)
        broker = self._broker

        async def _judge(messages):
            answer = broker.dispatch(capability, "chutes-score", {"messages": list(messages)})
            return answer.get("content", "")

        async def _fetch_pages(urls):
            answer = broker.dispatch(capability, "web-page-fetch", {"urls": list(urls)})
            return answer.get("pages", {})

        async def _rescrape(tweet_ids):
            answer = broker.dispatch(
                capability, "tweet-rescrape", {"tweet_ids": [str(i) for i in tweet_ids]})
            return answer.get("tweets", [])

        typed_tasks = tuple(build_task_from_input(task) for task in tasks)
        answers, process_times = {}, {}
        for attempt in attempts:
            answers[attempt["task_id"]] = _parse_answer(attempt["answer"])
            process_times[(0, attempt["task_id"])] = attempt["process_time"]

        score = asyncio.run(production_scorer.score_pool(
            pool=pool, tasks=typed_tasks, king_answers=answers, challenger_answers={},
            deep_task_ids=frozenset(
                task["task_id"] for task in tasks if task.get("deep")),
            judge=_judge, fetch_pages=_fetch_pages, rescrape_tweets=_rescrape,
            process_times=process_times))
        return score.king.as_dict(), score.credentials.as_dict()

    def _run_pool(self, *, bundle_root: str, tasks: list, broker: str,
                  capability: str, job_id: str) -> list:
        """Run every task in the pool, at most ``TASK_CONCURRENCY`` at a time.

        Bounded rather than unbounded: fifteen agent containers at once would contend for the
        room's memory and CPU, and a contestant that happened to run while another pool was busy
        would post a worse ``process_time`` for a reason that has nothing to do with its answer --
        and ``process_time`` is what the performance reward and the timeout penalty are measured
        against.

        Results are returned in TASK ORDER regardless of completion order, so two contestants'
        reports line up task for task.
        """
        import concurrent.futures

        attempts: list = [None] * len(tasks)

        def _one(index_and_task):
            index, task = index_and_task
            with tempfile.TemporaryDirectory() as workdir:
                started = time.monotonic()
                answer, stderr, timed_out, returncode, truncated = self._run_agent(
                    bundle_root=bundle_root,
                    workdir=workdir,
                    task=task,
                    broker=broker,
                    capability=capability,
                    job_id=job_id,
                    task_index=index,
                )
            return index, {
                "task_id": task.get("task_id"),
                "answer": answer,
                "stderr": stderr,
                "timed_out": timed_out,
                "returncode": returncode,
                "truncated": truncated,
                # What the room measured, not what the agent claimed. Upstream's performance reward
                # and timeout penalty are both computed from it.
                "process_time": round(time.monotonic() - started, 3),
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=TASK_CONCURRENCY) as pool_executor:
            for index, attempt in pool_executor.map(_one, list(enumerate(tasks))):
                attempts[index] = attempt
        return attempts

    def _run_agent(self, *, bundle_root: str, workdir: str, task: dict,
                   broker: str, capability: str, job_id: str,
                   task_index: int) -> tuple[str, str, bool, int, bool]:
        """Start the untrusted agent container and read its one JSON answer off stdout.

        Everything restrictive here is deliberate and mirrors SN60's profile: no network but the
        sealed inference network, a bundle copied into a read-only root filesystem, dropped
        capabilities, no privilege escalation, and a hard wall-clock bound. The agent talks to the
        gateway and to nothing else.

        ``docker cp`` into a daemon-managed volume is essential here. The Docker CLI runs inside the
        room container but talks to the host daemon through its socket; that daemon cannot resolve a
        bind source created in the room container's filesystem. The volume is populated through a
        stopped staging container, then mounted read-only into the read-only final container. A
        named final container also gives the timeout path something exact to kill instead of leaving
        a daemon-owned process running after the CLI is terminated.
        """
        image = self.agent_image()
        timeout = resolve_agent_execution_timeout_seconds()
        container = _agent_container_name(job_id, task_index, task.get("task_id"))
        staging = container + "-stage"
        volume = _bundle_volume_name(container)
        create_args = [
            "create",
            "--name", container,
            "--interactive",
            "--network", INF_NET,
            "--read-only",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--user", "65532:65532",
            "--memory", os.environ.get(AGENT_MEMORY_ENV, "").strip() or DEFAULT_AGENT_MEMORY,
            "--cpus", os.environ.get(AGENT_CPUS_ENV, "").strip() or DEFAULT_AGENT_CPUS,
            "--pids-limit", "64",
            "--log-driver", "none",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m,uid=65532,gid=65532,mode=700",
            "--tmpfs", "/work:rw,noexec,nosuid,size=16m,uid=65532,gid=65532,mode=700",
            "--mount", f"type=volume,source={volume},target=/bundle,readonly",
            "--workdir", "/work",
            "--env", f"SN22_BROKER_URL={broker}",
            # A CAPABILITY, not a credential. Everything an agent can do with what it is given here
            # is something the broker was going to allow anyway; there is no key in this container's
            # environment, argv, /proc or filesystem for it to find.
            "--env", f"SN22_BROKER_CAPABILITY={capability}",
            "--env", "SN22_PROTOCOL_VERSION=2",
            "--env", f"SN22_TASK_ID={task.get('task_id')}",
            image,
            # The BUNDLE PATH, appended to the image's own entry point -- the harness. The
            # submission is imported by reviewed code rather than executed as a program, so every
            # contestant's answer is framed by the same function. Running `python /bundle/agent.py`
            # here instead would put the framing back in each miner's hands.
            "/bundle/agent.py",
        ]
        stdout_path = os.path.join(workdir, "stdout")
        stderr_path = os.path.join(workdir, "stderr")
        _remove_agent_resources(container, staging, volume)
        try:
            created_volume = docker(
                ["volume", "create", "--label", "kata.agent=sn22", volume],
                timeout=30,
            )
            if created_volume.returncode != 0:
                raise RuntimeError(
                    "could not create the SN22 bundle volume: "
                    + _docker_error(created_volume)
                )

            created_staging = docker(
                [
                    "create",
                    "--name", staging,
                    "--network", "none",
                    "--mount", f"type=volume,source={volume},target=/bundle",
                    "--entrypoint", "python",
                    image,
                    "-c", "pass",
                ],
                timeout=120,
            )
            if created_staging.returncode != 0:
                raise RuntimeError(
                    "could not create the SN22 bundle staging container: "
                    + _docker_error(created_staging)
                )

            copied = docker(
                ["cp", os.path.join(bundle_root, "."), f"{staging}:/bundle"],
                timeout=120,
            )
            if copied.returncode != 0:
                raise RuntimeError(
                    "could not copy the SN22 submission into its bundle volume: "
                    + _docker_error(copied)
                )
            _remove_agent_container(staging, allow_missing=False)

            created = docker(create_args, timeout=120)
            if created.returncode != 0:
                raise RuntimeError(
                    "could not create the SN22 agent container: "
                    + _docker_error(created)
                )

            with open(stdout_path, "wb") as stdout, open(stderr_path, "wb") as stderr:
                try:
                    completed = docker(
                        ["start", "--attach", "--interactive", container],
                        stdin=json.dumps(task),
                        stdout=stdout,
                        stderr=stderr,
                        timeout=timeout,
                    )
                except subprocess.TimeoutExpired:
                    return "", "agent exceeded its execution timeout", True, 124, False
        finally:
            # This runs after success, agent failure, Docker failure, and TimeoutExpired. A cleanup
            # failure is an infrastructure fault, not something to hide behind a contestant score.
            _remove_agent_resources(container, staging, volume)
        with open(stdout_path, "rb") as stdout:
            raw_answer = stdout.read(MAX_AGENT_OUTPUT_BYTES + 1)
        with open(stderr_path, "rb") as stderr:
            raw_stderr = stderr.read(MAX_AGENT_STDERR_BYTES + 1)
        truncated = len(raw_answer) > MAX_AGENT_OUTPUT_BYTES
        answer = "" if truncated else raw_answer.decode("utf-8", errors="replace")
        return (
            answer,
            raw_stderr[:MAX_AGENT_STDERR_BYTES].decode("utf-8", errors="replace"),
            False,
            completed.returncode,
            truncated,
        )

    @staticmethod
    def _fixture_result(*, job_id: str, bundle_sha256: str) -> TeeJobResult:
        """The no-docker plumbing stub: proves the room wiring without running an agent."""
        return TeeJobResult(
            report={"schema_version": 1, "pool": "fixture", "tasks": [], "stderr_tail": ""},
            provenance={"profile": "sn22", "job_id": job_id, "bundle_sha256": bundle_sha256,
                        "fixture": True, "inference_funded_by": "miner"})


def _parse_answer(raw: str) -> dict:
    """The agent's answer document, or an empty one.

    An unparseable answer is not an error here: it is a contestant that produced nothing usable,
    which upstream's own penalties are what score. Raising would turn a bad agent into a broken
    room.
    """
    try:
        document = json.loads(raw) if raw else {}
    except ValueError:
        return {}
    return document if isinstance(document, dict) else {}


def main() -> int:  # pragma: no cover - convenience for `python -m`
    print("SN22 TEE profile; load via KATA_TEE_PROFILE=tee_profile:Sn22TeeProfile",
          file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
