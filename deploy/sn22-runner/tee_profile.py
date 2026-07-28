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

import json
import os
import subprocess
import sys
import tempfile

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
        """Run the miner's agent for one sealed SN22 task and return its answer plus provenance.

        ``project_key`` carries the lane's task descriptor (the JSON the agent receives on stdin),
        so the question comes from the validator and the answer's cost comes from the miner.
        """
        if project_key == self.fixture_project:
            return self._fixture_result(job_id=job_id, bundle_sha256=bundle_sha256)

        if not bundle_root:
            raise RuntimeError("no extracted submission bundle to run")
        if credential is None:
            raise RuntimeError(
                "SN22 TEE execution requires a miner-owned sealed inference credential")
        task = self._task_from(project_key)

        ensure_broker_network_once(self._broker)
        # The decrypted keys go into the BROKER, not into the agent. This is the whole of Phase C:
        # what the container receives is a capability, which is worth its remaining calls and
        # nothing else, and which is dead the moment close_job runs below.
        self._broker.open_job(job_id, dict(credential.credentials), contestant=bundle_sha256[:12])
        try:
            capability = self._broker.issue(job_id, role=ROLE_AGENT).token
            with tempfile.TemporaryDirectory() as workdir:
                os.chmod(workdir, 0o777)
                answer, stderr, timed_out, returncode, truncated = self._run_agent(
                    bundle_root=bundle_root, workdir=workdir, task=task,
                    broker=broker_url(), capability=capability)
        finally:
            # Always, including on an exception: a capability that outlived its job would let a
            # slow agent keep spending the miner's money after it had been scored.
            provider_records = self._broker.close_job(job_id)["records"]

        report = {
            "schema_version": 1,
            "task_id": task.get("task_id"),
            "answer": answer,
            "timed_out": timed_out,
            "returncode": returncode,
            "truncated": truncated,
            # Truncated and kept for the operator only. It is the miner's own process output and is
            # never scored -- a candidate that could influence scoring by what it printed to stderr
            # would be scoring itself.
            "stderr_tail": stderr[-4000:] if stderr else "",
        }
        provenance = {
            "profile": "sn22",
            "job_id": job_id,
            "bundle_sha256": bundle_sha256,
            "task_id": task.get("task_id"),
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
    def _task_from(project_key: str) -> dict:
        try:
            task = json.loads(project_key)
        except ValueError as exc:
            raise RuntimeError(f"SN22 project_key must be the task JSON: {exc}") from exc
        if not isinstance(task, dict) or not task.get("task_id"):
            raise RuntimeError("SN22 task JSON must be an object carrying a task_id")
        return task

    def _run_agent(self, *, bundle_root: str, workdir: str, task: dict,
                   broker: str, capability: str) -> tuple[str, str, bool, int, bool]:
        """Start the untrusted agent container and read its one JSON answer off stdout.

        Everything restrictive here is deliberate and mirrors SN60's profile: no network but the
        sealed inference network, a read-only bundle mount, dropped capabilities, no privilege
        escalation, and a hard wall-clock bound. The agent talks to the gateway and to nothing else.
        """
        image = self.agent_image()
        timeout = resolve_agent_execution_timeout_seconds()
        argv = [
            docker(), "run", "--rm", "-i",
            "--network", INF_NET,
            "--read-only",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--user", "65532:65532",
            "--memory", os.environ.get(AGENT_MEMORY_ENV, "").strip() or DEFAULT_AGENT_MEMORY,
            "--cpus", os.environ.get(AGENT_CPUS_ENV, "").strip() or DEFAULT_AGENT_CPUS,
            "--pids-limit", "64",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m,uid=65532,gid=65532,mode=700",
            "--mount", f"type=bind,source={bundle_root},target=/bundle,readonly",
            "--mount", f"type=bind,source={workdir},target=/work",
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
        with open(stdout_path, "wb") as stdout, open(stderr_path, "wb") as stderr:
            try:
                completed = subprocess.run(
                    argv,
                    input=json.dumps(task).encode("utf-8"),
                    stdout=stdout,
                    stderr=stderr,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return "", "agent exceeded its execution timeout", True, 124, False
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
            report={"schema_version": 1, "task_id": "fixture", "answer": "{}", "timed_out": False,
                    "returncode": 0, "truncated": False, "stderr_tail": ""},
            provenance={"profile": "sn22", "job_id": job_id, "bundle_sha256": bundle_sha256,
                        "fixture": True, "inference_funded_by": "miner"})


def main() -> int:  # pragma: no cover - convenience for `python -m`
    print("SN22 TEE profile; load via KATA_TEE_PROFILE=tee_profile:Sn22TeeProfile",
          file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
