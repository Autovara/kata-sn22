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
from pathlib import Path

from room.inference_network import (
    INF_NET,
    docker,
    ensure_inference_network_once,
    inference_gateway_url,
    start_inference_gateway_once,
)
from room.profile import (
    MinerInferenceCredential,
    TeeJobResult,
    resolve_agent_execution_timeout_seconds,
)

#: The agent image SN22 runs a submission in. An immutable digest, supplied by the deployer — never
#: a tag: a tag is a pointer someone else can move, and this one executes untrusted code.
AGENT_IMAGE_ENV = "KATA_SN22_TEE_AGENT_IMAGE"
#: Memory / CPU ceilings for the agent container. A search agent that allocates without bound would
#: take the room down for the contestant scheduled after it.
AGENT_MEMORY_ENV = "KATA_SN22_TEE_AGENT_MEMORY"
AGENT_CPUS_ENV = "KATA_SN22_TEE_AGENT_CPUS"
DEFAULT_AGENT_MEMORY = "2g"
DEFAULT_AGENT_CPUS = "2"


class Sn22TeeProfile:
    """Runs one SN22 task: hand the agent its sealed task, let it search, collect its answer."""

    #: Selects the no-docker plumbing stub the generic room uses for local tests.
    fixture_project = "fixture-task"

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
    def run(self, *, project_key: str, credential: MinerInferenceCredential | None,
            bundle_root: str | None, job_id: str, bundle_sha256: str) -> TeeJobResult:
        """Run the miner's agent for one sealed SN22 task and return its answer plus provenance.

        ``project_key`` carries the lane's task descriptor (the JSON the agent receives on stdin),
        so the question comes from the validator and the answer's cost comes from the miner.
        """
        if project_key == self.fixture_project:
            return self._fixture_result(job_id=job_id, bundle_sha256=bundle_sha256)

        if not bundle_root:
            raise RuntimeError("no extracted submission bundle to run")
        task = self._task_from(project_key)

        ensure_inference_network_once()
        start_inference_gateway_once()
        gateway = inference_gateway_url(job_id=job_id, credential=credential)

        with tempfile.TemporaryDirectory() as workdir:
            answer, stderr, timed_out = self._run_agent(
                bundle_root=bundle_root, workdir=workdir, task=task, gateway=gateway)

        report = {
            "schema_version": 1,
            "task_id": task.get("task_id"),
            "answer": answer,
            "timed_out": timed_out,
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
                   gateway: str) -> tuple[str, str, bool]:
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
            "--memory", os.environ.get(AGENT_MEMORY_ENV, "").strip() or DEFAULT_AGENT_MEMORY,
            "--cpus", os.environ.get(AGENT_CPUS_ENV, "").strip() or DEFAULT_AGENT_CPUS,
            "--mount", f"type=bind,source={bundle_root},target=/bundle,readonly",
            "--mount", f"type=bind,source={workdir},target=/work",
            "--workdir", "/work",
            "--env", f"SN22_INFERENCE_GATEWAY={gateway}",
            "--env", "SN22_PROTOCOL_VERSION=1",
            "--env", f"SN22_TASK_ID={task.get('task_id')}",
            image,
            "python", "/bundle/agent.py",
        ]
        try:
            completed = subprocess.run(
                argv, input=json.dumps(task).encode("utf-8"), capture_output=True,
                timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return "", "agent exceeded its execution timeout", True
        return (completed.stdout.decode("utf-8", errors="replace"),
                completed.stderr.decode("utf-8", errors="replace"),
                False)

    @staticmethod
    def _fixture_result(*, job_id: str, bundle_sha256: str) -> TeeJobResult:
        """The no-docker plumbing stub: proves the room wiring without running an agent."""
        return TeeJobResult(
            report={"schema_version": 1, "task_id": "fixture", "answer": "{}", "timed_out": False,
                    "stderr_tail": ""},
            provenance={"profile": "sn22", "job_id": job_id, "bundle_sha256": bundle_sha256,
                        "fixture": True, "inference_funded_by": "miner"})


def main() -> int:  # pragma: no cover - convenience for `python -m`
    print("SN22 TEE profile; load via KATA_TEE_PROFILE=tee_profile:Sn22TeeProfile",
          file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
