"""The SN22 (Desearch) subnet plugin (SN22-3).

A complete rewrite against the current ``kata.plugins`` ABI. The previous file was a skeleton whose
scorer read a ``# relevance=<float>`` comment out of the submitted ``agent.py`` — that is gone,
along with every ``kata.packages.*`` import, because a scorer a candidate can edit is not a scorer.

Everything of substance lives in the SN22-2 protocol modules; this file is the adapter between them
and the platform's King-of-the-Hill contract:

* ``sample_problems`` seals one challenge — queries drawn from a versioned pool, a frozen corpus,
  and the relay that serves it. Both contestants get byte-identical tasks from that sealed world.
* ``run_candidate`` executes a submission per task and classifies whatever comes back.
* ``score`` reduces the attempts to the seven ordered rank signals, measured against the sealed
  snapshot and the relay's own usage record — never against anything the candidate asserted.
* ``compare`` / ``beats_king`` apply the fixed lexicographic priority.

**Scoring profile is DETERMINISTIC, not NOISY.** The skeleton was NOISY because it imagined scoring
against the live web. A sealed snapshot removes that: inside one challenge the same submission
always scores the same, and ``benchmark_identity`` is the hash that says *which* sealed world it
was. A new round draws a new world and therefore a new identity, so nothing stale is ever reused —
but within a world, results are reproducible and auditable, which is what the plan asks for.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kata.plugins.contract import (
    EnvSpec,
    ProgressUpdate,
    RunContext,
    ScoreCard,
    ScoringProfile,
    SubnetPlugin,
)

from kata_sn22 import fixtures, relay_server, sandbox
from kata_sn22.gateway import GatewayDenied, Sn22Gateway
from kata_sn22.manifests import (
    QueryManifest,
    SnapshotManifest,
    UsageManifest,
    benchmark_identity,
)
from kata_sn22.protocol import (
    DEFAULT_RESULTS_PER_TASK,
    MAX_OUTPUT_BYTES,
    PROTOCOL_VERSION,
    ErrorClass,
    Limits,
    ProtocolError,
    Task,
    parse_task_output,
    validate_task,
)
from kata_sn22.scoring import RANK_SIGNALS, Signals, TaskAttempt, compare_signals, score_attempts

#: Identifies the scoring rules. Part of the benchmark identity, so changing the judge invalidates
#: every cached score rather than silently re-ranking history.
JUDGE_POLICY_ID = "sn22-sealed-overlap-v1"
#: The judge "model". Deterministic offline scoring until SN22-4 wires a real one.
MODEL_IDENTITY = "sn22-deterministic-judge-0"
#: The audited upstream this adapter tracks (plan §2; SN22-5 packages it).
UPSTREAM_COMMIT = "bea9712f58a5fc01c57ec441ce279499529d8bf6"
#: Bumped whenever this adapter's scoring surface changes.
PLUGIN_REVISION = "sn22-adapter-1"

#: Per-signal indifference bands for PROMOTION. A challenger must beat the king by MORE than these.
#:
#: Without them a crown changes hands on noise. Two byte-identical submissions do not produce
#: identical signals, because ``sn22_latency_seconds`` is wall clock measured by the lane:
#: whichever contestant happened to run on a quieter machine wins. That is exactly the
#: false-promotion mode §5.5 exists to bound, and a zero margin on a noisy signal guarantees it.
#:
#: These are DEFAULTS chosen to be obviously safe, not calibrated values. §5.5 replaces them from a
#: real calibration report before the lane goes live. Until then they err toward the incumbent,
#: which costs a good challenger one round and costs a bad promotion nothing.
PROMOTION_MARGINS: dict[str, float] = {
    "sn22_valid_query_rate": 0.0,     # a task is answered or it is not; nothing to jitter
    "sn22_weighted_quality": 0.01,    # judge noise
    "sn22_citation_precision": 0.01,
    "sn22_coverage": 0.01,
    "sn22_invalid_runs": 0.0,         # an integer count
    "sn22_cost_units": 1.0,           # one provider call of slack
    "sn22_latency_seconds": 2.0,      # wall clock, by far the noisiest signal here
}


@dataclass(frozen=True)
class Sn22Problems:
    """One sealed challenge: the questions, the corpus that answers them, and the tasks issued."""

    manifest: QueryManifest
    snapshot: SnapshotManifest
    tasks: tuple[Task, ...]
    challenge_id: str

    @property
    def identity(self) -> str:
        return benchmark_identity(
            query_commitment=self.manifest.as_commitment(),
            snapshot_digest=self.snapshot.digest(),
            judge_policy_id=JUDGE_POLICY_ID,
            model_identity=MODEL_IDENTITY,
            upstream_commit=UPSTREAM_COMMIT,
            plugin_revision=PLUGIN_REVISION,
        )


@dataclass(frozen=True)
class Sn22RawRun:
    """What one contestant actually did: its attempts plus the relay's billing for them."""

    variant: str
    agent_path: str
    attempts: tuple[TaskAttempt, ...]
    usage: UsageManifest
    #: Whether the submission ran under the sandbox. Recorded rather than assumed: a challenge run
    #: unconfined during calibration must never be mistaken afterwards for one that was isolated.
    isolated: bool = False


class Sn22AgentError(Exception):
    """The submission could not be executed at all."""


def _monotonic() -> float:
    """Wall clock for the LANE's own latency measurement, never the agent's self-report."""
    import time

    return time.monotonic()


#: Set in production. When the sandbox is unavailable the run is an ERROR rather than a quieter,
#: unconfined execution — plan §6.1 is not a preference.
REQUIRE_SANDBOX_ENV = "KATA_SN22_REQUIRE_SANDBOX"


def _sandbox_required() -> bool:
    return os.environ.get(REQUIRE_SANDBOX_ENV, "").strip().lower() not in ("", "0", "false", "no")


def _execute(agent_py: Path, task: Task, *, workdir: Path) -> tuple[bytes, ErrorClass | None]:
    """Run one submission against one task. Returns (stdout, classified failure or None).

    Deliberately narrow, however it runs:

    * **no candidate-selected command** — the interpreter and the entry file are both fixed, so a
      submission cannot choose what gets executed;
    * **a constructed environment** — built from nothing rather than filtered, so no provider
      credential exists to leak. The agent gets a relay capability instead;
    * **output is read under a cap** and never evaluated.

    Preferring the sandbox and falling back is a deliberate two-mode design, not a soft failure.
    Calibration (§5.5) needs thirty-plus paired challenges on hosts that may not offer user
    namespaces; production sets ``KATA_SN22_REQUIRE_SANDBOX`` and gets a hard refusal instead. What
    is never allowed is the two being indistinguishable afterwards, so the mode is recorded on the
    raw run and travels into the challenge result.
    """
    limits = sandbox.SandboxLimits(max_wall_seconds=task.limits.max_wall_seconds,
                                   max_output_bytes=MAX_OUTPUT_BYTES)
    task_input = task.as_input()

    if sandbox.available():
        try:
            run = sandbox.run_candidate(agent_py, task_input, workdir=workdir,
                                        relay_endpoint=task.relay_endpoint,
                                        capability=task.relay_capability, limits=limits)
        except sandbox.SandboxUnavailable:
            # The sandbox could not start, so the submission was never executed. That is a SHARED
            # infrastructure fault, not a candidate crash — scoring it against whoever happened to
            # be running is exactly what §5.4 forbids. Fall through to the two-mode decision below.
            if _sandbox_required():
                raise
        else:
            if run.timed_out:
                return b"", ErrorClass.TIMEOUT
            if run.truncated:
                return b"", ErrorClass.EXCESS_OUTPUT
            if run.returncode != 0 and not run.stdout:
                return b"", ErrorClass.CRASHED
            return run.stdout, None

    if _sandbox_required():
        raise Sn22AgentError(
            f"{sandbox.BWRAP} is unusable and {REQUIRE_SANDBOX_ENV} is set; refusing to run an "
            f"untrusted submission unconfined")

    env = sandbox.candidate_env(task_input=task_input, relay_endpoint=task.relay_endpoint,
                                capability=task.relay_capability, workdir=str(workdir))
    try:
        completed = subprocess.run(
            ["/usr/bin/python3", str(agent_py)],
            input=json.dumps(task_input).encode("utf-8"), capture_output=True,
            cwd=str(workdir), env=env, timeout=task.limits.max_wall_seconds, check=False,
        )
    except subprocess.TimeoutExpired:
        return b"", ErrorClass.TIMEOUT
    except OSError as exc:
        raise Sn22AgentError(f"cannot execute {agent_py}: {exc}") from exc
    if completed.returncode != 0 and not completed.stdout:
        return b"", ErrorClass.CRASHED
    # Truncation is itself the violation; do not silently score a half-response.
    if len(completed.stdout) > MAX_OUTPUT_BYTES:
        return b"", ErrorClass.EXCESS_OUTPUT
    return completed.stdout, None


class Sn22DesearchPlugin(SubnetPlugin):
    """Desearch (Bittensor SN22): a sealed, paired search-quality competition."""

    evaluator_id = "sn22_desearch"
    pack = "sn22__desearch"
    mode = "miner"
    # Sealed snapshot + fixed queries == reproducible within a challenge. See the module docstring.
    scoring_profile = ScoringProfile.DETERMINISTIC
    validator_identity = f"{PLUGIN_REVISION}/{JUDGE_POLICY_ID}"

    # ---- identity and environment ---------------------------------------------------------------
    def environment_spec(self) -> EnvSpec:
        """The candidate reaches the relay and nothing else, and holds no provider credential.

        ``required_secrets`` is deliberately EMPTY. SN22's providers (apify, openai, scrapingdog)
        are reached by the trusted gateway on the candidate's behalf under a short-lived
        capability, so a provider key never enters the sandbox — plan §6.1. Declaring one here
        would put it there.
        """
        return EnvSpec(
            network="relay_only",
            allowed_hosts=(),
            required_secrets=(),
            execution="sandbox",
            resources={"protocol_version": PROTOCOL_VERSION},
        )

    # ---- sealing a challenge --------------------------------------------------------------------
    def sample_problems(self, *, seed: str, config: dict[str, Any]) -> Sn22Problems:
        """Seal one challenge from the versioned pool and freeze the corpus that answers it."""
        count = int(config.get("task_count") or 4)
        manifest = fixtures.calibration_manifest(seed=seed, count=count)
        snapshot = fixtures.calibration_snapshot(manifest)
        limits = Limits(
            max_wall_seconds=int(config.get("max_wall_seconds") or 120),
            max_provider_calls=int(config.get("max_provider_calls") or 8),
            max_tokens=int(config.get("max_tokens") or 20_000),
            max_results=int(config.get("max_results") or DEFAULT_RESULTS_PER_TASK),
        )
        tasks = tuple(fixtures.tasks_for(manifest, limits=limits))
        for task in tasks:
            validate_task(task)   # a malformed task would misscore BOTH contestants
        return Sn22Problems(manifest=manifest, snapshot=snapshot, tasks=tasks,
                            challenge_id=f"sn22-{seed}")

    def benchmark_identity(self, problems: Sn22Problems) -> str:
        """NON-EMPTY: this challenge is reproducible, and this hash says which one it was."""
        return problems.identity

    # ---- running a contestant -------------------------------------------------------------------
    def run_candidate(self, *, agent_path: str, problems: Sn22Problems,
                      context: RunContext) -> Sn22RawRun:
        """Run one submission over every task, against the trusted gateway behind a unix relay.

        The gateway holds whatever credential the round needs and the candidate holds a capability;
        the relay socket is the only thing that crosses between them. The socket lives inside the
        run directory, which is the one writable path the sandbox has — a unix socket is a
        filesystem object, so it survives ``--unshare-net`` and the sandbox keeps no network at all.

        Isolation is REQUIRED where it is available and refused where it is not. On a host without
        `bwrap` the run falls back to a plain subprocess with the same constructed environment, and
        says so in the raw run: that is the calibration path, and a challenge that used it must not
        be mistaken for one that ran confined. Production sets ``KATA_SN22_REQUIRE_SANDBOX=1``, and
        then an unavailable sandbox is an error rather than a quieter run.
        """
        agent_py = Path(agent_path).expanduser().resolve() / "agent.py"
        if not agent_py.is_file():
            raise Sn22AgentError(f"submission has no agent.py at {agent_py}")

        variant = context.label or "candidate"
        gateway = Sn22Gateway(snapshot=problems.snapshot, challenge_id=problems.challenge_id,
                              lane_id=self.pack,
                              reservation_calls=self._reservation_calls(problems))
        workdir = sandbox.fresh_workdir(Path(context.output_root).expanduser().resolve(),
                                        f"sn22-{variant}")
        relay_server.install_client(workdir)
        # Staged once per contestant, then mounted read-only: every task runs the SAME bytes, and a
        # submission cannot rewrite its own entry point between tasks.
        staged_agent = sandbox.stage_bundle(agent_py.parent, workdir)

        attempts: list[TaskAttempt] = []
        with relay_server.RelayServer(gateway, workdir) as relay:
            for index, task in enumerate(problems.tasks, start=1):
                capability = gateway.issue(variant=variant, task_id=task.task_id,
                                           max_calls=task.limits.max_provider_calls)
                issued = Task(
                    task_id=task.task_id, query=task.query, search_type=task.search_type,
                    result_type=task.result_type, ai_mode=task.ai_mode,
                    relay_endpoint=relay.endpoint, relay_capability=capability.token,
                    limits=task.limits,
                )
                started = _monotonic()
                try:
                    raw, failure = _execute(staged_agent, issued, workdir=workdir)
                except GatewayDenied:
                    # The gateway refused outright: shared infrastructure, not the
                    # candidate's doing.
                    raw, failure = b"", ErrorClass.PROVIDER_UNAVAILABLE
                observed = _monotonic() - started

                if failure is not None:
                    attempts.append(TaskAttempt(task=task, error=failure,
                                                observed_seconds=observed))
                else:
                    try:
                        output = parse_task_output(raw, task=task)
                        attempts.append(TaskAttempt(task=task, output=output,
                                                    observed_seconds=observed))
                    except ProtocolError as exc:
                        attempts.append(TaskAttempt(task=task, error=exc.error_class,
                                                    observed_seconds=observed))
                if context.progress is not None:
                    context.progress(ProgressUpdate(
                        variant=variant, done=index, total=len(problems.tasks), state="scoring",
                        metrics={"usable": sum(1 for a in attempts if a.usable)}))
        # The context manager closed the gateway: capabilities die with the challenge, so nothing
        # spends after scoring has begun.
        return Sn22RawRun(variant=variant, agent_path=str(agent_path),
                          attempts=tuple(attempts), usage=gateway.usage_manifest(),
                          isolated=sandbox.available())

    @staticmethod
    def _reservation_calls(problems: Sn22Problems) -> int:
        """The gateway's HARD ceiling for this challenge: every task's full quota, once.

        Per-task quotas bound one task. Only this bounds the challenge, and the challenge is what
        was reserved against the lane's daily budget — so it is derived from the same tasks the
        runner is about to issue rather than from a default that could drift from them.
        """
        return sum(task.limits.max_provider_calls for task in problems.tasks) or 1

    # ---- scoring and ranking --------------------------------------------------------------------
    def score(self, raw: Sn22RawRun, problems: Sn22Problems) -> ScoreCard:
        signals = score_attempts(list(raw.attempts), snapshot=problems.snapshot,
                                 usage=raw.usage, variant=raw.variant)
        return ScoreCard(
            # A scalar for the core's coarse ordering; the real decision uses the payload below,
            # because seven ordered signals do not collapse into one number without losing the
            # priority that makes them meaningful.
            comparable=round(signals.sn22_weighted_quality, 8),
            passed=signals.sn22_valid_query_rate > 0.0,
            # ``isolated`` rides on the card so it reaches the published result. It is not a signal
            # and never ranks anything — it is the fact a canary must be able to check, because a
            # challenge that ran unconfined looks identical to one that did not until somebody asks.
            metrics={**signals.as_metrics(), "detail": signals.detail,
                     "isolated": raw.isolated},
            payload=signals,
        )

    def compare(self, a: ScoreCard, b: ScoreCard) -> int:
        """EXACT ordering, no margins.

        Margins are deliberately not applied here. An epsilon comparison is not transitive — a≈b and
        b≈c can hold while a<c — so a sort built on it is unstable and depends on input order. This
        stays a strict total order for ranking and display; the promotion decision below is where
        the noise band belongs, because that is where a wrong call actually costs something.
        """
        return compare_signals(a.payload, b.payload)

    def beats_king(self, candidate: ScoreCard, king: ScoreCard | None) -> bool:
        """Promotion, WITH the indifference bands. A challenger must be meaningfully better."""
        from kata_sn22.scoring import beats_king as _beats

        return _beats(candidate.payload, None if king is None else king.payload,
                      margins=PROMOTION_MARGINS)

    def conformance_scorecards(self) -> tuple[ScoreCard, ScoreCard]:
        """Side-effect-free weak/strong cards for the installer's ordering gate.

        SN22's comparator ranks the seven-signal payload, not ``comparable``, so a generic empty
        card would not exercise the real decision surface. These two run no agent, no relay and no
        provider call.
        """
        weak = Signals(0.5, 0.10, 0.5, 0.2, 2, 40.0, 30.0)
        strong = Signals(1.0, 0.90, 1.0, 1.0, 0, 10.0, 5.0)
        return (
            ScoreCard(comparable=weak.sn22_weighted_quality, passed=True,
                      metrics=weak.as_metrics(), payload=weak),
            ScoreCard(comparable=strong.sn22_weighted_quality, passed=True,
                      metrics=strong.as_metrics(), payload=strong),
        )

    # ---- cost ------------------------------------------------------------------------------------
    def capacity_estimate(self, *, config: dict[str, Any]) -> dict[str, float]:
        """A TRUE upper bound on what one challenge can cost, from this plugin's own config read.

        Both contestants, every task, the full per-task call quota — the most the relay can be made
        to spend before its own quotas refuse. Resolved from the same ``config`` the challenge runs
        with, so the reservation cannot diverge from the execution.
        """
        tasks = int(config.get("task_count") or 4)
        calls = int(config.get("max_provider_calls") or 8)
        tokens = int(config.get("max_tokens") or 20_000)
        variants = 2   # king and exactly one challenger
        return {
            "inference_calls": float(tasks * calls * variants),
            "tokens": float(tasks * tokens * variants),
        }

    # ---- screening and review --------------------------------------------------------------------
    def static_screen(self, submission_path: str) -> object | None:
        """Deterministic, offline checks before anything is executed or paid for.

        Cheap refusals that need no round: a missing entry point, an entry point that is not a
        regular file, an oversized submission, or source that tries to reach the network directly
        instead of through the relay.
        """
        root = Path(submission_path).expanduser().resolve()
        findings: list[str] = []
        agent_py = root / "agent.py"
        if not agent_py.exists():
            findings.append("submission has no agent.py entry point")
        elif agent_py.is_symlink() or not agent_py.is_file():
            findings.append("agent.py must be a regular file")
        elif agent_py.stat().st_size > 1_000_000:
            findings.append("agent.py exceeds 1 MB")

        # Direct egress is pointless under relay_only and signals a submission that expects to reach
        # providers itself. Named modules only -- this is a screen, not a sandbox.
        banned = ("import socket", "import requests", "import httpx", "urllib.request",
                  "http.client", "import subprocess")
        for path in sorted(root.rglob("*.py")):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for marker in banned:
                if marker in text:
                    findings.append(f"{path.relative_to(root)} uses {marker!r}; "
                                    f"reach providers through the relay capability instead")
        return {"findings": findings, "passed": not findings} if findings else None

    def benchmark_review(self, bundle_files, *, strict):
        """Anti-memorization: a submission must not ship the answers.

        A sealed corpus is only secret while it stays sealed. A submission carrying snapshot
        document ids has either seen a previous challenge's world or is guessing at ids, and
        both mean its citations were not earned by retrieval.
        """
        reject: list[str] = []
        review: list[str] = []
        known_ids = {document.doc_id for document in fixtures.SNAPSHOT_DOCUMENTS}
        for name, content in (bundle_files or {}).items():
            text = content if isinstance(content, str) else str(content)
            hits = sorted(doc_id for doc_id in known_ids if doc_id in text)
            if hits:
                message = f"{name} embeds sealed snapshot document id(s): {', '.join(hits[:5])}"
                (reject if strict else review).append(message)
        return reject, review, float(len(reject))

    # ---- provenance and freshness ----------------------------------------------------------------
    def record_promotion_provenance(self, *, entry, verification, summary,
                                    public_root: str | None = None) -> None:
        """Record WHICH sealed world produced this crown, next to the crown itself."""
        if public_root is None:
            return None
        identity = getattr(summary, "benchmark_identity", None) or ""
        target = Path(public_root).expanduser() / self.pack / "promotions"
        target.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": 1,
            "lane_id": self.pack,
            "evaluator_id": self.evaluator_id,
            "benchmark_identity": identity,
            "plugin_revision": PLUGIN_REVISION,
            "judge_policy_id": JUDGE_POLICY_ID,
            "upstream_commit": UPSTREAM_COMMIT,
            "protocol_version": PROTOCOL_VERSION,
            "entry": getattr(entry, "submission_id", None) or str(entry),
        }
        (target / f"{identity[:16] or 'unknown'}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        return None

    def benchmark_is_current(self, *, lane_id, summary, public_root=None) -> bool:
        """A crown won in a different sealed world does not carry over."""
        return bool(getattr(summary, "benchmark_identity", None))

    def extra_verification_reasons(self, *, lane_id, summary, public_root=None) -> list[str]:
        if not getattr(summary, "benchmark_identity", None):
            return ["challenge summary has no benchmark identity; the sealed world is unknown"]
        return []

    # ---- challenge config and serialization ------------------------------------------------------
    def add_challenge_arguments(self, parser) -> None:
        parser.add_argument("--sn22-task-count", type=int, default=4,
                            help="Queries per challenge (both contestants get the same set).")
        parser.add_argument("--sn22-max-provider-calls", type=int, default=8,
                            help="Relay call quota per task per contestant.")
        parser.add_argument("--sn22-max-tokens", type=int, default=20_000,
                            help="Token quota per task per contestant.")
        parser.add_argument("--sn22-max-wall-seconds", type=int, default=120,
                            help="Wall-clock ceiling per task.")
        parser.add_argument("--sn22-max-results", type=int, default=DEFAULT_RESULTS_PER_TASK,
                            help="Results each task asks for. Both a request and a ceiling: "
                                 "fewer takes the upstream count penalty, more is rejected.")

    def build_challenge_config(self, args) -> dict:
        """Explicit values only. A default that silently decides paid work is a defect (plan §4)."""
        return {
            "task_count": int(getattr(args, "sn22_task_count", 4)),
            "max_provider_calls": int(getattr(args, "sn22_max_provider_calls", 8)),
            "max_tokens": int(getattr(args, "sn22_max_tokens", 20_000)),
            "max_wall_seconds": int(getattr(args, "sn22_max_wall_seconds", 120)),
            "max_results": int(getattr(args, "sn22_max_results", DEFAULT_RESULTS_PER_TASK)),
        }

    def challenge_result_json(self, result) -> dict:
        """Publish the ordered rank signals, in priority order, with their directions.

        The order is the contract, so it is serialized explicitly rather than left to dict ordering:
        a consumer must be able to see WHY one contestant outranked the other, not just that it did.
        """
        def _card(card) -> dict | None:
            if card is None:
                return None
            signals = card.payload
            return {
                "comparable": card.comparable,
                "passed": card.passed,
                "isolated": bool((getattr(card, "metrics", None) or {}).get("isolated", False)),
                "signals": [
                    {"name": name, "value": getattr(signals, name),
                     "direction": "higher_is_better" if higher else "lower_is_better",
                     "priority": index}
                    for index, (name, higher) in enumerate(RANK_SIGNALS, start=1)
                ],
                "detail": signals.detail,
            }

        return {
            "schema_version": 1,
            "evaluator_id": self.evaluator_id,
            "lane_id": self.pack,
            "protocol_version": PROTOCOL_VERSION,
            "plugin_revision": PLUGIN_REVISION,
            "judge_policy_id": JUDGE_POLICY_ID,
            "upstream_commit": UPSTREAM_COMMIT,
            "benchmark_identity": getattr(result, "benchmark_identity", ""),
            "challenge_id": getattr(result, "challenge_id", "") or getattr(result, "run_id", ""),
            "king": _card(getattr(result, "king_card", None)),
            "challenger": _card(getattr(result, "candidate_card", None)),
            # BOTH sides, or the challenge was not isolated. A per-side flag would let a result
            # where only one contestant was confined read as a confined challenge.
            "isolated": all(
                bool((getattr(card, "metrics", None) or {}).get("isolated", False))
                for card in (getattr(result, "king_card", None),
                             getattr(result, "candidate_card", None))
                if card is not None) and any(
                getattr(result, name, None) is not None
                for name in ("king_card", "candidate_card")),
        }

    def render_challenge_text(self, result) -> str:
        document = self.challenge_result_json(result)
        lines = [f"SN22 challenge — benchmark {document['benchmark_identity'][:16] or '(none)'}"]
        for side in ("king", "challenger"):
            card = document.get(side)
            if card is None:
                lines.append(f"  {side:<10} (not scored)")
                continue
            summary = "  ".join(f"{s['name'].removeprefix('sn22_')}={s['value']}"
                                for s in card["signals"])
            lines.append(f"  {side:<10} {summary}")
        return "\n".join(lines)


__all__ = ["Sn22AgentError", "Sn22DesearchPlugin", "Sn22Problems", "Sn22RawRun"]
