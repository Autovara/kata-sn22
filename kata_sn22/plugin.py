"""The SN22 (Desearch) subnet plugin (SN22-3).

A complete rewrite against the current ``kata.plugins`` ABI. The previous file was a skeleton whose
scorer read a ``# relevance=<float>`` comment out of the submitted ``agent.py`` — that is gone,
along with every ``kata.packages.*`` import, because a scorer a candidate can edit is not a scorer.

Everything of substance lives in the SN22-2 protocol modules; this file is the adapter between them
and the platform's King-of-the-Hill contract:

* ``sample_problems`` draws one challenge's queries from a versioned pool. Both contestants get
  byte-identical tasks.
* ``run_candidate`` executes a submission per task and classifies whatever comes back.
* ``score`` VERIFIES what came back — fetching the pages itself, checking the miner's excerpts
  against them, judging what survives, re-scraping claimed tweets — and reduces that to the seven
  ordered rank signals. Never anything the candidate asserted.
* ``compare`` / ``beats_king`` apply the fixed lexicographic priority.

**Scoring profile is NOISY.** It was briefly DETERMINISTIC, on the strength of a sealed corpus this
adapter invented and upstream does not have. That corpus is gone. SN22 scores live sources with an
LLM judge, so the same submission does not score identically twice, and labelling it deterministic
would sanction a cross-challenge score cache that compares a stale king against a fresh challenger.

Fairness comes from somewhere else, and it is upstream's own answer: the validator fetches the
ground truth ITSELF and verifies both contestants against that same fetch. Freezing the world was
never required — verifying it independently was.
"""
from __future__ import annotations

import hashlib
import hmac
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
from kata_sn22 import judge as judge_module
from kata_sn22.execution import policy as execution_policy
from kata_sn22.execution import tee_execution_enabled
from kata_sn22.gateway import GatewayDenied, Sn22Gateway
from kata_sn22.manifests import (
    QueryManifest,
    UsageManifest,
    UsageRecord,
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

#: The packaged question pool a production epoch is built from. Overridable per challenge only so a
#: staging deployment can point at a different snapshot -- never at the calibration pool, which is
#: refused by kind rather than by name (see kata_sn22.question_pool).
PRODUCTION_QUESTION_POOL = "production"

#: Identifies the scoring rules. Part of the benchmark identity, so changing the judge invalidates
#: every cached score rather than silently re-ranking history.
JUDGE_POLICY_ID = "sn22-upstream-evidence-v1"
#: The judge behind the relevance and groundedness verdicts. Upstream's own model, so a Kata score
#: and an upstream score are produced by the same grader.
MODEL_IDENTITY = judge_module.JUDGE_MODEL
#: The audited upstream this adapter tracks (plan §2; SN22-5 packages it).
UPSTREAM_COMMIT = "bea9712f58a5fc01c57ec441ce279499529d8bf6"
#: Bumped whenever this adapter's scoring surface changes.
PLUGIN_REVISION = "sn22-adapter-2"

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
    # Wall clock, by far the noisiest signal here — and note it is a SUM OVER TASKS, not a per-task
    # figure. Per-task jitter therefore accumulates: at 8 tasks, 0.25s of jitter each already spends
    # this whole margin. So this number is not independent of `task_count`, and §5.5 must calibrate
    # the two TOGETHER rather than picking a margin and a task count separately.
    "sn22_latency_seconds": 2.0,
}


@dataclass(frozen=True)
class Sn22Problems:
    """One challenge: the questions both contestants answer, and the identity of the rules.

    There is no corpus. Fairness does not come from freezing the world -- it comes from the
    validator fetching the ground truth itself and verifying BOTH contestants against that same
    fetch (see :mod:`kata_sn22.verification`). That is upstream's own model, and it is what a sealed
    snapshot was standing in for.
    """

    manifest: QueryManifest
    tasks: tuple[Task, ...]
    challenge_id: str
    #: The production epoch: 60 tasks, four pools, three deep samples each, built from packaged
    #: upstream rows. Present under the ``tee`` backend and ``None`` under ``sandbox``, because the
    #: two paths have not converged yet -- Phase F is where the sandbox's scorer is replaced and
    #: ``tasks`` above stops existing. A duel NEVER runs without one; see ``sample_problems``.
    epoch: object = None

    @property
    def identity(self) -> str:
        """What makes two challenges comparable: the questions, and the rules used to score them.

        A snapshot digest used to be part of this. It cannot be any more, and should not be: the web
        moves between rounds, so binding identity to a copy of it would mean no two challenges were
        ever comparable. What must not move is the QUESTION SET, the judge policy, the model and the
        upstream commit -- change any of those and a score means something different.
        """
        return benchmark_identity(
            query_commitment=self.manifest.as_commitment(),
            snapshot_digest="",
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

#: Where the attested sealed room answers. Absent means no room, and under the ``tee`` backend that
#: is a refusal rather than a fallback.
ROOM_ENDPOINT_ENV = "KATA_SN22_ROOM_URL"
ROOM_MAX_ATTEMPTS_ENV = "KATA_SN22_ROOM_MAX_ATTEMPTS"


def _room_configured() -> bool:
    return bool(os.environ.get(ROOM_ENDPOINT_ENV, "").strip())


def _resolve_room_max_attempts() -> int:
    raw = os.environ.get(ROOM_MAX_ATTEMPTS_ENV, "3").strip()
    try:
        value = int(raw)
    except ValueError:
        return 3
    return value if 1 <= value <= 5 else 3


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
    """Desearch (Bittensor SN22): a paired, independently-verified search-quality competition."""

    evaluator_id = "sn22_desearch"
    pack = "sn22__desearch"
    mode = "miner"
    # Live sources, judged by an LLM: a submission's score drifts run to run. See the module
    # docstring for why this must not be DETERMINISTIC.
    scoring_profile = ScoringProfile.NOISY
    validator_identity = f"{PLUGIN_REVISION}/{JUDGE_POLICY_ID}"

    def __init__(self, *, page_transport=None, judge_client=None, tweet_scraper=None,
                 search_provider=None) -> None:
        """The three things the VALIDATOR pays for, injected rather than constructed.

        Verification needs to fetch pages, ask a judge, and re-scrape tweets. All three cost money
        and do network I/O, and the lane's runtime deliberately carries no HTTP client -- so each is
        a seam. Production wires live ones; calibration wires the cassettes in
        :mod:`kata_sn22.fetch`, :mod:`kata_sn22.judge` and :mod:`kata_sn22.tweets`.

        None of them defaults to something that "works". A missing seam raises when a round tries to
        verify, because the alternative -- scoring an unverified answer -- produces a number that
        looks like a score and means nothing.
        """
        self._page_transport = page_transport
        self._judge_client = judge_client
        self._tweet_scraper = tweet_scraper
        #: What answers a candidate's relay search. Under the TEE backend the miner funds its own
        #: calls inside its room and this stays unset; the gateway then refuses relay searches,
        #: which is correct — there is nothing for the lane to serve.
        self._search_provider = search_provider
        self._fetcher = None

    # ---- verification (what the validator establishes for itself) --------------------------------
    def _page_fetcher(self):
        """One fetcher per plugin instance, so KING AND CHALLENGER SHARE ITS CACHE.

        This is the fairness property, not an optimisation: both contestants must be judged against
        the same bytes. A per-variant fetcher could hand the second one a page that changed since
        the first was scored, and the difference would be indistinguishable from a scoring
        difference.
        """
        from kata_sn22.fetch import PageFetcher

        if self._fetcher is None:
            if self._page_transport is None:
                raise Sn22AgentError(
                    "no page transport is configured; a round cannot verify a miner's sources "
                    "without fetching them, and an unverified score is not a score")
            self._fetcher = PageFetcher(transport=self._page_transport)
        return self._fetcher

    def _verified(self, attempt: TaskAttempt) -> TaskAttempt:
        """Attach what the validator independently established about one attempt.

        An attempt with no output is returned unchanged: there is nothing to verify, and spending a
        fetch or a judge call on it would be paying to confirm an absence.
        """
        from dataclasses import replace

        from kata_sn22 import verification as verify

        if attempt.output is None:
            return attempt
        task = attempt.task
        if task.search_type == "x_search":
            if self._tweet_scraper is None:
                raise Sn22AgentError(
                    "no tweet scraper is configured; X results are verified by re-scraping them")
            result = verify.verify_x_search(
                attempt.output, scraper=self._tweet_scraper,
                start_date=task.start_date, end_date=task.end_date)
        else:
            if self._judge_client is None:
                raise Sn22AgentError(
                    "no judge is configured; source relevance is decided by the judge, not here")
            result = verify.verify_ai_search(
                attempt.output, query=task.query,
                fetcher=self._page_fetcher(), judge_client=self._judge_client)
        return replace(attempt, verification=result)

    # ---- identity and environment ---------------------------------------------------------------
    def environment_spec(self) -> EnvSpec:
        """The candidate reaches one gateway and nothing else, and holds no VALIDATOR credential.

        Two backends, and the credential story differs between them — which is exactly why the
        backend is declared here rather than assumed:

        * **``tee``** (production): the agent runs in the attested sealed room and funds its own
          search and inference with ITS OWN credential, sealed to its exact bundle and decrypted
          only inside the room. The validator handles ciphertext and pays for nothing.
        * **``sandbox``** (development and §5.5 calibration): the lane's own relay serves the sealed
          corpus locally under `bwrap`.

        ``required_secrets`` is EMPTY under both. Under ``tee`` the miner's key is a sealed bundle
        artifact the ROOM opens, never an environment variable the platform hands out; under
        ``sandbox`` there is no provider call to make. Declaring a secret here would put a validator
        credential within reach of candidate code, which is the one thing §6.1 forbids outright.
        """
        return EnvSpec(
            network="relay_only",
            allowed_hosts=(),
            required_secrets=(),
            execution="tee" if tee_execution_enabled() else "sandbox",
            resources={"protocol_version": PROTOCOL_VERSION},
        )

    # ---- sealing a challenge --------------------------------------------------------------------
    def sample_problems(self, *, seed: str, config: dict[str, Any]) -> Sn22Problems:
        """Draw one challenge's questions.

        No corpus is frozen. Both contestants search the live web with their own credentials, and
        the validator verifies both against sources IT fetches -- upstream's model, and the one a
        sealed snapshot was standing in for.

        **Under the production backend this builds the upstream epoch and refuses anything else.**
        The hand-written calibration pool below is six queries; it exists so a miner can iterate
        offline, and using it for a duel would mean a King defended its crown against a question set
        that has nothing to do with the subnet. There is deliberately no flag that re-enables it in
        production and no fallback when the packaged rows are missing -- the round fails first.
        """
        epoch = self._production_epoch(seed=seed, config=config) if tee_execution_enabled() \
            else None

        count = int(config.get("task_count") or 4)
        manifest = fixtures.calibration_manifest(seed=seed, count=count)
        limits = Limits(
            max_wall_seconds=int(config.get("max_wall_seconds") or 120),
            max_provider_calls=int(config.get("max_provider_calls") or 8),
            max_tokens=int(config.get("max_tokens") or 20_000),
            max_results=int(config.get("max_results") or DEFAULT_RESULTS_PER_TASK),
        )
        tasks = tuple(fixtures.tasks_for(manifest, limits=limits))
        for task in tasks:
            validate_task(task)   # a malformed task would misscore BOTH contestants
        return Sn22Problems(manifest=manifest, tasks=tasks, challenge_id=f"sn22-{seed}",
                            epoch=epoch)

    @staticmethod
    def _production_epoch(*, seed: str, config: dict[str, Any]):
        """The upstream epoch a real duel is scored on: 60 tasks, never fewer.

        ``task_count`` is deliberately not consulted. Upstream deep-scores 20% of a pool and DROPS
        any contestant with fewer than three deep samples, so a smaller epoch does not measure less
        -- it scores zero for a reason unrelated to the answers. A config asking for one is refused
        rather than quietly rounded up, because the operator who set it believes something false.
        """
        from kata_sn22.epoch_manifest import TASKS_PER_EPOCH, build_epoch
        from kata_sn22.question_pool import PoolError, load_pool

        requested = config.get("task_count")
        if requested is not None and int(requested) != TASKS_PER_EPOCH:
            raise Sn22AgentError(
                f"task_count={requested} cannot be scored: an upstream epoch is exactly "
                f"{TASKS_PER_EPOCH} tasks (15 per pool), because a pool with fewer than three "
                f"deep samples is DROPPED rather than scored low")
        name = str(config.get("question_pool") or PRODUCTION_QUESTION_POOL)
        try:
            # BOTH calls, not just the load. A missing pool and a development pool are the same
            # class of misconfiguration -- nobody snapshotted the questions -- and an operator who
            # got Sn22AgentError for one and a raw PoolError for the other would reasonably
            # conclude the second was a different, deeper problem.
            pool = load_pool(name)
            return build_epoch(seed=seed, pool=pool, production=True)
        except PoolError as exc:
            # Fail here, before either contestant has spent anything. Upstream's fallback is an LLM
            # call the validator pays for, and the validator holds no paid credential at all.
            raise Sn22AgentError(f"cannot build a production epoch: {exc}") from exc

    def benchmark_identity(self, problems: Sn22Problems) -> str:
        """NON-EMPTY: the question set plus the rules used to score it.

        Not a claim that the challenge is reproducible -- it is not, the web moves and the judge is
        an LLM. It is the identity of the RULES, so a summary scored under a different judge policy
        or a different upstream commit is never mistaken for a comparable one.
        """
        return problems.identity

    def execution_order(self, *, problems: Sn22Problems,
                        variants: tuple[str, ...]) -> tuple[str, ...]:
        """Permute who runs first, deterministically per challenge (plan §5.2 item 5).

        SN22 ranks ``sn22_latency_seconds``, measured by the lane's own clock. Contestants run one
        after another on one host, so the first to run meets a colder cache and a different
        neighbour than the second. Fixed king-then-challenger order puts that difference on the same
        side EVERY round — a systematic bias, not noise, and averaging more rounds does not remove
        a constant.

        Derived from the sealed challenge's own identity rather than from an RNG, for two reasons:

        * **an auditor must be able to reproduce it.** Re-running the challenge from its seed has to
          reproduce the whole challenge, order included, or "reproducible" is not true;
        * **a miner must not be able to predict it.** The identity hashes the secret query manifest,
          so predicting the order means already holding the queries — at which point the order is
          the least of the problems.

        Sorting by an HMAC keyed on the identity gives both: fixed for one challenge, unpredictable
        across them, and uniform over many.
        """
        if len(variants) < 2:
            return tuple(variants)
        key = problems.identity.encode("utf-8")
        return tuple(sorted(
            variants,
            key=lambda label: hmac.new(key, label.encode("utf-8"), hashlib.sha256).hexdigest(),
        ))

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

        # The declared backend and the executed backend must agree. A lane whose EnvSpec says "tee"
        # while its runner quietly executes locally is the worst of both: the operator believes the
        # miner is funding its own calls inside an attested room, and in fact untrusted code is
        # running on the validator host against the lane's own relay. Refuse instead.
        backend = self.environment_spec().execution
        if backend == "tee" and not _room_configured():
            raise Sn22AgentError(
                f"execution backend is 'tee' but no sealed room is configured "
                f"({ROOM_ENDPOINT_ENV} is unset). Set it, or select the development backend "
                f"explicitly with {execution_policy.EXECUTION_BACKEND_ENV}=sandbox — this lane "
                f"will not silently run an untrusted agent locally while declaring a TEE")
        if backend == "tee":
            return self._run_candidate_in_tee(
                agent_path=agent_path, problems=problems, context=context)

        variant = context.label or "candidate"
        gateway = Sn22Gateway(provider=self._search_provider,
                              challenge_id=problems.challenge_id,
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

    def _run_candidate_in_tee(
        self, *, agent_path: str, problems: Sn22Problems, context: RunContext
    ) -> Sn22RawRun:
        """Execute every task remotely and accept only quote-bound room answers."""
        from kata_sn22.execution.tee_room import (
            DcapQvlVerifier,
            HttpRoomLauncher,
            evaluate_candidate_in_room,
            hash_bundle,
            resolve_room_policy,
            sealed_key_for_bundle,
        )

        bundle_root = Path(agent_path).expanduser().resolve()
        variant = context.label or "candidate"
        launcher = HttpRoomLauncher(os.environ[ROOM_ENDPOINT_ENV].strip())
        policy = resolve_room_policy()
        verifier = DcapQvlVerifier()
        sealed_key = sealed_key_for_bundle(bundle_root)
        bundle_sha256 = hash_bundle(bundle_root)
        seen_nonces: set[bytes] = set()
        attempts: list[TaskAttempt] = []
        usage_records: list[UsageRecord] = []

        for index, task in enumerate(problems.tasks, start=1):
            project_key = json.dumps(task.as_input(), sort_keys=True, separators=(",", ":"))
            started = _monotonic()
            outcome = evaluate_candidate_in_room(
                agent_ref=str(bundle_root),
                project_key=project_key,
                sealed_key_ref=sealed_key,
                bundle_sha256=bundle_sha256,
                policy=policy,
                launcher=launcher,
                verifier=verifier,
                seen_nonces=seen_nonces,
            )
            observed = _monotonic() - started
            if not outcome.accepted:
                raise Sn22AgentError(f"sealed-room execution was rejected: {outcome.reason}")
            report = outcome.report
            if not isinstance(report, dict) or report.get("task_id") != task.task_id:
                raise Sn22AgentError("sealed room returned a report for the wrong SN22 task")
            if report.get("timed_out") is True:
                attempts.append(TaskAttempt(
                    task=task, error=ErrorClass.TIMEOUT, observed_seconds=observed))
            elif report.get("truncated") is True:
                attempts.append(TaskAttempt(
                    task=task, error=ErrorClass.EXCESS_OUTPUT, observed_seconds=observed))
            elif int(report.get("returncode", 0) or 0) != 0 and not report.get("answer"):
                attempts.append(TaskAttempt(
                    task=task, error=ErrorClass.CRASHED, observed_seconds=observed))
            else:
                answer = report.get("answer")
                if not isinstance(answer, str):
                    attempts.append(TaskAttempt(
                        task=task, error=ErrorClass.INVALID_SCHEMA, observed_seconds=observed))
                else:
                    try:
                        output = parse_task_output(answer.encode("utf-8"), task=task)
                        attempts.append(TaskAttempt(
                            task=task, output=output, observed_seconds=observed))
                    except ProtocolError as exc:
                        attempts.append(TaskAttempt(
                            task=task, error=exc.error_class, observed_seconds=observed))

            inference = (outcome.provenance or {}).get("inference_summary")
            if isinstance(inference, dict):
                usage_records.append(UsageRecord(
                    variant=variant,
                    task_id=task.task_id,
                    provider_calls=max(0, int(inference.get("requests", 0) or 0)),
                    tokens=max(0, int(inference.get("tokens", 0) or 0)),
                    spend_usd=0.0,
                ))
            if context.progress is not None:
                context.progress(ProgressUpdate(
                    variant=variant,
                    done=index,
                    total=len(problems.tasks),
                    state="scoring",
                    metrics={"usable": sum(1 for attempt in attempts if attempt.usable)},
                ))

        return Sn22RawRun(
            variant=variant,
            agent_path=str(agent_path),
            attempts=tuple(attempts),
            usage=UsageManifest(
                challenge_id=problems.challenge_id,
                records=tuple(usage_records),
            ),
            isolated=True,
        )

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
        attempts = [self._verified(attempt) for attempt in raw.attempts]
        signals = score_attempts(attempts, usage=raw.usage, variant=raw.variant)
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

    def preflight(self) -> list[dict[str, str]]:
        """Fail before paid work unless verification and the declared backend are usable."""
        from kata_sn22.execution.tee_room import (
            ROOM_AUTH_SECRET_ENV,
            DcapQvlVerifier,
            resolve_room_policy,
            validate_room_url,
            verify_room_identity,
        )
        issues: list[dict[str, str]] = []
        verification_mode = os.environ.get(
            "KATA_SN22_VERIFICATION_MODE", "live").strip().lower()
        if verification_mode not in {"live", "recorded"}:
            issues.append({
                "level": "error",
                "message": "KATA_SN22_VERIFICATION_MODE must be 'live' or 'recorded'.",
            })
        # NOTE: live verification deliberately requires NO validator provider credential.
        #
        # This check used to demand SCRAPINGDOG_API_KEY, APIFY_API_KEY and OPENAI_API_KEY, because
        # the validator once paid for its own page fetches, tweet re-scrapes and judging. Under the
        # miner-funded rule it pays for none of them: every one of those calls happens inside the
        # sealed room, with the four keys the CONTESTANT sealed to its bundle.
        #
        # Leaving the requirement in place was worse than useless -- a correctly configured lane,
        # with no provider keys anywhere by design, would fail preflight with three errors telling
        # the operator to add exactly the credentials that must not exist.
        try:
            backend = self.environment_spec().execution
        except ValueError as exc:
            issues.append({"level": "error", "message": str(exc)})
            return issues
        if backend == "sandbox":
            if verification_mode != "recorded":
                issues.append({
                    "level": "error",
                    "message": (
                        "The SN22 sandbox backend is a free canary/calibration path and requires "
                        "KATA_SN22_VERIFICATION_MODE=recorded."
                    ),
                })
            if _sandbox_required() and not sandbox.available():
                issues.append({
                    "level": "error",
                    "message": f"{REQUIRE_SANDBOX_ENV}=1 but {sandbox.BWRAP} is unavailable.",
                })
            return issues

        if verification_mode != "live":
            issues.append({
                "level": "error",
                "message": (
                    "The SN22 TEE backend requires KATA_SN22_VERIFICATION_MODE=live; "
                    "recorded verification is restricted to the free sandbox canary."
                ),
            })
        room_url = os.environ.get(ROOM_ENDPOINT_ENV, "").strip()
        if not room_url:
            issues.append({
                "level": "error",
                "message": f"{ROOM_ENDPOINT_ENV} is required for the SN22 TEE backend.",
            })
            return issues
        if not os.environ.get(ROOM_AUTH_SECRET_ENV, "").strip():
            issues.append({
                "level": "error",
                "message": f"{ROOM_AUTH_SECRET_ENV} is required to authenticate room runs.",
            })
        try:
            validate_room_url(room_url)
            policy = resolve_room_policy()
            verify_room_identity(room_url, policy=policy, verifier=DcapQvlVerifier())
        except RuntimeError as exc:
            issues.append({"level": "error", "message": str(exc)})
        return issues

    # ---- cost ------------------------------------------------------------------------------------
    def capacity_estimate(self, *, config: dict[str, Any]) -> dict[str, float]:
        """Hard upper bounds for one paired challenge.

        Candidate inference runs in the miner-funded room, so the validator's paid dimensions are
        its independent page fetches, judge calls and tweet re-scrapes. ``tee_runs`` still counts
        the finite room resource. Every bound assumes both contestants, every task and the
        costliest task type; the real mix can only be cheaper.
        """
        from kata_sn22.upstream_adapter import MAX_SAMPLED_LINKS

        tasks = int(config.get("task_count") or 4)
        variants = 2   # king and exactly one challenger
        if os.environ.get("KATA_SN22_VERIFICATION_MODE", "live").strip().lower() == "recorded":
            tee_runs = (
                tasks * variants * _resolve_room_max_attempts()
                if self.environment_spec().execution == "tee"
                else 0
            )
            return {
                "data_api_calls": 0.0,
                "inference_calls": 0.0,
                "scrape_units": 0.0,
                "tee_runs": float(tee_runs),
            }
        results = int(config.get("max_results") or DEFAULT_RESULTS_PER_TASK)
        return {
            # One fetch per returned web result. The shared PageFetcher cache may reduce this, but
            # unique URLs across every answer are the safe bound.
            "data_api_calls": float(tasks * results * variants),
            # At most MAX_SAMPLED_LINKS relevance calls plus one groundedness call per AI task.
            "inference_calls": float(tasks * (min(results, MAX_SAMPLED_LINKS) + 1) * variants),
            # An all-X challenge re-scrapes every returned tweet.
            "scrape_units": float(tasks * results * variants),
            "tee_runs": float(tasks * variants * _resolve_room_max_attempts()),
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

        if self.environment_spec().execution == "tee":
            sealed_key = root / "sealed_inference_key"
            try:
                ciphertext = bytes.fromhex(sealed_key.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                ciphertext = b""
            if len(ciphertext) < 32:
                findings.append(
                    "TEE submissions must include a non-trivial hexadecimal "
                    "sealed_inference_key bound to this bundle")

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
        """Anti-memorization: a submission must not ship the questions.

        There is no longer a corpus to memorize -- sources are live and the validator fetches them
        -- but the QUERY POOL is still finite and versioned. A bundle carrying the pool's queries
        verbatim has seen the question set, and an agent that recognises a query can answer it from
        a lookup table rather than by searching.
        """
        reject: list[str] = []
        review: list[str] = []
        known_queries = {query.strip().casefold()
                         for query in fixtures.query_pool() if len(query.strip()) >= 24}
        for name, content in (bundle_files or {}).items():
            text = (content if isinstance(content, str) else str(content)).casefold()
            hits = sorted(query for query in known_queries if query in text)
            if hits:
                message = (f"{name} embeds {len(hits)} pool query/queries verbatim: "
                           f"{hits[0][:60]!r}")
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
        """Publish SN22's own verdict in the shape the platform reads it in.

        The SCORING is entirely SN22's -- its tasks, its judge, its signals, its priority order. It
        is not shared with, derived from, or comparable to any other subnet's. What is shared is the
        *envelope*: the platform has to find out who won in order to merge the PR and crown a king,
        and it reads every lane the same way, by name.

        So each card carries its signals twice, identically valued:

        - ``signals``   -- ``direction``/``priority``, human-facing and read by the canary.
        - ``rank_signals`` -- ``higher_better``, the platform's promotion contract.

        and the contestants appear both as ``challenger`` (the best one, for display) and in
        ``entries`` keyed by ``submission_id`` (how the platform finds the PR it is deciding). The
        duplication is deliberate: dropping either spelling silently breaks a real consumer, and the
        two can never disagree because both are generated here from the same ``RANK_SIGNALS`` tuple.
        """
        def _card(variant) -> dict | None:
            if variant is None:
                return None
            card = variant.card
            signals = card.payload
            values = [(name, higher, getattr(signals, name)) for name, higher in RANK_SIGNALS]
            return {
                "submission_id": variant.label,
                "artifact_hash": self.hash_bundle(Path(variant.agent_path)),
                "comparable": card.comparable,
                "passed": card.passed,
                "isolated": bool((getattr(card, "metrics", None) or {}).get("isolated", False)),
                "signals": [
                    {"name": name, "value": value,
                     "direction": "higher_is_better" if higher else "lower_is_better",
                     "priority": index}
                    for index, (name, higher, value) in enumerate(values, start=1)
                ],
                "rank_signals": [
                    {"name": name, "value": value, "higher_better": higher}
                    for name, higher, value in values
                ],
                "detail": signals.detail,
            }

        outcome = getattr(result, "outcome", None)
        king_variant = getattr(outcome, "king", None)
        ranked = list(getattr(outcome, "ranked", None) or ())
        entries = [entry for entry in (_card(variant) for variant in ranked) if entry is not None]
        king = _card(king_variant)
        cards = [card for card in (king, *entries) if card is not None]

        return {
            "schema_version": 1,
            "evaluator_id": self.evaluator_id,
            "lane_id": self.pack,
            "protocol_version": PROTOCOL_VERSION,
            "plugin_revision": PLUGIN_REVISION,
            "judge_policy_id": JUDGE_POLICY_ID,
            "upstream_commit": UPSTREAM_COMMIT,
            "benchmark_identity": getattr(outcome, "benchmark_identity", "")
            or getattr(result, "benchmark_identity", ""),
            "challenge_id": getattr(result, "challenge_id", "") or getattr(result, "run_id", ""),
            "king": king,
            "challenger": entries[0] if entries else None,
            "entries": entries,
            # Subnet-neutral canary code enforces this declarative contract only in paid mode.
            # Positive quality/coverage plus attested provider calls prevents a byte-transparent
            # route to an incompatible endpoint from looking like a successful live canary.
            "canary_requirements": {
                "provider_calls_per_side_min": 1,
                "positive_signals": [
                    "sn22_weighted_quality",
                    "sn22_coverage",
                ],
            },
            # BOTH sides, or the challenge was not isolated. A per-side flag would let a result
            # where only one contestant was confined read as a confined challenge.
            "isolated": bool(cards) and all(card["isolated"] for card in cards),
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
