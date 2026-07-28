"""The attested pool report: what a sealed room says, and what the host will believe.

One report per pool per contestant — eight per duel. Each is quote-bound, so the host is not
trusting a JSON document; it is trusting a TDX quote that covers this document's hash.

**What a report is for.** The host does no paid work and re-runs nothing. It verifies eight reports
and does arithmetic. So a report has to carry everything a promotion decision rests on, and the host
has to be able to refuse one that does not match its sibling.

**The three outcomes, kept apart on purpose.** ``ok`` scores. ``credential_failure`` scores the
contestant **zero** — its own key failed, which is its own problem under the funding rule.
``infrastructure_failure`` **defers the whole duel** — nobody's answers were established, and a
score assigned on that basis would be a number with no evidence behind it. Collapsing the last two
would either punish an innocent miner for someone else's outage or hand a broken one a free pass.

**A refusal must still be attested.** A room that cannot run a contestant returns a quote-bound
``credential_failure`` report rather than an HTTP 500. A plain error is not evidence: it could come
from anywhere, including a validator that would rather the challenger lost.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum

from kata_sn22.credentials_v2 import CredentialReport
from kata_sn22.protocol_v2 import FIXED_SCORING_MODEL, UPSTREAM_COMMIT

REPORT_SCHEMA_VERSION = 2


class ReportStatus(str, Enum):
    OK = "ok"
    #: The contestant's own credential failed. Its score is zero; the duel still decides.
    CREDENTIAL_FAILURE = "credential_failure"
    #: Nobody's fault, and nothing was established. The duel defers.
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


#: The four pools, in the order a duel runs them. A contestant's epoch is exactly these.
POOLS: tuple[str, ...] = ("ai_search:fast", "ai_search:balanced", "ai_search:deep", "x_search")


class ReportError(ValueError):
    """A report is malformed, or a set of reports does not describe one comparable duel."""


@dataclass(frozen=True)
class PoolResult:
    """The upstream tuple for one pool, as ``_pool_raw_scores`` consumes it.

    Four numbers, and each has to survive the trip intact:

    * ``q_gate`` — the pool's quality gate input, compared against the search type's threshold;
    * ``q_weight`` — the quality that gets cubed;
    * ``volume`` — the task count, squared;
    * ``deep_count`` — deep samples actually scored. Below the minimum the pool is DROPPED, not
      scored low, so this is not a diagnostic.
    """

    q_gate: float
    q_weight: float
    volume: int
    deep_count: int

    def __post_init__(self) -> None:
        for name in ("q_gate", "q_weight"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ReportError(f"{name} must be a number")
            # NaN poisons every comparison downstream (`nan > x` is false for every x), so a duel
            # decided on one would silently keep the King forever.
            if value != value or value in (float("inf"), float("-inf")):
                raise ReportError(f"{name} must be finite, got {value!r}")
            if value < 0:
                raise ReportError(f"{name} must not be negative")
        for name in ("volume", "deep_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ReportError(f"{name} must be a non-negative integer")
        if self.deep_count > self.volume:
            raise ReportError("deep_count cannot exceed volume")

    def as_tuple(self) -> tuple:
        """The shape upstream's ``combine_pool_scores`` reads."""
        return (self.q_gate, self.q_weight, self.volume, self.deep_count)

    def as_dict(self) -> dict:
        return {"q_gate": self.q_gate, "q_weight": self.q_weight,
                "volume": self.volume, "deep_count": self.deep_count}


@dataclass(frozen=True)
class PoolReport:
    """One attested pool job.

    The identity fields are not provenance decoration. Each one is a thing that, if it differed
    between the two contestants, would make their scores incomparable — so the host checks every one
    of them before it compares anything.
    """

    pool: str
    status: ReportStatus
    contestant: str
    bundle_sha256: str
    task_manifest_sha256: str
    policy_hash: str
    agent_image_digest: str
    upstream_commit: str = UPSTREAM_COMMIT
    scoring_model: str = FIXED_SCORING_MODEL.value
    pool_result: PoolResult | None = None
    credentials: CredentialReport | None = None
    usage: dict = field(default_factory=dict)
    task_result_hashes: tuple[str, ...] = ()
    detail: str = ""

    def __post_init__(self) -> None:
        if self.pool not in POOLS:
            raise ReportError(f"unknown pool {self.pool!r}")
        if self.status is ReportStatus.OK and self.pool_result is None:
            raise ReportError("an ok report must carry a pool_result")
        if self.status is not ReportStatus.OK and self.pool_result is not None:
            # A failed pool with numbers attached invites someone to use them.
            raise ReportError(f"a {self.status.value} report must not carry a pool_result")

    def as_dict(self) -> dict:
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "status": self.status.value,
            "contestant": self.contestant,
            "pool": self.pool,
            "upstream_commit": self.upstream_commit,
            "scoring_model": self.scoring_model,
            "policy_hash": self.policy_hash,
            "task_manifest_sha256": self.task_manifest_sha256,
            "bundle_sha256": self.bundle_sha256,
            "agent_image_digest": self.agent_image_digest,
            "pool_result": self.pool_result.as_dict() if self.pool_result else None,
            "credential_status": self.credentials.as_dict() if self.credentials else {},
            "usage": dict(sorted(self.usage.items())),
            "task_result_hashes": list(self.task_result_hashes),
            "detail": self.detail,
        }

    def report_hash(self) -> str:
        """What the TDX quote binds. Any edit to any field moves it."""
        canonical = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    #: The identity every report in one duel must share.
    def identity(self) -> tuple:
        return (self.upstream_commit, self.scoring_model, self.policy_hash,
                self.task_manifest_sha256, self.agent_image_digest)


@dataclass(frozen=True)
class ContestantEpoch:
    """One contestant's four pool reports, verified to describe one epoch."""

    contestant: str
    reports: tuple

    @property
    def status(self) -> ReportStatus:
        """The epoch's outcome, worst-first.

        Infrastructure failure outranks credential failure: if any pool could not run, we do not
        actually know that the credential failure in another pool would have mattered, and a
        deferred duel can be re-run while a zeroed contestant cannot be un-zeroed.
        """
        statuses = {report.status for report in self.reports}
        if ReportStatus.INFRASTRUCTURE_FAILURE in statuses:
            return ReportStatus.INFRASTRUCTURE_FAILURE
        if ReportStatus.CREDENTIAL_FAILURE in statuses:
            return ReportStatus.CREDENTIAL_FAILURE
        return ReportStatus.OK

    def pool_results(self) -> dict:
        """``{pool: tuple}`` for upstream. Empty when the epoch did not score."""
        if self.status is not ReportStatus.OK:
            return {}
        return {report.pool: report.pool_result.as_tuple() for report in self.reports}


def assemble_epoch(reports, *, contestant: str) -> ContestantEpoch:
    """Check four pool reports describe ONE contestant's ONE epoch, or raise.

    Every check here is a way two reports could disagree while each looks fine alone -- which is
    exactly the class of problem a per-report check cannot catch.
    """
    reports = tuple(reports)
    if len(reports) != len(POOLS):
        raise ReportError(f"expected {len(POOLS)} pool reports, got {len(reports)}")

    wrong = [report.contestant for report in reports if report.contestant != contestant]
    if wrong:
        raise ReportError(f"reports name another contestant: {sorted(set(wrong))}")

    pools = [report.pool for report in reports]
    if sorted(pools) != sorted(POOLS):
        raise ReportError(f"reports do not cover every pool exactly once: {sorted(pools)}")

    identities = {report.identity() for report in reports}
    if len(identities) != 1:
        raise ReportError(
            "pool reports disagree on upstream commit, scoring model, policy, task manifest or "
            "agent image; they are not one epoch")

    bundles = {report.bundle_sha256 for report in reports}
    if len(bundles) != 1:
        raise ReportError("pool reports disagree on the bundle they ran")

    ordered = tuple(sorted(reports, key=lambda report: POOLS.index(report.pool)))
    return ContestantEpoch(contestant=contestant, reports=ordered)


def duel_is_comparable(king: ContestantEpoch, challenger: ContestantEpoch) -> None:
    """Both epochs were graded under the same rules, or raise.

    The bundles differ -- that is the whole point of a duel -- but everything about the *grading*
    must match. Without this, a challenger scored under a newer policy could beat a king scored
    under an older one, and the difference would look like skill.
    """
    king_identity = king.reports[0].identity()
    challenger_identity = challenger.reports[0].identity()
    if king_identity != challenger_identity:
        raise ReportError(
            "the two contestants were not graded under the same rules: upstream commit, scoring "
            "model, policy hash, task manifest or agent image differs")
    if king.contestant == challenger.contestant:
        raise ReportError("a duel needs two distinct contestants")


def parse_pool_report(raw: bytes | str | dict) -> PoolReport:
    """Parse one report as received from a room. Fail closed on anything unrecognised."""
    from kata_sn22.credentials_v2 import CredentialStatus

    if isinstance(raw, dict):
        document = raw
    else:
        payload = raw.encode("utf-8") if isinstance(raw, str) else raw
        try:
            document = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ReportError(f"report is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ReportError("report is not a JSON object")

    version = document.get("schema_version")
    if version != REPORT_SCHEMA_VERSION:
        raise ReportError(f"report schema_version {version!r} is not {REPORT_SCHEMA_VERSION}")

    allowed = {"schema_version", "status", "contestant", "pool", "upstream_commit",
               "scoring_model", "policy_hash", "task_manifest_sha256", "bundle_sha256",
               "agent_image_digest", "pool_result", "credential_status", "usage",
               "task_result_hashes", "detail"}
    unknown = sorted(set(document) - allowed)
    if unknown:
        raise ReportError(f"report has unknown field(s): {', '.join(unknown)}")

    try:
        status = ReportStatus(document.get("status"))
    except ValueError as exc:
        raise ReportError(f"unknown report status {document.get('status')!r}") from exc

    raw_result = document.get("pool_result")
    pool_result = None
    if raw_result is not None:
        if not isinstance(raw_result, dict) or set(raw_result) != {
                "q_gate", "q_weight", "volume", "deep_count"}:
            raise ReportError("pool_result must carry exactly q_gate, q_weight, volume, deep_count")
        pool_result = PoolResult(**raw_result)

    raw_credentials = document.get("credential_status") or {}
    if not isinstance(raw_credentials, dict):
        raise ReportError("credential_status must be an object")
    try:
        credentials = CredentialReport(
            {name: CredentialStatus(value) for name, value in raw_credentials.items()})
    except ValueError as exc:
        raise ReportError(f"unknown credential status: {exc}") from exc

    return PoolReport(
        pool=str(document.get("pool")), status=status,
        contestant=str(document.get("contestant") or ""),
        bundle_sha256=str(document.get("bundle_sha256") or ""),
        task_manifest_sha256=str(document.get("task_manifest_sha256") or ""),
        policy_hash=str(document.get("policy_hash") or ""),
        agent_image_digest=str(document.get("agent_image_digest") or ""),
        upstream_commit=str(document.get("upstream_commit") or ""),
        scoring_model=str(document.get("scoring_model") or ""),
        pool_result=pool_result, credentials=credentials,
        usage=dict(document.get("usage") or {}),
        task_result_hashes=tuple(document.get("task_result_hashes") or ()),
        detail=str(document.get("detail") or ""),
    )
