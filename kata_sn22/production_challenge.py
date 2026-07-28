"""One production duel: eight attested pool jobs, then one promotion decision.

This is the seam between the epoch (Phase E), the rooms (Phase C/F) and the verdict (Phase G). It
does four things and delegates everything that decides a number:

1. splits the 60-task manifest into four pool jobs per contestant;
2. sends each to a sealed room and **accepts only a quote-bound answer**;
3. turns each answer into a :class:`~kata_sn22.report_v2.PoolReport`;
4. hands all eight to :func:`kata_sn22.paired_scoring.decide`.

**Why eight jobs and not one.** Sixty tasks behind a single HTTP request is one proxy timeout away
from losing every answer the contestants already paid for. Four bounded jobs per side lose at most a
quarter, and each is separately attested — so a re-run repeats only what was lost.

**A rejected attestation defers; it never scores.** An unverifiable report is not a contestant doing
badly, it is the room failing to say what happened. Scoring it as zero would let anything that can
break attestation choose the winner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from kata_sn22 import paired_scoring
from kata_sn22.credentials_v2 import CredentialReport, CredentialStatus
from kata_sn22.epoch_manifest import POOLS, EpochManifest
from kata_sn22.paired_scoring import CHALLENGER, KING, DuelDeferred, Verdict
from kata_sn22.report_v2 import PoolReport, PoolResult, ReportStatus
from kata_sn22.scorer_policy import policy_hash
from kata_sn22.upstream_snapshot import UPSTREAM_COMMIT


@dataclass(frozen=True)
class Contestant:
    """One side of the duel: which bundle, and how the room is told to reach it."""

    label: str
    bundle_path: str
    bundle_sha256: str
    sealed_key: str = ""


@dataclass
class DuelRecord:
    """Everything a reviewer needs to see why a duel came out the way it did."""

    verdict: Verdict
    manifest_digest: str
    order: tuple
    reports: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "schema_version": 1,
            "manifest_sha256": self.manifest_digest,
            "execution_order": list(self.order),
            "upstream_commit": UPSTREAM_COMMIT,
            "scorer_policy_hash": policy_hash(),
            **self.verdict.as_dict(),
        }


def pool_job(manifest: EpochManifest, pool: str) -> str:
    """The JSON one room receives for one pool: the pool name and its fifteen tasks.

    ``deep`` rides on the job alongside each descriptor rather than inside it. The room needs it to
    score; the agent must never see it, and :meth:`as_agent_input` is what strips it.
    """
    tasks = manifest.tasks_by_pool().get(pool, ())
    return json.dumps({
        "pool": pool,
        "manifest_sha256": manifest.digest(),
        "tasks": [
            {**task.as_agent_input(), "deep": task.task_id in manifest.deep_task_ids}
            for task in tasks
        ],
    }, sort_keys=True, separators=(",", ":"))


def report_from_room(*, pool: str, contestant: str, manifest: EpochManifest,
                     bundle_sha256: str, outcome) -> PoolReport:
    """One room answer into one pool report.

    ``outcome`` is whatever the room client returned. A rejected attestation raises rather than
    producing a zero-scoring report: an unverifiable answer is not a bad answer.
    """
    if not getattr(outcome, "accepted", False):
        raise DuelDeferred(
            f"{contestant}'s {pool} report was not accepted: "
            f"{getattr(outcome, 'reason', 'unverified')}")

    report = getattr(outcome, "report", None)
    if not isinstance(report, dict):
        raise DuelDeferred(f"{contestant}'s {pool} room returned no report")
    if report.get("pool") != pool:
        raise DuelDeferred(
            f"{contestant}'s room answered for pool {report.get('pool')!r}, not {pool!r}")

    provenance = getattr(outcome, "provenance", None) or {}
    agent_image = str(provenance.get("agent_image") or "")

    raw_result = report.get("pool_result")
    credentials = _credentials(report.get("credential_status"))

    if raw_result is None:
        # The room ran and could not score. Which kind of failure it was is the credential report's
        # to say: a contestant fault zeroes it, anything else defers the whole duel.
        status = (ReportStatus.CREDENTIAL_FAILURE if credentials.contestant_at_fault
                  else ReportStatus.INFRASTRUCTURE_FAILURE)
        pool_result = None
    else:
        status = ReportStatus.OK
        pool_result = PoolResult(
            q_gate=float(raw_result["q_gate"]), q_weight=float(raw_result["q_weight"]),
            volume=int(raw_result["volume"]), deep_count=int(raw_result["deep_count"]))

    return PoolReport(
        pool=pool, status=status, contestant=contestant,
        bundle_sha256=bundle_sha256,
        task_manifest_sha256=manifest.digest(),
        policy_hash=policy_hash(),
        agent_image_digest=agent_image,
        pool_result=pool_result,
        credentials=credentials,
        usage=dict(provenance.get("inference_summary") or {}),
        detail=str(report.get("stderr_tail") or "")[:2000],
    )


def _credentials(raw) -> CredentialReport:
    if not isinstance(raw, dict) or not raw:
        # Nothing said means nothing observed, not "everything was fine".
        return CredentialReport({})
    statuses = {}
    for provider, value in raw.items():
        try:
            statuses[str(provider)] = CredentialStatus(value)
        except ValueError:
            statuses[str(provider)] = CredentialStatus.INVALID
    return CredentialReport(statuses)


def run_duel(*, manifest: EpochManifest, king: Contestant, challenger: Contestant,
             run_pool, challenge_id: str) -> DuelRecord:
    """Run all eight pool jobs and decide the duel.

    ``run_pool(contestant, pool, job)`` sends one job to a room and returns its verified outcome.
    Injected rather than built here so this function can be driven without a room -- and so the
    attestation policy stays in one place (``kata_sn22.execution.tee_room``) rather than being
    partly restated here.

    One contestant's four pools run before the other's, in a deterministic randomised order, so a
    provider having a bad ten minutes does not always land on the same side. Nothing about that
    order reaches the arithmetic; :mod:`kata_sn22.paired_scoring` is a dict lookup by UID.
    """
    order = paired_scoring.execution_order(challenge_id=challenge_id)
    sides = {KING: king, CHALLENGER: challenger}
    collected: dict = {KING: [], CHALLENGER: []}

    for label in order:
        contestant = sides[label]
        for pool in POOLS:
            outcome = run_pool(contestant, pool, pool_job(manifest, pool))
            collected[label].append(report_from_room(
                pool=pool, contestant=label, manifest=manifest,
                bundle_sha256=contestant.bundle_sha256, outcome=outcome))

    verdict = paired_scoring.decide(
        king_reports=collected[KING], challenger_reports=collected[CHALLENGER])
    return DuelRecord(
        verdict=verdict,
        manifest_digest=manifest.digest(),
        order=order,
        reports={label: [report.as_dict() for report in reports]
                 for label, reports in collected.items()},
    )
