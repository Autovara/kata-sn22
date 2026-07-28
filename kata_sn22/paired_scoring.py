"""The duel: eight attested pool reports in, one promotion decision out.

**One ranking signal.** ``sn22_combined_score``, higher wins. It is upstream's
``combine_pool_scores`` over both contestants' four pool tuples, and nothing else. Kata's
seven-signal lexicographic comparator — valid rate, then quality, then citation precision, then
coverage, then invalid runs, then cost, then latency — was calibration machinery. Using it would
mean a King defended its crown on a ranking upstream never described, decided by a tie-break order
Kata chose.

**Why the pools are normalised together.** ``combine_pool_scores`` divides each contestant's raw
pool score by the pool's *total across contestants*. Scoring the King alone and the Challenger
alone and comparing the two numbers would be a different arithmetic with a different answer. So the
two are combined in one call, as virtual UIDs 0 and 1.

**Order cannot matter.** The rooms run one contestant's four pools before the other's, in a
deterministic randomised order, so that a provider having a bad ten minutes does not always land on
the same side. The scoring is a dict lookup by UID, so nothing about that order reaches the
arithmetic — and :func:`decide` is tested by reversing both the input order and the execution order.

**A tie keeps the King.** Strictly greater promotes. That is not a preference about close calls: a
challenger that merely matched the King has not shown it is better, and the cost of a wrong
promotion is a worse King defending every future round.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from kata_sn22.credentials_v2 import CredentialReport
from kata_sn22.neuron_adapter import CHALLENGER_UID, KING_UID
from kata_sn22.report_v2 import POOLS, ContestantEpoch, ReportError, ReportStatus, assemble_epoch

#: The one signal a promotion is decided on. Higher wins.
RANK_SIGNAL = "sn22_combined_score"

KING = "king"
CHALLENGER = "challenger"


class DuelDeferred(Exception):
    """The duel cannot be decided, and no winner may be recorded.

    Distinct from "the challenger lost" on purpose. A deferred duel is re-run; a wrongly-decided one
    crowns the wrong King and every later round inherits it.
    """


@dataclass(frozen=True)
class Verdict:
    """Who won, on what number, and what each contestant's pools actually produced."""

    king_score: float
    challenger_score: float
    king_status: ReportStatus
    challenger_status: ReportStatus
    pools: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    @property
    def challenger_promotes(self) -> bool:
        """Strictly greater. A tie keeps the King."""
        return self.challenger_score > self.king_score

    def as_dict(self) -> dict:
        return {
            "rank_signal": RANK_SIGNAL,
            "king_score": self.king_score,
            "challenger_score": self.challenger_score,
            "king_status": self.king_status.value,
            "challenger_status": self.challenger_status.value,
            "challenger_promotes": self.challenger_promotes,
            "pools": dict(sorted(self.pools.items())),
            "diagnostics": dict(sorted(self.diagnostics.items())),
        }


def execution_order(*, challenge_id: str) -> tuple:
    """Which contestant's four pools run first. Deterministic, and derived from the challenge.

    Randomised so a provider having a bad ten minutes does not always land on the same contestant;
    deterministic so the order is reproducible from the record rather than being a thing that
    happened. Nothing about it reaches the arithmetic -- see the module docstring.
    """
    digest = hashlib.sha256(f"kata-sn22-order|{challenge_id}".encode("utf-8")).digest()
    return (KING, CHALLENGER) if digest[0] % 2 == 0 else (CHALLENGER, KING)


def assemble(reports, *, contestant: str) -> ContestantEpoch:
    """Four pool reports into one contestant's epoch, or raise.

    A thin wrapper so callers do not have to know that the identity checks -- same bundle, same
    manifest, same policy, same upstream, same image across all four -- live in
    :mod:`kata_sn22.report_v2`.
    """
    return assemble_epoch(reports, contestant=contestant)


def combine_epochs(king: ContestantEpoch, challenger: ContestantEpoch) -> dict:
    """Upstream's ``combine_pool_scores``, called ONCE with both contestants.

    Not once per contestant. The function normalises within each pool by the total across UIDs, so
    two separate calls would each divide by their own total and produce two numbers that cannot be
    compared -- both would come out at the pool's full share.
    """
    from kata_sn22 import production_scorer

    king_pools = king.pool_results()
    challenger_pools = challenger.pool_results()

    qualities: dict = {}
    for pool in POOLS:
        entry: dict = {}
        if pool in king_pools:
            entry[KING_UID] = king_pools[pool]
        if pool in challenger_pools:
            entry[CHALLENGER_UID] = challenger_pools[pool]
        if entry:
            qualities[production_scorer.pool_share_key(pool)] = entry

    scheduler = production_scorer.upstream_module(
        "neurons.validators.scoring.query_scheduler")
    return scheduler.combine_pool_scores(qualities)


def decide(*, king_reports, challenger_reports) -> Verdict:
    """The whole promotion decision.

    Raises :class:`DuelDeferred` when the duel cannot be decided at all -- a missing or
    unverifiable report, pools that disagree on the rules they were scored under, or an
    infrastructure failure in any pool on either side. Everything else is a score, including zero.
    """
    try:
        king = assemble(king_reports, contestant=KING)
        challenger = assemble(challenger_reports, contestant=CHALLENGER)
    except ReportError as exc:
        # Malformed or mismatched reports are not a contestant losing -- they are the room failing
        # to say what happened, and a promotion decided on that is decided on nothing.
        raise DuelDeferred(f"the pool reports do not describe one duel: {exc}") from exc

    _require_same_rules(king, challenger)

    for epoch in (king, challenger):
        if epoch.status is ReportStatus.INFRASTRUCTURE_FAILURE:
            raise DuelDeferred(
                f"{epoch.contestant} had an infrastructure failure; nobody's answers were "
                f"established, and a score assigned on that basis would be a number with no "
                f"evidence behind it")

    # A credential failure is the contestant's own and scores it ZERO -- it does not defer. Under
    # the miner-funded rule, a contestant that cannot pay for its own evaluation has not been
    # evaluated, and the King is not made to wait for it.
    scores = combine_epochs(king, challenger)
    return Verdict(
        king_score=_score_for(king, scores, KING_UID),
        challenger_score=_score_for(challenger, scores, CHALLENGER_UID),
        king_status=king.status,
        challenger_status=challenger.status,
        pools=_pool_table(king, challenger),
        diagnostics=_diagnostics(king, challenger),
    )


def _require_same_rules(king: ContestantEpoch, challenger: ContestantEpoch) -> None:
    """Both sides were asked the same questions and graded by the same rules, or defer."""
    from kata_sn22.report_v2 import duel_is_comparable

    try:
        duel_is_comparable(king, challenger)
    except ReportError as exc:
        raise DuelDeferred(
            f"the two contestants were not graded under the same rules: {exc}") from exc


def _score_for(epoch: ContestantEpoch, scores: dict, uid: int) -> float:
    """A contestant that did not score is a zero, not an absence.

    ``combine_pool_scores`` omits a UID whose raw score summed to nothing. Reporting that as
    "missing" would let a caller treat it as unknown, and unknown is what defers a duel -- so a
    credential failure would silently become a deferral and the King would wait forever.
    """
    if epoch.status is not ReportStatus.OK:
        return 0.0
    return float(scores.get(uid, 0.0))


def _pool_table(king: ContestantEpoch, challenger: ContestantEpoch) -> dict:
    """What each pool produced for each side. Diagnostic only -- never a tie-breaker."""
    king_pools, challenger_pools = king.pool_results(), challenger.pool_results()
    table: dict = {}
    for pool in POOLS:
        table[pool] = {
            KING: list(king_pools.get(pool, ())),
            CHALLENGER: list(challenger_pools.get(pool, ())),
        }
    return table


def _diagnostics(king: ContestantEpoch, challenger: ContestantEpoch) -> dict:
    """Published separately from the ranking signal, and deliberately not comparable.

    Valid rate, cost and latency used to be ranked. They are useful for a reviewer and useless as a
    tie-break: upstream's aggregation already accounts for what it accounts for, and adding a second
    ordering on top would decide duels on a rule nobody upstream wrote down.
    """
    def _for(epoch: ContestantEpoch) -> dict:
        results = epoch.pool_results()
        return {
            "status": epoch.status.value,
            "pools_scored": len(results),
            "deep_samples": sum(result[3] for result in results.values()),
            "tasks": sum(result[2] for result in results.values()),
            "credentials": _credentials(epoch).as_dict(),
        }

    return {KING: _for(king), CHALLENGER: _for(challenger)}


def _credentials(epoch: ContestantEpoch) -> CredentialReport:
    """The worst status per provider across the epoch's four pools.

    Worst rather than last: a Chutes key that failed in one pool and worked in three did fail, and
    a report that showed only the final pool's status would say otherwise.
    """
    merged: dict = {}
    for report in epoch.reports:
        for provider, status in (report.credentials.statuses if report.credentials else {}).items():
            current = merged.get(provider)
            if current is None or _severity(status) > _severity(current):
                merged[provider] = status
    return CredentialReport(merged)


def _severity(status) -> int:
    from kata_sn22.credentials_v2 import CONTESTANT_FAULT_STATUSES, DEFER_STATUSES, CredentialStatus

    if status in DEFER_STATUSES:
        return 3
    if status in CONTESTANT_FAULT_STATUSES:
        return 2
    return 1 if status is not CredentialStatus.OK else 0
