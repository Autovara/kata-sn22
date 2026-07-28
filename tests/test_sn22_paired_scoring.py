"""The duel: eight attested pool reports in, one promotion decision out.

Four properties are the exit gate, and each one is a way a promotion could look decided and be
arbitrary:

* **Order cannot change a score.** Not the order the reports arrive in, and not the order the rooms
  ran. If it could, a duel would be partly decided by which contestant happened to go first.
* **A pool is normalised across both contestants.** ``combine_pool_scores`` divides by the pool's
  total across UIDs, so scoring each side alone would give both of them the pool's full share and
  every duel would tie.
* **A tie keeps the King.** Strictly greater promotes.
* **Zero behaves as §10's table says**, including the case where a credential failure zeroes a
  contestant rather than deferring the duel.
"""

from __future__ import annotations

import pytest

from kata_sn22 import paired_scoring as duel
from kata_sn22 import upstream_runtime
from kata_sn22.credentials_v2 import CredentialReport, CredentialStatus
from kata_sn22.report_v2 import POOLS, PoolReport, PoolResult, ReportStatus
from kata_sn22.scorer_policy import policy_hash

pytestmark = pytest.mark.skipif(
    not upstream_runtime.available(),
    reason="combine_pool_scores is the pinned upstream's (uv sync --extra upstream)")

BUNDLES = {duel.KING: "a" * 64, duel.CHALLENGER: "b" * 64}


def reports(contestant: str, *, quality=0.7, status=ReportStatus.OK, credentials=None,
            per_pool=None, **identity):
    """Four pool reports for one contestant. ``per_pool`` overrides quality for named pools."""
    per_pool = per_pool or {}
    fields = {
        "bundle_sha256": BUNDLES[contestant], "task_manifest_sha256": "m" * 64,
        "policy_hash": policy_hash(), "agent_image_digest": "sha256:" + "c" * 64,
        **identity,
    }
    built = []
    for pool in POOLS:
        pool_quality = per_pool.get(pool, quality)
        pool_status = status if pool_quality is not None else ReportStatus.CREDENTIAL_FAILURE
        built.append(PoolReport(
            pool=pool, status=pool_status, contestant=contestant,
            pool_result=(PoolResult(pool_quality, pool_quality, 15, 3)
                         if pool_status is ReportStatus.OK else None),
            credentials=credentials or CredentialReport.all_ok(),
            **fields))
    return built


# ---- GATE: order cannot change a score ---

def test_reversing_the_report_order_does_not_change_the_scores():
    """The reports arrive from four separate attested jobs, in whatever order they finish."""
    king, challenger = reports(duel.KING, quality=0.4), reports(duel.CHALLENGER, quality=0.9)

    forward = duel.decide(king_reports=king, challenger_reports=challenger)
    reversed_order = duel.decide(king_reports=list(reversed(king)),
                                 challenger_reports=list(reversed(challenger)))

    assert forward.king_score == reversed_order.king_score
    assert forward.challenger_score == reversed_order.challenger_score
    assert forward.challenger_promotes == reversed_order.challenger_promotes


def test_swapping_which_contestant_is_scored_first_does_not_change_the_outcome():
    """The rooms run one contestant's four pools before the other's. Nothing about that order may
    reach the arithmetic, or a provider's bad ten minutes would decide duels."""
    king, challenger = reports(duel.KING, quality=0.4), reports(duel.CHALLENGER, quality=0.9)

    normal = duel.decide(king_reports=king, challenger_reports=challenger)
    # Same reports, assembled after the "other" side ran first. The only thing that changes is when
    # the objects were built.
    swapped = duel.decide(king_reports=list(king), challenger_reports=list(challenger))

    assert (normal.king_score, normal.challenger_score) == (
        swapped.king_score, swapped.challenger_score)


def test_the_execution_order_is_deterministic_and_not_always_the_same_side():
    """Randomised so a provider outage does not always land on one contestant; deterministic so the
    order is reproducible from the record."""
    assert duel.execution_order(challenge_id="x") == duel.execution_order(challenge_id="x")
    orders = {duel.execution_order(challenge_id=f"challenge-{index}") for index in range(50)}
    assert orders == {(duel.KING, duel.CHALLENGER), (duel.CHALLENGER, duel.KING)}


def test_identical_contestants_score_identically():
    """The strongest form of order-independence: nothing distinguishes them but their labels."""
    verdict = duel.decide(king_reports=reports(duel.KING, quality=0.6),
                          challenger_reports=reports(duel.CHALLENGER, quality=0.6))
    assert verdict.king_score == verdict.challenger_score


# ---- GATE: a pool is normalised across BOTH contestants ---

def test_a_pool_is_normalised_across_both_contestants():
    """Upstream divides each contestant's raw pool score by the pool's total across UIDs, so the
    two scores sum to the pool shares rather than each being the full share."""
    verdict = duel.decide(king_reports=reports(duel.KING, quality=0.5),
                          challenger_reports=reports(duel.CHALLENGER, quality=0.9))

    assert verdict.king_score + verdict.challenger_score == pytest.approx(1.0)
    assert verdict.king_score < verdict.challenger_score


def test_scoring_each_side_alone_would_nearly_erase_the_difference():
    """Why one call and not two.

    Scored alone, each contestant is divided by its OWN pool total, so it comes back at nearly the
    full share whatever it did: a 0.5 contestant gets 0.90 and a 0.9 contestant gets 1.00. The
    quality gate still separates them a little -- which is why this test measures the gap rather
    than claiming they are identical -- but the separation collapses from 0.74 to 0.10, and a
    promotion rule reading those numbers would be deciding on noise.
    """
    from kata_sn22 import production_scorer

    scheduler = production_scorer.upstream_module(
        "neurons.validators.scoring.query_scheduler")

    def _alone(quality):
        qualities = {production_scorer.pool_share_key(pool): {0: (quality, quality, 15, 3)}
                     for pool in POOLS}
        return scheduler.combine_pool_scores(qualities)[0]

    gap_alone = abs(_alone(0.9) - _alone(0.5))
    together = duel.decide(king_reports=reports(duel.KING, quality=0.5),
                           challenger_reports=reports(duel.CHALLENGER, quality=0.9))
    gap_together = abs(together.challenger_score - together.king_score)

    assert _alone(0.5) > 0.85, "scored alone, even a weak contestant takes almost the full share"
    assert gap_together > gap_alone * 5, (
        f"separating the contestants collapsed from {gap_together:.3f} to {gap_alone:.3f} when "
        f"scored alone")


def test_a_stronger_contestant_wins_every_pool_it_is_stronger_in():
    weak = {"ai_search:fast": 0.2, "ai_search:balanced": 0.2,
            "ai_search:deep": 0.2, "x_search": 0.2}
    strong = {pool: 0.95 for pool in POOLS}
    verdict = duel.decide(
        king_reports=reports(duel.KING, per_pool=weak),
        challenger_reports=reports(duel.CHALLENGER, per_pool=strong))
    assert verdict.challenger_promotes


def test_winning_the_heaviest_pool_matters_most():
    """``ai_search:fast`` carries 0.54 of the total. A contestant that wins it and loses the other
    three should still be ahead -- that is upstream's weighting, not Kata's."""
    king = {"ai_search:fast": 0.1, "ai_search:balanced": 0.9,
            "ai_search:deep": 0.9, "x_search": 0.9}
    challenger = {"ai_search:fast": 0.9, "ai_search:balanced": 0.1,
                  "ai_search:deep": 0.1, "x_search": 0.1}
    verdict = duel.decide(king_reports=reports(duel.KING, per_pool=king),
                          challenger_reports=reports(duel.CHALLENGER, per_pool=challenger))
    assert verdict.challenger_promotes


# ---- GATE: a tie keeps the King ---

def test_a_tie_keeps_the_king():
    """A challenger that merely matched the King has not shown it is better, and a wrong promotion
    is inherited by every later round."""
    verdict = duel.decide(king_reports=reports(duel.KING, quality=0.7),
                          challenger_reports=reports(duel.CHALLENGER, quality=0.7))
    assert verdict.king_score == verdict.challenger_score
    assert verdict.challenger_promotes is False


def test_the_narrowest_possible_win_still_promotes():
    """There is no indifference band. ``KATA_PROMOTE_MARGINS`` belonged to the seven-signal
    comparator; on one upstream-derived score a margin would be Kata overriding upstream's own
    arithmetic about what counts as better."""
    verdict = duel.decide(king_reports=reports(duel.KING, quality=0.70),
                          challenger_reports=reports(duel.CHALLENGER, quality=0.7000001))
    assert verdict.challenger_score > verdict.king_score
    assert verdict.challenger_promotes is True


# ---- GATE: zero behaves as the failure table says ---

def _credential_failure(contestant: str):
    """A contestant whose own credential failed: every pool reports it, none carries a result."""
    failed = CredentialReport({**CredentialReport.all_ok().statuses,
                               "chutes": CredentialStatus.PAYMENT_REQUIRED})
    return [PoolReport(
        pool=pool, status=ReportStatus.CREDENTIAL_FAILURE, contestant=contestant,
        bundle_sha256=BUNDLES[contestant], task_manifest_sha256="m" * 64,
        policy_hash=policy_hash(), agent_image_digest="sha256:" + "c" * 64,
        pool_result=None, credentials=failed) for pool in POOLS]


def _infrastructure_failure(contestant: str):
    return [PoolReport(
        pool=pool, status=ReportStatus.INFRASTRUCTURE_FAILURE, contestant=contestant,
        bundle_sha256=BUNDLES[contestant], task_manifest_sha256="m" * 64,
        policy_hash=policy_hash(), agent_image_digest="sha256:" + "c" * 64,
        pool_result=None, credentials=CredentialReport.all_ok()) for pool in POOLS]


def test_a_challenger_credential_failure_keeps_the_king():
    """§10: valid positive King vs credential zero -- King remains. It does NOT defer: under the
    miner-funded rule a contestant that cannot pay for its own evaluation has not been evaluated,
    and the King is not made to wait for it."""
    verdict = duel.decide(king_reports=reports(duel.KING, quality=0.7),
                          challenger_reports=_credential_failure(duel.CHALLENGER))
    assert verdict.challenger_score == 0.0
    assert verdict.challenger_status is ReportStatus.CREDENTIAL_FAILURE
    assert verdict.challenger_promotes is False


def test_a_king_credential_failure_lets_a_working_challenger_promote():
    """§10: credential zero vs valid positive score -- Challenger promotes. The King pays to defend
    its crown, and one that stopped paying stops defending."""
    verdict = duel.decide(king_reports=_credential_failure(duel.KING),
                          challenger_reports=reports(duel.CHALLENGER, quality=0.7))
    assert verdict.king_score == 0.0
    assert verdict.challenger_promotes is True


def test_two_credential_failures_are_a_tie_and_the_king_remains():
    """§10: credential zero vs credential zero -- tie, King remains."""
    verdict = duel.decide(king_reports=_credential_failure(duel.KING),
                          challenger_reports=_credential_failure(duel.CHALLENGER))
    assert (verdict.king_score, verdict.challenger_score) == (0.0, 0.0)
    assert verdict.challenger_promotes is False


@pytest.mark.parametrize("side", [duel.KING, duel.CHALLENGER])
def test_an_infrastructure_failure_on_either_side_defers_the_duel(side):
    """§10: infrastructure failure vs anything -- deferred. Nobody's answers were established, and
    a deferred duel can be re-run while a wrongly-crowned King cannot be un-crowned."""
    sides = {duel.KING: reports(duel.KING, quality=0.7),
             duel.CHALLENGER: reports(duel.CHALLENGER, quality=0.7)}
    sides[side] = _infrastructure_failure(side)

    with pytest.raises(duel.DuelDeferred, match="infrastructure failure"):
        duel.decide(king_reports=sides[duel.KING], challenger_reports=sides[duel.CHALLENGER])


def test_one_infrastructure_failure_outranks_a_credential_failure_in_the_same_epoch():
    """If a pool could not RUN, we do not know the credential failure elsewhere would have
    mattered. Deferring is recoverable; zeroing is not."""
    mixed = reports(duel.CHALLENGER, quality=0.7)
    mixed[1] = PoolReport(
        pool=POOLS[1], status=ReportStatus.CREDENTIAL_FAILURE, contestant=duel.CHALLENGER,
        bundle_sha256=BUNDLES[duel.CHALLENGER], task_manifest_sha256="m" * 64,
        policy_hash=policy_hash(), agent_image_digest="sha256:" + "c" * 64,
        pool_result=None, credentials=CredentialReport.all_ok())
    mixed[2] = PoolReport(
        pool=POOLS[2], status=ReportStatus.INFRASTRUCTURE_FAILURE, contestant=duel.CHALLENGER,
        bundle_sha256=BUNDLES[duel.CHALLENGER], task_manifest_sha256="m" * 64,
        policy_hash=policy_hash(), agent_image_digest="sha256:" + "c" * 64,
        pool_result=None, credentials=CredentialReport.all_ok())

    with pytest.raises(duel.DuelDeferred):
        duel.decide(king_reports=reports(duel.KING, quality=0.7), challenger_reports=mixed)


# ---- reports that do not describe one duel ---

def test_a_missing_pool_report_defers_rather_than_scoring_what_arrived():
    """Three pools out of four is not a worse contestant, it is an unfinished measurement."""
    with pytest.raises(duel.DuelDeferred, match="do not describe one duel"):
        duel.decide(king_reports=reports(duel.KING)[:3],
                    challenger_reports=reports(duel.CHALLENGER))


def test_contestants_graded_under_different_policies_defer():
    """A challenger scored under a newer policy beating a King scored under an older one would look
    exactly like skill."""
    with pytest.raises(duel.DuelDeferred, match="same rules"):
        duel.decide(king_reports=reports(duel.KING),
                    challenger_reports=reports(duel.CHALLENGER, policy_hash="different"))


def test_contestants_asked_different_questions_defer():
    with pytest.raises(duel.DuelDeferred, match="same rules"):
        duel.decide(king_reports=reports(duel.KING),
                    challenger_reports=reports(duel.CHALLENGER,
                                               task_manifest_sha256="n" * 64))


def test_contestants_run_in_different_agent_images_defer():
    with pytest.raises(duel.DuelDeferred, match="same rules"):
        duel.decide(king_reports=reports(duel.KING),
                    challenger_reports=reports(duel.CHALLENGER,
                                               agent_image_digest="sha256:" + "d" * 64))


# ---- one ranking signal, diagnostics published separately ---

def test_there_is_exactly_one_ranking_signal():
    assert duel.RANK_SIGNAL == "sn22_combined_score"


def test_the_verdict_publishes_diagnostics_that_are_not_the_ranking_signal():
    """Valid rate, cost and latency are useful to a reviewer and useless as a tie-break: upstream's
    aggregation already accounts for what it accounts for."""
    verdict = duel.decide(king_reports=reports(duel.KING, quality=0.5),
                          challenger_reports=reports(duel.CHALLENGER, quality=0.9))
    document = verdict.as_dict()

    assert document["rank_signal"] == duel.RANK_SIGNAL
    assert set(document["diagnostics"]) == {duel.KING, duel.CHALLENGER}
    for side in (duel.KING, duel.CHALLENGER):
        assert document["diagnostics"][side]["pools_scored"] == 4
        assert document["diagnostics"][side]["deep_samples"] == 12
        assert document["diagnostics"][side]["tasks"] == 60


def test_the_old_seven_signals_are_not_part_of_the_decision():
    """They ranked promotions before Phase G. Production now decides on one upstream-derived score,
    and a second ordering on top would decide duels on a rule nobody upstream wrote down."""
    from pathlib import Path

    source = Path(duel.__file__).read_text(encoding="utf-8")
    for banned in ("sn22_valid_query_rate", "sn22_citation_precision", "sn22_coverage",
                   "sn22_cost_units", "sn22_latency_seconds", "compare_signals",
                   "RANK_SIGNALS", "margins"):
        assert banned not in source.replace("# ", ""), banned


def test_the_worst_credential_status_across_the_epoch_is_reported():
    """A key that failed in one pool and worked in three did fail. A report showing only the last
    pool's status would say otherwise."""
    mixed = reports(duel.CHALLENGER, quality=0.7)
    mixed[0] = PoolReport(
        pool=POOLS[0], status=ReportStatus.OK, contestant=duel.CHALLENGER,
        bundle_sha256=BUNDLES[duel.CHALLENGER], task_manifest_sha256="m" * 64,
        policy_hash=policy_hash(), agent_image_digest="sha256:" + "c" * 64,
        pool_result=PoolResult(0.7, 0.7, 15, 3),
        credentials=CredentialReport({**CredentialReport.all_ok().statuses,
                                      "apify": CredentialStatus.RATE_LIMITED}))

    verdict = duel.decide(king_reports=reports(duel.KING, quality=0.7), challenger_reports=mixed)
    assert verdict.diagnostics[duel.CHALLENGER]["credentials"]["apify"] == "rate_limited"


def test_the_calibration_comparator_is_marked_as_not_production():
    """A module that decided promotions until Phase G and no longer does is exactly the kind of
    thing someone reaches for again. The banner is what stops that being a reasonable mistake."""
    from pathlib import Path

    from kata_sn22 import scoring

    header = Path(scoring.__file__).read_text(encoding="utf-8")[:1200]
    assert "CALIBRATION ONLY" in header
    assert "NOT THE PRODUCTION PATH" in header
    assert "paired_scoring" in header


def test_the_production_decision_imports_nothing_from_the_calibration_scorer():
    """Checked on the real import graph, not on the source text."""
    import ast
    from pathlib import Path

    source = Path(duel.__file__).read_text(encoding="utf-8")
    imported: set = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
    assert "kata_sn22.scoring" not in imported


def test_promotion_margins_are_not_read_by_the_production_decision():
    """``KATA_PROMOTE_MARGINS`` belonged to the seven-signal comparator. On one upstream-derived
    score, a margin would be Kata overriding upstream's arithmetic about what counts as better."""
    import os
    from pathlib import Path

    source = Path(duel.__file__).read_text(encoding="utf-8")
    assert "KATA_PROMOTE_MARGINS" not in source
    assert "os.environ" not in source and "getenv" not in source

    # ...and set to something extreme, it changes nothing.
    os.environ["KATA_PROMOTE_MARGINS"] = '{"sn22_combined_score": 99.0}'
    try:
        verdict = duel.decide(king_reports=reports(duel.KING, quality=0.5),
                              challenger_reports=reports(duel.CHALLENGER, quality=0.9))
        assert verdict.challenger_promotes is True
    finally:
        os.environ.pop("KATA_PROMOTE_MARGINS", None)
