"""Eight attested pool jobs into one promotion decision.

The room is injected, so these tests drive the real splitting, the real report construction and the
real verdict without a sealed room. What is NOT faked is anything that decides a number.

Two properties carry the weight:

* **An unverifiable report defers; it never scores.** If a rejected attestation became a zero,
  anything able to break attestation could choose the winner.
* **The agent never sees which tasks are deep-scored**, even though the room must.
"""

from __future__ import annotations

import json
import types

import pytest

from kata_sn22 import paired_scoring, upstream_runtime
from kata_sn22 import production_challenge as duel
from kata_sn22.epoch_manifest import POOLS, build_epoch
from kata_sn22.question_pool import load_pool
from kata_sn22.report_v2 import ReportStatus

pytestmark = pytest.mark.skipif(
    not upstream_runtime.available(),
    reason="the verdict uses the pinned upstream (uv sync --extra upstream)")

KING_BUNDLE = "a" * 64
CHALLENGER_BUNDLE = "b" * 64
AGENT_IMAGE = "sha256:" + "c" * 64


@pytest.fixture
def manifest():
    return build_epoch(seed="duel", pool=load_pool("development"), production=False)


@pytest.fixture
def contestants():
    return (duel.Contestant(label="king", bundle_path="/k", bundle_sha256=KING_BUNDLE),
            duel.Contestant(label="challenger", bundle_path="/c",
                            bundle_sha256=CHALLENGER_BUNDLE))


def _ok(pool: str, quality: float):
    return types.SimpleNamespace(
        accepted=True, provenance={"agent_image": AGENT_IMAGE},
        report={"pool": pool,
                "pool_result": {"q_gate": quality, "q_weight": quality,
                                "volume": 15, "deep_count": 3},
                "credential_status": {"scrapingdog": "ok", "apify": "ok",
                                      "openai": "unused", "chutes": "ok"}})


def _room(quality_by_bundle: dict, *, record=None):
    def _run_pool(contestant, pool, job):
        if record is not None:
            record.append((contestant.label, pool, job))
        return _ok(pool, quality_by_bundle[contestant.bundle_sha256])
    return _run_pool


# ---- the eight jobs ---

def test_a_duel_is_eight_pool_jobs(manifest, contestants):
    """Four per contestant. Sixty tasks behind one request is one timeout away from losing every
    answer already paid for."""
    king, challenger = contestants
    sent: list = []
    duel.run_duel(manifest=manifest, king=king, challenger=challenger,
                  run_pool=_room({KING_BUNDLE: 0.5, CHALLENGER_BUNDLE: 0.9}, record=sent),
                  challenge_id="x")

    assert len(sent) == 8
    assert {(label, pool) for label, pool, _job in sent} == {
        (label, pool) for label in ("king", "challenger") for pool in POOLS}


def test_one_contestants_pools_all_run_before_the_others(manifest, contestants):
    """So a provider having a bad ten minutes does not straddle both sides of the comparison."""
    king, challenger = contestants
    sent: list = []
    record = duel.run_duel(manifest=manifest, king=king, challenger=challenger,
                           run_pool=_room({KING_BUNDLE: 0.5, CHALLENGER_BUNDLE: 0.9},
                                          record=sent),
                           challenge_id="x")

    labels = [label for label, _pool, _job in sent]
    assert labels[:4] == [record.order[0]] * 4
    assert labels[4:] == [record.order[1]] * 4


def test_each_pool_job_carries_exactly_its_fifteen_tasks(manifest, contestants):
    king, challenger = contestants
    sent: list = []
    duel.run_duel(manifest=manifest, king=king, challenger=challenger,
                  run_pool=_room({KING_BUNDLE: 0.5, CHALLENGER_BUNDLE: 0.9}, record=sent),
                  challenge_id="x")

    for _label, pool, job in sent:
        document = json.loads(job)
        assert document["pool"] == pool
        assert len(document["tasks"]) == 15
        assert document["manifest_sha256"] == manifest.digest()


def test_the_room_is_told_which_tasks_are_deep_but_the_agent_is_not(manifest, contestants):
    """The room needs it to score. An agent that knew would work hardest on exactly those, and the
    20% sample would stop measuring the other 80%."""
    king, _challenger = contestants
    job = json.loads(duel.pool_job(manifest, "ai_search:fast"))

    assert sum(1 for task in job["tasks"] if task["deep"]) == 3
    # ...and `deep` rides ALONGSIDE the descriptor the agent is handed, never inside it.
    for task in manifest.tasks_by_pool()["ai_search:fast"]:
        assert "deep" not in task.as_agent_input()


# ---- an unverifiable report defers ---

def test_a_rejected_attestation_defers_rather_than_scoring_zero(manifest, contestants):
    """If it scored zero, anything able to break attestation could choose the winner."""
    king, challenger = contestants

    def _run_pool(contestant, pool, job):
        if contestant.label == "challenger" and pool == "x_search":
            return types.SimpleNamespace(accepted=False, reason="quote verification failed",
                                         report=None, provenance={})
        return _ok(pool, 0.7)

    with pytest.raises(duel.DuelDeferred, match="not accepted"):
        duel.run_duel(manifest=manifest, king=king, challenger=challenger,
                      run_pool=_run_pool, challenge_id="x")


def test_a_room_answering_for_the_wrong_pool_defers(manifest, contestants):
    """Two reports for one pool and none for another would score a contestant on a pool it never
    ran, at the weight of the pool it did."""
    king, challenger = contestants

    def _run_pool(contestant, pool, job):
        return _ok("ai_search:fast", 0.7)          # always the same pool

    with pytest.raises(duel.DuelDeferred, match="answered for pool"):
        duel.run_duel(manifest=manifest, king=king, challenger=challenger,
                      run_pool=_run_pool, challenge_id="x")


def test_a_room_returning_no_report_defers(manifest, contestants):
    king, challenger = contestants

    def _run_pool(contestant, pool, job):
        return types.SimpleNamespace(accepted=True, report=None, provenance={})

    with pytest.raises(duel.DuelDeferred, match="no report"):
        duel.run_duel(manifest=manifest, king=king, challenger=challenger,
                      run_pool=_run_pool, challenge_id="x")


# ---- what the room could not score ---

def test_a_contestant_fault_becomes_a_credential_failure(manifest):
    """The room ran and could not score. Which kind of failure it was is the credential report's to
    say -- a contestant fault zeroes it."""
    report = duel.report_from_room(
        pool="ai_search:fast", contestant="challenger", manifest=manifest,
        bundle_sha256=CHALLENGER_BUNDLE,
        outcome=types.SimpleNamespace(
            accepted=True, provenance={"agent_image": AGENT_IMAGE},
            report={"pool": "ai_search:fast", "pool_result": None,
                    "credential_status": {"chutes": "payment_required"}}))

    assert report.status is ReportStatus.CREDENTIAL_FAILURE
    assert report.pool_result is None


def test_anything_else_becomes_an_infrastructure_failure(manifest):
    """Nobody's fault and nothing established. Deferring is recoverable; zeroing is not."""
    report = duel.report_from_room(
        pool="ai_search:fast", contestant="challenger", manifest=manifest,
        bundle_sha256=CHALLENGER_BUNDLE,
        outcome=types.SimpleNamespace(
            accepted=True, provenance={"agent_image": AGENT_IMAGE},
            report={"pool": "ai_search:fast", "pool_result": None,
                    "credential_status": {"apify": "provider_outage"}}))

    assert report.status is ReportStatus.INFRASTRUCTURE_FAILURE


def test_a_silent_room_is_an_infrastructure_failure_not_an_all_clear(manifest):
    """Nothing said means nothing observed. Reading an empty credential report as "everything was
    fine" would turn a room that failed quietly into a contestant that scored zero."""
    report = duel.report_from_room(
        pool="x_search", contestant="king", manifest=manifest, bundle_sha256=KING_BUNDLE,
        outcome=types.SimpleNamespace(
            accepted=True, provenance={},
            report={"pool": "x_search", "pool_result": None, "credential_status": {}}))

    assert report.status is ReportStatus.INFRASTRUCTURE_FAILURE


# ---- the decision ---

def test_the_stronger_contestant_promotes(manifest, contestants):
    king, challenger = contestants
    record = duel.run_duel(
        manifest=manifest, king=king, challenger=challenger,
        run_pool=_room({KING_BUNDLE: 0.5, CHALLENGER_BUNDLE: 0.9}), challenge_id="x")

    assert record.verdict.challenger_promotes is True
    assert record.verdict.challenger_score > record.verdict.king_score


def test_the_record_binds_the_manifest_the_policy_and_the_upstream(manifest, contestants):
    """A duel record that did not say what was asked or how it was graded would be a number with no
    way to check it later."""
    king, challenger = contestants
    document = duel.run_duel(
        manifest=manifest, king=king, challenger=challenger,
        run_pool=_room({KING_BUNDLE: 0.5, CHALLENGER_BUNDLE: 0.9}),
        challenge_id="x").as_dict()

    from kata_sn22.scorer_policy import policy_hash
    from kata_sn22.upstream_snapshot import UPSTREAM_COMMIT

    assert document["manifest_sha256"] == manifest.digest()
    assert document["scorer_policy_hash"] == policy_hash()
    assert document["upstream_commit"] == UPSTREAM_COMMIT
    assert document["rank_signal"] == paired_scoring.RANK_SIGNAL


def test_the_execution_order_is_recorded(manifest, contestants):
    king, challenger = contestants
    record = duel.run_duel(
        manifest=manifest, king=king, challenger=challenger,
        run_pool=_room({KING_BUNDLE: 0.7, CHALLENGER_BUNDLE: 0.7}), challenge_id="x")

    assert set(record.order) == {"king", "challenger"}
    assert record.as_dict()["execution_order"] == list(record.order)


def test_both_contestants_run_the_same_manifest(manifest, contestants):
    """The jobs are built from one manifest, so the two sides cannot be asked different things."""
    king, challenger = contestants
    sent: list = []
    duel.run_duel(manifest=manifest, king=king, challenger=challenger,
                  run_pool=_room({KING_BUNDLE: 0.7, CHALLENGER_BUNDLE: 0.7}, record=sent),
                  challenge_id="x")

    by_pool: dict = {}
    for label, pool, job in sent:
        by_pool.setdefault(pool, set()).add(job)
    for pool, jobs in by_pool.items():
        assert len(jobs) == 1, f"the two contestants got different {pool} jobs"
