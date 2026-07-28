"""One whole production round, driven through ``run_challenge``.

This is the test the production wiring exists to satisfy. Everything under it was built and tested
in isolation across Phases E, F and G — the 60-task epoch, the in-room scorer, the paired verdict —
and none of it was reachable from the entry point the platform actually calls. Each piece was
correct and the chain was not, which is the failure this repository keeps producing.

So this drives the real ``run_challenge`` and asserts the whole path:

    60-task epoch  ->  8 pool jobs  ->  verified reports  ->  upstream verdict  ->  promotion

The **room** is faked, because a real one costs money and needs an enclave. Nothing else is: the
epoch is real, the manifest digest is real, the verdict comes from the pinned upstream's own
``combine_pool_scores``, and the published document is the one kata-bot reads.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from kata_sn22 import question_pool as qp
from kata_sn22 import upstream_runtime
from kata_sn22.execution import policy as execution_policy
from kata_sn22.plugin import Sn22AgentError

pytestmark = pytest.mark.skipif(
    not upstream_runtime.available(),
    reason="the verdict comes from the pinned upstream (uv sync --extra upstream)")

AGENT_IMAGE = "sha256:" + "c" * 64


@pytest.fixture
def production(monkeypatch, tmp_path):
    """A lane configured exactly as production is, with the room replaced.

    The development question rows are relabelled as a snapshot so the test does not depend on an
    operator having run ``tools/snapshot_questions.py`` in this checkout.
    """
    import dataclasses

    from kata_sn22 import plugin as plugin_module

    monkeypatch.setenv(execution_policy.EXECUTION_BACKEND_ENV, "tee")
    monkeypatch.setenv("KATA_SN22_VERIFICATION_MODE", "live")
    monkeypatch.setenv("KATA_SN22_ROOM_URL", "https://room.example")
    monkeypatch.setenv("KATA_ROOM_AUTH_SECRET", "secret")

    pool = dataclasses.replace(qp.load_pool("development"), kind=qp.KIND_UPSTREAM_SNAPSHOT)
    monkeypatch.setattr(qp, "load_pool", lambda _name: pool)
    monkeypatch.setattr(plugin_module, "PRODUCTION_QUESTION_POOL", "development")
    return plugin_module.Sn22DesearchPlugin()


def _bundle(root: Path, name: str) -> str:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "agent.py").write_text("# a submission\n", encoding="utf-8")
    # Realistic length: the room refuses ciphertext under 32 bytes, and a fixture that skipped
    # that check would not be exercising the path a real submission takes.
    (directory / "sealed_inference_key").write_text("ab" * 96, encoding="utf-8")
    return str(directory)


def _side(contestant) -> str:
    """``king`` or ``challenger``.

    The challenger's Contestant carries its SUBMISSION ID as a label (``pr-7``), because that is
    what the platform uses to find the pull request. Only the King is labelled by role.
    """
    return "king" if contestant.label == "king" else "challenger"


def _room(quality_by_label: dict, *, record=None, override=None):
    """Answer every pool job as a healthy room would, at a per-contestant quality."""
    def _run(contestant, pool, job):
        if record is not None:
            record.append((_side(contestant), pool, job))
        if override is not None:
            replaced = override(contestant, pool)
            if replaced is not None:
                return replaced
        quality = quality_by_label[_side(contestant)]
        return types.SimpleNamespace(
            accepted=True,
            provenance={"agent_image": AGENT_IMAGE, "inference_summary": {"requests": 4}},
            report={"pool": pool,
                    "pool_result": {"q_gate": quality, "q_weight": quality,
                                    "volume": 15, "deep_count": 3},
                    "credential_status": {"scrapingdog": "ok", "apify": "ok",
                                          "openai": "unused", "chutes": "ok"}})
    return _run


def _challenge(plugin, tmp_path, room, *, king_quality=0.5, challenger_quality=0.9,
               monkeypatch=None):
    monkeypatch.setattr(plugin, "_run_pool_in_room", room)
    return plugin.run_challenge(
        king_agent_path=_bundle(tmp_path, "king"),
        candidates=[("pr-7", _bundle(tmp_path, "challenger"))],
        config={},
        output_root=str(tmp_path / "out"),
        run_id="challenge-e2e",
    )


# ---- THE end-to-end round ---

def test_a_whole_round_runs_and_the_better_contestant_promotes(production, tmp_path, monkeypatch):
    sent: list = []
    result = _challenge(
        production, tmp_path,
        _room({"king": 0.5, "challenger": 0.9}, record=sent), monkeypatch=monkeypatch)

    assert len(sent) == 8, f"expected 8 pool jobs, got {len(sent)}"

    document = production.challenge_result_json(result)
    assert document["challenger"]["beats_king_pairwise"] is True
    assert document["king"]["beats_king_pairwise"] is False


def test_the_round_asks_sixty_questions_split_across_four_pools(production, tmp_path, monkeypatch):
    sent: list = []
    _challenge(production, tmp_path, _room({"king": 0.7, "challenger": 0.7}, record=sent),
               monkeypatch=monkeypatch)

    per_contestant: dict = {}
    for label, pool, job in sent:
        per_contestant.setdefault(label, {})[pool] = len(json.loads(job)["tasks"])
    for label, pools in per_contestant.items():
        assert sorted(pools) == ["ai_search:balanced", "ai_search:deep",
                                 "ai_search:fast", "x_search"], label
        assert sum(pools.values()) == 60, label


def test_both_contestants_are_asked_the_same_questions(production, tmp_path, monkeypatch):
    """The single most important fairness property. If it failed, one side could be handed an
    easier epoch and the score would look like skill."""
    sent: list = []
    _challenge(production, tmp_path, _room({"king": 0.7, "challenger": 0.7}, record=sent),
               monkeypatch=monkeypatch)

    by_pool: dict = {}
    for _label, pool, job in sent:
        by_pool.setdefault(pool, set()).add(job)
    for pool, jobs in by_pool.items():
        assert len(jobs) == 1, f"the two contestants received different {pool} jobs"


def test_the_promotion_uses_the_upstream_combined_score_only(production, tmp_path, monkeypatch):
    """Not the seven calibration signals, and not a margin."""
    from kata_sn22.paired_scoring import RANK_SIGNAL

    result = _challenge(production, tmp_path, _room({"king": 0.5, "challenger": 0.9}),
                        monkeypatch=monkeypatch)
    document = production.challenge_result_json(result)

    for side in ("king", "challenger"):
        names = [signal["name"] for signal in document[side]["rank_signals"]]
        assert names == [RANK_SIGNAL], f"{side} publishes {names}"

    king = document["king"]["rank_signals"][0]["value"]
    challenger = document["challenger"]["rank_signals"][0]["value"]
    assert challenger > king
    assert king + challenger == pytest.approx(1.0), "pools were not normalised across both sides"


def test_a_tie_keeps_the_king(production, tmp_path, monkeypatch):
    result = _challenge(production, tmp_path, _room({"king": 0.7, "challenger": 0.7}),
                        monkeypatch=monkeypatch)
    document = production.challenge_result_json(result)

    assert document["challenger"]["beats_king_pairwise"] is False
    assert result.outcome.winner is None


def test_the_published_document_binds_the_manifest_and_the_policy(production, tmp_path,
                                                                  monkeypatch):
    """A published score with no record of what was asked or how it was graded cannot be checked
    later, which is the entire point of publishing it."""
    from kata_sn22.scorer_policy import policy_hash
    from kata_sn22.upstream_snapshot import UPSTREAM_COMMIT

    result = _challenge(production, tmp_path, _room({"king": 0.5, "challenger": 0.9}),
                        monkeypatch=monkeypatch)
    document = production.challenge_result_json(result)

    assert document["upstream_commit"] == UPSTREAM_COMMIT
    assert document["scorer_policy_hash"] == policy_hash()
    assert len(document["task_manifest_sha256"]) == 64
    assert document["benchmark_identity"]


# ---- failure behaves as section 10 says ---

def test_a_challenger_credential_failure_scores_zero_and_keeps_the_king(production, tmp_path,
                                                                        monkeypatch):
    def _fail(contestant, pool):
        if _side(contestant) != "challenger":
            return None
        return types.SimpleNamespace(
            accepted=True, provenance={"agent_image": AGENT_IMAGE},
            report={"pool": pool, "pool_result": None,
                    "credential_status": {"chutes": "payment_required"}})

    result = _challenge(production, tmp_path,
                        _room({"king": 0.7, "challenger": 0.7}, override=_fail),
                        monkeypatch=monkeypatch)
    document = production.challenge_result_json(result)

    assert document["challenger"]["rank_signals"][0]["value"] == 0.0
    assert document["challenger"]["beats_king_pairwise"] is False
    assert document["challenger"]["failure_category"] == "credential_payment_required"


def test_a_king_credential_failure_lets_a_working_challenger_promote(production, tmp_path,
                                                                     monkeypatch):
    """The King pays to defend its crown. One that stopped paying stops defending."""
    def _fail(contestant, pool):
        if _side(contestant) != "king":
            return None
        return types.SimpleNamespace(
            accepted=True, provenance={"agent_image": AGENT_IMAGE},
            report={"pool": pool, "pool_result": None,
                    "credential_status": {"chutes": "unauthorized"}})

    result = _challenge(production, tmp_path,
                        _room({"king": 0.7, "challenger": 0.7}, override=_fail),
                        monkeypatch=monkeypatch)
    document = production.challenge_result_json(result)

    assert document["king"]["rank_signals"][0]["value"] == 0.0
    assert document["challenger"]["beats_king_pairwise"] is True


def test_an_infrastructure_failure_defers_rather_than_deciding(production, tmp_path, monkeypatch):
    """Nobody's answers were established. A promotion decided on that would be decided on
    nothing."""
    def _outage(contestant, pool):
        if pool != "x_search":
            return None
        return types.SimpleNamespace(
            accepted=True, provenance={"agent_image": AGENT_IMAGE},
            report={"pool": pool, "pool_result": None,
                    "credential_status": {"apify": "provider_outage"}})

    with pytest.raises(Sn22AgentError, match="deferred"):
        _challenge(production, tmp_path,
                   _room({"king": 0.7, "challenger": 0.7}, override=_outage),
                   monkeypatch=monkeypatch)


def test_a_rejected_attestation_defers_rather_than_scoring_zero(production, tmp_path, monkeypatch):
    """If it scored zero, anything able to break attestation could choose the winner."""
    def _reject(contestant, pool):
        if _side(contestant) == "challenger" and pool == "ai_search:deep":
            return types.SimpleNamespace(accepted=False, reason="quote verification failed",
                                         report=None, provenance={})
        return None

    with pytest.raises(Sn22AgentError, match="deferred"):
        _challenge(production, tmp_path,
                   _room({"king": 0.7, "challenger": 0.7}, override=_reject),
                   monkeypatch=monkeypatch)


# ---- the old path is gone from production ---

def test_production_never_falls_back_to_the_calibration_path(production, tmp_path, monkeypatch):
    """The six-query pool and the seven-signal comparator are calibration machinery. A production
    round that quietly used either would look exactly like a working round."""
    from kata_sn22 import fixtures

    def _refuse(*_args, **_kwargs):
        raise AssertionError("production reached the calibration question pool")

    monkeypatch.setattr(fixtures, "calibration_manifest", _refuse)
    result = _challenge(production, tmp_path, _room({"king": 0.5, "challenger": 0.9}),
                        monkeypatch=monkeypatch)
    assert production.challenge_result_json(result)["challenger"]["beats_king_pairwise"] is True


def test_the_production_card_carries_no_calibration_signal(production, tmp_path, monkeypatch):
    result = _challenge(production, tmp_path, _room({"king": 0.5, "challenger": 0.9}),
                        monkeypatch=monkeypatch)
    document = production.challenge_result_json(result)

    published = json.dumps(document)
    for calibration in ("sn22_valid_query_rate", "sn22_citation_precision", "sn22_cost_units",
                        "sn22_latency_seconds"):
        assert calibration not in published, calibration


def test_beats_king_uses_the_duel_verdict_not_the_margins(production, tmp_path, monkeypatch):
    """``beats_king`` used to apply ``PROMOTION_MARGINS`` to seven signals. On one
    upstream-derived score a margin would be Kata overriding upstream's arithmetic."""
    result = _challenge(production, tmp_path, _room({"king": 0.70, "challenger": 0.7000001}),
                        monkeypatch=monkeypatch)
    ranked = result.outcome.ranked
    assert production.beats_king(ranked[0].card, result.outcome.king.card) is True
