"""The adapter as the lane actually uses it (SN22-5).

`test_sn22_parity.py` proves the adapter agrees with the pinned upstream. This proves the lane wires
it up correctly: that ``sn22_weighted_quality`` really is the upstream reward, that the Kata-to-
upstream translation preserves what each check was written to catch, and that the two components
Kata excludes are excluded on purpose and said so out loud.
"""
from __future__ import annotations

import pytest

from kata_sn22 import fixtures, scoring
from kata_sn22 import upstream_adapter as adapter
from kata_sn22.manifests import UsageManifest, UsageRecord
from kata_sn22.protocol import parse_task_output
from kata_sn22.scoring import TaskAttempt, score_attempts

OBSERVED = {"weak": 3.0, "medium": 2.0, "strong": 1.0, "invalid": 5.0, "malicious": 4.0}


@pytest.fixture
def world():
    manifest = fixtures.calibration_manifest(count=4)
    snapshot = fixtures.calibration_snapshot(manifest)
    return fixtures.tasks_for(manifest), snapshot


def _attempts(kind, tasks, snapshot):
    attempts = []
    for task, raw in zip(tasks, fixtures.reference_responses(kind, tasks, snapshot), strict=True):
        try:
            attempts.append(TaskAttempt(task=task, output=parse_task_output(raw, task=task),
                                        observed_seconds=OBSERVED[kind]))
        except Exception as exc:                       # noqa: BLE001 - classified below
            attempts.append(TaskAttempt(task=task, error=exc.error_class,
                                        observed_seconds=OBSERVED[kind]))
    return attempts


def _usage(tasks):
    return UsageManifest(challenge_id="c", records=tuple(
        UsageRecord("v", task.task_id, 1, 250, 0.002) for task in tasks))


def _signals(kind, tasks, snapshot):
    return score_attempts(_attempts(kind, tasks, snapshot), snapshot=snapshot,
                          usage=_usage(tasks), variant="v")


# ---- the weights are the pinned ones, not a second copy----------------------------------------

def test_scoring_reexports_the_adapter_tables_rather_than_restating_them():
    assert scoring.SEARCH_TYPE_WEIGHTS is adapter.SEARCH_TYPE_WEIGHTS
    assert scoring.AI_MODE_WEIGHTS is adapter.AI_MODE_WEIGHTS
    assert scoring.AI_QUALITY_WEIGHTS == {"content_relevance": 0.60, "summary_relevance": 0.40}


def test_task_weight_is_the_upstream_pool_share(world):
    tasks, _snapshot = world
    for task in tasks:
        weight = scoring._task_weight(task)
        if task.search_type == "x_search":
            assert weight == 0.10
        else:
            assert weight == pytest.approx(0.90 * adapter.AI_MODE_WEIGHTS[task.ai_mode])


# ---- the quality signal is the upstream reward-------------------------------------------------

def test_the_ladder_still_orders_on_the_upstream_reward(world):
    tasks, snapshot = world
    weak = _signals("weak", tasks, snapshot)
    medium = _signals("medium", tasks, snapshot)
    strong = _signals("strong", tasks, snapshot)
    assert weak.sn22_weighted_quality < medium.sn22_weighted_quality < strong.sn22_weighted_quality
    # A submission that returns everything asked for, all of it relevant, takes no penalty at all.
    assert strong.sn22_weighted_quality == pytest.approx(1.0)


def test_a_strong_submission_takes_no_upstream_penalty(world):
    tasks, snapshot = world
    detail = _signals("strong", tasks, snapshot).detail
    for row in detail["per_task"]:
        assert row["penalties"] == {}, row


def test_a_short_result_list_takes_the_count_penalty(world):
    """Upstream's count penalty, reached through the Kata protocol's own ``max_results``."""
    tasks, snapshot = world
    rows = _signals("medium", tasks, snapshot).detail["per_task"]
    assert all(row["penalties"]["count_penalty"] > 0 for row in rows)
    # One of five requested results.
    assert rows[0]["penalties"]["count_penalty"] == pytest.approx(0.8)


def test_a_summary_with_no_citations_takes_the_structure_penalty(world):
    """Kata's citations ARE the upstream summary's links; a summary with neither is penalised."""
    tasks, snapshot = world
    rows = [row for row in _signals("weak", tasks, snapshot).detail["per_task"]
            if row["search_type"] == "ai_search"]
    assert rows
    assert all(row["penalties"]["summary_structure_penalty"] == 1.0 for row in rows)


def test_citing_a_document_it_never_returned_is_an_unsourced_link(world):
    """The malicious fixture cites the answers without retrieving them. Two independent checks must
    catch it: Kata's citation precision, and the upstream summary-structure penalty."""
    tasks, snapshot = world
    malicious = _signals("malicious", tasks, snapshot)
    assert malicious.sn22_citation_precision == 0.0
    # Only AI search carries a summary; X search has no summary component upstream, which is why
    # the citation-precision signal is the check that covers BOTH search types.
    rows = [row for row in malicious.detail["per_task"] if row["search_type"] == "ai_search"]
    assert rows
    assert all(row["penalties"]["summary_structure_penalty"] == 1.0 for row in rows)
    assert malicious.sn22_weighted_quality == 0.0


def test_an_invalid_run_is_not_sent_through_the_upstream_components(world):
    """A run with no output has no response shape; a penalty for one would be a fiction."""
    tasks, snapshot = world
    rows = _signals("invalid", tasks, snapshot).detail["per_task"]
    assert rows and all("penalties" not in row for row in rows)
    assert all(row["reward"] == 0.0 and row["reason"] for row in rows)


# ---- what Kata excludes, and why---------------------------------------------------------------

def test_excluded_components_are_declared_in_the_result(world):
    tasks, snapshot = world
    detail = _signals("strong", tasks, snapshot).detail
    assert detail["upstream_penalties_excluded"] == ["timeout_penalty",
                                                     "min_realistic_time_penalty"]
    assert detail["upstream_performance_multiplier_applied"] is False
    assert detail["upstream_commit"] == adapter_commit()


def adapter_commit() -> str:
    from kata_sn22.upstream_snapshot import UPSTREAM_COMMIT

    return UPSTREAM_COMMIT


def test_applied_and_excluded_penalties_partition_the_upstream_set():
    """No penalty may be silently in neither list — that is how a check disappears unnoticed."""
    declared = set(scoring.KATA_APPLICABLE_PENALTIES) | set(scoring.KATA_EXCLUDED_PENALTIES)
    assert declared == set(adapter.PENALTY_FUNCTIONS)
    assert not (set(scoring.KATA_APPLICABLE_PENALTIES) & set(scoring.KATA_EXCLUDED_PENALTIES))


def test_latency_does_not_leak_into_the_quality_signal(world):
    """Latency is signal 7. If it also moved signal 2, a fast agent would outrank a better one."""
    tasks, snapshot = world
    fast = score_attempts(
        [TaskAttempt(task=a.task, output=a.output, observed_seconds=0.01)
         for a in _attempts("strong", tasks, snapshot)],
        snapshot=snapshot, usage=_usage(tasks), variant="v")
    slow = score_attempts(
        [TaskAttempt(task=a.task, output=a.output, observed_seconds=119.0)
         for a in _attempts("strong", tasks, snapshot)],
        snapshot=snapshot, usage=_usage(tasks), variant="v")
    assert fast.sn22_weighted_quality == slow.sn22_weighted_quality
    assert fast.sn22_latency_seconds < slow.sn22_latency_seconds


def test_the_default_adapter_path_still_applies_everything():
    """The narrowing is a caller's choice. The default — the one under parity — is untouched."""
    response = adapter.UpstreamResponse(
        kind="ai_search", mode="fast", count=10, tools=("Web Search",),
        search_results=(), texts={}, process_time=0.1, max_execution_time=5, timeout=12.0)
    full = adapter.score_response(response, (0.9, 0.9))
    assert set(full.penalties) == set(adapter.AI_PENALTIES)
    assert full.perf_multiplier < 1.0

    narrowed = adapter.score_response(response, (0.9, 0.9),
                                      penalty_names=scoring.KATA_APPLICABLE_PENALTIES,
                                      apply_performance=False)
    assert "timeout_penalty" not in narrowed.penalties
    assert narrowed.perf_multiplier == 1.0


def test_a_narrowing_cannot_add_a_penalty_the_search_type_lacks():
    """X search's sort-order penalty on an AI response would be a check on an absent field."""
    response = adapter.UpstreamResponse(kind="ai_search", mode="fast", count=1,
                                        tools=("Web Search",),
                                        search_results=({"title": "t", "link": "https://a.test/1",
                                                         "snippet": "s"},),
                                        texts={"summary": "**x** [t](https://a.test/1)"},
                                        process_time=3.0, max_execution_time=5, timeout=12.0)
    score = adapter.score_response(response, (0.5, 0.5), penalty_names=("sort_order_penalty",))
    assert score.penalties == {}


# ---- the translation seam----------------------------------------------------------------------

def test_snapshot_urls_are_stable_and_inert():
    assert scoring.snapshot_url("doc-a") == "sn22-snapshot://doc-a"
    # Lowercase and slash-free, so upstream's URL normalization is a no-op rather than a surprise.
    assert adapter.normalize_source_url(scoring.snapshot_url("doc-a")) == "sn22-snapshot://doc-a"


def test_x_search_tasks_are_scored_on_content_relevance_alone(world):
    tasks, snapshot = world
    x_tasks = [task for task in tasks if task.search_type == "x_search"]
    if not x_tasks:
        pytest.skip("this seed drew no X search task")
    attempts = {a.task.task_id: a for a in _attempts("strong", tasks, snapshot)}
    for task in x_tasks:
        response = scoring._upstream_response(attempts[task.task_id], snapshot=snapshot)
        assert response.kind == "x_search"
        assert adapter.reward_weights_for(response) == (1.0,)
        # Every synthesized tweet must pass the upstream schema check, or the lane would be
        # charging a candidate for the shape of its own translation.
        assert adapter.result_schema_penalty(response) == 0.0


def test_a_links_only_task_drops_the_summary_component(world):
    tasks, snapshot = world
    from dataclasses import replace

    task = replace(tasks[0], result_type="links")
    attempt = TaskAttempt(task=task, observed_seconds=1.0, output=parse_task_output(
        fixtures.reference_responses("strong", [task], snapshot)[0], task=task))
    response = scoring._upstream_response(attempt, snapshot=snapshot)
    assert response.result_type == adapter.RESULT_TYPE_ONLY_LINKS
    assert adapter.reward_weights_for(response) == (1.0, 0.0)
    assert adapter.summary_structure_penalty(response) == 0.0
