"""The adapter as the lane actually uses it (SN22-5).

`test_sn22_parity.py` proves the adapter agrees with the pinned upstream. This proves the lane wires
it up correctly: that ``sn22_weighted_quality`` really is the upstream reward, that the Kata-to-
upstream translation preserves what each check was written to catch, and that the two components
Kata excludes are excluded on purpose and said so out loud.
"""
from __future__ import annotations

import random

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
    return fixtures.tasks_for(manifest), _plugin()


def _plugin():
    from kata_sn22.fetch import RecordedPages
    from kata_sn22.plugin import Sn22DesearchPlugin

    recorded = fixtures.recorded_tweets()
    return Sn22DesearchPlugin(
        page_transport=RecordedPages(records=fixtures.recorded_pages()),
        judge_client=fixtures.scripted_judge(),
        tweet_scraper=lambda ids: {tid: recorded[tid] for tid in ids if tid in recorded})


def _attempts(kind, tasks, plugin):
    """Parsed AND verified: scoring reads only what the validator established for itself, so an
    attempt that skipped verification would score zero quality and look like a bad answer."""
    attempts = []
    for task, raw in zip(tasks, fixtures.reference_responses(kind, tasks), strict=True):
        try:
            attempt = TaskAttempt(task=task, output=parse_task_output(raw, task=task),
                                  observed_seconds=OBSERVED[kind])
        except Exception as exc:                       # noqa: BLE001 - classified below
            attempt = TaskAttempt(task=task, error=exc.error_class,
                                  observed_seconds=OBSERVED[kind])
        attempts.append(plugin._verified(attempt))
    return attempts


def _usage(tasks):
    return UsageManifest(challenge_id="c", records=tuple(
        UsageRecord("v", task.task_id, 1, 250, 0.002) for task in tasks))


def _signals(kind, tasks, plugin):
    return score_attempts(_attempts(kind, tasks, plugin),
                          usage=_usage(tasks), variant="v")


# ---- the weights are the pinned ones, not a second copy----------------------------------------

def test_scoring_reexports_the_adapter_tables_rather_than_restating_them():
    assert scoring.SEARCH_TYPE_WEIGHTS is adapter.SEARCH_TYPE_WEIGHTS
    assert scoring.AI_MODE_WEIGHTS is adapter.AI_MODE_WEIGHTS
    assert scoring.AI_QUALITY_WEIGHTS == {"content_relevance": 0.60, "summary_relevance": 0.40}


def test_task_weight_is_the_upstream_pool_share(world):
    tasks, _plugin = world
    for task in tasks:
        weight = scoring._task_weight(task)
        if task.search_type == "x_search":
            assert weight == 0.10
        else:
            assert weight == pytest.approx(0.90 * adapter.AI_MODE_WEIGHTS[task.ai_mode])


# ---- the quality signal is the upstream reward-------------------------------------------------

def test_the_ladder_still_orders_on_the_upstream_reward(world):
    tasks, plugin = world
    weak = _signals("weak", tasks, plugin)
    medium = _signals("medium", tasks, plugin)
    strong = _signals("strong", tasks, plugin)
    assert weak.sn22_weighted_quality < medium.sn22_weighted_quality < strong.sn22_weighted_quality
    # NOT asserted to be 1.0. A perfect score is no longer reachable, and that is the point: quality
    # is now a judge's verdict on a SAMPLE of verified sources, so it reflects what a grader thought
    # of real excerpts rather than recall against an answer key we wrote.
    assert 0.0 < strong.sn22_weighted_quality <= 1.0


def test_a_strong_submission_takes_no_upstream_penalty(world):
    tasks, plugin = world
    detail = _signals("strong", tasks, plugin).detail
    for row in detail["per_task"]:
        assert row["penalties"] == {}, row


def test_a_short_result_list_takes_the_count_penalty(world):
    """Upstream's count penalty, reached through the Kata protocol's own ``max_results``."""
    tasks, plugin = world
    rows = [row for row in _signals("weak", tasks, plugin).detail["per_task"]
            if "count_penalty" in row["penalties"]]
    assert rows, "the weak reference must return fewer results than were asked for"
    # One of five requested results.
    assert rows[0]["penalties"]["count_penalty"] == pytest.approx(0.8)


def test_a_summary_with_no_citations_takes_the_structure_penalty(world):
    """Kata's citations ARE the upstream summary's links; a summary with neither is penalised."""
    tasks, plugin = world
    rows = [row for row in _signals("weak", tasks, plugin).detail["per_task"]
            if row["search_type"] == "ai_search"]
    assert rows
    assert all(row["penalties"]["summary_structure_penalty"] == 1.0 for row in rows)


def test_citing_a_document_it_never_returned_is_an_unsourced_link(world):
    """The malicious fixture cites the answers without retrieving them. Two independent checks must
    catch it: Kata's citation precision, and the upstream summary-structure penalty."""
    tasks, plugin = world
    malicious = _signals("malicious", tasks, plugin)
    assert malicious.sn22_citation_precision == 0.0
    # Only AI search carries a summary; X search has no summary component upstream, which is why
    # the citation-precision signal is the check that covers BOTH search types.
    rows = [row for row in malicious.detail["per_task"] if row["search_type"] == "ai_search"]
    assert rows
    assert all(row["penalties"]["summary_structure_penalty"] == 1.0 for row in rows)
    assert malicious.sn22_weighted_quality == 0.0


def test_an_invalid_run_is_not_sent_through_the_upstream_components(world):
    """A run with no output has no response shape; a penalty for one would be a fiction."""
    tasks, plugin = world
    rows = _signals("invalid", tasks, plugin).detail["per_task"]
    assert rows and all("penalties" not in row for row in rows)
    assert all(row["reward"] == 0.0 and row["reason"] for row in rows)


# ---- what Kata excludes, and why---------------------------------------------------------------

def test_excluded_components_are_declared_in_the_result(world):
    tasks, plugin = world
    detail = _signals("strong", tasks, plugin).detail
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
    tasks, plugin = world
    fast = score_attempts(
        [TaskAttempt(task=a.task, output=a.output, observed_seconds=0.01)
         for a in _attempts("strong", tasks, plugin)],
        usage=_usage(tasks), variant="v")
    slow = score_attempts(
        [TaskAttempt(task=a.task, output=a.output, observed_seconds=119.0)
         for a in _attempts("strong", tasks, plugin)],
        usage=_usage(tasks), variant="v")
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

def test_a_result_reaches_the_upstream_components_as_its_own_live_link() -> None:
    """There is no translation layer any more. A source is identified by the URL the miner returned,
    which is also the URL the validator fetched -- so "did the miner return this source" means the
    same thing on both sides by construction rather than by a mapping that could drift."""
    task = fixtures.tasks_for(fixtures.calibration_manifest())[0]
    attempt = TaskAttempt(task=task, observed_seconds=1.0, output=parse_task_output(
        fixtures.reference_responses("strong", [task])[0], task=task))
    response = scoring._upstream_response(attempt)
    if response.kind == "ai_search":
        links = {item["link"] for item in response.search_results}
        assert links == {result.link for result in attempt.output.results}
        assert all(link.startswith("https://") for link in links)


def test_x_search_tasks_are_scored_on_content_relevance_alone(world):
    tasks, plugin = world
    x_tasks = [task for task in tasks if task.search_type == "x_search"]
    if not x_tasks:
        pytest.skip("this seed drew no X search task")
    attempts = {a.task.task_id: a for a in _attempts("strong", tasks, plugin)}
    for task in x_tasks:
        response = scoring._upstream_response(attempts[task.task_id])
        assert response.kind == "x_search"
        assert adapter.reward_weights_for(response) == (1.0,)
        # Every synthesized tweet must pass the upstream schema check, or the lane would be
        # charging a candidate for the shape of its own translation.
        assert adapter.result_schema_penalty(response) == 0.0


def test_a_links_only_task_drops_the_summary_component(world):
    tasks, plugin = world
    from dataclasses import replace

    task = replace(tasks[0], result_type="links")
    attempt = TaskAttempt(task=task, observed_seconds=1.0, output=parse_task_output(
        fixtures.reference_responses("strong", [task])[0], task=task))
    response = scoring._upstream_response(attempt)
    assert response.result_type == adapter.RESULT_TYPE_ONLY_LINKS
    assert adapter.reward_weights_for(response) == (1.0, 0.0)
    assert adapter.summary_structure_penalty(response) == 0.0


# ---- evidence: what a miner must prove before anything it says is judged -------------------------
#
# Parity proves these agree with the pinned upstream. What parity cannot say is WHY each rule is
# there, and that matters most here: this is the layer a miner has the most to gain from defeating,
# and every rule below corresponds to a specific way of faking a source. A future reader relaxing
# one of them should have to argue with the attack, not just with a test name.

def test_highlights_must_appear_in_order() -> None:
    """A membership check ("are these phrases on the page?") is satisfied by a miner that scrapes a
    page's vocabulary and reassembles it into an excerpt nobody wrote. Requiring order means the
    miner had to read a contiguous span."""
    body = "First the alpha section. Then the gamma section."
    assert adapter.highlights_in_order(["the alpha", "the gamma"], body) is True
    assert adapter.highlights_in_order(["the gamma", "the alpha"], body) is False


def test_one_sentence_cannot_satisfy_two_highlights() -> None:
    """The cursor advances past each match, so a miner cannot pass by quoting one real sentence
    twice and calling it two pieces of evidence."""
    assert adapter.highlights_in_order(["alpha", "alpha"], "alpha appears once") is False
    assert adapter.highlights_in_order(["alpha", "alpha"], "alpha and alpha") is True


def test_matching_survives_how_a_page_is_actually_written() -> None:
    """Fuzzy in the direction that protects the HONEST miner: a page rendering an apostrophe as an
    entity, or breaking a sentence across whitespace, must not fail a correct quote."""
    assert adapter.highlights_in_order(["it's here"], "IT&#39;S     HERE") is True
    assert adapter.highlights_in_order(["café résumé"], "Café, Résumé!") is True


def test_evidence_is_checked_against_both_the_page_and_the_miners_own_answer() -> None:
    """Two different lies, two directions.

    Against the validator's fetched body: the miner did not invent the excerpt.
    Against the miner's own text: the miner actually used what it quoted, rather than pasting real
    excerpts beside an answer written from somewhere else entirely.
    """
    assert adapter.link_meets_evidence(["alpha"], "we found alpha", "the page says alpha") is True
    assert adapter.link_meets_evidence(
        ["alpha"], "unrelated answer", "the page says alpha") is False
    assert adapter.link_meets_evidence(["alpha"], "we found alpha", "unrelated page") is False


@pytest.mark.parametrize(("highlights", "text"), [([], "alpha"), (["alpha"], ""), ([""], "alpha")])
def test_a_link_with_nothing_to_check_fails_evidence(highlights, text) -> None:
    """No highlights is not "nothing to object to" -- it is a link that has proved nothing, and it
    must not reach the paid judge."""
    assert adapter.link_meets_evidence(highlights, text, "alpha") is False


def test_the_spot_check_always_draws_an_uncited_link() -> None:
    """Judging costs money, so only a few links are judged. Sampling only CITED links would tell a
    miner exactly which ones are worth making real -- it could return ten links and fake the eight
    it never cites. One uncited link is always drawn when any exists."""
    urls = [f"https://s{i}.test" for i in range(6)]
    cited = {"https://s0.test", "https://s1.test"}
    for seed in range(25):
        random.seed(seed)
        picks = adapter.sample_cited_and_uncited(urls, cited, adapter.MAX_CITED_SAMPLE,
                                                 adapter.MAX_SAMPLED_LINKS)
        assert len(picks) == adapter.MAX_SAMPLED_LINKS
        assert len(set(picks)) == len(picks), "a link must not be judged twice"
        assert any(url not in cited for url in picks)


def test_the_spot_check_cannot_ask_for_more_links_than_exist() -> None:
    random.seed(0)
    assert adapter.sample_cited_and_uncited(["https://a.test"], {"https://a.test"}, 2, 3) == [
        "https://a.test"]
    assert adapter.sample_cited_and_uncited([], set(), 2, 3) == []


def test_the_richest_body_wins_when_a_source_is_fetched_twice() -> None:
    """Two fetches of the same page can differ in completeness; the judge should read the fuller
    one. Sameness is by ``source_key``, so tracking params make no phantom second copy."""
    bodies = adapter.dedup_richest([
        {"url": "https://example.com/a?utm_source=x", "text": "short"},
        {"url": "https://www.example.com/a/", "text": "a considerably longer body"},
    ])
    assert [body["text"] for body in bodies] == ["a considerably longer body"]


def test_citation_markers_are_renumbered_to_the_order_the_judge_sees() -> None:
    """The groundedness judge is asked whether a value is supported by the source the answer cites
    FOR it. If [2] in the answer and [2] in the rendered sources are different pages, the judge
    answers a different question -- and would fail an honest answer."""
    summary = "worth [2](https://b.test) and [1](https://a.test)"
    aligned = adapter.align_citation_markers(
        summary, [{"url": "https://a.test"}, {"url": "https://b.test"}])
    assert aligned == "worth [2](https://b.test) and [1](https://a.test)"

    reordered = adapter.align_citation_markers(
        summary, [{"url": "https://b.test"}, {"url": "https://a.test"}])
    assert reordered == "worth [1](https://b.test) and [2](https://a.test)"


def test_a_marker_for_an_unjudged_source_is_left_alone() -> None:
    """Renumbering it to something would point the judge at the wrong body; dropping it would hide
    that the answer cited a source nobody checked."""
    assert adapter.align_citation_markers(
        "see [9](https://z.test)", [{"url": "https://a.test"}]) == "see [9](https://z.test)"


def test_cited_bodies_come_only_from_what_the_validator_fetched() -> None:
    """The groundedness judge exists to check the answer against INDEPENDENTLY obtained text. A
    body sourced from the miner would let it grade its own homework."""
    bodies = adapter.collect_cited_bodies(
        validator_links=[{"link": "https://a.test", "title": "A", "body": "fetched body"}],
        validator_tweets=[],
        cited_urls=["https://a.test", "https://never-fetched.test"],
    )
    assert [body["url"] for body in bodies] == ["https://a.test"]
    assert bodies[0]["text"] == "fetched body"


def test_a_cited_tweet_is_resolved_by_its_id() -> None:
    """A summary cites a tweet by its status URL; the validator holds it keyed by id."""
    bodies = adapter.collect_cited_bodies(
        validator_links=[],
        validator_tweets=[{"id": "12345", "text": "the claim", "url": "https://x.com/a/status/12345",
                           "user": {"username": "alice"}}],
        cited_urls=["https://x.com/anyone/status/12345"],
    )
    assert bodies[0]["text"] == "the claim"
    assert bodies[0]["title"] == "Tweet by @alice"


def test_a_quoted_tweet_travels_with_the_tweet_that_quotes_it() -> None:
    """A reply reading "this" above a quoted claim is only judgeable with the quote attached."""
    text = adapter.tweet_relevance_text(
        {"text": "this", "quote": {"text": "emissions rose 12%", "user": {"username": "bob"}}})
    assert text == "this\n\nQuoted tweet (@bob): emissions rose 12%"
    assert adapter.tweet_relevance_text({"text": "standalone"}) == "standalone"


def test_a_source_that_could_not_be_fetched_is_shown_as_empty_not_omitted() -> None:
    """The judge is told not to fail an answer over a body it cannot see. It can only apply that
    rule if it knows the body is missing, rather than silently seeing a shorter list."""
    rendered = adapter.render_cited_sources([{"url": "https://a.test", "title": "A", "text": ""}])
    assert "[no body could be fetched for this source]" in rendered
    assert "[1] https://a.test" in rendered
