"""What the validator establishes for itself, and what it refuses to establish.

This is the module that replaced the invented one. The old scorer measured a miner against a corpus
we wrote, with a judge we wrote; this one runs upstream's workflow — fetch the page, check the
excerpts against it, judge only what survives, re-scrape the tweets.

The tests are organised around the three ways that can go wrong, in descending order of how badly:

1. **Scoring something that was never verified.** A source whose page would not load, or a judge
   that could not be reached, must never turn into a low score — that punishes a miner for the
   validator's problem, and it looks identical to a miner that did badly.
2. **Paying for nothing.** Judge calls cost money. A source that failed evidence, or a link outside
   the sample, must not reach the judge at all.
3. **Letting a claim count as evidence.** Anything the miner said about a page is a claim; only
   what the validator found on the page is evidence.
"""

from __future__ import annotations

import pytest

from kata_sn22 import verification
from kata_sn22.fetch import FetchUnavailable, PageFetcher
from kata_sn22.judge import JudgeUnavailable
from kata_sn22.protocol import Limits, SearchResult, Task, TaskOutput, ToolUsage, TweetResult

ARTICLE = ("The measured figure is 28 percent, recorded in July 2026, and the mechanism is "
           "described in detail below. ") * 5
HIGHLIGHT = "The measured figure is 28 percent, recorded in July 2026"

TASK = Task(task_id="t000", query="what is the figure?", search_type="ai_search", ai_mode="fast",
            limits=Limits(max_results=5))


def _output(results=(), summary="", tweets=()) -> TaskOutput:
    return TaskOutput(protocol_version=1, task_id="t000", summary=summary, results=tuple(results),
                      tweets=tuple(tweets), citations=(), usage=ToolUsage())


def _result(link="https://a.test/x", highlights=(HIGHLIGHT,), text=None) -> SearchResult:
    return SearchResult(link=link, title="A Title", snippet="s", highlights=tuple(highlights),
                        text=HIGHLIGHT if text is None else text)


def _fetcher(pages=None, *, fetched=None):
    pages = {"https://a.test/x": ARTICLE} if pages is None else pages

    def _transport(urls):
        if fetched is not None:
            fetched.extend(urls)
        return {url: {"text": pages[url], "title": "A Title"} for url in urls if url in pages}

    return PageFetcher(transport=_transport)


def _judge(reply="Verdict: HIGH", *, seen=None):
    def _client(messages):
        if seen is not None:
            seen.append(messages)
        return reply

    return _client


# ---- failing to verify is not verifying a failure ------------------------------------------------

def test_a_page_that_will_not_load_is_recorded_not_scored_against() -> None:
    """The failure is the validator's. A miner whose perfectly good source happened to 503 must not
    be marked down for it -- it simply earns nothing from that source."""
    result = verification.verify_ai_search(
        _output([_result()]), query=TASK.query, fetcher=_fetcher(pages={}),
        judge_client=_judge())
    assert result.content_relevance == 0.0
    assert [source.reason for source in result.sources] == ["no article"]
    assert result.detail["note"] == "no source survived the evidence check"


def test_a_judge_outage_stops_the_round_rather_than_scoring_zero() -> None:
    """"This source is bad" and "we could not find out" must not be the same number. Ranking a
    contestant on evidence that was never gathered is the one thing this whole module prevents."""
    def _down(_messages):
        raise JudgeUnavailable("connection refused")

    with pytest.raises(JudgeUnavailable):
        verification.verify_ai_search(_output([_result()], summary="answer"), query=TASK.query,
                                      fetcher=_fetcher(), judge_client=_down)


def test_a_fetcher_outage_stops_the_round() -> None:
    def _broken(_urls):
        raise ConnectionError("dns failure")

    with pytest.raises(FetchUnavailable):
        verification.verify_ai_search(_output([_result()]), query=TASK.query,
                                      fetcher=PageFetcher(transport=_broken),
                                      judge_client=_judge())


def test_an_answer_with_no_sources_costs_nothing() -> None:
    def _explode(*_args, **_kwargs):
        raise AssertionError("nothing to verify must cost no fetch and no judge call")

    result = verification.verify_ai_search(
        _output(), query=TASK.query, fetcher=PageFetcher(transport=_explode),
        judge_client=_explode)
    assert result.content_relevance == 0.0
    assert result.detail["note"] == "the answer returned no sources"


# ---- only what the validator found counts ------------------------------------------------------

def test_a_source_whose_excerpts_are_not_on_the_page_never_reaches_the_judge() -> None:
    """The anti-fabrication check, and the money it saves. A fabricated source is not scored LOW --
    it is dropped before a judge call is spent on it, and it does not dilute the mean either."""
    seen: list = []
    result = verification.verify_ai_search(
        _output([_result(highlights=("a sentence nobody wrote",))]),
        query=TASK.query, fetcher=_fetcher(), judge_client=_judge(seen=seen))

    assert seen == [], "a source that failed evidence must not be judged"
    assert result.sources[0].evidence_ok is False
    assert "not in the fetched page" in result.sources[0].reason


def test_a_source_the_miner_did_not_use_in_its_own_answer_fails_evidence() -> None:
    """The second direction: real excerpts pasted beside an answer written somewhere else."""
    result = verification.verify_ai_search(
        _output([_result(text="an unrelated answer")]),
        query=TASK.query, fetcher=_fetcher(), judge_client=_judge())
    assert result.sources[0].evidence_ok is False


def test_the_judge_sees_the_verified_excerpts_not_the_page() -> None:
    """A miner is graded on what it PROVED it read. Showing the judge the whole page would let a
    miner earn credit for a page it never opened that happens to be relevant."""
    seen: list = []
    verification.verify_ai_search(
        _output([_result()], summary="answer"), query=TASK.query,
        fetcher=_fetcher(), judge_client=_judge(seen=seen))

    body_call = seen[0][1]["content"]
    assert HIGHLIGHT in body_call
    assert "the mechanism is described in detail below" not in body_call


def test_only_a_sample_of_sources_is_judged() -> None:
    """Judging is paid. A response with more sources than the sample size must not cost one call
    per source, or a miner could raise the lane's bill by returning more links."""
    from kata_sn22.upstream_adapter import MAX_SAMPLED_LINKS

    links = [f"https://s{index}.test/x" for index in range(8)]
    seen: list = []
    result = verification.verify_ai_search(
        _output([_result(link=link) for link in links]),
        query=TASK.query,
        fetcher=_fetcher(pages={link: ARTICLE for link in links}),
        judge_client=_judge(seen=seen))

    # One groundedness call is possible on top of the sampled body calls; the summary here cites
    # nothing, so there is none.
    assert len(seen) == MAX_SAMPLED_LINKS
    assert result.detail["judged_sources"] == MAX_SAMPLED_LINKS
    assert result.detail["verified_sources"] == 8


def test_every_source_is_reported_whether_judged_or_not() -> None:
    """An operator reading a lost round has to be able to tell "fabricated" from "not sampled" from
    "would not load". They are three different facts and they produce similar scores."""
    links = [f"https://s{index}.test/x" for index in range(5)]
    result = verification.verify_ai_search(
        _output([_result(link=link) for link in links]),
        query=TASK.query, fetcher=_fetcher(pages={link: ARTICLE for link in links}),
        judge_client=_judge())
    assert len(result.sources) == 5
    assert {source.judged for source in result.sources} == {True, False}


# ---- groundedness ------------------------------------------------------------------------------

def test_the_summary_is_judged_against_the_bodies_the_validator_fetched() -> None:
    """Never against what the miner supplied -- that would let the answer grade its own homework."""
    seen: list = []
    verification.verify_ai_search(
        _output([_result()], summary="the figure is 28 percent [1](https://a.test/x)"),
        query=TASK.query, fetcher=_fetcher(), judge_client=_judge(seen=seen))

    grounding = [call for call in seen if "CitedSources" in call[1]["content"]]
    assert grounding, "a summary that cites a source must be checked for groundedness"
    assert ARTICLE[:50] in grounding[0][1]["content"]


def test_a_summary_that_cites_nothing_costs_no_groundedness_call() -> None:
    seen: list = []
    verification.verify_ai_search(
        _output([_result()], summary="the figure is 28 percent"),
        query=TASK.query, fetcher=_fetcher(), judge_client=_judge(seen=seen))
    assert not any("CitedSources" in call[1]["content"] for call in seen)


def test_a_summary_citing_a_source_that_was_never_fetched_is_not_grounded() -> None:
    result = verification.verify_ai_search(
        _output([_result()], summary="the figure is 28 percent [1](https://elsewhere.test/y)"),
        query=TASK.query, fetcher=_fetcher(), judge_client=_judge())
    assert result.summary_relevance == 0.0


# ---- X search ----------------------------------------------------------------------------------

def _tweet(tweet_id: str, text: str = "a real tweet about the subject") -> dict:
    from kata_sn22.tweets import normalize_scraped_date
    from kata_sn22.upstream_adapter import synthetic_created_at

    return {"id": tweet_id, "text": text,
            "created_at": normalize_scraped_date(synthetic_created_at(int(tweet_id) % 10)),
            "reply_count": 1, "retweet_count": 2, "like_count": 3, "quote_count": 0,
            "bookmark_count": 0, "url": f"https://x.com/a/status/{tweet_id}",
            "is_quote_tweet": False, "is_retweet": False,
            "user": {"id": "u-1", "username": "alice"}}


def _scraped(tweet_id: str, text: str = "a real tweet about the subject") -> dict:
    from kata_sn22.upstream_adapter import synthetic_created_at

    return {"text": text, "created_at": synthetic_created_at(int(tweet_id) % 10)}


def test_x_search_is_scored_on_the_tweets_that_survive_re_scraping() -> None:
    output = _output(tweets=[TweetResult(_tweet("1")), TweetResult(_tweet("2")),
                             TweetResult(_tweet("3", "edited by the miner"))])
    result = verification.verify_x_search(
        output, scraper=lambda ids: {tid: _scraped(tid) for tid in ids})
    assert result.content_relevance == pytest.approx(2 / 3)
    assert result.detail["returned_tweets"] == 3


def test_x_search_carries_no_summary_component() -> None:
    """Upstream weights X search on content relevance alone. Inventing a summary score here would
    be inventing a component the subnet does not have."""
    output = _output(tweets=[TweetResult(_tweet("1")), TweetResult(_tweet("2"))],
                     summary="an answer")
    result = verification.verify_x_search(
        output, scraper=lambda ids: {tid: _scraped(tid) for tid in ids})
    assert result.summary_relevance == 0.0


def test_every_tweet_verdict_records_why() -> None:
    output = _output(tweets=[TweetResult(_tweet("1")), TweetResult(_tweet("2", "edited"))])
    result = verification.verify_x_search(
        output, scraper=lambda ids: {tid: _scraped(tid) for tid in ids})
    reasons = {source.link: source.reason for source in result.sources}
    assert "matches" in reasons["tweet:1"]
    assert "does not match" in reasons["tweet:2"]
