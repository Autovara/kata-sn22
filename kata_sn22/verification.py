"""Turning a miner's claims into a verified quality score, the way upstream does.

This is the step that used to be invented. It measured a miner against a sealed corpus we wrote,
with a token-overlap judge we wrote, and neither existed upstream. Both are gone. What runs now is
upstream's own workflow, assembled from the pieces ported and parity-proven in
:mod:`kata_sn22.upstream_adapter`, :mod:`kata_sn22.fetch`, :mod:`kata_sn22.tweets` and
:mod:`kata_sn22.judge`.

**AI search**, per source, in this order and for these reasons:

1. The validator **fetches the page itself**. Nothing the miner says about a page is evidence.
2. :func:`~kata_sn22.upstream_adapter.link_meets_evidence` — the miner's highlights must appear in
   order in the validator's fetched body *and* in the miner's own text. A source that fails this is
   dropped before any money is spent on it.
3. A **sample** of the survivors is judged, always including one link the summary does not cite.
   Judging is paid; sampling only cited links would tell a miner which ones are worth making real.
4. Each sampled link's verified excerpts — never the whole page — go to the relevance judge.
5. Separately, the summary goes to the **groundedness** judge against the bodies of the sources it
   cites, which is what catches citing a real page for a value that page never states.

**X search**: the validator re-scrapes each claimed tweet by id and compares field by field
(:mod:`kata_sn22.tweets`). A tweet is short enough to quote whole, so there is no excerpt to reason
about.

Both paths share one rule: **failing to verify is not the same as verifying a failure.** A source
that cannot be fetched, or a judge that cannot be reached, must never be scored as though the miner
produced something bad. The first is recorded and skipped; the second aborts the round.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from kata_sn22 import judge as judge_module
from kata_sn22.upstream_adapter import (
    MAX_CITED_SAMPLE,
    MAX_SAMPLED_LINKS,
    align_citation_markers,
    cited_urls_normalized,
    collect_cited_bodies,
    link_meets_evidence,
    render_cited_sources,
    sample_cited_and_uncited,
)


@dataclass
class SourceVerification:
    """What happened to one claimed source, and why. Kept per source because an operator reading a
    lost round needs to distinguish "the miner fabricated this" from "the page would not load"."""

    link: str
    evidence_ok: bool
    judged: bool
    relevance: float = 0.0
    reason: str = ""

    def as_dict(self) -> dict:
        return {"link": self.link, "evidence_ok": self.evidence_ok, "judged": self.judged,
                "relevance": self.relevance, "reason": self.reason}


@dataclass
class Verification:
    """One task's verified quality: the two components upstream's arithmetic then weights."""

    content_relevance: float = 0.0
    summary_relevance: float = 0.0
    sources: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"content_relevance": self.content_relevance,
                "summary_relevance": self.summary_relevance,
                "sources": [source.as_dict() for source in self.sources],
                **self.detail}


def verify_ai_search(output, *, query: str, fetcher, judge_client, rng=None) -> Verification:
    """Verify and score one AI-search answer.

    ``fetcher`` is a :class:`kata_sn22.fetch.PageFetcher`; ``judge_client`` anything satisfying
    :class:`kata_sn22.judge.JudgeClient`. Both are seams so a calibration run can replay cassettes,
    and both may raise -- a round that could not gather evidence must stop rather than rank on none.
    """
    results = list(getattr(output, "results", ()) or ())
    if not results:
        return Verification(detail={"note": "the answer returned no sources"})

    pages = fetcher.get_many([result.link for result in results])

    verified: list = []
    sources: list = []
    for result in results:
        page = pages.get(result.link)
        body = page.text if page is not None else ""
        if not body:
            sources.append(SourceVerification(
                link=result.link, evidence_ok=False, judged=False,
                reason=(page.error if page is not None else "not fetched") or "no article"))
            continue
        if not link_meets_evidence(result.highlights, result.text, body):
            # The anti-fabrication check. Not a low score -- the source never becomes evidence at
            # all, so it does not dilute the mean either.
            sources.append(SourceVerification(
                link=result.link, evidence_ok=False, judged=False,
                reason="the claimed excerpts are not in the fetched page, in order"))
            continue
        verified.append(result)

    if not verified:
        return Verification(
            sources=sources,
            detail={"note": "no source survived the evidence check", "verified_sources": 0})

    cited = cited_urls_normalized(output.summary)
    sampled = set(sample_cited_and_uncited(
        [result.link for result in verified], cited, MAX_CITED_SAMPLE, MAX_SAMPLED_LINKS,
        rng=rng or random))

    scores: list = []
    for result in verified:
        if result.link not in sampled:
            sources.append(SourceVerification(
                link=result.link, evidence_ok=True, judged=False,
                reason="passed evidence; not in the judged sample"))
            continue
        # The judge sees the VERIFIED EXCERPTS, not the page. The miner is graded on what it proved
        # it read, which is also what stops a miner from earning credit for a page it never opened
        # that happens to be relevant.
        judged_body = "\n\n".join(result.highlights)
        relevance = judge_module.judge_relevance(
            judge_client,
            judge_module.build_body_relevance_messages(
                query, result.link, result.title, judged_body))
        scores.append(relevance)
        sources.append(SourceVerification(
            link=result.link, evidence_ok=True, judged=True, relevance=relevance,
            reason="judged on its verified excerpts"))

    content_relevance = sum(scores) / len(scores) if scores else 0.0
    summary_relevance = _summary_groundedness(
        output, query=query, pages=pages, judge_client=judge_client)

    return Verification(
        content_relevance=content_relevance,
        summary_relevance=summary_relevance,
        sources=sources,
        detail={"verified_sources": len(verified), "judged_sources": len(scores),
                "returned_sources": len(results)})


def _summary_groundedness(output, *, query: str, pages, judge_client) -> float:
    """Whether the answer's values are supported by the sources it cites FOR them.

    The bodies come from what the validator fetched, never from the miner -- a groundedness check
    against miner-supplied text would let the answer grade its own homework. Citation markers are
    renumbered to the order the judge sees, or the judge would be asked about the wrong source.
    """
    cited = [url for url, _title in _markdown_links(output.summary)]
    if not cited:
        # Upstream's rubric FAILs an answer that addresses nothing; an answer citing nothing has
        # nothing to be grounded in, so there is no call worth paying for.
        return 0.0
    validator_links = [
        {"link": url, "title": page.title, "body": page.text}
        for url, page in pages.items() if page.text
    ]
    bodies = collect_cited_bodies(
        validator_links=validator_links, validator_tweets=[], cited_urls=cited)
    if not bodies:
        return 0.0
    aligned = align_citation_markers(output.summary, bodies)
    reply = judge_client(judge_module.build_summary_groundedness_messages(
        query, aligned, render_cited_sources(bodies)))
    return judge_module.verdict_score(reply) / judge_module.VERDICT_SCALE


def _markdown_links(summary: str):
    from kata_sn22.upstream_adapter import extract_markdown_links

    return [(url, text) for text, url in extract_markdown_links(summary or "")]


def verify_x_search(output, *, scraper, start_date=None, end_date=None) -> Verification:
    """Verify one X-search answer by re-scraping the tweets it claims.

    X search carries a single quality component upstream (content relevance at weight 1.0), so the
    summary side stays zero rather than being invented.
    """
    from kata_sn22.tweets import verify_miner_tweets

    miner_tweets = [tweet.as_dict() for tweet in getattr(output, "tweets", ()) or ()]
    result = verify_miner_tweets(
        miner_tweets=miner_tweets, scraper=scraper, start_date=start_date, end_date=end_date)
    sources = [
        SourceVerification(link=f"tweet:{verdict.tweet_id}", evidence_ok=verdict.score > 0,
                           judged=True, relevance=verdict.score, reason=verdict.reason)
        for verdict in result.verdicts
    ]
    detail = {"scraped_tweets": result.scraped_count, "returned_tweets": len(miner_tweets)}
    if result.reason:
        detail["note"] = result.reason
    return Verification(content_relevance=result.score, sources=sources, detail=detail)
