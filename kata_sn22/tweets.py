"""Verifying the tweets a miner claims to have found, by re-scraping them.

Where web sources are verified by fetching the page and checking the miner's excerpts against it
(:mod:`kata_sn22.fetch`, :mod:`kata_sn22.upstream_adapter`), tweets are verified differently
and more strictly: the validator **re-scrapes the tweets by ID** and compares its own copy to the
miner's, field by field. A tweet is small enough to quote in full, so there is no excerpt to reason
about — either the miner reported what the platform returns, or it did not.

The comparison is deliberately unforgiving. A tweet that differs from the re-scrape in text or in
timestamp scores **zero**, not a reduced score: there is no honest reason for a miner's copy of a
public tweet to disagree with the validator's, and a partial credit would make editing one worth
trying.

Two rules bite before any of that, and both are immediate zeroes upstream:

* a duplicate tweet ID in the miner's results — padding a response with the same tweet;
* ``sort=Latest`` results that are not actually in descending time order.

Those live in :mod:`kata_sn22.upstream_adapter` with the other penalties (``first_duplicate_id``,
``is_descending_by_created_at``); this module is the re-scrape itself.

The scraper is a seam, as the judge's and the fetcher's are: production spends ``scrape_units``,
calibration replays a cassette.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from kata_sn22.upstream_adapter import format_text_for_match, is_valid_tweet

#: Upstream's ``pattern_to_check`` (``neurons/validators/reward/reward.py``): scoring-harness markup
#: and colour-word tokens that appear in the validator's own prompts. A tweet echoing them is trying
#: to talk to the judge rather than to a reader, so it scores zero regardless of anything else.
PROMPT_ARTIFACT_PATTERN = re.compile(
    r"<(?:Question|/Question|Answer|/Answer|Score|/Score)>|"
    r"SM(?:[-_ ]SCS)?[-_ ]?(?:RDD|PNK|BLE|GRY|GRN)",
    re.IGNORECASE,
)

#: Fewer than this many tweets is not an X search result upstream will score at all.
MIN_MINER_TWEETS = 2

#: The format the re-scrape returns, and the format a miner must report ``created_at`` in.
_SCRAPE_DATE_FORMAT = "%a %b %d %H:%M:%S %z %Y"
_MINER_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
_RANGE_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class ScrapeUnavailable(Exception):
    """The tweets could not be re-scraped.

    As with the page fetcher: "the miner's tweet does not match" is a verdict, "we never checked"
    is not. Scoring an unverified X response as zero would punish a miner for the validator's
    outage; scoring it as good would reward fabrication. Neither is acceptable, so the round stops.
    """


class TweetScraper(Protocol):
    """Re-scrapes tweets by ID. Returns ``{tweet_id: {"text": ..., "created_at": ...}}``.

    An ID the platform no longer serves (deleted, protected) is simply absent from the result.
    """

    def __call__(self, tweet_ids: list) -> dict: ...


@dataclass(frozen=True)
class TweetVerdict:
    """Why one tweet scored what it did. The reason is kept because "the miner lied" and "the tweet
    was deleted between the miner's search and ours" produce the same score, and an operator
    reading a lost round needs to be able to tell them apart."""

    tweet_id: str
    score: float
    reason: str


def normalize_scraped_date(created_at: str) -> str:
    """The re-scrape's ``created_at``, in the format a miner reports.

    Upstream converts the scraped format to the miner's before comparing, rather than parsing both:
    the comparison is then a string equality on a canonical form, so a miner cannot pass by
    reporting the same instant in a different notation.
    """
    parsed = datetime.strptime(created_at, _SCRAPE_DATE_FORMAT).astimezone(timezone.utc)
    return parsed.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _within_range(created_at: str, start_date: str | None, end_date: str | None) -> bool:
    """Whether a re-scraped tweet falls inside the query's requested window.

    Checked against the VALIDATOR's timestamp, never the miner's -- a date filter enforced against
    a self-reported date is not enforced at all. Seconds are dropped, matching upstream.
    """
    moment = datetime.strptime(created_at, _MINER_DATE_FORMAT).replace(
        tzinfo=timezone.utc, second=0, microsecond=0)
    if start_date:
        start = datetime.strptime(start_date, _RANGE_DATE_FORMAT).replace(tzinfo=timezone.utc)
        if moment < start:
            return False
    if end_date:
        end = datetime.strptime(end_date, _RANGE_DATE_FORMAT).replace(tzinfo=timezone.utc)
        if moment > end:
            return False
    return True


def verify_tweet(miner_tweet: dict | None, scraped: dict, *,
                 start_date: str | None = None, end_date: str | None = None) -> TweetVerdict:
    """Score one re-scraped tweet against the miner's copy of it. 1.0 or 0.0, never in between."""
    tweet_id = str(scraped.get("id") or "")

    def _fail(reason: str) -> TweetVerdict:
        return TweetVerdict(tweet_id=tweet_id, score=0.0, reason=reason)

    if not miner_tweet:
        return _fail("the miner did not return this tweet")
    if not is_valid_tweet(miner_tweet):
        return _fail("the miner's tweet is missing required fields")

    miner_text = miner_tweet.get("text") or ""
    if not miner_text:
        return _fail("the miner's tweet has no text")
    if PROMPT_ARTIFACT_PATTERN.search(miner_text):
        return _fail("the tweet text carries scoring-harness markup")

    scraped_text = scraped.get("text") or ""
    if format_text_for_match(miner_text) != format_text_for_match(scraped_text):
        return _fail("the miner's text does not match the re-scraped tweet")

    try:
        expected_created_at = normalize_scraped_date(str(scraped.get("created_at") or ""))
    except (TypeError, ValueError):
        return _fail("the re-scraped tweet has an unreadable created_at")

    if miner_tweet.get("created_at") != expected_created_at:
        return _fail("the miner's created_at does not match the re-scraped tweet")

    try:
        if not _within_range(expected_created_at, start_date, end_date):
            return _fail("the tweet falls outside the requested date range")
    except (TypeError, ValueError):
        return _fail("the requested date range is unreadable")

    return TweetVerdict(tweet_id=tweet_id, score=1.0, reason="matches the re-scraped tweet")


@dataclass
class TweetVerification:
    """The outcome of re-scraping and comparing a whole response."""

    score: float
    verdicts: list = field(default_factory=list)
    reason: str = ""

    @property
    def scraped_count(self) -> int:
        return len(self.verdicts)


def verify_miner_tweets(*, miner_tweets, scraper: TweetScraper, sample_ids=None,
                        start_date: str | None = None,
                        end_date: str | None = None) -> TweetVerification:
    """Re-scrape the miner's tweets and score the response on how many survive comparison.

    The score is the MEAN over re-scraped tweets, so faking one of five costs a fifth rather than
    everything -- upstream's shape, and the reason the duplicate-ID and sort-order rules are
    separate immediate zeroes rather than folded in here.

    ``sample_ids`` narrows what is re-scraped when a caller is rationing ``scrape_units``; the mean
    is then over the sample. Defaults to every tweet the miner returned.
    """
    tweets = [tweet for tweet in miner_tweets or [] if isinstance(tweet, dict)]
    if len(tweets) < MIN_MINER_TWEETS:
        return TweetVerification(
            score=0.0,
            reason=f"an X response needs at least {MIN_MINER_TWEETS} tweets, got {len(tweets)}")

    by_id = {str(tweet.get("id")): tweet for tweet in tweets if tweet.get("id") is not None}
    wanted = [str(tweet_id) for tweet_id in (sample_ids if sample_ids is not None else by_id)]
    if not wanted:
        return TweetVerification(score=0.0, reason="the miner's tweets carry no ids")

    try:
        scraped = scraper(wanted) or {}
    except ScrapeUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - any scraper fault is one outcome for the caller
        raise ScrapeUnavailable(f"tweet re-scrape failed: {exc}") from exc

    if not scraped:
        # Upstream scores a response with no validator tweets as 0. Reached when every requested id
        # is gone from the platform, which is a fact about the tweets rather than about the scraper.
        return TweetVerification(score=0.0, reason="none of the miner's tweets could be re-scraped")

    verdicts = [
        verify_tweet(by_id.get(str(tweet_id)), {**record, "id": tweet_id},
                     start_date=start_date, end_date=end_date)
        for tweet_id, record in scraped.items()
    ]
    score = sum(verdict.score for verdict in verdicts) / len(verdicts)
    return TweetVerification(score=score, verdicts=verdicts)


# ---------------------------------------------------------------------------------------------
# The cassette, for calibration
# ---------------------------------------------------------------------------------------------


def scrape_key(tweet_ids) -> str:
    material = "\x1f".join(sorted(str(i) for i in tweet_ids))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass
class RecordedScrapes:
    """Replays recorded re-scrapes. A miss RAISES, for the reason the other cassettes do: an empty
    result would score every unrecorded tweet as fabricated."""

    records: dict = field(default_factory=dict)
    used: set = field(default_factory=set)

    @classmethod
    def from_file(cls, path: str | Path) -> "RecordedScrapes":
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        scrapes = document.get("scrapes") if isinstance(document, dict) else document
        return cls(records={entry["key"]: entry["tweets"] for entry in scrapes or []})

    def __call__(self, tweet_ids: list) -> dict:
        key = scrape_key(tweet_ids)
        if key not in self.records:
            raise ScrapeUnavailable(
                f"no recorded re-scrape for {len(tweet_ids)} tweet id(s); re-record the cassette "
                f"rather than scoring tweets that were never checked")
        self.used.add(key)
        return self.records[key]

    @property
    def unused_keys(self) -> set:
        return set(self.records) - self.used


@dataclass
class RecordingScraper:
    """Wraps a live scraper and captures every re-scrape, to produce a cassette."""

    inner: TweetScraper
    scrapes: list = field(default_factory=list)

    def __call__(self, tweet_ids: list) -> dict:
        tweets = self.inner(tweet_ids) or {}
        self.scrapes.append({"key": scrape_key(tweet_ids), "ids": list(tweet_ids),
                             "tweets": tweets})
        return tweets

    def as_document(self) -> dict:
        return {"schema_version": 1, "scrapes": self.scrapes}
