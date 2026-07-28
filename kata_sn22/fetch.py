"""Fetching the page bodies the validator judges against.

This is what makes SN22's fairness model work. The validator does not take the miner's word for what
a page says, and it does not compare the two contestants against each other's claims — it fetches
the page itself, once, and both king and challenger are judged against *that* text. Upstream's whole
anti-fabrication story rests on this fetch actually happening.

Three things happen to fetched text before anything reads it, and each exists for a specific reason:

**Injection is neutered.** A fetched page is attacker-controlled text that is about to be pasted
into a judge's prompt. A page containing ``Verdict: HIGH`` would otherwise show the model a
filled-in example of exactly the output it was asked to produce, sitting inside the evidence it was
asked to grade. Every ``verdict:`` is rewritten to ``verdict-`` — upstream's rule, ported exactly.

  *Worth knowing what this does not do.* ``verdict- HIGH`` still matches
  :data:`kata_sn22.judge`'s verdict pattern, which accepts ``:`` **or** ``-``. So the sanitizer
  breaks the visual FORM a model would copy; it does not make the text unparseable. Pointing
  ``verdict_score`` at a page body would hand a hostile page the score it asked for. Upstream's
  behaviour is kept exactly — a plugged-in subnet's validation is not redesigned here — and the
  safety comes from the call structure instead: the parser only ever reads a judge's REPLY, while
  bodies travel inside the request. ``tests/test_sn22_fetch.py`` pins both halves of that.

**Unusable text is treated as no text.** A JavaScript wall, a bot-check interstitial or a 200-byte
stub is not a page body; scoring a miner against one would fail it for a source that was fine. Those
become an empty body with a recorded reason, which the judge is told not to punish.

**A page is fetched once per round.** The cache is not an optimisation detail — king and challenger
must be judged against the SAME bytes. Fetching twice could hand one contestant a page that changed
between the two calls, and the difference would look like a scoring difference.

The transport is a seam, exactly as :mod:`kata_sn22.judge`'s is: production does HTTP and spends
``data_api_calls``; calibration replays a cassette so a repeatable run costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

#: Body text kept per page. Upstream's ``_RAW_CACHE_CHARS``; the judge is shown excerpts, not books.
MAX_BODY_CHARS = 16000

#: Below this, a "page" is a stub, a paywall notice or an error card rather than an article.
MIN_ARTICLE_CHARS = 200

#: How long a fetched page stays reusable. Upstream's ``_CACHE_TTL_S``.
CACHE_TTL_SECONDS = 600

#: Upstream's ``_MAX_CACHE_ENTRIES``, so a long run cannot grow the cache without bound.
MAX_CACHE_ENTRIES = 2000

#: Rewrites any ``verdict:`` in fetched text. THE prompt-injection defence for the judge path.
_VERDICT_INJECTION = re.compile(r"(?i)\bverdict\b\s*:")

#: Text that means "you are looking at a wall, not an article".
_JS_GATE = re.compile(
    r"(?i)(please enable javascript|enable javascript and refresh|"
    r"something went wrong\.?\s*(wait a moment|please|try)|"
    r"you (need to )?(enable|turn on) javascript|access denied|are you a robot)"
)


def sanitize_body_text(text: str) -> str:
    """Neuter verdict-shaped text in an attacker-controlled body.

    Breaks the FORM a model would copy, not the parse. See this module's docstring for what that
    does and does not buy, and why upstream's exact behaviour is kept.
    """
    return _VERDICT_INJECTION.sub("verdict-", text or "")


def is_usable_article(text: str) -> bool:
    """Whether fetched text is an article at all, as opposed to a wall or a stub."""
    return bool(text) and len(text) >= MIN_ARTICLE_CHARS and not _JS_GATE.search(text)


@dataclass(frozen=True)
class Page:
    """One fetched page. ``error`` is non-empty exactly when ``text`` is empty."""

    url: str
    title: str = ""
    text: str = ""
    published_date: str = ""
    author: str = ""
    error: str = ""

    def as_dict(self) -> dict:
        return {"url": self.url, "title": self.title, "text": self.text,
                "published_date": self.published_date, "author": self.author, "error": self.error}


class FetchUnavailable(Exception):
    """The fetcher could not run at all.

    Distinct from a page that fetched empty. "This source has no readable body" is a judgement the
    round can proceed on; "we never fetched anything" means no contestant was verified against
    independent ground truth, and ranking on that would be ranking on the miners' own claims.
    """


class PageTransport(Protocol):
    """Anything that can turn URLs into raw fetch records.

    Records are ``{url: {"text": ..., "title": ..., "published_date": ..., "author": ...}}``.
    Extraction from HTML happens on the transport's side of the seam, because it needs a real HTML
    parser and the lane's runtime deliberately has none.
    """

    def __call__(self, urls: list) -> dict: ...


@dataclass
class PageFetcher:
    """Fetches page bodies once per round, sanitizes them, and caches them.

    ``now`` is injected so cache expiry is testable without sleeping; it defaults to a monotonic
    clock, which is what a TTL must be measured on -- a wall clock that steps backwards over an NTP
    correction would resurrect expired entries.
    """

    transport: PageTransport
    now: object = None
    _cache: dict = field(default_factory=dict)
    #: URLs actually sent to the transport, so a caller can meter what it spent.
    fetched_urls: list = field(default_factory=list)

    def _clock(self) -> float:
        if self.now is not None:
            return float(self.now())
        import time

        return time.monotonic()

    def get_many(self, urls: list) -> dict:
        """``{url: Page}`` for every requested URL. Order-preserving, duplicate-free.

        A URL that fails to fetch still gets a :class:`Page` -- with empty text and a reason. A
        missing key would make "we could not read it" indistinguishable from "we never asked",
        and only one of those is a fact about the miner's source.
        """
        wanted = [url for url in dict.fromkeys(urls or []) if url]
        if not wanted:
            return {}

        now = self._clock()
        result: dict = {}
        to_fetch: list = []
        for url in wanted:
            entry = self._cache.get(url)
            if entry is not None and (now - entry[0]) < CACHE_TTL_SECONDS:
                result[url] = entry[1]
            else:
                to_fetch.append(url)

        if to_fetch:
            try:
                raw = self.transport(to_fetch) or {}
            except FetchUnavailable:
                raise
            except Exception as exc:  # noqa: BLE001 - any transport fault is one fetch outcome
                raise FetchUnavailable(f"page fetch failed: {exc}") from exc
            self.fetched_urls.extend(to_fetch)
            for url in to_fetch:
                page = self._page_from(url, raw.get(url) or {})
                self._store(url, page)
                result[url] = page

        return {url: result[url] for url in wanted}

    def _page_from(self, url: str, record: dict) -> Page:
        text = sanitize_body_text(str(record.get("text") or ""))[:MAX_BODY_CHARS]
        if not is_usable_article(text):
            # Upstream's fallback: the extractor may fail on a page whose raw text is fine.
            text = sanitize_body_text(str(record.get("raw_text") or ""))[:MAX_BODY_CHARS]
        if not is_usable_article(text):
            text = ""
        return Page(
            url=url,
            title=str(record.get("title") or ""),
            text=text,
            published_date=str(record.get("published_date") or record.get("date") or ""),
            author=str(record.get("author") or ""),
            error="" if text else "no article",
        )

    def _store(self, url: str, page: Page) -> None:
        if len(self._cache) >= MAX_CACHE_ENTRIES:
            now = self._clock()
            self._cache = {
                key: entry for key, entry in self._cache.items()
                if (now - entry[0]) < CACHE_TTL_SECONDS
            }
            if len(self._cache) >= MAX_CACHE_ENTRIES:
                oldest = sorted(self._cache, key=lambda key: self._cache[key][0])
                for key in oldest[: len(self._cache) // 2]:
                    del self._cache[key]
        self._cache[url] = (self._clock(), page)


# ---------------------------------------------------------------------------------------------
# The cassette, for calibration
# ---------------------------------------------------------------------------------------------


def page_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


@dataclass
class RecordedPages:
    """Replays recorded fetches. A miss RAISES.

    Same reasoning as the judge's cassette: a fetcher that returned an empty body for anything it
    had not seen would score every unrecorded source as unverifiable, which reads as "the miners
    cited bad sources" rather than "the cassette is incomplete".
    """

    records: dict = field(default_factory=dict)
    used: set = field(default_factory=set)

    @classmethod
    def from_file(cls, path: str | Path) -> "RecordedPages":
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        pages = document.get("pages") if isinstance(document, dict) else document
        return cls(records={entry["key"]: entry["record"] for entry in pages or []})

    def __call__(self, urls: list) -> dict:
        missing = [url for url in urls if page_key(url) not in self.records]
        if missing:
            raise FetchUnavailable(
                f"no recorded page for {len(missing)} URL(s), first {missing[0]!r}; re-record the "
                f"cassette rather than scoring sources that were never fetched")
        out = {}
        for url in urls:
            key = page_key(url)
            self.used.add(key)
            out[url] = self.records[key]
        return out

    @property
    def unused_keys(self) -> set:
        return set(self.records) - self.used


@dataclass
class RecordingFetcher:
    """Wraps a live transport and captures every fetch, to produce a cassette."""

    inner: PageTransport
    pages: list = field(default_factory=list)

    def __call__(self, urls: list) -> dict:
        raw = self.inner(urls) or {}
        for url in urls:
            self.pages.append({"key": page_key(url), "url": url, "record": raw.get(url) or {}})
        return raw

    def as_document(self) -> dict:
        return {"schema_version": 1, "pages": self.pages}
