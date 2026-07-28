"""The frozen SN22 submission contract (plan §5.1, SN22-2).

A miner submits an agent; the lane runs it and scores what comes back. Everything crossing that
boundary is defined here, versioned, and validated before it is allowed to influence a score.

Three rules shape the whole module, and each exists because of a specific way this could go wrong:

* **Everything a candidate produces is untrusted DATA.** Not instructions, not code, not a path.
  A submission's output is parsed under a size cap, schema-checked, and never executed, formatted
  into a prompt as a command, or used to build a filesystem path. Retrieved web/X content is doubly
  untrusted: it was written by a third party and reaches the judge, so it is the natural home for a
  prompt injection.
* **Failures are CLASSIFIED, never silently zero.** A candidate that times out, returns malformed
  JSON, or blows a quota must be distinguishable from one that answered badly — otherwise "crash"
  and "poor answer" both score 0.0 and a broken agent looks merely mediocre. Every rejection here
  returns a named :class:`ErrorClass`.
* **The version is part of the contract.** A submission declaring a protocol version this lane does
  not implement is rejected rather than interpreted leniently, because a lenient read of an unknown
  schema is how a field silently stops being checked.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kata_sn22.upstream_adapter import source_key

#: Bumped whenever a field's meaning changes. A submission built against a different version is
#: refused, not coerced.
PROTOCOL_VERSION = 1

#: Hard ceiling on a single agent response, before parsing. Applied to BYTES, not to the parsed
#: object: a 500 MB JSON document is an availability problem while it is still being decoded, so the
#: check has to happen before ``json.loads`` ever sees it.
MAX_OUTPUT_BYTES = 256 * 1024
#: Ceilings on the shape of a response. Each bounds work the judge would otherwise have to do.
MAX_RESULTS_PER_TASK = 20
#: How many results a task ASKS for by default. Distinct in meaning from the ceiling above, and
#: since SN22-5 it is load-bearing in both directions: returning more is an ``EXCESS_OUTPUT``
#: violation, and returning fewer is the upstream count penalty. Five rather than twenty because a
#: request has to be satisfiable — asking for more results than the sealed corpus can supply would
#: make the count penalty a constant, which is a penalty that ranks nobody.
DEFAULT_RESULTS_PER_TASK = 5
MAX_CITATIONS_PER_TASK = 40
MAX_TEXT_CHARS = 8_000

#: Search modes, mirroring the audited upstream. ``ai_mode`` applies only to AI search.
SEARCH_TYPES = ("ai_search", "x_search")
AI_MODES = ("fast", "balanced", "deep")
RESULT_TYPES = ("summary", "links", "both")

#: A document identifier must be STABLE — the same document is the same id in the sealed
#: snapshot and in a citation — and inert, because it is used as a dict key and printed into
#: reports.
#: A result's live URL. Only http(s): a scheme the validator cannot fetch is a source it cannot
#: verify, and an unverifiable source must not reach the judge.
HTTP_URL_RE = re.compile(r"^https?://[^\s<>\"]{1,2040}$")

#: Excerpts a miner may claim per source. A cap because each one is checked against the fetched
#: body, and an unbounded list is an unbounded amount of the lane's work per result.
MAX_HIGHLIGHTS_PER_RESULT = 8
TASK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class ErrorClass(str, Enum):
    """Why a run did not produce a usable answer. Deterministic: the same fault always maps here.

    Kept distinct from a low score on purpose. ``sn22_invalid_runs`` counts these, and it is a
    separate ranked signal from answer quality (plan §5.4), so an agent cannot hide crashes inside a
    mediocre average.
    """

    TIMEOUT = "timeout"                  # exceeded the round's wall clock
    INVALID_SCHEMA = "invalid_schema"    # unparseable or contract-violating output
    EXCESS_OUTPUT = "excess_output"      # over MAX_OUTPUT_BYTES or a shape ceiling
    EXCESS_CALLS = "excess_calls"        # blew a tool/token quota
    PROVIDER_UNAVAILABLE = "provider_unavailable"  # SHARED infrastructure fault, not the agent's
    CRASHED = "crashed"                  # non-zero exit, no parseable output

    @property
    def candidate_caused(self) -> bool:
        """Whether this failure is the CANDIDATE's fault.

        The distinction decides a promotion, so it is explicit rather than inferred. A provider
        outage hits whichever contestant happened to run during it; charging that to the candidate
        would let infrastructure flakiness decide a crown. Plan §5.2 goes further and refuses to
        decide the challenge at all when shared infrastructure was incomplete for either side.
        """
        return self is not ErrorClass.PROVIDER_UNAVAILABLE


class ProtocolError(Exception):
    """A submission violated the contract. Always carries the class it maps to."""

    def __init__(self, error_class: ErrorClass, message: str):
        super().__init__(message)
        self.error_class = error_class


@dataclass(frozen=True)
class Limits:
    """The quotas a task runs under. Handed to the agent so it can plan, and enforced anyway.

    ``max_results`` is the number of results the task asks for. It is both a request and a ceiling:
    fewer is a shortfall the upstream count penalty charges for, more is a contract violation. One
    number for both because "return this many" is one instruction, and splitting it into a floor and
    a ceiling would let a submission satisfy one while ignoring the other.
    """

    max_wall_seconds: int = 120
    max_provider_calls: int = 8
    max_tokens: int = 20_000
    max_results: int = DEFAULT_RESULTS_PER_TASK

    def as_dict(self) -> dict:
        return {"max_wall_seconds": self.max_wall_seconds,
                "max_provider_calls": self.max_provider_calls,
                "max_tokens": self.max_tokens,
                "max_results": self.max_results}


@dataclass(frozen=True)
class Task:
    """One unit of work handed to an agent. Both contestants receive byte-identical tasks."""

    task_id: str
    query: str
    search_type: str
    result_type: str = "both"
    ai_mode: str | None = None
    #: X search only. ``"Latest"`` additionally requires the results to be in descending time order,
    #: which upstream scores as an immediate zero when broken.
    sort: str | None = None
    #: Optional window the results must fall inside. Enforced against the VALIDATOR's re-scraped
    #: timestamps, never the miner's -- a date filter checked against a self-reported date is not
    #: a filter at all.
    start_date: str | None = None
    end_date: str | None = None
    #: The round-scoped relay capability: a short-lived, challenge/variant/task-bound token,
    #: never a provider API key. See plan §6.1; the credential boundary itself is SN22-4.
    relay_endpoint: str = ""
    relay_capability: str = ""
    limits: Limits = field(default_factory=Limits)

    def as_input(self) -> dict:
        """The exact JSON document an agent receives on stdin."""
        return {
            "protocol_version": PROTOCOL_VERSION,
            "task_id": self.task_id,
            "query": self.query,
            "search_type": self.search_type,
            "result_type": self.result_type,
            "ai_mode": self.ai_mode,
            "sort": self.sort,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "relay": {"endpoint": self.relay_endpoint, "capability": self.relay_capability},
            "limits": self.limits.as_dict(),
        }


def validate_task(task: Task) -> None:
    """Refuse a malformed task before an agent sees it: a bad task misscores BOTH contestants."""
    if not TASK_ID_RE.fullmatch(task.task_id):
        raise ProtocolError(ErrorClass.INVALID_SCHEMA, f"invalid task_id {task.task_id!r}")
    if not task.query.strip():
        raise ProtocolError(ErrorClass.INVALID_SCHEMA, "task query must not be empty")
    if len(task.query) > MAX_TEXT_CHARS:
        raise ProtocolError(ErrorClass.EXCESS_OUTPUT, "task query exceeds the text ceiling")
    if task.search_type not in SEARCH_TYPES:
        raise ProtocolError(ErrorClass.INVALID_SCHEMA, f"unknown search_type {task.search_type!r}")
    if task.result_type not in RESULT_TYPES:
        raise ProtocolError(ErrorClass.INVALID_SCHEMA, f"unknown result_type {task.result_type!r}")
    if task.search_type == "ai_search":
        if task.ai_mode not in AI_MODES:
            raise ProtocolError(ErrorClass.INVALID_SCHEMA,
                                f"ai_search requires ai_mode in {AI_MODES}, got {task.ai_mode!r}")
    elif task.ai_mode is not None:
        raise ProtocolError(ErrorClass.INVALID_SCHEMA, "ai_mode applies only to ai_search")


@dataclass(frozen=True)
class Citation:
    """A claim the agent says a source supports, named by that source's live URL."""

    link: str
    claim: str

    def as_dict(self) -> dict:
        return {"link": self.link, "claim": self.claim}


@dataclass(frozen=True)
class SearchResult:
    """One retrieved web source, with the evidence the miner must supply for it to be judged.

    ``highlights`` are excerpts the miner claims appear on the page, and ``text`` is what the miner
    itself wrote about the source. Neither is decoration: the validator fetches the page and
    requires the highlights to appear IN ORDER in both its own copy and the miner's text before the
    source is worth paying a judge to score (see
    :func:`kata_sn22.upstream_adapter.link_meets_evidence`). A result with no highlights has proved
    nothing and earns nothing.
    """

    link: str
    title: str
    snippet: str
    highlights: tuple[str, ...] = ()
    text: str = ""

    def as_dict(self) -> dict:
        return {"link": self.link, "title": self.title, "snippet": self.snippet,
                "highlights": list(self.highlights), "text": self.text}


@dataclass(frozen=True)
class TweetResult:
    """One retrieved tweet, in the shape the validator re-scrapes and compares field by field.

    Carried as the raw fields rather than a normalized object because the comparison IS field by
    field: anything this layer tidied up would be a difference the check could no longer see.
    """

    fields: dict

    @property
    def tweet_id(self) -> str:
        return str(self.fields.get("id") or "")

    def as_dict(self) -> dict:
        return dict(self.fields)


@dataclass(frozen=True)
class ToolUsage:
    """What the agent spent. Cross-checked against the relay's own usage manifest, never trusted
    alone — a candidate reporting its own cost has every reason to under-report it."""

    provider_calls: int = 0
    tokens: int = 0
    elapsed_seconds: float = 0.0

    def as_dict(self) -> dict:
        return {"provider_calls": self.provider_calls, "tokens": self.tokens,
                "elapsed_seconds": self.elapsed_seconds}


@dataclass(frozen=True)
class TaskOutput:
    """A validated agent response for one task."""

    protocol_version: int
    task_id: str
    summary: str
    results: tuple[SearchResult, ...]
    citations: tuple[Citation, ...]
    usage: ToolUsage
    #: X search only. Verified by re-scraping each tweet by id, never by excerpt.
    tweets: tuple[TweetResult, ...] = ()

    def as_dict(self) -> dict:
        return {
            "protocol_version": self.protocol_version,
            "task_id": self.task_id,
            "summary": self.summary,
            "results": [r.as_dict() for r in self.results],
            "tweets": [t.as_dict() for t in self.tweets],
            "citations": [c.as_dict() for c in self.citations],
            "usage": self.usage.as_dict(),
        }


def _require(condition: bool, error_class: ErrorClass, message: str) -> None:
    if not condition:
        raise ProtocolError(error_class, message)


def _text(value: Any, where: str) -> str:
    _require(isinstance(value, str), ErrorClass.INVALID_SCHEMA, f"{where} must be a string")
    _require(len(value) <= MAX_TEXT_CHARS, ErrorClass.EXCESS_OUTPUT,
             f"{where} exceeds {MAX_TEXT_CHARS} characters")
    return value


def parse_task_output(raw: bytes | str, *, task: Task) -> TaskOutput:
    """Parse and validate one agent response, or raise a CLASSIFIED :class:`ProtocolError`.

    The size check runs on bytes, before decoding, because an oversized document is a cost the lane
    pays during parsing. Everything after that is shape validation: no field is coerced, defaulted
    or repaired, since a repaired response is one the candidate did not actually produce.
    """
    payload = raw.encode("utf-8") if isinstance(raw, str) else raw
    _require(len(payload) <= MAX_OUTPUT_BYTES, ErrorClass.EXCESS_OUTPUT,
             f"response is {len(payload)} bytes, over the {MAX_OUTPUT_BYTES} ceiling")
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProtocolError(ErrorClass.INVALID_SCHEMA,
                            f"response is not valid JSON: {exc}") from exc
    _require(isinstance(document, dict), ErrorClass.INVALID_SCHEMA, "response is not a JSON object")

    version = document.get("protocol_version")
    _require(version == PROTOCOL_VERSION, ErrorClass.INVALID_SCHEMA,
             f"protocol_version {version!r} is not {PROTOCOL_VERSION}")
    # The response must name the task it answers. Without this a slow agent could answer task 1 and
    # have it counted against task 2, which is both a correctness bug and an easy way to game a
    # per-task score by answering only the easy ones.
    _require(document.get("task_id") == task.task_id, ErrorClass.INVALID_SCHEMA,
             f"response task_id {document.get('task_id')!r} != requested {task.task_id!r}")

    summary = _text(document.get("summary", ""), "summary")

    raw_results = document.get("results", [])
    _require(isinstance(raw_results, list), ErrorClass.INVALID_SCHEMA, "results must be a list")
    _require(len(raw_results) <= task.limits.max_results, ErrorClass.EXCESS_OUTPUT,
             f"{len(raw_results)} results exceeds the limit of {task.limits.max_results}")
    results = []
    seen_links: set[str] = set()
    for index, item in enumerate(raw_results):
        where = f"results[{index}]"
        _require(isinstance(item, dict), ErrorClass.INVALID_SCHEMA, f"{where} must be an object")
        link = item.get("link")
        _require(isinstance(link, str) and bool(HTTP_URL_RE.fullmatch(link)),
                 ErrorClass.INVALID_SCHEMA, f"{where}.link {link!r} is not an http(s) URL")
        # Duplicates would inflate coverage for free: ten copies of one page is one page. Compared
        # by upstream's own source key, so tracking parameters cannot disguise a repeat.
        key = source_key(link)
        _require(key not in seen_links, ErrorClass.INVALID_SCHEMA,
                 f"{where}.link {link!r} repeats an earlier result")
        seen_links.add(key)
        raw_highlights = item.get("highlights", [])
        _require(isinstance(raw_highlights, list), ErrorClass.INVALID_SCHEMA,
                 f"{where}.highlights must be a list")
        _require(len(raw_highlights) <= MAX_HIGHLIGHTS_PER_RESULT, ErrorClass.EXCESS_OUTPUT,
                 f"{where}.highlights exceeds {MAX_HIGHLIGHTS_PER_RESULT}")
        highlights = tuple(_text(h, f"{where}.highlights[{i}]")
                           for i, h in enumerate(raw_highlights))
        results.append(SearchResult(
            link=link,
            title=_text(item.get("title", ""), f"{where}.title"),
            snippet=_text(item.get("snippet", ""), f"{where}.snippet"),
            highlights=highlights,
            text=_text(item.get("text", ""), f"{where}.text")))

    raw_tweets = document.get("tweets", [])
    _require(isinstance(raw_tweets, list), ErrorClass.INVALID_SCHEMA, "tweets must be a list")
    _require(len(raw_tweets) <= task.limits.max_results, ErrorClass.EXCESS_OUTPUT,
             f"{len(raw_tweets)} tweets exceeds the limit of {task.limits.max_results}")
    tweets = []
    for index, item in enumerate(raw_tweets):
        where = f"tweets[{index}]"
        _require(isinstance(item, dict), ErrorClass.INVALID_SCHEMA, f"{where} must be an object")
        # Shape is NOT validated here beyond "it is an object with an id". Upstream's own
        # `is_valid_tweet` decides validity during scoring, and a tweet rejected there scores zero
        # rather than invalidating the whole response -- which is the difference between a miner
        # that returned one bad tweet and one that returned nothing usable.
        _require(isinstance(item.get("id"), str) and bool(item.get("id")),
                 ErrorClass.INVALID_SCHEMA, f"{where}.id must be a non-empty string")
        tweets.append(TweetResult(fields=dict(item)))

    raw_citations = document.get("citations", [])
    _require(isinstance(raw_citations, list), ErrorClass.INVALID_SCHEMA, "citations must be a list")
    _require(len(raw_citations) <= MAX_CITATIONS_PER_TASK, ErrorClass.EXCESS_OUTPUT,
             f"{len(raw_citations)} citations exceeds {MAX_CITATIONS_PER_TASK}")
    citations = []
    for index, item in enumerate(raw_citations):
        where = f"citations[{index}]"
        _require(isinstance(item, dict), ErrorClass.INVALID_SCHEMA, f"{where} must be an object")
        link = item.get("link")
        _require(isinstance(link, str) and bool(HTTP_URL_RE.fullmatch(link)),
                 ErrorClass.INVALID_SCHEMA, f"{where}.link {link!r} is not an http(s) URL")
        citations.append(Citation(link=link,
                                  claim=_text(item.get("claim", ""), f"{where}.claim")))

    raw_usage = document.get("usage", {})
    _require(isinstance(raw_usage, dict), ErrorClass.INVALID_SCHEMA, "usage must be an object")
    usage = ToolUsage(
        provider_calls=_non_negative_int(raw_usage.get("provider_calls", 0),
                                         "usage.provider_calls"),
        tokens=_non_negative_int(raw_usage.get("tokens", 0), "usage.tokens"),
        elapsed_seconds=_non_negative_float(raw_usage.get("elapsed_seconds", 0.0),
                                            "usage.elapsed_seconds"),
    )
    return TaskOutput(protocol_version=PROTOCOL_VERSION, task_id=task.task_id, summary=summary,
                      results=tuple(results), tweets=tuple(tweets), citations=tuple(citations),
                      usage=usage)


def _non_negative_int(value: Any, where: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
             ErrorClass.INVALID_SCHEMA, f"{where} must be a non-negative integer")
    return int(value)


def _non_negative_float(value: Any, where: str) -> float:
    import math

    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             ErrorClass.INVALID_SCHEMA, f"{where} must be a number")
    value = float(value)
    # NaN and inf poison every downstream comparison: `nan > x` is False for every x, so a NaN
    # latency would quietly win every tie-break it touched.
    _require(math.isfinite(value) and value >= 0.0, ErrorClass.INVALID_SCHEMA,
             f"{where} must be finite and non-negative")
    return value
