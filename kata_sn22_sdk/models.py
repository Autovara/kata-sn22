"""What a submission receives and what it returns.

These are the *agent's* view of the version-2 protocol. They deliberately mirror
:mod:`kata_sn22.protocol_v2` without importing it: that module also carries the scoring surface,
the parser and the fixed judge identity, and none of that belongs in an image that runs untrusted
code. The harness on the trusted side is what reconciles the two, and
``tests/test_sn22_sdk.py`` holds them to the same shape.

**A synapse carries no ``deep`` flag.** Whether a task is one of the pool's deep-scored samples is
the manifest's business. An agent that knew would work hardest on exactly those, and the 20% sample
would stop measuring the other 80%.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

PROTOCOL_VERSION = 2


class SearchType(str, Enum):
    AI_SEARCH = "ai_search"
    X_SEARCH = "x_search"


class SearchMode(str, Enum):
    """Each mode is its own scoring pool, and each has its own execution budget."""

    FAST = "fast"
    BALANCED = "balanced"
    DEEP = "deep"


class ResultType(str, Enum):
    """``ONLY_LINKS`` means there is no summary to judge.

    Worth reading as an instruction: a task that asked for no summary is not marked down for not
    having one, and writing one anyway spends tokens on something nobody grades.
    """

    ONLY_LINKS = "ONLY_LINKS"
    LINKS_WITH_FINAL_SUMMARY = "LINKS_WITH_FINAL_SUMMARY"


class ScraperTextRole(str, Enum):
    """Which text block a streamed chunk belongs to.

    ``FINAL_SUMMARY`` is the one the groundedness judge reads. The others exist because a streamed
    answer emits them and the streaming penalty counts what was emitted.
    """

    INTRO = "intro"
    TWITTER_SUMMARY = "twitter_summary"
    SEARCH_SUMMARY = "search_summary"
    REDDIT_SUMMARY = "reddit_summary"
    HACKER_NEWS_SUMMARY = "hacker_news_summary"
    FINAL_SUMMARY = "summary"


class SdkError(Exception):
    """The task was not something this SDK can present, or the answer not something it can frame."""


@dataclass(frozen=True)
class Limits:
    """Bounds both contestants receive identically. Not miner input.

    ``max_execution_time`` is upstream's own serving budget for the mode. The performance multiplier
    and the timeout penalty are both measured against it, so answering fast is worth points and
    overrunning is worth less than answering badly on time.
    """

    max_execution_time: int = 30
    max_provider_calls: int = 32
    max_tokens: int = 60_000
    max_output_bytes: int = 2_000_000


@dataclass(frozen=True)
class AiSearchSynapse:
    """One AI-search task. ``smart_scraper`` receives this."""

    task_id: str
    prompt: str
    mode: SearchMode
    result_type: ResultType
    tools: tuple[str, ...]
    count: int = 10
    system_message: str = ""
    include_domains: tuple[str, ...] = ()
    exclude_domains: tuple[str, ...] = ()
    start_date: str | None = None
    end_date: str | None = None
    date_filter_type: str | None = None
    limits: Limits = field(default_factory=Limits)

    search_type: SearchType = SearchType.AI_SEARCH

    @property
    def wants_summary(self) -> bool:
        """Whether this task is graded on a final summary at all."""
        return self.result_type is ResultType.LINKS_WITH_FINAL_SUMMARY


@dataclass(frozen=True)
class XSearchSynapse:
    """One Basic X-search task. ``twitter_search`` receives this."""

    task_id: str
    query: str
    count: int = 10
    sort: str | None = None
    user: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    lang: str | None = None
    verified: bool | None = None
    blue_verified: bool | None = None
    is_quote: bool | None = None
    is_video: bool | None = None
    is_image: bool | None = None
    min_retweets: int | None = None
    min_replies: int | None = None
    min_likes: int | None = None
    limits: Limits = field(default_factory=Limits)

    search_type: SearchType = SearchType.X_SEARCH


Synapse = AiSearchSynapse | XSearchSynapse


@dataclass
class AiSearchResult:
    """What ``smart_scraper`` returns.

    Mutable and default-empty on purpose: a submission fills in what it produced, and anything it
    leaves alone is derived from what it streamed. See :class:`kata_sn22_sdk.agent.Emit`.
    """

    #: ``{title, link, snippet}`` per web result, in upstream's own shape. The validator fetches
    #: every link itself, so a link that is not real is worse than one fewer link.
    search_results: list = field(default_factory=list)
    #: Raw tweet objects. The validator re-scrapes each one and compares it field by field, so an
    #: edited tweet scores zero rather than less.
    miner_tweets: list = field(default_factory=list)
    #: Set to override the completion derived from the streamed chunks. Rarely what you want.
    completion: str | None = None
    #: Set to override the per-role texts derived from the streamed chunks.
    texts: dict | None = None


@dataclass
class XSearchResult:
    """What ``twitter_search`` returns."""

    results: list = field(default_factory=list)


def _enum(kind, value, field_name: str):
    try:
        return kind(value)
    except ValueError as exc:
        raise SdkError(f"unknown {field_name} {value!r}") from exc


def _limits(document: dict) -> Limits:
    raw = document.get("limits") or {}
    if not isinstance(raw, dict):
        raise SdkError("limits must be an object")
    known = {name: raw[name] for name in
             ("max_execution_time", "max_provider_calls", "max_tokens", "max_output_bytes")
             if name in raw}
    return Limits(**known)


def synapse_from_input(document: dict) -> Synapse:
    """Build the synapse for one task descriptor, or raise.

    Fails closed on an unknown protocol version rather than guessing: a lenient read of a schema
    nobody recognises is how a field stops being checked.
    """
    if not isinstance(document, dict):
        raise SdkError("a task must be a JSON object")
    version = document.get("protocol_version")
    if version != PROTOCOL_VERSION:
        raise SdkError(f"unsupported protocol_version {version!r}; this SDK speaks "
                       f"{PROTOCOL_VERSION}")
    task_id = document.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise SdkError("a task must carry a task_id")

    search_type = _enum(SearchType, document.get("search_type"), "search_type")
    if search_type is SearchType.X_SEARCH:
        return XSearchSynapse(
            task_id=task_id,
            query=str(document.get("query") or ""),
            count=int(document.get("count") or 10),
            sort=document.get("sort"),
            user=document.get("user"),
            start_date=document.get("start_date"),
            end_date=document.get("end_date"),
            lang=document.get("lang"),
            **{name: document[name] for name in
               ("verified", "blue_verified", "is_quote", "is_video", "is_image",
                "min_retweets", "min_replies", "min_likes") if name in document},
            limits=_limits(document),
        )
    return AiSearchSynapse(
        task_id=task_id,
        prompt=str(document.get("prompt") or ""),
        mode=_enum(SearchMode, document.get("mode"), "mode"),
        result_type=_enum(ResultType, document.get("result_type"), "result_type"),
        tools=tuple(document.get("tools") or ()),
        count=int(document.get("count") or 10),
        system_message=str(document.get("system_message") or ""),
        include_domains=tuple(document.get("include_domains") or ()),
        exclude_domains=tuple(document.get("exclude_domains") or ()),
        start_date=document.get("start_date"),
        end_date=document.get("end_date"),
        date_filter_type=document.get("date_filter_type"),
        limits=_limits(document),
    )
