"""Protocol version 2: the upstream SN22 task and response surface, carried faithfully.

Version 1 (:mod:`kata_sn22.protocol`) defined Kata's *own* task and answer shapes. That was the
right size for a sealed offline challenge and is the wrong size for production: upstream's scorer
reads fields v1 never carried, so a v1 answer cannot be scored by upstream code at all.

**What this module is.** The exact surface the pinned upstream scoring path reads off a
``ScraperStreamingSynapse`` or a ``TwitterSearchSynapse`` — no more and no less. The field set was
not chosen by reading the synapse definitions, which carry ~40 transport and presentation fields
between them; it was *derived* from what the reward, penalty and scheduler modules actually access.
:func:`scoring_surface` records that derivation so it can be re-checked against a new upstream.

**Why not just import the upstream models.** They are pydantic models on a `bittensor.Synapse` base,
so importing them drags a chain client into whatever process touches a task. These are plain
dataclasses that *serialize into* the upstream models inside the trusted runner, where that
dependency is acceptable and attested. The lane, the bot and a miner's tooling stay dependency-free.

**Fail closed on anything unrecognised.** An unknown field or a version that is not 2 is an error,
never a silently ignored extra. A task a miner does not fully understand is a task it should refuse,
and an answer carrying fields the scorer will not read is an answer whose author believed something
false about how it would be graded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

#: This module's version. Bumped only when the SCORED surface changes; adding a transport field
#: that no reward component reads is not a protocol change.
PROTOCOL_VERSION = 2

#: The upstream tree this surface was derived from. A different commit may read different fields,
#: so a task manifest carrying a different one must not be scored with this module.
UPSTREAM_COMMIT = "bea9712f58a5fc01c57ec441ce279499529d8bf6"


class SearchType(str, Enum):
    """Upstream's two task families. The pool weights hang off this."""

    AI_SEARCH = "ai_search"
    X_SEARCH = "x_search"


class SearchMode(str, Enum):
    """``desearch.protocol.SearchMode``. AI search only; each mode is its own scoring pool."""

    FAST = "fast"
    BALANCED = "balanced"
    DEEP = "deep"


class ResultType(str, Enum):
    """``desearch.protocol.ResultType``.

    ``ONLY_LINKS`` reweights the AI quality split to (1.0, 0.0) -- there is no summary to judge, so
    a task that asked for none must not be marked down for not having one.
    """

    ONLY_LINKS = "ONLY_LINKS"
    LINKS_WITH_FINAL_SUMMARY = "LINKS_WITH_FINAL_SUMMARY"


class ScraperTextRole(str, Enum):
    """``desearch.protocol.ScraperTextRole``: which text block a completion chunk belongs to.

    The scorer reads ``texts[FINAL_SUMMARY]`` for groundedness. The other roles exist because a
    streamed answer emits them, and the streaming penalty counts what was emitted.
    """

    INTRO = "intro"
    TWITTER_SUMMARY = "twitter_summary"
    SEARCH_SUMMARY = "search_summary"
    REDDIT_SUMMARY = "reddit_summary"
    HACKER_NEWS_SUMMARY = "hacker_news_summary"
    FINAL_SUMMARY = "summary"


class ScoringModel(str, Enum):
    """``desearch.protocol.ScoringModel``.

    Production pins :data:`FIXED_SCORING_MODEL`. The others are listed because upstream defines them
    and a report naming one of them must be *recognised and rejected*, not treated as unparseable.
    """

    OPENAI_GPT4_1_NANO = "openai/gpt-4.1-nano"
    QWEN3_6_27B = "Qwen/Qwen3.6-27B-TEE"
    QWEN3_5_397B = "Qwen/Qwen3.5-397B-A17B-TEE"


#: The scorer every contestant is graded by, fixed for the whole duel. Upstream's validator default
#: (``neurons/validators/config.py``). Upstream falls back to gpt-4.1-nano when Chutes fails; Kata
#: disables that fallback, because one contestant graded by Qwen and another by GPT is not one
#: competition. A contestant whose Chutes credential fails scores zero instead.
FIXED_SCORING_MODEL = ScoringModel.QWEN3_5_397B

#: Upstream's minimum, and its default (``ScraperStreamingSynapse.count``: ``Field(10, ge=10)``).
#: v1 defaulted to 5, which upstream's own model would reject as below its minimum.
DEFAULT_RESULT_COUNT = 10
MIN_RESULT_COUNT = 10
MAX_RESULT_COUNT = 200


class ProtocolV2Error(ValueError):
    """A task or answer does not satisfy the version-2 contract."""


# ---------------------------------------------------------------------------------------------
# The scoring surface
# ---------------------------------------------------------------------------------------------

#: Fields the pinned upstream scoring path reads off a response, derived by scanning every
#: ``response.<field>`` / ``synapse.<field>`` access in ``neurons/validators/{reward,penalty,
#: scrapers,scoring}``. Kept as data so a test can re-derive it against the vendored tree and fail
#: when a new upstream reads something this module does not carry.
SCORED_RESPONSE_FIELDS: tuple[str, ...] = (
    "completion",
    "count",
    "end_date",
    "exclude_domains",
    "include_domains",
    "max_execution_time",
    "miner_tweets",
    "prompt",
    "result_type",
    "results",
    "search_results",
    "sort",
    "start_date",
    "system_message",
    "text_chunks",
    "texts",
    "tools",
    "validator_links",
    "validator_tweets",
)

#: Read off the task rather than the answer: the lane sets these, the miner cannot.
SCORED_TASK_FIELDS: tuple[str, ...] = (
    "count", "end_date", "exclude_domains", "include_domains", "max_execution_time", "mode",
    "prompt", "query", "result_type", "sort", "start_date", "tools",
)


def scoring_surface() -> dict[str, tuple[str, ...]]:
    """The derivation, as data. See :data:`SCORED_RESPONSE_FIELDS`."""
    return {"response": SCORED_RESPONSE_FIELDS, "task": SCORED_TASK_FIELDS}


# ---------------------------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Limits:
    """Bounds both contestants receive identically. Not miner input.

    ``max_execution_time`` is upstream's own serving budget for the mode, and the performance
    multiplier and timeout penalty are both measured against it -- so it is part of the task, not a
    Kata convenience.
    """

    max_execution_time: int
    max_provider_calls: int = 32
    max_tokens: int = 60_000
    max_output_bytes: int = 2_000_000

    def as_dict(self) -> dict:
        return {"max_execution_time": self.max_execution_time,
                "max_provider_calls": self.max_provider_calls,
                "max_tokens": self.max_tokens, "max_output_bytes": self.max_output_bytes}


@dataclass(frozen=True)
class AiSearchTask:
    """One upstream AI-search task (``ScraperStreamingSynapse`` input side)."""

    task_id: str
    prompt: str
    mode: SearchMode
    result_type: ResultType
    tools: tuple[str, ...]
    count: int = DEFAULT_RESULT_COUNT
    system_message: str = ""
    include_domains: tuple[str, ...] = ()
    exclude_domains: tuple[str, ...] = ()
    start_date: str | None = None
    end_date: str | None = None
    date_filter_type: str | None = None
    limits: Limits = field(default_factory=lambda: Limits(max_execution_time=30))
    #: Whether this task is one of the pool's deep-scored samples. Set by the manifest, identical
    #: for both contestants, and never visible to the agent -- an agent that knew which tasks were
    #: deep-scored would work hardest on exactly those.
    deep: bool = False

    search_type: SearchType = SearchType.AI_SEARCH

    def __post_init__(self) -> None:
        _require(bool(self.task_id), "task_id is required")
        _require(bool(self.prompt.strip()), "prompt must not be empty")
        _require(bool(self.tools), "an AI-search task must name at least one tool")
        _require(MIN_RESULT_COUNT <= self.count <= MAX_RESULT_COUNT,
                 f"count must be {MIN_RESULT_COUNT}..{MAX_RESULT_COUNT}, got {self.count}")

    def as_agent_input(self) -> dict:
        """What the agent receives. Deliberately WITHOUT ``deep``."""
        return {
            "protocol_version": PROTOCOL_VERSION,
            "task_id": self.task_id,
            "search_type": self.search_type.value,
            "prompt": self.prompt,
            "mode": self.mode.value,
            "result_type": self.result_type.value,
            "tools": list(self.tools),
            "count": self.count,
            "system_message": self.system_message,
            "include_domains": list(self.include_domains),
            "exclude_domains": list(self.exclude_domains),
            "start_date": self.start_date,
            "end_date": self.end_date,
            "date_filter_type": self.date_filter_type,
            "limits": self.limits.as_dict(),
        }


@dataclass(frozen=True)
class XSearchTask:
    """One upstream Basic X-search task (``TwitterSearchSynapse`` input side)."""

    task_id: str
    query: str
    count: int = DEFAULT_RESULT_COUNT
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
    limits: Limits = field(default_factory=lambda: Limits(max_execution_time=30))
    deep: bool = False

    search_type: SearchType = SearchType.X_SEARCH

    def __post_init__(self) -> None:
        _require(bool(self.task_id), "task_id is required")
        _require(bool(self.query.strip()), "query must not be empty")
        _require(MIN_RESULT_COUNT <= self.count <= MAX_RESULT_COUNT,
                 f"count must be {MIN_RESULT_COUNT}..{MAX_RESULT_COUNT}, got {self.count}")
        # `sort=Latest` is scored: results not in descending time order are an immediate zero.
        _require(self.sort in (None, "Latest", "Top"), f"unknown sort {self.sort!r}")

    def as_agent_input(self) -> dict:
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "task_id": self.task_id,
            "search_type": self.search_type.value,
            "query": self.query,
            "count": self.count,
            "sort": self.sort,
            "user": self.user,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "lang": self.lang,
            "limits": self.limits.as_dict(),
        }
        for name in ("verified", "blue_verified", "is_quote", "is_video", "is_image",
                     "min_retweets", "min_replies", "min_likes"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        return payload


Task = AiSearchTask | XSearchTask


# ---------------------------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamChunk:
    """One streamed completion chunk, in emission order.

    v1 returned a single JSON document, so upstream's ``streaming_penalty`` -- which counts tokens
    per chunk -- had no input and was excluded from the adapted set. Production restores it, which
    means the answer has to record what was streamed rather than only what it added up to.
    """

    role: ScraperTextRole
    text: str

    def as_dict(self) -> dict:
        return {"role": self.role.value, "text": self.text}


@dataclass(frozen=True)
class AiSearchAnswer:
    """One agent answer to an AI-search task (``ScraperStreamingSynapse`` output side)."""

    task_id: str
    completion: str
    texts: dict[str, str]
    chunks: tuple[StreamChunk, ...]
    #: Upstream's ``search_results``: ``{title, link, snippet}`` per web result. Left as dicts
    #: because the scorer reads them as dicts and any tidying here is a difference it cannot see.
    search_results: tuple[dict, ...] = ()
    #: Upstream's ``miner_tweets``: raw tweet objects, re-scraped and compared field by field.
    miner_tweets: tuple[dict, ...] = ()
    text_chunks: dict[str, list[str]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "task_id": self.task_id,
            "completion": self.completion,
            "texts": dict(self.texts),
            "chunks": [chunk.as_dict() for chunk in self.chunks],
            "search_results": [dict(item) for item in self.search_results],
            "miner_tweets": [dict(item) for item in self.miner_tweets],
            "text_chunks": {k: list(v) for k, v in self.text_chunks.items()},
        }


@dataclass(frozen=True)
class XSearchAnswer:
    """One agent answer to a Basic X-search task (``TwitterSearchSynapse`` output side)."""

    task_id: str
    results: tuple[dict, ...] = ()

    def as_dict(self) -> dict:
        return {"protocol_version": PROTOCOL_VERSION, "task_id": self.task_id,
                "results": [dict(item) for item in self.results]}


Answer = AiSearchAnswer | XSearchAnswer


# ---------------------------------------------------------------------------------------------
# Parsing, fail-closed
# ---------------------------------------------------------------------------------------------

_AI_ANSWER_FIELDS = {"protocol_version", "task_id", "completion", "texts", "chunks",
                     "search_results", "miner_tweets", "text_chunks"}
_X_ANSWER_FIELDS = {"protocol_version", "task_id", "results"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolV2Error(message)


def _object(raw: bytes | str | dict, *, max_bytes: int) -> dict:
    if isinstance(raw, dict):
        return raw
    payload = raw.encode("utf-8") if isinstance(raw, str) else raw
    # Size is checked on BYTES before decoding: an oversized document costs the lane its parse time
    # whether or not it is valid, so it has to be refused before that is spent.
    _require(len(payload) <= max_bytes,
             f"answer is {len(payload)} bytes, over the {max_bytes} ceiling")
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProtocolV2Error(f"answer is not valid JSON: {exc}") from exc
    _require(isinstance(document, dict), "answer is not a JSON object")
    return document


def _check_envelope(document: dict, task: Task, allowed: set[str]) -> None:
    version = document.get("protocol_version")
    _require(version == PROTOCOL_VERSION,
             f"protocol_version {version!r} is not {PROTOCOL_VERSION}")
    _require(document.get("task_id") == task.task_id,
             f"answer task_id {document.get('task_id')!r} != requested {task.task_id!r}")
    # Unknown fields are an ERROR, not extras to ignore. An agent sending a field the scorer will
    # never read believed something false about how it would be graded, and silently dropping it
    # would let that belief survive to the next round.
    unknown = sorted(set(document) - allowed)
    _require(not unknown, f"answer carries unknown field(s): {', '.join(unknown)}")


def _dicts(value: Any, where: str) -> tuple[dict, ...]:
    _require(isinstance(value, list), f"{where} must be a list")
    for index, item in enumerate(value):
        _require(isinstance(item, dict), f"{where}[{index}] must be an object")
    return tuple(value)


def parse_answer(raw: bytes | str | dict, *, task: Task) -> Answer:
    """Parse one agent answer, or raise :class:`ProtocolV2Error`.

    Nothing is coerced, defaulted or repaired: a repaired answer is one the contestant did not
    produce, and scoring it would score a document the lane wrote.
    """
    document = _object(raw, max_bytes=task.limits.max_output_bytes)
    if isinstance(task, XSearchTask):
        _check_envelope(document, task, _X_ANSWER_FIELDS)
        return XSearchAnswer(task_id=task.task_id,
                             results=_dicts(document.get("results", []), "results"))

    _check_envelope(document, task, _AI_ANSWER_FIELDS)
    texts = document.get("texts", {})
    _require(isinstance(texts, dict), "texts must be an object")
    for role in texts:
        _require(role in {member.value for member in ScraperTextRole},
                 f"texts has unknown role {role!r}")
    _require(all(isinstance(v, str) for v in texts.values()), "every texts value must be a string")

    raw_chunks = document.get("chunks", [])
    _require(isinstance(raw_chunks, list), "chunks must be a list")
    chunks = []
    for index, item in enumerate(raw_chunks):
        _require(isinstance(item, dict), f"chunks[{index}] must be an object")
        _require(set(item) <= {"role", "text"}, f"chunks[{index}] has unknown fields")
        try:
            role = ScraperTextRole(item.get("role"))
        except ValueError as exc:
            raise ProtocolV2Error(f"chunks[{index}].role {item.get('role')!r} is unknown") from exc
        _require(isinstance(item.get("text"), str), f"chunks[{index}].text must be a string")
        chunks.append(StreamChunk(role=role, text=item["text"]))

    text_chunks = document.get("text_chunks", {})
    _require(isinstance(text_chunks, dict), "text_chunks must be an object")
    for key, value in text_chunks.items():
        _require(isinstance(value, list) and all(isinstance(v, str) for v in value),
                 f"text_chunks[{key!r}] must be a list of strings")

    completion = document.get("completion", "")
    _require(isinstance(completion, str), "completion must be a string")
    return AiSearchAnswer(
        task_id=task.task_id, completion=completion, texts=dict(texts), chunks=tuple(chunks),
        search_results=_dicts(document.get("search_results", []), "search_results"),
        miner_tweets=_dicts(document.get("miner_tweets", []), "miner_tweets"),
        text_chunks={k: list(v) for k, v in text_chunks.items()},
    )
