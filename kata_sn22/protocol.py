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

#: Bumped whenever a field's meaning changes. A submission built against a different version is
#: refused, not coerced.
PROTOCOL_VERSION = 1

#: Hard ceiling on a single agent response, before parsing. Applied to BYTES, not to the parsed
#: object: a 500 MB JSON document is an availability problem while it is still being decoded, so the
#: check has to happen before ``json.loads`` ever sees it.
MAX_OUTPUT_BYTES = 256 * 1024
#: Ceilings on the shape of a response. Each bounds work the judge would otherwise have to do.
MAX_RESULTS_PER_TASK = 20
MAX_CITATIONS_PER_TASK = 40
MAX_TEXT_CHARS = 8_000

#: Search modes, mirroring the audited upstream. ``ai_mode`` applies only to AI search.
SEARCH_TYPES = ("ai_search", "x_search")
AI_MODES = ("fast", "balanced", "deep")
RESULT_TYPES = ("summary", "links", "both")

#: A document identifier must be STABLE — the same document is the same id in the sealed
#: snapshot and in a citation — and inert, because it is used as a dict key and printed into
#: reports.
DOC_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,127}$")
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
    """The quotas a task runs under. Handed to the agent so it can plan, and enforced anyway."""

    max_wall_seconds: int = 120
    max_provider_calls: int = 8
    max_tokens: int = 20_000
    max_results: int = MAX_RESULTS_PER_TASK

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
    """A claim the agent says the snapshot supports. ``doc_id`` must exist in that snapshot."""

    doc_id: str
    claim: str

    def as_dict(self) -> dict:
        return {"doc_id": self.doc_id, "claim": self.claim}


@dataclass(frozen=True)
class SearchResult:
    """One retrieved item. ``doc_id`` ties it to the sealed snapshot rather than to a live URL."""

    doc_id: str
    title: str
    snippet: str

    def as_dict(self) -> dict:
        return {"doc_id": self.doc_id, "title": self.title, "snippet": self.snippet}


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

    def as_dict(self) -> dict:
        return {
            "protocol_version": self.protocol_version,
            "task_id": self.task_id,
            "summary": self.summary,
            "results": [r.as_dict() for r in self.results],
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
    seen_docs: set[str] = set()
    for index, item in enumerate(raw_results):
        where = f"results[{index}]"
        _require(isinstance(item, dict), ErrorClass.INVALID_SCHEMA, f"{where} must be an object")
        doc_id = item.get("doc_id")
        _require(isinstance(doc_id, str) and bool(DOC_ID_RE.fullmatch(doc_id)),
                 ErrorClass.INVALID_SCHEMA, f"{where}.doc_id {doc_id!r} is not a stable identifier")
        # Duplicates would inflate coverage for free: ten copies of one document is one document.
        _require(doc_id not in seen_docs, ErrorClass.INVALID_SCHEMA,
                 f"{where}.doc_id {doc_id!r} is repeated")
        seen_docs.add(doc_id)
        results.append(SearchResult(doc_id=doc_id,
                                    title=_text(item.get("title", ""), f"{where}.title"),
                                    snippet=_text(item.get("snippet", ""), f"{where}.snippet")))

    raw_citations = document.get("citations", [])
    _require(isinstance(raw_citations, list), ErrorClass.INVALID_SCHEMA, "citations must be a list")
    _require(len(raw_citations) <= MAX_CITATIONS_PER_TASK, ErrorClass.EXCESS_OUTPUT,
             f"{len(raw_citations)} citations exceeds {MAX_CITATIONS_PER_TASK}")
    citations = []
    for index, item in enumerate(raw_citations):
        where = f"citations[{index}]"
        _require(isinstance(item, dict), ErrorClass.INVALID_SCHEMA, f"{where} must be an object")
        doc_id = item.get("doc_id")
        _require(isinstance(doc_id, str) and bool(DOC_ID_RE.fullmatch(doc_id)),
                 ErrorClass.INVALID_SCHEMA, f"{where}.doc_id {doc_id!r} is not a stable identifier")
        citations.append(Citation(doc_id=doc_id,
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
                      results=tuple(results), citations=tuple(citations), usage=usage)


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
