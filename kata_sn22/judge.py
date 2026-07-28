"""SN22's judge: the paid LLM that decides whether a source answers the question.

Upstream scores an answer in two LLM passes, and Kata runs the same two:

* **body relevance** — for each link that CLEARED EVIDENCE (see
  :func:`kata_sn22.upstream_adapter.link_meets_evidence`), how useful the verified excerpts are for
  the question. Only the excerpts are shown, never the whole page — the miner is graded on what it
  proved it read.
* **summary groundedness** — whether every number, date and name in the answer is supported by the
  very source the answer cites for it. This is the check that catches citing a real page for a value
  that page never gives.

**The prompts below are upstream's, verbatim.** They are not documentation of the policy, they ARE
the policy: reword a rule and the lane scores differently from the subnet it claims to follow. They
were extracted from the pinned tree rather than retyped, and
``tests/test_sn22_judge.py`` asserts each one is still byte-identical to the vendored source. A
reader who wants to change how SN22 scores must change SN22, not this file.

**What this module does not do:** call anything. A judge call costs money and takes network, so the
transport is a seam (:class:`JudgeClient`). Production passes an HTTP client; calibration passes
:class:`RecordedJudge`, which replays a cassette so a 30-run calibration costs nothing and gives the
same answer twice. The scoring logic is identical on both paths — only where the text comes from
differs.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from kata_sn22.judge_prompts import (
    SYSTEM_BODY_LINK_RELEVANCE_TEMPLATE,
    SYSTEM_SUMMARY_GROUNDEDNESS_TEMPLATE,
    SYSTEM_TWEET_RELEVANCE_TEMPLATE,
    USER_BODY_LINK_RELEVANCE_TEMPLATE,
    USER_SUMMARY_GROUNDEDNESS_TEMPLATE,
)

#: Upstream's model for scoring calls (``desearch/utils.py::call_scoring_llm``). Recorded here so
#: the cassette and any live client name the same judge; a different model is a different policy.
JUDGE_MODEL = "gpt-4.1-nano"

#: Upstream calls with this temperature: not 0, but low enough to be near-deterministic. Copied
#: rather than rounded to 0 — the judge's own variance is part of what makes SN22 NOISY, and
#: pretending otherwise would justify a score cache upstream does not have.
JUDGE_TEMPERATURE = 0.0001

#: The verdict vocabulary and its scores, from ``prompts.py``. HIGH/MEDIUM/LOW for relevance;
#: HIGH/MEDIUM/FAIL for groundedness. Both map through one table because upstream uses one.
VERDICT_SCORE: dict[str, float] = {"HIGH": 3.0, "MEDIUM": 1.5, "FAIL": 0.0, "LOW": 0.0}

#: Upstream divides the groundedness verdict by 3.0 to land it in 0..1. The relevance verdicts are
#: divided by the same scale where they are combined.
VERDICT_SCALE = 3.0

_VERDICT_RE = re.compile(r"(?im)\bverdict\b\s*[:\-]?\s*([A-Z]+)")
_LABEL_RE = re.compile(r"(?i)\b(HIGH|MEDIUM|FAIL)\b")


# ---------------------------------------------------------------------------------------------
# Reading a verdict
# ---------------------------------------------------------------------------------------------


def verdict_score(response: str) -> float:
    """The numeric verdict in a judge reply, or 0.0 when there isn't one.

    Two passes, and the order matters. An explicit ``Verdict: X`` line wins, because the prompt asks
    for one and a reply that gives it means it. Only if there is none does a bare HIGH/MEDIUM/FAIL
    token count -- and the LAST match is taken either way, since a model that reasons before
    answering mentions the labels on its way to a conclusion.

    An unparseable reply scores 0.0 rather than raising: a judge that returns nonsense must not stop
    the round, and the source simply earns nothing.
    """
    if not response:
        return 0.0
    if matches := _VERDICT_RE.findall(response):
        return VERDICT_SCORE.get(matches[-1].upper(), 0.0)
    if matches := _LABEL_RE.findall(response):
        return VERDICT_SCORE[matches[-1].upper()]
    return 0.0


def verdict_relevance(response: str) -> str:
    """The verdict label, collapsing anything unrecognised to ``LOW``. Fail closed: an unreadable
    reply is not evidence that a source was good."""
    if matches := _VERDICT_RE.findall(response or ""):
        label = matches[-1].upper()
        if label in ("HIGH", "MEDIUM"):
            return label
    return "LOW"


# ---------------------------------------------------------------------------------------------
# Building the calls
# ---------------------------------------------------------------------------------------------


def build_body_relevance_messages(prompt: str, url: str, title: str, body: str) -> list | None:
    """Messages for judging one web source. ``None`` when there is no body to judge -- an empty
    body is not a LOW verdict, it is a call not worth paying for."""
    if not body:
        return None
    return [
        {"role": "system", "content": SYSTEM_BODY_LINK_RELEVANCE_TEMPLATE},
        {"role": "user", "content": USER_BODY_LINK_RELEVANCE_TEMPLATE.format(prompt, url, title,
                                                                             body)},
    ]


def build_tweet_relevance_messages(prompt: str, url: str, title: str, body: str) -> list | None:
    """Messages for judging one tweet. Same user template as a web source, different system prompt:
    a tweet that points at an answer is useful in a way a stub article is not, and judging it by the
    article rubric would mark every tweet LOW."""
    if not body:
        return None
    return [
        {"role": "system", "content": SYSTEM_TWEET_RELEVANCE_TEMPLATE},
        {"role": "user", "content": USER_BODY_LINK_RELEVANCE_TEMPLATE.format(prompt, url, title,
                                                                             body)},
    ]


def build_summary_groundedness_messages(prompt: str, summary: str, cited_sources: str) -> list:
    """Messages for judging whether the answer is grounded in the sources it cites."""
    return [
        {"role": "system", "content": SYSTEM_SUMMARY_GROUNDEDNESS_TEMPLATE},
        {"role": "user", "content": USER_SUMMARY_GROUNDEDNESS_TEMPLATE.format(prompt, summary,
                                                                              cited_sources)},
    ]


# ---------------------------------------------------------------------------------------------
# The transport seam
# ---------------------------------------------------------------------------------------------


class JudgeUnavailable(Exception):
    """The judge could not be reached or refused the call.

    Distinct from a LOW verdict on purpose. "The source is bad" and "we could not find out" must not
    be the same outcome: the first is a score, the second is a reason to abandon the round rather
    than rank a contestant on evidence that was never gathered.
    """


class JudgeClient(Protocol):
    """Anything that can turn judge messages into the judge's reply text."""

    def __call__(self, messages: list) -> str: ...


@dataclass
class JudgeCall:
    """One recorded judge exchange. The key is over the MESSAGES, so a cassette entry can only be
    replayed for the exact question it was recorded for."""

    key: str
    reply: str

    @staticmethod
    def key_for(messages: list) -> str:
        canonical = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class RecordedJudge:
    """A judge that replays a cassette instead of paying for a call.

    Calibration needs 30+ paired runs (plan §5.5). A live judge would make that both expensive and
    unrepeatable -- the thing being measured would move while it was measured. A cassette makes it
    free and deterministic.

    A miss RAISES rather than falling back to a default verdict. A cassette that silently answered
    LOW for anything it had not seen would turn "this calibration is incomplete" into "these agents
    scored badly", which is the same number with the opposite meaning.
    """

    replies: dict[str, str] = field(default_factory=dict)
    #: Keys actually replayed, so a calibration can report what its cassette did not cover.
    used: set = field(default_factory=set)

    @classmethod
    def from_file(cls, path: str | Path) -> "RecordedJudge":
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = document.get("calls") if isinstance(document, dict) else document
        return cls(replies={entry["key"]: entry["reply"] for entry in entries or []})

    def __call__(self, messages: list) -> str:
        key = JudgeCall.key_for(messages)
        if key not in self.replies:
            raise JudgeUnavailable(
                f"no recorded judge reply for {key[:12]}...; re-record the cassette rather than "
                f"scoring against a default verdict")
        self.used.add(key)
        return self.replies[key]

    @property
    def unused_keys(self) -> set:
        return set(self.replies) - self.used


@dataclass
class RecordingJudge:
    """Wraps a live judge and captures every exchange, to produce a cassette.

    Used once, deliberately, under an approved budget -- never on the scoring path.
    """

    inner: JudgeClient
    calls: list = field(default_factory=list)

    def __call__(self, messages: list) -> str:
        reply = self.inner(messages)
        self.calls.append({"key": JudgeCall.key_for(messages), "reply": reply})
        return reply

    def as_document(self) -> dict:
        return {"schema_version": 1, "model": JUDGE_MODEL, "temperature": JUDGE_TEMPERATURE,
                "calls": self.calls}


def judge_relevance(client: JudgeClient, messages: list | None) -> float:
    """Score one source in 0..1. ``None`` messages (nothing to judge) score 0.0 without a call."""
    if not messages:
        return 0.0
    return verdict_score(client(messages)) / VERDICT_SCALE
