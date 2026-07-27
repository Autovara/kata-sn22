"""The SN22 scoring schema and comparator (plan §5.3–§5.4, SN22-2).

Two things live here, and the separation matters.

**The signals** are the seven ordered quantities a challenge publishes (§5.4). They are computed
from the sealed snapshot, not from anything the candidate asserts: a citation counts only if the
snapshot actually contains the document, and relevance is measured against the snapshot's own
ground truth.
An agent cannot improve its score by claiming a better one.

**The comparator** decides a crown from those signals by fixed lexicographic priority — validity
first, then quality, then precision, coverage, invalid runs, cost, latency. Lexicographic rather
than a weighted sum on purpose: a weighted sum lets a candidate buy a quality win with unlimited
spend, whereas priority order says plainly that an agent which answers fewer queries validly does
not win by being prettier on the ones it did answer.

**Upstream weights** (§5.3) are kept where the corresponding component exists: AI search 0.90 / X
search 0.10; within AI search fast 0.60, balanced 0.20, deep 0.20; AI quality content 0.60 / summary
0.40. This deliberately does NOT claim to reproduce on-chain emissions, which are pool-relative and
depend on miner population — Kata compares exactly two agents on one sealed challenge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from kata_sn22.manifests import SnapshotManifest, UsageManifest
from kata_sn22.protocol import ErrorClass, Task, TaskOutput

#: Upstream search-type weights (§5.3).
SEARCH_TYPE_WEIGHTS = {"ai_search": 0.90, "x_search": 0.10}
#: Within AI search, by mode.
AI_MODE_WEIGHTS = {"fast": 0.60, "balanced": 0.20, "deep": 0.20}
#: Within AI quality: how much of the score is the retrieved content vs the written summary.
AI_QUALITY_WEIGHTS = {"content_relevance": 0.60, "summary_relevance": 0.40}

#: The published rank signals, in promotion priority order. ``higher`` says which direction wins.
#: This tuple IS the contract: the comparator walks it in order, so reordering it changes
#: promotions.
RANK_SIGNALS: tuple[tuple[str, bool], ...] = (
    ("sn22_valid_query_rate", True),
    ("sn22_weighted_quality", True),
    ("sn22_citation_precision", True),
    ("sn22_coverage", True),
    ("sn22_invalid_runs", False),
    ("sn22_cost_units", False),
    ("sn22_latency_seconds", False),
)


@dataclass(frozen=True)
class TaskAttempt:
    """One contestant's attempt at one task: either a validated output, or a classified failure.

    ``observed_seconds`` is the LANE's own measurement of how long the attempt took. It is not the
    agent's ``usage.elapsed_seconds``, and the difference is the whole point: latency is a ranked
    signal, and any ranked signal taken from a candidate's self-report is a signal the candidate can
    win by lying about. The self-reported figure is kept for display and cross-checking only.
    """

    task: Task
    output: TaskOutput | None = None
    error: ErrorClass | None = None
    observed_seconds: float = 0.0

    def __post_init__(self) -> None:
        if (self.output is None) == (self.error is None):
            raise ValueError("a task attempt is exactly one of an output or a classified error")
        if not math.isfinite(self.observed_seconds) or self.observed_seconds < 0:
            raise ValueError("observed_seconds must be finite and non-negative")

    @property
    def usable(self) -> bool:
        return self.output is not None

    @property
    def infrastructure_fault(self) -> bool:
        """A shared-infrastructure failure, which is nobody's candidate's fault."""
        return self.error is not None and not self.error.candidate_caused


@dataclass(frozen=True)
class Signals:
    """The seven published signals for one contestant, plus display detail."""

    sn22_valid_query_rate: float
    sn22_weighted_quality: float
    sn22_citation_precision: float
    sn22_coverage: float
    sn22_invalid_runs: int
    sn22_cost_units: float
    sn22_latency_seconds: float
    detail: dict = field(default_factory=dict)

    def as_metrics(self) -> dict:
        return {name: getattr(self, name) for name, _higher in RANK_SIGNALS}

    def rank_tuple(self) -> tuple[float, ...]:
        """Signals oriented so that GREATER IS ALWAYS BETTER, for a plain lexicographic compare.

        Lower-is-better signals are negated here rather than special-cased in the comparator, so
        there is exactly one place where a signal's direction is expressed.
        """
        return tuple(float(getattr(self, name)) if higher else -float(getattr(self, name))
                     for name, higher in RANK_SIGNALS)


def _relevance(output: TaskOutput, snapshot: SnapshotManifest, task: Task) -> tuple[float, float]:
    """(content_relevance, summary_relevance) for one answered task, measured against the snapshot.

    Content relevance is recall over the sealed ground truth: of the documents that genuinely answer
    this query, how many did the agent return. Summary relevance is how much of that same ground
    truth the written summary actually reflects, checked by title token overlap — a stand-in for the
    LLM judge, chosen because it is deterministic and offline, which is what SN22-2 requires.
    """
    truth = snapshot.relevant(task.task_id)
    if not truth:
        return 0.0, 0.0
    returned = {result.doc_id for result in output.results}
    content = len(truth & returned) / len(truth)

    summary_words = {word.strip(".,:;!?").lower() for word in output.summary.split()}
    covered = 0
    for doc_id in truth:
        document = snapshot.document(doc_id)
        if document is None:
            continue
        title_words = {word.strip(".,:;!?").lower() for word in document.title.split()}
        # A title is "reflected" when the summary mentions most of its distinctive words. Simple and
        # deterministic; the production judge replaces this behind the same interface.
        if title_words and len(title_words & summary_words) / len(title_words) >= 0.5:
            covered += 1
    return content, covered / len(truth)


def _task_weight(task: Task) -> float:
    """The upstream weight for one task's category (§5.3)."""
    weight = SEARCH_TYPE_WEIGHTS.get(task.search_type, 0.0)
    if task.search_type == "ai_search":
        weight *= AI_MODE_WEIGHTS.get(task.ai_mode or "", 0.0)
    return weight


def score_attempts(attempts: list[TaskAttempt], *, snapshot: SnapshotManifest,
                   usage: UsageManifest, variant: str) -> Signals:
    """Turn one contestant's attempts into the seven published signals.

    ``usage`` is REQUIRED, and it is the relay's record rather than the agent's. Plan §5.4 calls
    signal 6 "*enforced* candidate-side cost" for this reason: the relay is the party that made the
    calls and did the billing, while the candidate is the party with a motive to report zero. Making
    the argument optional would leave a path where a challenge is ranked on numbers the challenger
    chose, so there is no such path.
    """
    if not attempts:
        raise ValueError("cannot score an empty attempt set")

    # Infrastructure faults are excluded from the denominator entirely. Counting them would penalise
    # whichever contestant happened to be running when a provider blipped.
    scorable = [a for a in attempts if not a.infrastructure_fault]
    if not scorable:
        return Signals(0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0,
                       detail={"note": "every task hit a shared-infrastructure fault"})

    usable = [a for a in scorable if a.usable]
    valid_rate = len(usable) / len(scorable)
    invalid_runs = sum(1 for a in scorable if not a.usable)

    weighted_quality = 0.0
    total_weight = 0.0
    cited = 0
    cited_supported = 0
    coverage_sum = 0.0
    latency = 0.0
    self_reported_calls = 0
    self_reported_tokens = 0

    for attempt in scorable:
        weight = _task_weight(attempt.task)
        total_weight += weight
        # The LANE's clock, not the agent's. Counted even for an invalid run: a candidate that burns
        # the full timeout and then returns garbage did consume that time.
        latency += attempt.observed_seconds
        if attempt.output is None:
            continue   # an invalid run contributes 0 quality but still consumes its task weight
        output = attempt.output
        self_reported_calls += output.usage.provider_calls
        self_reported_tokens += output.usage.tokens

        if attempt.task.search_type == "ai_search":
            content, summary = _relevance(output, snapshot, attempt.task)
            quality = (AI_QUALITY_WEIGHTS["content_relevance"] * content
                       + AI_QUALITY_WEIGHTS["summary_relevance"] * summary)
        else:
            # X quality is content relevance only, per the upstream breakdown.
            quality, _ = _relevance(output, snapshot, attempt.task)
        weighted_quality += weight * quality

        truth = snapshot.relevant(attempt.task.task_id)
        returned = {result.doc_id for result in output.results}
        coverage_sum += (len(truth & returned) / len(truth)) if truth else 0.0

        for citation in output.citations:
            cited += 1
            # Three conditions, and dropping any one of them opens a way to cite without searching:
            #   * the snapshot holds the document      -> no fabricated ids;
            #   * it genuinely answers THIS task       -> no citing a real document for any query;
            #   * the agent actually RETURNED it       -> no claiming support for documents it never
            #     retrieved, which is the cheapest attack of the three because the relevant ids can
            #     often be guessed from the query alone.
            if (snapshot.contains(citation.doc_id) and citation.doc_id in truth
                    and citation.doc_id in returned):
                cited_supported += 1

    quality = weighted_quality / total_weight if total_weight > 0 else 0.0
    # An agent that made NO claims has none that failed, so precision is 1.0 rather than 0.0.
    # Scoring silence as 0.0 would rank an honest agent that cited nothing BELOW a liar whose
    # fabricated citations happened to include a few real ids. Safe to do because silence already
    # loses at higher priority: citing nothing means retrieving nothing, and quality (2) and
    # coverage (4) both outrank precision (3).
    precision = (cited_supported / cited) if cited else 1.0
    coverage = coverage_sum / len(scorable)

    totals = usage.totals(variant)
    cost_units = totals["provider_calls"] + totals["tokens"] / 1000.0

    return Signals(
        sn22_valid_query_rate=round(valid_rate, 6),
        sn22_weighted_quality=round(quality, 6),
        sn22_citation_precision=round(precision, 6),
        sn22_coverage=round(coverage, 6),
        sn22_invalid_runs=invalid_runs,
        sn22_cost_units=round(cost_units, 6),
        sn22_latency_seconds=round(latency, 6),
        detail={
            "scorable_tasks": len(scorable),
            "usable_tasks": len(usable),
            "infrastructure_faults": len(attempts) - len(scorable),
            "citations_made": cited,
            "citations_supported": cited_supported,
            "usage_source": "relay",
            # Kept for review, never ranked. A large gap between what the agent said it spent and
            # what the relay billed is itself worth looking at.
            "self_reported_calls": self_reported_calls,
            "self_reported_tokens": self_reported_tokens,
            "relay_billed_calls": totals["provider_calls"],
            "relay_billed_tokens": totals["tokens"],
        },
    )


def compare_signals(left: Signals, right: Signals, *, margins: dict | None = None) -> int:
    """Lexicographic comparison by RANK_SIGNALS priority: -1, 0 or +1 for left vs right.

    ``margins`` sets a per-signal indifference band: a difference smaller than the margin is treated
    as a tie and the next signal decides. Without that, floating-point noise on a high-priority
    signal would settle a crown that nothing else got to influence — which is exactly the
    false-promotion mode the §5.5 calibration is meant to bound.

    Antisymmetric by construction: ``compare(a, b) == -compare(b, a)`` for every pair, so the order
    contestants are passed in cannot change the verdict.
    """
    margins = margins or {}
    for name, higher in RANK_SIGNALS:
        a = float(getattr(left, name))
        b = float(getattr(right, name))
        if not (math.isfinite(a) and math.isfinite(b)):
            raise ValueError(f"signal {name} is not finite: {a!r} vs {b!r}")
        margin = abs(float(margins.get(name, 0.0)))
        if abs(a - b) <= margin:
            continue
        if a == b:
            continue
        wins = a > b if higher else a < b
        return 1 if wins else -1
    return 0


def beats_king(candidate: Signals, king: Signals | None, *, margins: dict | None = None) -> bool:
    """Whether a challenger takes the crown. Ties go to the incumbent.

    A challenger that merely equals the king does not promote: churn costs a paid round and hands
    the crown to whoever submitted most recently rather than to whoever is better.
    """
    if king is None:
        # An empty throne still requires a valid run. Promoting an agent that answered nothing would
        # seed the ledger with a king no challenger needs to beat on quality.
        return candidate.sn22_valid_query_rate > 0.0
    return compare_signals(candidate, king, margins=margins) > 0
