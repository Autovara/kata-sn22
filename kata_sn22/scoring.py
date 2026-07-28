"""The SN22 scoring schema and comparator (plan §5.3–§5.4, SN22-2).

Two things live here, and the separation matters.

**The signals** are the seven ordered quantities a challenge publishes (§5.4). They are computed
from what the VALIDATOR independently established -- pages it fetched itself, excerpts it checked
against those pages, verdicts a judge gave on them, tweets it re-scraped (see
:mod:`kata_sn22.verification`) -- never from anything the candidate asserts. A citation counts only
if the source was returned AND survived the evidence check; quality is what the judge said about
verified excerpts. An agent cannot improve its score by claiming a better one.

**The comparator** decides a crown from those signals by fixed lexicographic priority — validity
first, then quality, then precision, coverage, invalid runs, cost, latency. Lexicographic rather
than a weighted sum on purpose: a weighted sum lets a candidate buy a quality win with unlimited
spend, whereas priority order says plainly that an agent which answers fewer queries validly does
not win by being prettier on the ones it did answer.

**Upstream weights** (§5.3) are not restated here. Since SN22-5 the whole weighting, penalty and
combination path is :mod:`kata_sn22.upstream_adapter` — a port of the pinned upstream that
:mod:`kata_sn22.parity` proves, by execution, computes what the real upstream computes. A second
copy of "0.90 / 0.10" living in this file is a copy that can drift silently, so there is not one.

What that buys, concretely: ``sn22_weighted_quality`` is the upstream reward for the components it
covers — the AI content / summary split, the ONLY_LINKS reweighting, the component floors, the
applicable penalties, and the pool shares.

**It is NOT upstream-complete, and must not be described as such.** Three things are missing, and
each changes a score:

* the TIMING components — ``timeout_penalty``, ``min_realistic_time_penalty`` and the performance
  multiplier — are excluded here and published as a separate ranked signal instead;
* ``streaming_penalty`` has no input, because protocol v1 returns one JSON document rather than a
  stream;
* pool scores are computed PER CONTESTANT. Upstream's ``combine_pool_scores`` normalizes a pool
  across the whole population, so two agents scored separately and then compared are not being
  compared the way upstream would compare them.

The production path replaces this: it executes the pinned upstream validators inside the sealed room
and runs ``combine_pool_scores`` once over both contestants as two virtual UIDs. This module remains
the calibration and fixture scorer, where an offline, dependency-free score is what is wanted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from kata_sn22 import upstream_adapter as upstream
from kata_sn22.manifests import UsageManifest
from kata_sn22.protocol import ErrorClass, Task, TaskOutput

#: Re-exported from the adapter so there is exactly one definition of each weight in the package.
SEARCH_TYPE_WEIGHTS = upstream.SEARCH_TYPE_WEIGHTS
AI_MODE_WEIGHTS = upstream.AI_MODE_WEIGHTS
AI_QUALITY_WEIGHTS = {"content_relevance": upstream.AI_CONTENT_WEIGHT,
                      "summary_relevance": upstream.AI_SUMMARY_WEIGHT}

#: The upstream penalties whose inputs a sealed Kata challenge actually carries (plan §5.3: retain
#: the upstream components that are *included*). Two are deliberately absent:
#:
#: * ``timeout_penalty`` and ``min_realistic_time_penalty`` are about provider latency, which Kata
#:   already publishes as its own ranked signal, ``sn22_latency_seconds``. Folding latency into
#:   quality (priority 2) as well would let a fast agent outrank a better one on the signal that is
#:   meant to be about answers.
#:
#: The performance multiplier is excluded for the same reason. Both exclusions are recorded in the
#: score detail rather than left implicit, so a reader of a challenge result can see what was and
#: was not applied.
KATA_APPLICABLE_PENALTIES: tuple[str, ...] = (
    "count_penalty",
    "duplicate_results_penalty",
    "result_schema_penalty",
    "domain_filter_penalty",
    "date_range_penalty",
    "sort_order_penalty",
    "summary_structure_penalty",
)
KATA_EXCLUDED_PENALTIES: tuple[str, ...] = ("timeout_penalty", "min_realistic_time_penalty")


def upstream_commit() -> str:
    """The pinned upstream this scorer's components came from.

    Imported lazily so the scoring module still loads where the vendored snapshot is absent — the
    comparator and the signal schema are useful to a reviewer with only the source, and refusing to
    import over a missing tree would make them unreadable rather than safe. The lane itself fails
    closed elsewhere: `verify_snapshot` runs at conformance and install time.
    """
    from kata_sn22.upstream_snapshot import UPSTREAM_COMMIT

    return UPSTREAM_COMMIT


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
    #: What the VALIDATOR established about this answer by fetching, checking evidence, judging and
    #: re-scraping (:mod:`kata_sn22.verification`). Attached to the attempt rather than passed
    #: alongside it so that scoring stays a pure function of what was verified -- there is no path
    #: where a signal is computed from something the candidate merely asserted.
    verification: object | None = None

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


def _task_weight(task: Task) -> float:
    """The upstream pool share for one task's category (§5.3), from the pinned tables."""
    if task.search_type == "x_search":
        return upstream.POOL_SHARES[("x_search", None)]
    return upstream.POOL_SHARES.get(("ai_search", task.ai_mode), 0.0)


def _upstream_summary(output: TaskOutput) -> str:
    """Kata's summary rendered the way the upstream summary checks expect to read it.

    Upstream miners return prose with markdown links, and `summary_structure_penalty` asks whether
    every link is one the miner itself returned. Kata's protocol carries the same claim in a
    structured ``citations`` array as well, so the citations are appended as markdown links rather
    than the penalty being dropped.

    Appended, not substituted: a summary that already contains its own markdown links keeps them,
    because those are what the groundedness judge reads and what `cited_urls_normalized` extracts.
    """
    links = " ".join(f"[{index}]({citation.link})"
                     for index, citation in enumerate(output.citations, 1))
    return f"{output.summary}\n\n{links}".strip() if links else output.summary


def _upstream_response(attempt: TaskAttempt) -> upstream.UpstreamResponse:
    """One Kata attempt as the response shape the pinned upstream components score.

    Only the fields the adapted components read are populated. The results are the miner's OWN
    claims -- which is correct here and only here: these components are the cheap structural
    penalties (duplicate ids, schema, sort order, domain filter), and their entire job is to inspect
    what the miner said. Whether any of it is TRUE is decided elsewhere, by fetching and judging.

    ``count`` is the number of results the task ASKED for, so the count penalty measures a shortfall
    against a published request rather than against a secret.
    """
    task = attempt.task
    output = attempt.output
    results = output.results if output is not None else ()
    summary = _upstream_summary(output) if output is not None else ""

    if task.search_type == "x_search":
        tweets = tuple(tweet.as_dict() for tweet in (output.tweets if output is not None else ()))
        return upstream.UpstreamResponse(
            kind="x_search", count=task.limits.max_results, results=tweets,
            sort=task.sort,
            max_execution_time=task.limits.max_wall_seconds,
            process_time=attempt.observed_seconds, successful=output is not None)

    search_results = tuple({"title": result.title, "link": result.link,
                            "snippet": result.snippet} for result in results)
    return upstream.UpstreamResponse(
        kind="ai_search", mode=task.ai_mode, count=task.limits.max_results,
        tools=("Web Search",),
        result_type=(upstream.RESULT_TYPE_ONLY_LINKS if task.result_type == "links"
                     else upstream.RESULT_TYPE_LINKS_WITH_FINAL_SUMMARY),
        search_results=search_results, texts={"summary": summary},
        max_execution_time=task.limits.max_wall_seconds,
        process_time=attempt.observed_seconds, successful=output is not None)


def upstream_score_for(attempt: TaskAttempt, *,
                       components: tuple[float, ...]) -> upstream.UpstreamScore:
    """Score one attempt through the pinned upstream components."""
    response = _upstream_response(attempt)
    return upstream.score_response(response, components,
                                   penalty_names=KATA_APPLICABLE_PENALTIES,
                                   apply_performance=False)


def score_attempts(attempts: list[TaskAttempt], *,
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
    per_task: list[dict] = []
    gated_tasks = 0

    for attempt in scorable:
        weight = _task_weight(attempt.task)
        total_weight += weight
        # The LANE's clock, not the agent's. Counted even for an invalid run: a candidate that burns
        # the full timeout and then returns garbage did consume that time.
        latency += attempt.observed_seconds
        if attempt.output is None:
            # An invalid run contributes 0 quality but still consumes its task weight. It is not
            # sent through the upstream components at all: there is no response to score, and
            # feeding an empty one would report a penalty for a shape the candidate never produced.
            per_task.append({"task_id": attempt.task.task_id, "reward": 0.0,
                             "reason": (attempt.error.value if attempt.error else "no output")})
            continue
        output = attempt.output
        self_reported_calls += output.usage.provider_calls
        self_reported_tokens += output.usage.tokens

        # What the VALIDATOR established, not what the candidate claimed. An attempt with no
        # verification scores zero quality rather than defaulting to something generous: it means
        # nothing was checked, and an unchecked answer has earned nothing.
        verification = attempt.verification
        content = float(getattr(verification, "content_relevance", 0.0) or 0.0)
        summary_relevance = float(getattr(verification, "summary_relevance", 0.0) or 0.0)
        # X quality is content relevance only, per the upstream breakdown; AI search carries both
        # and the adapter applies the 0.60/0.40 split (or (1.0, 0.0) for a links-only request).
        components = ((content,) if attempt.task.search_type == "x_search"
                      else (content, summary_relevance))
        score = upstream_score_for(attempt, components=components)
        weighted_quality += weight * score.reward
        if score.quality_gate <= 0.0 < score.reward:
            gated_tasks += 1
        per_task.append({
            "task_id": attempt.task.task_id,
            "search_type": attempt.task.search_type,
            "ai_mode": attempt.task.ai_mode,
            "pool_share": round(score.pool_share, 6),
            "content_relevance": round(content, 6),
            "summary_relevance": round(summary_relevance, 6),
            "reward": round(score.reward, 6),
            "quality_gate": round(score.quality_gate, 6),
            "penalties": {name: round(value, 6) for name, value in score.penalties.items()
                          if value > 0},
        })

        # Coverage: the share of what the agent CLAIMED that survived independent verification.
        # With live search there is no sealed answer key to measure recall against -- upstream has
        # none either -- so the honest question is not "how much of the truth did you find" but
        # "how much of what you reported turned out to be real".
        verified_links = {
            source.link for source in getattr(verification, "sources", []) or []
            if source.evidence_ok
        }
        returned = {result.link for result in output.results} or {
            f"tweet:{tweet.tweet_id}" for tweet in output.tweets}
        coverage_sum += (len(verified_links & returned) / len(returned)) if returned else 0.0

        for citation in output.citations:
            cited += 1
            # Two conditions, and dropping either opens a way to cite without searching:
            #   * the agent actually RETURNED the source -> no claiming support from a page it
            #     never retrieved, which is the cheapest attack because a plausible URL is free;
            #   * that source PASSED EVIDENCE           -> no citing a page whose claimed excerpts
            #     are not on it, which is citing a real URL for text nobody wrote.
            if citation.link in returned and citation.link in verified_links:
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
            # Which upstream components produced the quality signal, and which did not. Stated in
            # the result rather than only in this file's comments, because a reader comparing a
            # Kata score to an upstream one needs to know what was applied without reading the
            # source.
            "upstream_commit": upstream_commit(),
            "upstream_penalties_applied": list(KATA_APPLICABLE_PENALTIES),
            "upstream_penalties_excluded": list(KATA_EXCLUDED_PENALTIES),
            "upstream_performance_multiplier_applied": False,
            "quality_gated_tasks": gated_tasks,
            "per_task": per_task,
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
