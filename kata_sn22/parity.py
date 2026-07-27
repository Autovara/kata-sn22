"""The upstream parity contract (SN22-5 exit gate).

The plan asks for one thing that is easy to claim and hard to earn: *executed* evidence that the
Kata adapter computes what the pinned upstream computes. Not "it imports", not "the constants look
the same" — the same numbers, from the same inputs, produced by running both.

Three artefacts make that checkable, and each covers a hole the others leave:

1. **The recorded inputs** — :data:`PARITY_CASES`. Fixed response shapes chosen so every adapted
   component fires at least once, including the ways each one fails.
2. **The recorded outputs** — `parity_expectations.json`, produced by `tools/record_parity.py`,
   which imports the *real* upstream under the shim and runs it over those inputs. Nothing in the
   normal test path can regenerate it; it is evidence, and evidence a build can rewrite is not
   evidence.
3. **The source pins** — for every adapted symbol, the upstream file's digest and the digest of
   that symbol's own source text, extracted from the vendored tree by `ast`. The file digest
   catches any edit; the symbol digest says *which* adapted component an edit touched.

Together they give the property the exit gate asks for: change one upstream byte and the tree
digest moves, so the recorded expectations no longer match the tree they claim to come from, and
the parity evidence — and with it the bundle digest — is invalid until a reviewer re-records.

**What is executed, and what is pinned only.** Every penalty, the performance curve, the weight
tables, the response checks and the validity predicates are executed against the real upstream. One
component is not: the reward-combination arithmetic in
`neurons/validators/scrapers/base_scraper_validator.py:compute_rewards_and_penalties`. It is a
method on a live validator that logs to W&B, writes a metagraph-sized score array and awaits a
neuron config, so running it would mean reconstructing a validator rather than a scorer. It is
pinned by source digest, its steps are transcribed in
:func:`kata_sn22.upstream_adapter.score_response` with the order preserved, and every *input* to it
is executed-verified. That boundary is recorded in the report rather than left for a reader to
discover.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from kata_sn22 import upstream_adapter as adapter
from kata_sn22.upstream_snapshot import (
    UPSTREAM_COMMIT,
    UPSTREAM_REPO,
    SnapshotError,
    load_manifest,
    snapshot_root,
    verify_snapshot,
)

PARITY_SCHEMA_VERSION = 1

#: Absolute tolerance for a float comparison. The adapter and the upstream do the same arithmetic in
#: the same order, so agreement is normally exact; the tolerance exists because upstream computes
#: penalties in ``numpy.float32`` arrays while the adapter stays in Python floats, and a float32
#: round-trip of a value like ``1 - 3/7`` differs in the eighth decimal. 1e-6 is far below any
#: promotion margin in §5.5, so a difference this size cannot move a crown.
FLOAT_TOLERANCE = 1e-6

EXPECTATIONS_NAME = "parity_expectations.json"


class ParityError(Exception):
    """The parity evidence is missing, stale, or contradicted by the adapter."""


# ---------------------------------------------------------------------------------------------
# The registry of adapted components
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AdaptedComponent:
    """One upstream symbol this adapter reproduces, and where it came from."""

    name: str                 # the parity report's key, and the adapter's public symbol
    upstream_path: str        # relative to the pinned snapshot root
    upstream_symbol: str      # dotted path inside that file
    executed: bool = True     # False -> pinned by source only, with a reason
    note: str = ""

    def as_dict(self) -> dict:
        return {"name": self.name, "upstream_path": self.upstream_path,
                "upstream_symbol": self.upstream_symbol, "executed": self.executed,
                "note": self.note}


_PENALTY = "neurons/validators/penalty"
_REWARD = "neurons/validators/reward"
_UTILS = "neurons/validators/utils"
_SCRAPERS = "neurons/validators/scrapers"

COMPONENTS: tuple[AdaptedComponent, ...] = (
    # -- weight tables -------------------------------------------------------------------------
    AdaptedComponent("SEARCH_TYPE_WEIGHTS", "neurons/validators/scoring/constants.py",
                     "SEARCH_TYPE_WEIGHTS"),
    AdaptedComponent("AI_MODE_WEIGHTS", "neurons/validators/scoring/constants.py",
                     "AI_MODE_WEIGHTS"),
    AdaptedComponent("POOL_SHARES", "neurons/validators/scoring/constants.py", "POOL_SHARES"),
    AdaptedComponent("QUALITY_THRESHOLDS", "neurons/validators/scoring/constants.py",
                     "QUALITY_THRESHOLDS"),
    AdaptedComponent("AI_CONTENT_WEIGHT", f"{_SCRAPERS}/advanced_scraper_validator.py",
                     "AdvancedScraperValidator.__init__",
                     note="content 0.60 / summary 0.40 and the (0.30, 0.30) component floors"),
    AdaptedComponent("X_CONTENT_WEIGHT", f"{_SCRAPERS}/x_scraper_validator.py",
                     "XScraperValidator.__init__", note="single-component X search weighting"),
    AdaptedComponent("reward_weights_for", f"{_SCRAPERS}/advanced_scraper_validator.py",
                     "AdvancedScraperValidator.compute_reward_weights_matrix",
                     note="ONLY_LINKS reweights to (1.0, 0.0)"),

    # -- performance ---------------------------------------------------------------------------
    AdaptedComponent("performance_reward", f"{_REWARD}/performance_reward.py",
                     "PerformanceRewardModel.reward"),
    AdaptedComponent("perf_factor", f"{_REWARD}/performance_reward.py", "perf_factor"),
    AdaptedComponent("perf_floor_for", f"{_REWARD}/performance_reward.py", "perf_floor_for"),
    AdaptedComponent("MODE_PERF_FLOORS", f"{_REWARD}/performance_reward.py", "MODE_PERF_FLOORS"),
    AdaptedComponent("resolve_scoring_budget", f"{_REWARD}/performance_reward.py",
                     "resolve_scoring_budget"),
    AdaptedComponent("min_realistic_for_budget", f"{_REWARD}/performance_reward.py",
                     "min_realistic_for_budget"),

    # -- penalties -----------------------------------------------------------------------------
    AdaptedComponent("count_penalty", f"{_PENALTY}/count_penalty.py",
                     "CountPenaltyModel.penalty_for"),
    AdaptedComponent("duplicate_results_penalty", f"{_PENALTY}/duplicate_results_penalty.py",
                     "DuplicateResultsPenaltyModel.penalty_for"),
    AdaptedComponent("result_schema_penalty", f"{_PENALTY}/result_schema_penalty.py",
                     "ResultSchemaPenaltyModel.penalty_for"),
    AdaptedComponent("domain_filter_penalty", f"{_PENALTY}/domain_filter_penalty.py",
                     "DomainFilterPenaltyModel.penalty_for"),
    AdaptedComponent("date_range_penalty", f"{_PENALTY}/date_range_penalty.py",
                     "DateRangePenaltyModel.penalty_for"),
    AdaptedComponent("sort_order_penalty", f"{_PENALTY}/sort_order_penalty.py",
                     "SortOrderPenaltyModel.penalty_for"),
    AdaptedComponent("min_realistic_time_penalty", f"{_PENALTY}/min_realistic_time_penalty.py",
                     "MinRealisticTimePenaltyModel.penalty_for"),
    AdaptedComponent("summary_structure_penalty", f"{_PENALTY}/summary_structure_penalty.py",
                     "SummaryStructurePenaltyModel.penalty_for"),
    AdaptedComponent("timeout_penalty", f"{_PENALTY}/timeout_penalty.py",
                     "TimeoutPenaltyModel.calculate_penalties"),
    AdaptedComponent("applied_penalty", f"{_PENALTY}/penalty.py",
                     "BasePenaltyModel.apply_penalties",
                     note="the clip-and-invert tail that turns a penalty into a multiplier"),

    # -- response checks -----------------------------------------------------------------------
    AdaptedComponent("normalize_source_url", f"{_UTILS}/response_checks.py",
                     "normalize_source_url"),
    AdaptedComponent("source_key", f"{_UTILS}/response_checks.py", "source_key"),
    AdaptedComponent("first_duplicate_id", f"{_UTILS}/response_checks.py", "first_duplicate_id"),
    AdaptedComponent("extract_markdown_links", f"{_UTILS}/response_checks.py",
                     "extract_markdown_links"),
    AdaptedComponent("check_markdown_structure", f"{_UTILS}/response_checks.py",
                     "check_markdown_structure"),
    AdaptedComponent("collect_summary_sources", f"{_UTILS}/response_checks.py",
                     "collect_summary_sources"),
    AdaptedComponent("tweet_date_in_range", f"{_UTILS}/response_checks.py", "tweet_date_in_range"),
    AdaptedComponent("is_descending_by_created_at", f"{_UTILS}/response_checks.py",
                     "is_descending_by_created_at"),
    AdaptedComponent("normalize_domains", f"{_UTILS}/web_query_operators.py", "normalize_domains"),
    AdaptedComponent("host_in_domains", f"{_UTILS}/web_query_operators.py", "host_in_domains"),
    AdaptedComponent("parse_web_query", f"{_UTILS}/web_query_operators.py", "parse_web_query"),

    # -- validity predicates -------------------------------------------------------------------
    AdaptedComponent("format_text_for_match", "desearch/utils.py", "format_text_for_match"),
    AdaptedComponent("is_valid_tweet", "desearch/utils.py", "is_valid_tweet"),
    AdaptedComponent("is_valid_web_search_result", "desearch/utils.py",
                     "is_valid_web_search_result"),
    AdaptedComponent("MODE_BUDGETS", "desearch/utils.py", "MODE_BUDGETS"),

    # -- pinned, not executed --------------------------------------------------------------------
    AdaptedComponent(
        "score_response", f"{_SCRAPERS}/base_scraper_validator.py",
        "BaseScraperValidator.compute_rewards_and_penalties", executed=False,
        note="combination arithmetic only: weighted sum, component-floor gate, performance "
             "multiplier on the reward, penalty multiplier on both. Not executed because the "
             "upstream method is a live validator step (W&B logging, metagraph-sized score array, "
             "neuron config); every input it combines IS executed above."),
)

COMPONENTS_BY_NAME = {component.name: component for component in COMPONENTS}


# ---------------------------------------------------------------------------------------------
# The recorded inputs
# ---------------------------------------------------------------------------------------------

def _tweet(tweet_id: str, *, text: str, created_at: str, username: str = "alice",
           url: str | None = None, **overrides) -> dict:
    """A schema-complete tweet. Fixtures start from valid and break one thing at a time, so a
    penalty that fires can only be firing for the reason the case is named after."""
    tweet = {
        "id": tweet_id,
        "text": text,
        "reply_count": 0,
        "retweet_count": 1,
        "like_count": 2,
        "quote_count": 0,
        "bookmark_count": 0,
        "url": url if url is not None else f"https://x.com/{username}/status/{tweet_id}",
        "created_at": created_at,
        "is_quote_tweet": False,
        "is_retweet": False,
        "user": {"id": f"u{tweet_id}", "username": username},
    }
    tweet.update(overrides)
    return tweet


_T0 = "Mon Dec 29 12:00:00 +0000 2025"
_T1 = "Mon Dec 29 11:00:00 +0000 2025"
_T2 = "Mon Dec 29 10:00:00 +0000 2025"
_OUT_OF_RANGE = "Sat Jan 04 09:00:00 +0000 2025"

def _web(index: int, host: str = "example.com") -> dict:
    return {"title": f"Doc {index}", "link": f"https://{host}/{index}",
            "snippet": f"snippet {index}"}


def _summary_for(results) -> str:
    """A summary that satisfies the structure penalty for exactly these results.

    Upstream requires bold-not-hash headers, at least one markdown link, and every link to be one
    the miner itself returned. Deriving it from the results means a case that changes its result set
    does not accidentally start failing a penalty it was not written to test.
    """
    links = " ".join(f"[{r['title']}]({r['link']})" for r in results)
    return f"**Findings**\n\nThe retrieved sources cover the query. {links}"


_WEB_A = _web(0)
_WEB_B = _web(1, host="example.org")
#: Ten results, so the default ``count=10`` is satisfied and the count penalty stays out of the way
#: of the case actually under test.
_WEB_RESULTS = tuple(_web(i) for i in range(10))
_GOOD_SUMMARY = _summary_for(_WEB_RESULTS)
#: Five on each host, for the include/exclude filter case: exactly half must violate.
_MIXED_HOST_RESULTS = tuple(_web(i) if i < 5 else _web(i, host="example.org") for i in range(10))

#: Nine schema-complete tweets and one that is missing most of its required fields, so the schema
#: penalty lands on a clean 1/10 rather than on some fraction that also depends on the count.
_MIXED_TWEETS = tuple(
    [_tweet(str(20 + i), text=f"tweet body {i}", created_at=_T0) for i in range(9)]
    + [{"id": "29", "text": "no counts, no author, no timestamp"}]
)


def _tweet_summary(tweets) -> str:
    """A summary citing tweets by the URL `collect_summary_sources` derives from author and id.

    Deliberately NOT the tweet's own ``url`` field: upstream reconstructs the canonical
    `x.com/<username>/status/<id>` and compares against that, so a fixture built from the ``url``
    field would pass or fail for a reason unrelated to what the check does.
    """
    links = " ".join(
        f"[{t['id']}](https://x.com/{t['user']['username']}/status/{t['id']})" for t in tweets)
    return f"**Findings**\n\nThe retrieved posts cover the query. {links}"


#: The shape every AI case starts from: ten results, a summary that cites exactly them, and timing
#: comfortably inside the fast budget. Each case then breaks ONE thing, so a penalty that fires can
#: only be firing for the reason the case is named after.
_AI_BASE: dict = {
    "kind": "ai_search", "mode": "fast", "count": 10, "tools": ("Web Search",),
    "search_results": _WEB_RESULTS, "texts": {"summary": _GOOD_SUMMARY},
    "process_time": 3.0, "max_execution_time": 5, "timeout": 12.0,
}

#: Fixed inputs. ``response`` is the keyword set for both
#: :class:`kata_sn22.upstream_adapter.UpstreamResponse` and the real upstream synapse; ``quality``
#: are the judge's relevance components. Every adapted penalty has at least one case where it fires
#: and one where it does not, because a penalty only ever observed at 0.0 is untested in both
#: directions.
PARITY_CASES: tuple[dict, ...] = (
    {
        "id": "ai-fast-clean",
        "response": {**_AI_BASE},
        "quality": (0.80, 0.70),
        "why": "healthy AI search: every penalty reads 0.0 and performance is inside the budget",
    },
    {
        "id": "ai-fast-count-shortfall",
        "response": {**_AI_BASE, "search_results": _WEB_RESULTS[:1],
                     "texts": {"summary": _summary_for(_WEB_RESULTS[:1])}},
        "quality": (0.80, 0.70),
        "why": "one of ten requested results: the count penalty scales with the shortfall, and "
               "the summary cites only what was actually returned so nothing else fires",
    },
    {
        "id": "ai-balanced-only-links",
        "response": {**_AI_BASE, "mode": "balanced", "result_type": "ONLY_LINKS", "texts": {},
                     "process_time": 9.0, "max_execution_time": 15, "timeout": 20.0},
        "quality": (0.75, 0.0),
        "why": "ONLY_LINKS reweights to (1.0, 0.0) and exempts the summary-structure penalty",
    },
    {
        "id": "ai-deep-duplicate-links",
        "response": {**_AI_BASE, "mode": "deep", "search_results": tuple([_WEB_A] * 10),
                     "texts": {"summary": _summary_for((_WEB_A,))},
                     "process_time": 18.0, "max_execution_time": 30, "timeout": 40.0},
        "quality": (0.60, 0.55),
        "why": "ten copies of one link: the duplicate penalty is all-or-nothing while the count "
               "is nominally satisfied — padding a result list must not buy a clean score",
    },
    {
        "id": "ai-fast-domain-violation",
        "response": {**_AI_BASE, "search_results": _MIXED_HOST_RESULTS,
                     "include_domains": ("example.com",),
                     "texts": {"summary": _summary_for(_MIXED_HOST_RESULTS)}},
        "quality": (0.80, 0.70),
        "why": "exactly half the links are off the requested domain: the penalty is that fraction",
    },
    {
        "id": "ai-fast-bad-summary-markdown",
        "response": {**_AI_BASE,
                     "texts": {"summary": "# Findings\n\nEmissions are distributed per subnet."}},
        "quality": (0.80, 0.70),
        "why": "'#' headers and no markdown links: full summary-structure penalty, nothing else",
    },
    {
        "id": "ai-fast-unsourced-summary-link",
        "response": {**_AI_BASE,
                     "texts": {"summary": "**Findings**\n\nSee [elsewhere](https://evil.test/x)."}},
        "quality": (0.80, 0.70),
        "why": "a summary link the miner never returned as a source: a citation it did not earn",
    },
    {
        "id": "ai-twitter-tool-invalid-tweets",
        "response": {**_AI_BASE, "tools": ("Twitter Search",), "search_results": (),
                     "miner_tweets": _MIXED_TWEETS,
                     "texts": {"summary": _tweet_summary(_MIXED_TWEETS[:9])}},
        "quality": (0.50, 0.50),
        "why": "one of ten tweets fails the schema: the penalty is the invalid fraction, and the "
               "count group for the Twitter tool is the tweets rather than the web results",
    },
    {
        "id": "ai-fast-timeout-overrun",
        "response": {**_AI_BASE, "process_time": 8.5},
        "quality": (0.80, 0.70),
        "why": "3.5s over a 5s ceiling inside a 7s grace window; performance has decayed to 0 too",
    },
    {
        "id": "ai-fast-implausibly-quick",
        "response": {**_AI_BASE, "process_time": 0.2},
        "quality": (0.80, 0.70),
        "why": "faster than 0.3x the budget: cached, so full min-realistic penalty and zero perf",
    },
    {
        "id": "ai-fast-untimed",
        "response": {**_AI_BASE, "process_time": None, "successful": False},
        "quality": (0.80, 0.70),
        "why": "no timing at all: the timeout penalty is FULL, not zero — 'we could not measure "
               "this' is not evidence that it was fast",
    },
    {
        "id": "ai-fast-low-content-component",
        "response": {**_AI_BASE},
        "quality": (0.10, 0.90),
        "why": "content below the 0.30 floor: the quality gate zeroes, the reward does not",
    },
    {
        "id": "ai-deep-slow-but-inside-budget",
        "response": {**_AI_BASE, "mode": "deep", "process_time": 24.0,
                     "max_execution_time": 30, "timeout": 40.0},
        "quality": (0.80, 0.70),
        "why": "past 60% of a 30s budget but under it: performance decays linearly, and deep's "
               "0.85 floor keeps the multiplier high",
    },
    {
        "id": "x-latest-clean",
        "response": {"kind": "x_search", "count": 3, "sort": "Latest",
                     "results": (_tweet("11", text="one", created_at=_T0),
                                 _tweet("12", text="two", created_at=_T1),
                                 _tweet("13", text="three", created_at=_T2)),
                     "process_time": 6.0, "max_execution_time": 10, "timeout": 15.0},
        "quality": (0.85,),
        "why": "healthy X search, correctly sorted newest first",
    },
    {
        "id": "x-latest-misordered",
        "response": {"kind": "x_search", "count": 3, "sort": "Latest",
                     "results": (_tweet("11", text="one", created_at=_T2),
                                 _tweet("12", text="two", created_at=_T0),
                                 _tweet("13", text="three", created_at=_T1)),
                     "process_time": 6.0, "max_execution_time": 10, "timeout": 15.0},
        "quality": (0.85,),
        "why": "sort=Latest but not descending: full sort-order penalty",
    },
    {
        "id": "x-date-range-violation",
        "response": {"kind": "x_search", "count": 2, "sort": "Top",
                     "start_date": "2025-12-28T00:00:00Z", "end_date": "2025-12-30T00:00:00Z",
                     "results": (_tweet("11", text="one", created_at=_T0),
                                 _tweet("12", text="two", created_at=_OUT_OF_RANGE)),
                     "process_time": 6.0, "max_execution_time": 10, "timeout": 15.0},
        "quality": (0.85,),
        "why": "one of two tweets outside the requested window",
    },
    {
        "id": "x-duplicate-text",
        "response": {"kind": "x_search", "count": 2, "sort": "Top",
                     "results": (_tweet("11", text="same body", created_at=_T0),
                                 _tweet("12", text="same body", created_at=_T1)),
                     "process_time": 6.0, "max_execution_time": 10, "timeout": 15.0},
        "quality": (0.85,),
        "why": "distinct ids and urls but identical normalized text",
    },
    {
        "id": "x-count-shortfall-and-slow",
        "response": {"kind": "x_search", "count": 10, "sort": "Top",
                     "results": (_tweet("11", text="only one", created_at=_T0),),
                     "process_time": 9.5, "max_execution_time": 10, "timeout": 15.0},
        "quality": (0.85,),
        "why": "nine of ten results missing, and slow enough that performance has decayed",
    },
)

#: Scalar probes for components a response-shaped case cannot isolate. Each is
#: ``(component, args)``; the recorder calls the upstream symbol with the same arguments.
SCALAR_PROBES: tuple[tuple[str, tuple], ...] = (
    ("normalize_source_url", ("https://WWW.Example.com/Path/",)),
    ("normalize_source_url", ("http://www.example.com/x",)),
    ("normalize_source_url", ("",)),
    ("source_key", ("https://example.com/a?utm_source=x&id=7&fbclid=z",)),
    ("source_key", ("https://www.example.com/a/",)),
    ("source_key", ("https://example.com/a?b=&c=1",)),
    ("extract_markdown_links", ("see [one](https://a.test) and [two](https://b.test)",)),
    ("extract_markdown_links", ("no links here",)),
    ("check_markdown_structure", ("**Bold** heading",)),
    ("check_markdown_structure", ("# Hash heading",)),
    ("check_markdown_structure", ("   ",)),
    ("format_text_for_match", ("@bob @carol Hello  https://t.co/abc world&amp;more",)),
    ("format_text_for_match", ("x" * 400,)),
    ("normalize_domains", (["HTTPS://Example.com/path", "example.com", " other.org. "],)),
    ("normalize_domains", ([],)),
    ("host_in_domains", ("https://sub.example.com/x", ["example.com"])),
    ("host_in_domains", ("https://notexample.com/x", ["example.com"])),
    ("host_in_domains", ("https://example.com/x", [])),
    ("parse_web_query", ("bittensor emissions site:example.com site:foo.org/bar",)),
    ("parse_web_query", ("",)),
    ("tweet_date_in_range", (_T0, "2025-12-28T00:00:00Z", "2025-12-30T00:00:00Z")),
    ("tweet_date_in_range", (_OUT_OF_RANGE, "2025-12-28T00:00:00Z", "2025-12-30T00:00:00Z")),
    ("tweet_date_in_range", ("not a date", None, None)),
    ("min_realistic_for_budget", (5.0,)),
    ("min_realistic_for_budget", (0.0,)),
    ("min_realistic_for_budget", (30.0,)),
    ("perf_factor", (0.0, 0.5)),
    ("perf_factor", (1.0, 0.85)),
    ("performance_reward", (0.5, 5.0)),
    ("performance_reward", (3.0, 5.0)),
    ("performance_reward", (4.0, 5.0)),
    ("performance_reward", (7.5, 5.0)),
    ("performance_reward", (2.0, 0.0)),
    ("performance_reward", (9.0, 0.0)),
    ("applied_penalty", (0.0,)),
    ("applied_penalty", (0.4,)),
    ("applied_penalty", (1.7,)),
    ("applied_penalty", (-0.3,)),
    ("first_duplicate_id", ([{"id": "a"}, {"id": "b"}, {"id": "a"}], "id")),
    ("first_duplicate_id", ([{"id": "a"}, {"id": "b"}], "id")),
    ("first_duplicate_id", ([{"link": "https://example.com/x?utm_source=q"},
                             {"link": "https://www.example.com/x/"}], "link")),
    ("is_descending_by_created_at", ([{"created_at": _T0}, {"created_at": _T1}],)),
    ("is_descending_by_created_at", ([{"created_at": _T1}, {"created_at": _T0}],)),
    ("is_descending_by_created_at", ([{"created_at": ""}],)),
    ("is_valid_tweet", (_tweet("9", text="ok", created_at=_T0),)),
    ("is_valid_tweet", ({"id": "9", "text": "missing counts"},)),
    # Deliberately no non-dict probe: upstream's `is_valid_tweet` splats its argument and would
    # raise TypeError, because its only caller guards `isinstance(item, dict)` first. The adapter
    # keeps that guard in `_tweet_item_ok`; probing the unreachable path would compare against
    # behaviour upstream does not have.
    ("is_valid_web_search_result", (_WEB_A,)),
    ("is_valid_web_search_result", ({"title": "no link"},)),
)


# ---------------------------------------------------------------------------------------------
# Running the adapter side
# ---------------------------------------------------------------------------------------------

#: Components computed for every response case. Names match :data:`COMPONENTS`.
CASE_COMPONENTS = ("count_penalty", "duplicate_results_penalty", "result_schema_penalty",
                   "domain_filter_penalty", "date_range_penalty", "sort_order_penalty",
                   "min_realistic_time_penalty", "summary_structure_penalty", "timeout_penalty",
                   "resolve_scoring_budget", "performance_reward", "perf_floor_for",
                   "reward_weights_for", "collect_summary_sources")


def build_response(case: dict) -> adapter.UpstreamResponse:
    return adapter.UpstreamResponse(**case["response"])


def adapter_case_outputs(case: dict) -> dict:
    """Every per-case component value the adapter produces, plus the combined score."""
    response = build_response(case)
    outputs: dict[str, object] = {}
    for name in CASE_COMPONENTS:
        if name in adapter.PENALTY_FUNCTIONS:
            outputs[name] = adapter.PENALTY_FUNCTIONS[name](response)
        elif name == "resolve_scoring_budget":
            outputs[name] = adapter.resolve_scoring_budget(response)
        elif name == "performance_reward":
            outputs[name] = adapter.performance_reward(
                adapter.response_time_for(response), adapter.resolve_scoring_budget(response))
        elif name == "perf_floor_for":
            outputs[name] = adapter.perf_floor_for(
                response, adapter.perf_floor_default_for(response))
        elif name == "reward_weights_for":
            outputs[name] = list(adapter.reward_weights_for(response))
        elif name == "collect_summary_sources":
            outputs[name] = sorted(adapter.collect_summary_sources(response))
    score = adapter.score_response(response, case["quality"])
    outputs["score"] = {"reward": score.reward, "quality_gate": score.quality_gate,
                        "perf_multiplier": score.perf_multiplier,
                        "penalty_multiplier": score.penalty_multiplier,
                        "pool_share": score.pool_share}
    return outputs


_SCALAR_ADAPTER_FUNCTIONS = {
    "normalize_source_url": adapter.normalize_source_url,
    "source_key": adapter.source_key,
    "extract_markdown_links": adapter.extract_markdown_links,
    "check_markdown_structure": adapter.check_markdown_structure,
    "format_text_for_match": adapter.format_text_for_match,
    "normalize_domains": adapter.normalize_domains,
    "host_in_domains": adapter.host_in_domains,
    "parse_web_query": adapter.parse_web_query,
    "tweet_date_in_range": adapter.tweet_date_in_range,
    "min_realistic_for_budget": adapter.min_realistic_for_budget,
    "perf_factor": adapter.perf_factor,
    "performance_reward": adapter.performance_reward,
    "applied_penalty": adapter.applied_penalty,
    # ``(items, key)``, with URL-shaped keys normalized through ``source_key`` — the exact call
    # `duplicate_results_penalty` makes. The recorder applies the same rule to the upstream side, so
    # the probe compares the two implementations rather than two calling conventions.
    "first_duplicate_id": lambda items, key: adapter.first_duplicate_id(
        items, key=key, normalize=adapter.source_key if key in ("link", "url") else None),
    "is_descending_by_created_at": adapter.is_descending_by_created_at,
    "is_valid_tweet": adapter.is_valid_tweet,
    "is_valid_web_search_result": adapter.is_valid_web_search_result,
}

#: The weight tables, compared value-for-value against the pinned upstream. A constant is the
#: cheapest thing to get wrong and the most expensive to notice: a wrong 0.20 changes every score
#: the lane ever produces and breaks no test that does not check the number itself.
def adapter_constants() -> dict:
    return normalize_value({
        "SEARCH_TYPE_WEIGHTS": adapter.SEARCH_TYPE_WEIGHTS,
        "AI_MODE_WEIGHTS": adapter.AI_MODE_WEIGHTS,
        "POOL_SHARES": {f"{search_type}:{mode or '-'}": share
                        for (search_type, mode), share in adapter.POOL_SHARES.items()},
        "QUALITY_THRESHOLDS": adapter.QUALITY_THRESHOLDS,
        "MODE_BUDGETS": adapter.MODE_BUDGETS,
        "MODE_PERF_FLOORS": adapter.MODE_PERF_FLOORS,
        "AI_CONTENT_WEIGHT": adapter.AI_CONTENT_WEIGHT,
        "AI_SUMMARY_WEIGHT": adapter.AI_SUMMARY_WEIGHT,
        "AI_COMPONENT_FLOORS": list(adapter.AI_COMPONENT_FLOORS),
        "X_CONTENT_WEIGHT": adapter.X_CONTENT_WEIGHT,
        "AI_PERF_FLOOR": adapter.AI_PERF_FLOOR,
        "X_PERF_FLOOR": adapter.X_PERF_FLOOR,
    })


def normalize_value(value):
    """One JSON shape per result, so an upstream tuple and an adapter list compare equal.

    Structural sameness is what parity is about; a `WebQueryOperators` dataclass on one side and a
    pydantic-free one on the other are the same answer, and a comparison that said otherwise would
    be testing the container rather than the arithmetic.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, (list, tuple)):
        return [normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(k): normalize_value(v)
                for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if hasattr(value, "text") and hasattr(value, "sites"):        # WebQueryOperators, either side
        return {"text": value.text, "sites": normalize_value(list(value.sites))}
    return value


def adapter_scalar_outputs() -> list[dict]:
    results = []
    for name, args in SCALAR_PROBES:
        function = _SCALAR_ADAPTER_FUNCTIONS[name]
        results.append({"component": name, "args": normalize_value(list(args)),
                        "value": normalize_value(function(*args))})
    return results


def adapter_outputs() -> dict:
    """The complete adapter side of the parity comparison."""
    return {
        "constants": adapter_constants(),
        "cases": {case["id"]: normalize_value(adapter_case_outputs(case))
                  for case in PARITY_CASES},
        "scalars": adapter_scalar_outputs(),
    }


# ---------------------------------------------------------------------------------------------
# Source pins
# ---------------------------------------------------------------------------------------------


def _extract_symbol_source(tree: ast.AST, source: str, dotted: str) -> str | None:
    """The source text of ``dotted`` inside a parsed module, or ``None`` when absent."""
    parts = dotted.split(".")
    node: ast.AST = tree
    for index, part in enumerate(parts):
        found = None
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if child.name == part:
                    found = child
                    break
            elif isinstance(child, ast.Assign) and index == len(parts) - 1:
                names = [t.id for t in child.targets if isinstance(t, ast.Name)]
                if part in names:
                    found = child
                    break
            elif isinstance(child, ast.AnnAssign) and index == len(parts) - 1:
                if isinstance(child.target, ast.Name) and child.target.id == part:
                    found = child
                    break
        if found is None:
            return None
        node = found
    return ast.get_source_segment(source, node)


def source_pins(root: Path | None = None) -> dict:
    """Per-component digests of the upstream file AND of the adapted symbol's own source.

    Two digests rather than one because they answer different questions. The file digest is what the
    tree digest is built from, so it is what invalidates the bundle. The symbol digest tells a
    reviewer *which adapted component* a change touched — a comment edit elsewhere in
    `response_checks.py` moves the file digest but leaves every symbol digest alone, and knowing
    that is the difference between a five-minute re-review and a full re-record.
    """
    root = Path(root).resolve() if root is not None else snapshot_root()
    manifest = load_manifest(root)
    manifest_files: dict[str, str] = dict(manifest.get("files") or {})
    parsed: dict[str, tuple[ast.AST, str]] = {}
    pins: dict[str, dict] = {}

    for component in COMPONENTS:
        path = root / component.upstream_path
        entry: dict = {"upstream_path": component.upstream_path,
                       "upstream_symbol": component.upstream_symbol,
                       "executed": component.executed}
        if not path.is_file():
            entry["error"] = "upstream file missing from the pinned snapshot"
            pins[component.name] = entry
            continue
        entry["file_sha256"] = manifest_files.get(component.upstream_path, "")
        if component.upstream_path not in parsed:
            source = path.read_text(encoding="utf-8")
            parsed[component.upstream_path] = (ast.parse(source), source)
        tree, source = parsed[component.upstream_path]
        segment = _extract_symbol_source(tree, source, component.upstream_symbol)
        if segment is None:
            entry["error"] = f"symbol {component.upstream_symbol} not found in the pinned file"
        else:
            entry["symbol_sha256"] = hashlib.sha256(segment.encode("utf-8")).hexdigest()
            entry["symbol_lines"] = segment.count("\n") + 1
        pins[component.name] = entry
    return dict(sorted(pins.items()))


# ---------------------------------------------------------------------------------------------
# Comparison and reporting
# ---------------------------------------------------------------------------------------------


def expectations_path() -> Path:
    return Path(__file__).resolve().parent / EXPECTATIONS_NAME


def load_expectations() -> dict:
    path = expectations_path()
    if not path.is_file():
        raise ParityError(f"no recorded parity evidence at {path}; "
                          f"run tools/record_parity.py against the pinned snapshot")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != PARITY_SCHEMA_VERSION:
        raise ParityError(f"parity evidence schema {document.get('schema_version')!r} "
                          f"is not {PARITY_SCHEMA_VERSION}")
    return document


def _diff(path: str, expected, observed, findings: list[str]) -> None:
    if isinstance(expected, bool) or isinstance(observed, bool):
        if expected != observed:
            findings.append(f"{path}: upstream {expected!r} != adapter {observed!r}")
        return
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        if not math.isfinite(float(expected)) or not math.isfinite(float(observed)):
            findings.append(f"{path}: non-finite value ({expected!r} vs {observed!r})")
        elif abs(float(expected) - float(observed)) > FLOAT_TOLERANCE:
            findings.append(f"{path}: upstream {expected!r} != adapter {observed!r} "
                            f"(tolerance {FLOAT_TOLERANCE})")
        return
    if isinstance(expected, list) and isinstance(observed, list):
        if len(expected) != len(observed):
            findings.append(f"{path}: length {len(expected)} != {len(observed)}")
            return
        for index, (left, right) in enumerate(zip(expected, observed)):
            _diff(f"{path}[{index}]", left, right, findings)
        return
    if isinstance(expected, dict) and isinstance(observed, dict):
        for key in sorted(set(expected) | set(observed)):
            if key not in expected:
                findings.append(f"{path}.{key}: adapter produced a key upstream did not")
            elif key not in observed:
                findings.append(f"{path}.{key}: upstream produced a key the adapter did not")
            else:
                _diff(f"{path}.{key}", expected[key], observed[key], findings)
        return
    if expected != observed:
        findings.append(f"{path}: upstream {expected!r} != adapter {observed!r}")


def compare_against_expectations(expectations: dict | None = None) -> list[str]:
    """Run the adapter and diff it against the recorded upstream outputs. Empty means parity."""
    expectations = expectations or load_expectations()
    findings: list[str] = []
    observed = adapter_outputs()

    _diff("constants", expectations.get("constants") or {}, observed["constants"], findings)

    expected_cases = expectations.get("cases") or {}
    observed_cases = observed["cases"]
    for case_id in sorted(set(expected_cases) | set(observed_cases)):
        if case_id not in expected_cases:
            findings.append(f"case {case_id}: no recorded upstream output "
                            f"(the case set changed; re-record)")
        elif case_id not in observed_cases:
            findings.append(f"case {case_id}: recorded but no longer in PARITY_CASES")
        else:
            _diff(f"case[{case_id}]", expected_cases[case_id], observed_cases[case_id], findings)

    expected_scalars = expectations.get("scalars") or []
    observed_scalars = observed["scalars"]
    if len(expected_scalars) != len(observed_scalars):
        findings.append(f"scalar probes: recorded {len(expected_scalars)}, "
                        f"adapter has {len(observed_scalars)}; re-record")
    else:
        for index, (left, right) in enumerate(zip(expected_scalars, observed_scalars)):
            if (left.get("component") != right.get("component")
                    or left.get("args") != right.get("args")):
                findings.append(f"scalar[{index}]: probe changed; re-record")
                continue
            _diff(f"scalar[{index}]({left['component']})", left.get("value"), right.get("value"),
                  findings)
    return findings


def evidence_is_current() -> list[str]:
    """Whether the recorded evidence still describes the tree that is on disk.

    This is the exit gate's teeth. The recorded document names the upstream commit, the tree digest
    and every adapted symbol's digest; if any of them moved, the evidence describes code that is no
    longer installed, and saying otherwise would be the whole point of pinning thrown away.
    """
    findings: list[str] = []
    try:
        expectations = load_expectations()
    except ParityError as exc:
        return [str(exc)]

    try:
        verification = verify_snapshot()
    except SnapshotError as exc:
        return [str(exc)]
    if not verification.ok:
        findings.extend(f"snapshot: {finding}" for finding in verification.findings)

    if expectations.get("upstream_commit") != UPSTREAM_COMMIT:
        findings.append(f"evidence records upstream commit "
                        f"{expectations.get('upstream_commit')!r}, adapter targets "
                        f"{UPSTREAM_COMMIT!r}")
    if expectations.get("upstream_tree_sha256") != verification.observed_tree_sha256:
        findings.append("evidence was recorded against a different upstream tree "
                        f"({str(expectations.get('upstream_tree_sha256'))[:12]} vs "
                        f"{verification.observed_tree_sha256[:12]}); re-record parity")

    recorded_pins = expectations.get("source_pins") or {}
    current_pins = source_pins()
    for name in sorted(set(recorded_pins) | set(current_pins)):
        if name not in recorded_pins:
            findings.append(f"component {name}: adapted but not present in the recorded evidence")
        elif name not in current_pins:
            findings.append(f"component {name}: recorded but no longer registered")
        elif recorded_pins[name] != current_pins[name]:
            findings.append(f"component {name}: upstream source moved since the evidence "
                            f"was recorded")
    return findings


def parity_report() -> dict:
    """The reviewable parity document (plan §8 SN22-9 "parity report")."""
    verification = verify_snapshot()
    findings = evidence_is_current()
    comparison = compare_against_expectations() if not findings else []
    executed = [c.name for c in COMPONENTS if c.executed]
    pinned_only = [c.as_dict() for c in COMPONENTS if not c.executed]
    return {
        "schema_version": PARITY_SCHEMA_VERSION,
        "upstream_repo": UPSTREAM_REPO,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_tree_sha256": verification.observed_tree_sha256,
        "snapshot_ok": verification.ok,
        "snapshot_findings": list(verification.findings),
        "float_tolerance": FLOAT_TOLERANCE,
        "component_count": len(COMPONENTS),
        "executed_components": executed,
        "pinned_only_components": pinned_only,
        "case_count": len(PARITY_CASES),
        "scalar_probe_count": len(SCALAR_PROBES),
        "evidence_findings": findings,
        "comparison_findings": comparison,
        "ok": not findings and not comparison and verification.ok,
    }


__all__ = [
    "CASE_COMPONENTS",
    "COMPONENTS",
    "COMPONENTS_BY_NAME",
    "FLOAT_TOLERANCE",
    "PARITY_CASES",
    "PARITY_SCHEMA_VERSION",
    "SCALAR_PROBES",
    "AdaptedComponent",
    "ParityError",
    "adapter_case_outputs",
    "adapter_outputs",
    "adapter_scalar_outputs",
    "build_response",
    "compare_against_expectations",
    "evidence_is_current",
    "expectations_path",
    "load_expectations",
    "normalize_value",
    "parity_report",
    "source_pins",
]
