#!/usr/bin/env python3
"""Record the upstream side of the parity evidence by EXECUTING the pinned upstream (SN22-5).

This is the only thing that writes `kata_sn22/parity_expectations.json`, and it is run by a
reviewer, never by a build or a test. That split is the point: a test that could regenerate its own
expected values would pass for any adapter, including a wrong one.

What it does, per recorded case:

* builds the REAL upstream synapse (`desearch/protocol.py`, validated by real pydantic) from the
  case's response fields;
* runs the REAL upstream penalty models, performance curve, response checks and validity predicates
  over it, under `tools/upstream_shim.py`;
* writes the results, the upstream tree digest, and every adapted symbol's source digest.

Run it after re-vendoring at a new upstream commit, and review the diff. A changed number there is
an upstream behaviour change, and it must be understood before the adapter is taught to agree with
it — the point of parity is to notice, not to converge automatically.

    uv run --extra parity python tools/record_parity.py
    uv run --extra parity python tools/record_parity.py --check   # record nothing, just compare
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import upstream_shim  # noqa: E402

from kata_sn22 import parity  # noqa: E402
from kata_sn22.upstream_snapshot import (  # noqa: E402
    UPSTREAM_COMMIT,
    UPSTREAM_REPO,
    require_intact,
    snapshot_root,
)

# ---------------------------------------------------------------------------------------------
# Building the real upstream synapse from a recorded case
# ---------------------------------------------------------------------------------------------


def _build_synapse(case: dict, upstream):
    """One recorded case as the actual upstream synapse the validator would score.

    Note ``text_chunks`` rather than ``texts``: upstream's ``texts`` is a derived property that
    joins the streamed chunks, so writing the chunk is the only way to set it — and going through
    the real property is what makes the summary-structure comparison meaningful.
    """
    fields = case["response"]
    kind = fields["kind"]
    protocol = upstream["protocol"]
    dendrite = upstream_shim.Dendrite(
        process_time=fields.get("process_time"),
        status_code=200 if fields.get("successful", True) else 500,
    )

    if kind == "ai_search":
        summary = (fields.get("texts") or {}).get("summary")
        synapse = protocol.ScraperStreamingSynapse(
            prompt="sn22 parity case",
            count=fields.get("count") or 10,
            tools=list(fields.get("tools") or ()),
            result_type=protocol.ResultType(
                fields.get("result_type") or "LINKS_WITH_FINAL_SUMMARY"),
            mode=protocol.SearchMode(fields["mode"]) if fields.get("mode") else None,
            miner_tweets=[dict(t) for t in fields.get("miner_tweets") or ()],
            search_results=[dict(r) for r in fields.get("search_results") or ()],
            include_domains=list(fields.get("include_domains") or ()),
            exclude_domains=list(fields.get("exclude_domains") or ()),
            start_date=fields.get("start_date"),
            end_date=fields.get("end_date"),
            max_execution_time=fields.get("max_execution_time"),
            text_chunks={"summary": [summary]} if summary else {},
            timeout=float(fields.get("timeout") or 12.0),
        )
    else:
        synapse = protocol.TwitterSearchSynapse(
            query="sn22 parity case",
            sort=fields.get("sort"),
            count=fields.get("count") or 20,
            start_date=fields.get("start_date"),
            end_date=fields.get("end_date"),
            results=[dict(r) for r in fields.get("results") or ()],
            max_execution_time=fields.get("max_execution_time"),
            timeout=float(fields.get("timeout") or 12.0),
        )

    synapse.dendrite = dendrite
    synapse.axon = upstream_shim.Axon()
    return synapse


# ---------------------------------------------------------------------------------------------
# Reading the weights out of the pinned scraper constructors
# ---------------------------------------------------------------------------------------------


def _self_assignments(path: Path, class_name: str, method: str) -> dict:
    """Literal ``self.x = <constant>`` assignments in a method, read from the pinned source.

    The scraper constructors take a live neuron and build an LLM client, so they cannot be executed
    here — but the weights they set are plain literals, and reading them out of the pinned AST is
    exact. A non-literal assignment is simply absent from the result rather than guessed at.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.FunctionDef) or child.name != method:
                continue
            for statement in ast.walk(child):
                if not isinstance(statement, ast.Assign):
                    continue
                for target in statement.targets:
                    if (isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"):
                        try:
                            found[target.attr] = ast.literal_eval(statement.value)
                        except (ValueError, SyntaxError):
                            continue
    return found


def _component_floors(path: Path, class_name: str) -> list:
    """The ``component_floors=[...]`` argument of the scraper's ``super().__init__`` call."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            for keyword in call.keywords:
                if keyword.arg == "component_floors":
                    try:
                        value = ast.literal_eval(keyword.value)
                    except (ValueError, SyntaxError):
                        continue
                    if value is not None:
                        return list(value)
    return []


# ---------------------------------------------------------------------------------------------
# Executing the upstream
# ---------------------------------------------------------------------------------------------


def _import_upstream(root: Path) -> dict:
    upstream_shim.install(root)

    import numpy
    from desearch import protocol
    from desearch import utils as desearch_utils
    from neurons.validators.penalty import penalty as penalty_base
    from neurons.validators.penalty.count_penalty import CountPenaltyModel
    from neurons.validators.penalty.date_range_penalty import DateRangePenaltyModel
    from neurons.validators.penalty.domain_filter_penalty import DomainFilterPenaltyModel
    from neurons.validators.penalty.duplicate_results_penalty import DuplicateResultsPenaltyModel
    from neurons.validators.penalty.min_realistic_time_penalty import MinRealisticTimePenaltyModel
    from neurons.validators.penalty.result_schema_penalty import ResultSchemaPenaltyModel
    from neurons.validators.penalty.sort_order_penalty import SortOrderPenaltyModel
    from neurons.validators.penalty.summary_structure_penalty import SummaryStructurePenaltyModel
    from neurons.validators.penalty.timeout_penalty import TimeoutPenaltyModel
    from neurons.validators.reward import performance_reward as perf
    from neurons.validators.scoring import constants as scoring_constants
    from neurons.validators.scrapers import advanced_scraper_validator as advanced
    from neurons.validators.utils import response_checks, source_bodies, web_query_operators

    return {
        "numpy": numpy,
        "protocol": protocol,
        "desearch_utils": desearch_utils,
        "penalty_base": penalty_base,
        "response_checks": response_checks,
        "web_query_operators": web_query_operators,
        "source_bodies": source_bodies,
        "perf": perf,
        "scoring_constants": scoring_constants,
        "advanced": advanced,
        "penalties": {
            "count_penalty": CountPenaltyModel(max_penalty=1.0),
            "duplicate_results_penalty": DuplicateResultsPenaltyModel(max_penalty=1.0),
            "result_schema_penalty": ResultSchemaPenaltyModel(max_penalty=1.0),
            "domain_filter_penalty": DomainFilterPenaltyModel(max_penalty=1.0),
            "date_range_penalty": DateRangePenaltyModel(max_penalty=1.0),
            "sort_order_penalty": SortOrderPenaltyModel(max_penalty=1.0),
            "min_realistic_time_penalty": MinRealisticTimePenaltyModel(max_penalty=1.0),
            "summary_structure_penalty": SummaryStructurePenaltyModel(max_penalty=1.0),
        },
        "timeout_penalty_model": TimeoutPenaltyModel(max_penalty=1.0),
    }


def _upstream_applied_penalty(upstream, raw: float) -> float:
    """Run the REAL ``BasePenaltyModel.apply_penalties`` clip-and-invert tail on one value."""
    numpy = upstream["numpy"]
    base = upstream["penalty_base"]

    class _Probe(base.BasePenaltyModel):
        is_deep = False

        @property
        def name(self) -> str:
            return "parity_probe"

        async def calculate_penalties(self, responses, additional_params=None):
            return numpy.array(list(responses), dtype=numpy.float32)

    _, _, applied = asyncio.run(_Probe(max_penalty=1.0).apply_penalties([raw], uids=[0]))
    return float(applied[0])


def _upstream_case_outputs(case: dict, upstream, weights) -> dict:
    from kata_sn22 import upstream_adapter as adapter

    numpy = upstream["numpy"]
    perf = upstream["perf"]
    synapse = _build_synapse(case, upstream)
    response = parity.build_response(case)
    kind = case["response"]["kind"]

    outputs: dict = {}
    for name, model in upstream["penalties"].items():
        outputs[name] = float(model.penalty_for(synapse))
    outputs["timeout_penalty"] = float(asyncio.run(
        upstream["timeout_penalty_model"].calculate_penalties([synapse]))[0])

    budget = float(perf.resolve_scoring_budget(synapse))
    outputs["resolve_scoring_budget"] = budget
    # The axon time is Kata's own seam: upstream reads a dendrite status code to decide whether a
    # response counts, and Kata's equivalent is "the output parsed against the protocol". The CURVE
    # is what parity tests, so both sides are fed the same resolved time.
    axon_time = adapter.response_time_for(response)
    outputs["performance_reward"] = float(
        perf.PerformanceRewardModel.reward(None, axon_time, budget))
    default_floor = perf.AI_PERF_FLOOR if kind == "ai_search" else perf.X_PERF_FLOOR
    outputs["perf_floor_for"] = float(perf.perf_floor_for(synapse, default_floor))

    if kind == "ai_search":
        # The real reweighting method, with a stand-in ``self`` carrying the pinned weights.
        holder = type("_Weights", (), {
            "reward_weights": numpy.array([weights["ai_content"], weights["ai_summary"]],
                                          dtype=numpy.float32),
            "content_weight": weights["ai_content"],
            "summary_relevance_weight": weights["ai_summary"]})()
        matrix = upstream["advanced"].AdvancedScraperValidator.compute_reward_weights_matrix(
            holder, [synapse])
        outputs["reward_weights_for"] = [float(v) for v in matrix[0]]
        outputs["collect_summary_sources"] = sorted(
            upstream["response_checks"].collect_summary_sources(synapse))
    else:
        outputs["reward_weights_for"] = [float(weights["x_content"])]
        outputs["collect_summary_sources"] = []

    # The combination arithmetic itself is pinned, not executed (see kata_sn22.parity). It is
    # reproduced here from the executed pieces above so the recorded score is still upstream's
    # numbers rather than the adapter's.
    components = tuple(float(c) for c in case["quality"])
    component_weights = tuple(outputs["reward_weights_for"])
    reward = sum(w * c for w, c in zip(component_weights, components))
    quality_gate = reward
    floors = weights["ai_floors"] if kind == "ai_search" else [0.0]
    for index, floor in enumerate(floors):
        if floor > 0 and index < len(components) and components[index] < floor \
                and component_weights[index] > 0:
            quality_gate = 0.0
    perf_multiplier = float(perf.perf_factor(outputs["performance_reward"],
                                             outputs["perf_floor_for"]))
    reward *= perf_multiplier

    names = adapter.AI_PENALTIES if kind == "ai_search" else adapter.X_PENALTIES
    penalty_multiplier = 1.0
    for name in names:
        penalty_multiplier *= _upstream_applied_penalty(upstream, outputs[name])
    reward *= penalty_multiplier
    quality_gate *= penalty_multiplier

    pool_shares = upstream["scoring_constants"].POOL_SHARES
    protocol = upstream["protocol"]
    if kind == "ai_search":
        mode = protocol.SearchMode(case["response"]["mode"])
        pool_share = float(pool_shares[("ai_search", mode)])
    else:
        pool_share = float(pool_shares[("x_search", None)])

    outputs["score"] = {"reward": reward, "quality_gate": quality_gate,
                        "perf_multiplier": perf_multiplier,
                        "penalty_multiplier": penalty_multiplier, "pool_share": pool_share}
    return parity.normalize_value(outputs)


#: Seed used for the one sampled component, on both the recording and checking sides.
PARITY_SAMPLE_SEED = 20260101


def _reward_module():
    """`reward.py` imports bittensor at module scope; reached lazily through the shim."""
    from neurons.validators.reward import reward

    return reward


def _body_fetch(upstream):
    """`body_fetch` imports bittensor at module scope, so it is reached through the shim and
    resolved lazily -- the same treatment `search_content_relevance` gets."""
    from neurons.validators.apify import body_fetch

    return body_fetch


def _link_meets_evidence_upstream(upstream):
    """`link_meets_evidence` lives in the reward model's module, which imports bittensor at module
    scope. It is reached through the shim like everything else, but resolved lazily so importing the
    recorder does not pull the whole reward stack in."""
    from neurons.validators.reward.search_content_relevance import link_meets_evidence

    return link_meets_evidence


def _upstream_scalar(upstream, component: str, args: tuple):
    checks = upstream["response_checks"]
    operators = upstream["web_query_operators"]
    perf = upstream["perf"]
    utils = upstream["desearch_utils"]
    bodies = upstream["source_bodies"]

    if component == "sample_cited_and_uncited":
        # The only sampled component. Upstream draws from the module-level `random`, so both sides
        # are seeded identically before the call and the recorded value is the sequence that seed
        # produces -- which is what makes a sampling function parity-checkable at all.
        import random as _random
        _random.seed(PARITY_SAMPLE_SEED)
        return bodies.sample_cited_and_uncited(*args)

    if component == "first_duplicate_id":
        items, key = args
        normalize = checks.source_key if key in ("link", "url") else None
        return checks.first_duplicate_id(items, key=key, normalize=normalize)
    if component == "applied_penalty":
        return _upstream_applied_penalty(upstream, args[0])
    if component == "performance_reward":
        return perf.PerformanceRewardModel.reward(None, *args)

    table = {
        "normalize_source_url": checks.normalize_source_url,
        "source_key": checks.source_key,
        "extract_markdown_links": checks.extract_markdown_links,
        "check_markdown_structure": checks.check_markdown_structure,
        "tweet_date_in_range": checks.tweet_date_in_range,
        "is_descending_by_created_at": checks.is_descending_by_created_at,
        "normalize_domains": operators.normalize_domains,
        "host_in_domains": operators.host_in_domains,
        "parse_web_query": operators.parse_web_query,
        "format_text_for_match": utils.format_text_for_match,
        "is_valid_tweet": utils.is_valid_tweet,
        "is_valid_web_search_result": utils.is_valid_web_search_result,
        "min_realistic_for_budget": perf.min_realistic_for_budget,
        "perf_factor": perf.perf_factor,
        "sanitize_body_text": _body_fetch(upstream).sanitize_body_text,
        "is_usable_article": _body_fetch(upstream).is_usable_article,
        "highlights_in_order": bodies.highlights_in_order,
        "highlight_subset_of_body": bodies.highlight_subset_of_body,
        "cited_urls_normalized": bodies.cited_urls_normalized,
        "dedup_richest": bodies.dedup_richest,
        "align_citation_markers": bodies.align_citation_markers,
        "link_meets_evidence": _link_meets_evidence_upstream(upstream),
    }
    return table[component](*args)


def _upstream_constants(upstream, weights) -> dict:
    constants = upstream["scoring_constants"]
    perf = upstream["perf"]
    utils = upstream["desearch_utils"]
    return parity.normalize_value({
        "SEARCH_TYPE_WEIGHTS": {k.value if hasattr(k, "value") else str(k): v
                                for k, v in constants.SEARCH_TYPE_WEIGHTS.items()},
        "AI_MODE_WEIGHTS": {k.value if hasattr(k, "value") else str(k): v
                            for k, v in constants.AI_MODE_WEIGHTS.items()},
        "POOL_SHARES": {
            f"{(k[0].value if hasattr(k[0], 'value') else str(k[0]))}:"
            f"{(k[1].value if hasattr(k[1], 'value') else '-')}": v
            for k, v in constants.POOL_SHARES.items()},
        "QUALITY_THRESHOLDS": {k.value if hasattr(k, "value") else str(k): v
                               for k, v in constants.QUALITY_THRESHOLDS.items()},
        "MODE_BUDGETS": {k.value if hasattr(k, "value") else str(k): v
                         for k, v in utils.MODE_BUDGETS.items()},
        "MODE_PERF_FLOORS": dict(perf.MODE_PERF_FLOORS),
        "AI_CONTENT_WEIGHT": weights["ai_content"],
        "AI_SUMMARY_WEIGHT": weights["ai_summary"],
        "AI_COMPONENT_FLOORS": list(weights["ai_floors"]),
        "X_CONTENT_WEIGHT": weights["x_content"],
        "AI_PERF_FLOOR": perf.AI_PERF_FLOOR,
        "X_PERF_FLOOR": perf.X_PERF_FLOOR,
        "MIN_ARTICLE_CHARS": _body_fetch(upstream)._MIN_ARTICLE_CHARS,
        "MAX_BODY_CHARS": _body_fetch(upstream)._RAW_CACHE_CHARS,
        "CACHE_TTL_SECONDS": _body_fetch(upstream)._CACHE_TTL_S,
        "MAX_CACHE_ENTRIES": _body_fetch(upstream)._MAX_CACHE_ENTRIES,
        "PROMPT_ARTIFACT_PATTERN": _reward_module().pattern_to_check,
        # Upstream's floor is a literal `< 2` inside check_tweet_content rather than a constant, so
        # it is transcribed here alongside the symbol digest that pins the method it came from.
        "MIN_MINER_TWEETS": 2,
    })


def record(root: Path) -> dict:
    upstream = _import_upstream(root)
    scrapers = root / "neurons" / "validators" / "scrapers"
    advanced_fields = _self_assignments(scrapers / "advanced_scraper_validator.py",
                                        "AdvancedScraperValidator", "__init__")
    x_fields = _self_assignments(scrapers / "x_scraper_validator.py",
                                 "XScraperValidator", "__init__")
    weights = {
        "ai_content": float(advanced_fields["content_weight"]),
        "ai_summary": float(advanced_fields["summary_relevance_weight"]),
        "ai_floors": [float(f) for f in _component_floors(
            scrapers / "advanced_scraper_validator.py", "AdvancedScraperValidator")],
        "x_content": float(x_fields["twitter_content_weight"]),
    }

    return {
        "schema_version": parity.PARITY_SCHEMA_VERSION,
        "recorded_by": "tools/record_parity.py",
        "upstream_repo": UPSTREAM_REPO,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_tree_sha256": require_intact(root),
        "float_tolerance": parity.FLOAT_TOLERANCE,
        "source_pins": parity.source_pins(root),
        "constants": _upstream_constants(upstream, weights),
        "cases": {case["id"]: _upstream_case_outputs(case, upstream, weights)
                  for case in parity.PARITY_CASES},
        "scalars": [
            {"component": component, "args": parity.normalize_value(list(args)),
             "value": parity.normalize_value(_upstream_scalar(upstream, component, args))}
            for component, args in parity.SCALAR_PROBES
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Compare against the stored evidence without writing it.")
    parser.add_argument("--root", default=None, help="Pinned snapshot root.")
    args = parser.parse_args()
    root = Path(args.root).resolve() if args.root else snapshot_root()

    try:
        document = record(root)
    except upstream_shim.ShimUnavailable as exc:
        print(f"cannot execute the pinned upstream: {exc}", file=sys.stderr)
        return 2

    target = parity.expectations_path()
    serialized = json.dumps(document, indent=2, sort_keys=True) + "\n"

    if args.check:
        if not target.is_file():
            print("no stored parity evidence", file=sys.stderr)
            return 1
        if target.read_text(encoding="utf-8") == serialized:
            print("stored parity evidence matches a fresh recording")
            return 0
        print("stored parity evidence DIFFERS from a fresh recording", file=sys.stderr)
        return 1

    target.write_text(serialized, encoding="utf-8")
    findings = parity.compare_against_expectations(document)
    print(f"recorded {len(document['cases'])} cases and {len(document['scalars'])} scalar probes")
    print(f"  upstream commit  {document['upstream_commit']}")
    print(f"  upstream tree    {document['upstream_tree_sha256']}")
    print(f"  wrote            {target}")
    if findings:
        print(f"  ADAPTER DISAGREES with the upstream on {len(findings)} value(s):")
        for finding in findings:
            print(f"    {finding}")
        return 1
    print("  adapter agrees with the pinned upstream on every recorded value")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
