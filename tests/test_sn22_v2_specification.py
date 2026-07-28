"""Phase A exit gates: schemas round-trip, unknown input fails closed, the policy hash moves.

Phase A freezes a *specification*, so these tests are the specification's teeth. Nothing here runs a
round; what they establish is that the contract cannot drift quietly — a field upstream reads and
this repo does not carry, a prompt edited by a character, a v1 credential accepted as v2.

The most valuable tests in this file are the ones that derive an expectation from the **vendored
upstream** rather than restating a constant. A test that asserts ``0.20 == 0.20`` proves nothing; a
test that reads ``DEEP_SAMPLE_RATE`` out of the pinned tree and compares it to ours catches the day
upstream changes it.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from kata_sn22 import credentials_v2 as cred
from kata_sn22 import protocol_v2 as v2
from kata_sn22 import report_v2 as rep
from kata_sn22 import scorer_policy as policy
from kata_sn22.upstream_snapshot import snapshot_root

GOLDEN = Path(__file__).resolve().parents[1] / "kata_sn22" / "golden"


def _upstream_constant(relative: str, name: str):
    """Read a module-level constant out of the pinned tree by AST.

    By AST rather than by import: these modules import ``bittensor`` at module scope, and the point
    of the dependency-free port is that nothing in this repo's test path needs it.
    """
    source = (snapshot_root() / relative).read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{relative} no longer defines {name}")


# ---- the numbers come from upstream, not from us ---

def test_the_deep_sample_rate_matches_the_pinned_upstream():
    """The pool size falls out of this. If upstream changes it, 15 tasks stop producing 3 deep
    samples and every pool silently drops out of scoring."""
    assert policy.DEEP_SAMPLE_RATE == _upstream_constant(
        "neurons/validators/scoring/query_scheduler.py", "DEEP_SAMPLE_RATE")


def test_the_minimum_deep_samples_matches_the_pinned_upstream():
    assert policy.MIN_DEEP_SAMPLES_PER_POOL == _upstream_constant(
        "neurons/validators/scoring/constants.py", "MIN_DEEP_SAMPLES_PER_POOL")


def test_the_pool_size_is_derived_not_chosen():
    """15 is not a preference. ``_pool_raw_scores`` DROPS a UID with fewer than the minimum deep
    samples, so a smaller pool scores zero for a reason unrelated to the answers — which is why the
    deployed ``task_count=8`` could never have worked."""
    assert policy.minimum_tasks_per_pool() == 15
    assert policy.TASKS_PER_POOL == policy.minimum_tasks_per_pool()


def test_the_quality_and_volume_exponents_match_upstream():
    for name in ("QUALITY_EXPONENT", "VOLUME_EXPONENT", "GATE_RAMP"):
        assert getattr(policy, name) == _upstream_constant(
            "neurons/validators/scoring/constants.py", name), name


def test_the_result_count_is_upstreams_minimum_not_ours():
    """v1 defaulted to 5. Upstream declares ``Field(10, ge=10)``, so 5 was not a different default
    — it was input upstream's own model rejects."""
    assert v2.DEFAULT_RESULT_COUNT == 10
    assert v2.MIN_RESULT_COUNT == 10


def test_the_pool_weights_sum_to_one_and_match_the_adapter():
    from kata_sn22 import upstream_adapter as adapter

    assert sum(policy.POOL_WEIGHTS.values()) == pytest.approx(1.0)
    assert policy.POOL_WEIGHTS["ai_search:fast"] == pytest.approx(
        adapter.SEARCH_TYPE_WEIGHTS["ai_search"] * adapter.AI_MODE_WEIGHTS["fast"])
    assert policy.POOL_WEIGHTS["x_search"] == pytest.approx(
        adapter.SEARCH_TYPE_WEIGHTS["x_search"])


def test_the_fixed_scorer_is_the_model_upstream_defaults_to():
    """Upstream's validator config defaults to Qwen; several helper constructors default to
    gpt-4.1-nano and there is a Chutes->OpenAI fallback. Kata pins Qwen and disables the fallback,
    because one contestant graded by Qwen and another by GPT is not one competition."""
    source = (snapshot_root() / "desearch" / "protocol.py").read_text(encoding="utf-8")
    assert v2.FIXED_SCORING_MODEL.value in source
    assert policy.PRODUCTION_POLICY.scorer_fallback_enabled is False


# ---- the scoring surface is complete ---

def test_every_field_the_upstream_scorer_reads_is_carried():
    """Derived, not asserted: scan every ``response.<field>`` access in the reward, penalty,
    scraper and scoring packages and require this module to know about each one.

    This is the test that catches a new upstream reading a field v2 does not carry — which would
    otherwise surface as a score computed from a default nobody chose.
    """
    import re

    accessed: set[str] = set()
    for package in ("reward", "penalty", "scrapers", "scoring"):
        directory = snapshot_root() / "neurons" / "validators" / package
        if not directory.is_dir():
            continue
        for path in directory.glob("*.py"):
            accessed |= set(re.findall(r"(?:response|synapse)\.([a-z_]+)",
                                       path.read_text(encoding="utf-8")))

    # Transport, helper methods and pydantic plumbing -- not scored content.
    not_content = {
        "get", "dendrite", "axon", "model_copy", "model_dump", "id", "urls",
        "get_links_from_search_results", "get_links_from_tweets", "get_search_results_by_tools",
        "is_success", "status_code", "deserialize", "name", "timeout", "process_time",
    }
    known = set(v2.SCORED_RESPONSE_FIELDS) | set(v2.SCORED_TASK_FIELDS) | not_content
    unknown = sorted(accessed - known)
    assert not unknown, (
        f"the pinned upstream scorer reads field(s) protocol_v2 does not carry: {unknown}. "
        f"Add them to SCORED_RESPONSE_FIELDS and to the models, or to the not-content list with a "
        f"reason.")


# ---- schemas round-trip ---

@pytest.mark.parametrize("name", [
    "ai_web_links_with_summary", "ai_twitter_links_with_summary", "ai_only_links", "x_basic"])
def test_every_golden_example_round_trips(name):
    """All four pools and both result types. Golden files are generated FROM the models, so a
    change that breaks the contract breaks this rather than quietly rewriting the examples."""
    document = json.loads((GOLDEN / f"{name}.json").read_text(encoding="utf-8"))
    task_input = document["task"]
    assert task_input["protocol_version"] == v2.PROTOCOL_VERSION

    if task_input["search_type"] == v2.SearchType.X_SEARCH.value:
        task = v2.XSearchTask(task_id=task_input["task_id"], query=task_input["query"],
                              count=task_input["count"], sort=task_input["sort"])
    else:
        task = v2.AiSearchTask(
            task_id=task_input["task_id"], prompt=task_input["prompt"],
            mode=v2.SearchMode(task_input["mode"]),
            result_type=v2.ResultType(task_input["result_type"]),
            tools=tuple(task_input["tools"]), count=task_input["count"])

    answer = v2.parse_answer(document["answer"], task=task)
    assert answer.task_id == task.task_id
    # Re-serializing must produce the same document: a lossy round-trip means the report hash a
    # room computes and the one the host recomputes would differ.
    assert answer.as_dict() == document["answer"]


def test_the_golden_examples_cover_every_pool_and_both_result_types():
    seen_modes, seen_types = set(), set()
    for path in GOLDEN.glob("*.json"):
        task = json.loads(path.read_text(encoding="utf-8"))["task"]
        seen_modes.add(task.get("mode") or "x_search")
        seen_types.add(task.get("result_type"))
    assert seen_modes == {"fast", "balanced", "deep", "x_search"}
    assert {"ONLY_LINKS", "LINKS_WITH_FINAL_SUMMARY"} <= seen_types


def test_the_deep_flag_never_reaches_the_agent():
    """An agent that knew which tasks were deep-scored would work hardest on exactly those, and the
    20% sample would stop measuring the other 80%."""
    for path in GOLDEN.glob("*.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        assert "deep" not in document["task"], path.name
        assert "deep" in document, f"{path.name} must still RECORD it for the manifest"


# ---- unknown input fails closed ---

def _task():
    return v2.AiSearchTask(task_id="t1", prompt="q", mode=v2.SearchMode.FAST,
                           result_type=v2.ResultType.LINKS_WITH_FINAL_SUMMARY,
                           tools=("Web Search",))


@pytest.mark.parametrize("version", [1, 3, "2", None])
def test_a_wrong_protocol_version_is_refused(version):
    with pytest.raises(v2.ProtocolV2Error, match="protocol_version"):
        v2.parse_answer({"protocol_version": version, "task_id": "t1"}, task=_task())


def test_an_unknown_answer_field_is_refused_not_ignored():
    """An agent sending a field the scorer will never read believed something false about how it
    would be graded. Dropping it silently lets that belief survive to the next round."""
    with pytest.raises(v2.ProtocolV2Error, match="unknown field"):
        v2.parse_answer({"protocol_version": 2, "task_id": "t1", "confidence": 0.9},
                        task=_task())


def test_an_unknown_text_role_or_chunk_role_is_refused():
    for payload in (
        {"protocol_version": 2, "task_id": "t1", "texts": {"invented": "x"}},
        {"protocol_version": 2, "task_id": "t1", "chunks": [{"role": "invented", "text": "x"}]},
    ):
        with pytest.raises(v2.ProtocolV2Error):
            v2.parse_answer(payload, task=_task())


def test_an_answer_for_another_task_is_refused():
    with pytest.raises(v2.ProtocolV2Error, match="task_id"):
        v2.parse_answer({"protocol_version": 2, "task_id": "other"}, task=_task())


@pytest.mark.parametrize("count", [0, 5, 9, 201])
def test_a_result_count_outside_upstreams_range_is_refused(count):
    with pytest.raises(v2.ProtocolV2Error, match="count"):
        v2.AiSearchTask(task_id="t1", prompt="q", mode=v2.SearchMode.FAST,
                        result_type=v2.ResultType.ONLY_LINKS, tools=("Web Search",), count=count)


def test_an_oversized_answer_is_refused_before_it_is_parsed():
    task = v2.AiSearchTask(task_id="t1", prompt="q", mode=v2.SearchMode.FAST,
                           result_type=v2.ResultType.ONLY_LINKS, tools=("Web Search",),
                           limits=v2.Limits(max_execution_time=5, max_output_bytes=64))
    with pytest.raises(v2.ProtocolV2Error, match="over the"):
        v2.parse_answer(json.dumps({"protocol_version": 2, "task_id": "t1",
                                    "completion": "x" * 500}), task=task)


# ---- credentials ---

def _payload(**over):
    document = {
        "version": 2, "credential_profile": cred.CREDENTIAL_PROFILE,
        "credentials": {name: {"api_key": "k" * 24} for name in cred.REQUIRED_PROVIDERS},
        "bundle_binding": "a" * 64,
    }
    document.update(over)
    return document


def test_a_complete_credential_set_parses():
    parsed = cred.parse_credential_payload(_payload())
    assert sorted(parsed.keys) == sorted(cred.REQUIRED_PROVIDERS)


def test_a_version_1_credential_cannot_be_read_as_version_2():
    """v1 sealed ONE key for SN60. Reading it as "one of four provided" would start a round that
    cannot finish, three pools in."""
    with pytest.raises(cred.CredentialError, match="version-1"):
        cred.parse_credential_payload({"version": 1, "provider": "openai", "api_key": "k" * 24})


@pytest.mark.parametrize("missing", cred.REQUIRED_PROVIDERS)
def test_every_provider_is_required(missing):
    """All four, because a production epoch covers all four pools. Requiring them at sealing time
    turns "you fail on pool three, an hour in" into "your submission is rejected at intake"."""
    payload = _payload()
    del payload["credentials"][missing]
    with pytest.raises(cred.CredentialError, match=missing):
        cred.parse_credential_payload(payload)


def test_an_unknown_provider_is_refused():
    payload = _payload()
    payload["credentials"]["anthropic"] = {"api_key": "k" * 24}
    with pytest.raises(cred.CredentialError, match="unknown provider"):
        cred.parse_credential_payload(payload)


@pytest.mark.parametrize("bad", ["", "short", "k" * 600, "has space", "has\nnewline"])
def test_an_unusable_key_is_refused(bad):
    payload = _payload()
    payload["credentials"]["openai"] = {"api_key": bad}
    with pytest.raises(cred.CredentialError):
        cred.parse_credential_payload(payload)


def test_no_error_message_ever_contains_a_key():
    """These exceptions are logged, attested and shown to a miner."""
    secret = "sk-super-secret-value-12345678"
    payload = _payload()
    payload["credentials"]["openai"] = {"api_key": secret, "extra": 1}
    with pytest.raises(cred.CredentialError) as raised:
        cred.parse_credential_payload(payload)
    assert secret not in str(raised.value)


def test_the_credential_set_never_renders_its_keys():
    """A dataclass holding secrets that renders them in a traceback is a secret in every log that
    catches an exception."""
    parsed = cred.parse_credential_payload(_payload())
    for rendered in (repr(parsed), str(parsed), f"{parsed}"):
        assert "kkkk" not in rendered


def test_the_binding_is_compared_against_the_actual_bundle():
    parsed = cred.parse_credential_payload(_payload())
    assert cred.bundle_binding_matches(parsed, "a" * 64)
    assert not cred.bundle_binding_matches(parsed, "b" * 64)


def test_the_binding_covers_every_file_except_the_ciphertext():
    """The ciphertext cannot commit to itself; everything else must, so editing agent.py or a
    helper invalidates the seal and forces a reseal."""
    files = {"agent.py": b"print(1)", "helpers/util.py": b"x = 1",
             "submission.json": b"{}", cred.SEALED_FILENAME: b"deadbeef"}
    baseline = cred.compute_bundle_binding(files)

    assert cred.compute_bundle_binding({**files, cred.SEALED_FILENAME: b"cafe"}) == baseline
    for changed in ("agent.py", "helpers/util.py", "submission.json"):
        assert cred.compute_bundle_binding({**files, changed: b"tampered"}) != baseline, changed


def test_the_agent_can_never_spend_the_judge_credential():
    """An agent that could reach the judge could grade itself."""
    parsed = cred.parse_credential_payload(_payload())
    assert "chutes" not in parsed.for_role("agent")
    assert "chutes" in parsed.for_role("evaluator")
    # ...and the evaluator has no reason to hold the agent's summary key.
    assert "openai" not in parsed.for_role("evaluator")


@pytest.mark.parametrize("status", sorted(cred.CONTESTANT_FAULT_STATUSES, key=lambda s: s.value))
def test_a_contestant_fault_zeroes_that_contestant(status):
    report = cred.CredentialReport({**cred.CredentialReport.all_ok().statuses, "apify": status})
    assert report.contestant_at_fault is True
    assert report.defer is False


def test_a_provider_outage_defers_rather_than_zeroing():
    """Punishing a miner for someone else's outage is the failure this separation prevents."""
    report = cred.CredentialReport({**cred.CredentialReport.all_ok().statuses,
                                    "apify": cred.CredentialStatus.PROVIDER_OUTAGE})
    assert report.defer is True
    assert report.contestant_at_fault is False


# ---- the policy hash ---

def test_the_policy_hash_is_stable_for_an_unchanged_policy():
    assert policy.policy_hash() == policy.ScorerPolicy().policy_hash()


@pytest.mark.parametrize(("field_name", "value"), [
    ("upstream_commit", "b" * 40),
    ("scoring_model", "openai/gpt-4.1-nano"),
    ("judge_temperature", 0.0),
    ("scorer_fallback_enabled", True),
    ("result_count", 20),
    ("tasks_per_pool", 20),
    ("deep_sample_rate", 0.25),
    ("min_deep_samples_per_pool", 4),
    ("quality_exponent", 2.0),
    ("volume_exponent", 3.0),
    ("gate_ramp", 0.1),
    ("ai_content_weight", 0.5),
    ("ai_summary_weight", 0.5),
    ("apify_tweet_actor", "someOtherActor"),
])
def test_changing_any_scoring_input_moves_the_policy_hash(field_name, value):
    """THE Phase A exit gate. Two contestants are only comparable if this matched."""
    import dataclasses

    changed = dataclasses.replace(policy.PRODUCTION_POLICY, **{field_name: value})
    assert changed.policy_hash() != policy.policy_hash(), field_name


def test_editing_a_judge_prompt_by_one_character_moves_the_policy_hash(monkeypatch):
    """The rubric IS the policy. A reworded prompt is a different scoring policy, and it would look
    like a tidy-up in review."""
    from kata_sn22 import judge_prompts

    before = policy.policy_hash()
    monkeypatch.setattr(judge_prompts, "SYSTEM_SUMMARY_GROUNDEDNESS_TEMPLATE",
                        judge_prompts.SYSTEM_SUMMARY_GROUNDEDNESS_TEMPLATE + " ")
    assert policy.ScorerPolicy().policy_hash() != before


def test_changing_a_provider_route_moves_the_policy_hash():
    """A different route means a different provider answered, and a score produced against a
    different provider is not comparable."""
    routes = {**policy.PROVIDER_ROUTES, "evaluator.judge": "chutes:some-other-model"}
    import dataclasses

    assert dataclasses.replace(
        policy.PRODUCTION_POLICY, provider_routes=routes).policy_hash() != policy.policy_hash()


def test_operational_tuning_is_deliberately_NOT_in_the_policy():
    """Timeouts, retries and concurrency change how long a round takes, not what an answer is
    worth. Folding them in would make the hash churn on tuning and stop a mismatch meaning
    anything."""
    document = policy.PRODUCTION_POLICY.as_document()
    for name in ("timeout", "retries", "concurrency", "round_gap_sec", "image_tag"):
        assert not any(name in key for key in document), name


# ---- reports ---

def _report(pool, *, contestant="king", status=rep.ReportStatus.OK, **over):
    fields = dict(
        pool=pool, status=status, contestant=contestant, bundle_sha256="b" * 64,
        task_manifest_sha256="m" * 64, policy_hash=policy.policy_hash(),
        agent_image_digest="sha256:" + "a" * 64,
        pool_result=rep.PoolResult(0.7, 0.6, 15, 3) if status is rep.ReportStatus.OK else None,
        credentials=cred.CredentialReport.all_ok())
    fields.update(over)
    return rep.PoolReport(**fields)


def test_a_pool_report_round_trips():
    original = _report("ai_search:fast")
    assert rep.parse_pool_report(original.as_dict()).as_dict() == original.as_dict()


def test_an_unknown_report_field_is_refused():
    document = {**_report("ai_search:fast").as_dict(), "bonus": 1}
    with pytest.raises(rep.ReportError, match="unknown field"):
        rep.parse_pool_report(document)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_a_non_finite_or_negative_pool_value_is_refused(bad):
    """NaN poisons every comparison downstream — ``nan > x`` is false for every x — so a duel
    decided on one would silently keep the King forever."""
    with pytest.raises(rep.ReportError):
        rep.PoolResult(bad, 0.6, 15, 3)


def test_more_deep_samples_than_tasks_is_refused():
    with pytest.raises(rep.ReportError, match="deep_count"):
        rep.PoolResult(0.7, 0.6, 15, 16)


def test_a_failed_report_must_not_carry_numbers():
    """A failed pool with a score attached invites someone to use it."""
    with pytest.raises(rep.ReportError, match="must not carry"):
        _report("ai_search:fast", status=rep.ReportStatus.CREDENTIAL_FAILURE,
                pool_result=rep.PoolResult(0.7, 0.6, 15, 3))


def test_an_epoch_needs_every_pool_exactly_once():
    with pytest.raises(rep.ReportError, match="every pool"):
        rep.assemble_epoch([_report("ai_search:fast")] * 4, contestant="king")


@pytest.mark.parametrize("drift", ["policy_hash", "task_manifest_sha256", "agent_image_digest",
                                   "upstream_commit", "scoring_model", "bundle_sha256"])
def test_pool_reports_that_disagree_on_identity_are_not_one_epoch(drift):
    """Each of these, if it differed between pools, would mean the four were not produced under one
    set of rules — and the total would be a sum of incomparable things."""
    reports = [_report(pool) for pool in rep.POOLS]
    import dataclasses

    reports[2] = dataclasses.replace(reports[2], **{drift: "drifted"})
    with pytest.raises(rep.ReportError):
        rep.assemble_epoch(reports, contestant="king")


def test_infrastructure_failure_outranks_credential_failure():
    """If any pool could not RUN, we do not know the credential failure elsewhere would have
    mattered. A deferred duel can be re-run; a zeroed contestant cannot be un-zeroed."""
    reports = [_report(pool) for pool in rep.POOLS]
    reports[1] = _report(rep.POOLS[1], status=rep.ReportStatus.CREDENTIAL_FAILURE)
    reports[2] = _report(rep.POOLS[2], status=rep.ReportStatus.INFRASTRUCTURE_FAILURE)
    epoch = rep.assemble_epoch(reports, contestant="king")
    assert epoch.status is rep.ReportStatus.INFRASTRUCTURE_FAILURE
    assert epoch.pool_results() == {}


def test_a_duel_is_refused_when_the_two_sides_were_graded_differently():
    """A challenger scored under a newer policy beating a king scored under an older one would look
    exactly like skill."""
    king = rep.assemble_epoch([_report(p, contestant="king") for p in rep.POOLS],
                              contestant="king")
    challenger = rep.assemble_epoch(
        [_report(p, contestant="pr-7", policy_hash="different") for p in rep.POOLS],
        contestant="pr-7")
    with pytest.raises(rep.ReportError, match="same rules"):
        rep.duel_is_comparable(king, challenger)


def test_a_comparable_duel_passes():
    king = rep.assemble_epoch([_report(p, contestant="king") for p in rep.POOLS],
                              contestant="king")
    challenger = rep.assemble_epoch(
        [_report(p, contestant="pr-7", bundle_sha256="c" * 64) for p in rep.POOLS],
        contestant="pr-7")
    # The BUNDLES differ -- that is the duel. Everything about the grading matches.
    rep.duel_is_comparable(king, challenger)
    assert len(king.pool_results()) == 4
