"""SN22-2: the frozen protocol, the sealed manifests, and the reference ladder.

The exit gate has two halves and both are pinned here:

  "The protocol and fixtures are reviewable **without network access** and deterministically order
   **weak < medium < strong in both comparator directions**."

The second half is the subtle one. "Both directions" is not a restatement of antisymmetry for its
own sake — it is the fairness property from plan §5.2 item 5, that the order contestants are run and
compared in must not decide the crown. Every ladder test below therefore checks the reversal too.
"""
from __future__ import annotations

import json
import socket

import pytest

from kata_sn22.fake_provider import FakeRelay, RelayDenied
from kata_sn22.fixtures import (
    CALIBRATION_SEED,
    QUERY_POOL,
    QUERY_SOURCE_ID,
    QUERY_SOURCE_VERSION,
    calibration_manifest,
    calibration_snapshot,
    reference_responses,
    tasks_for,
)
from kata_sn22.manifests import (
    ManifestError,
    SnapshotDocument,
    SnapshotManifest,
    UsageManifest,
    UsageRecord,
    benchmark_identity,
    derive_query_manifest,
)
from kata_sn22.protocol import (
    MAX_OUTPUT_BYTES,
    PROTOCOL_VERSION,
    ErrorClass,
    Limits,
    ProtocolError,
    Task,
    parse_task_output,
    validate_task,
)
from kata_sn22.scoring import (
    RANK_SIGNALS,
    Signals,
    TaskAttempt,
    beats_king,
    compare_signals,
    score_attempts,
)


# ---- the whole suite must be reviewable OFFLINE --------------------------------------------------
@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Make any socket use a hard failure for every test in this module.

    The exit gate says the protocol and fixtures are reviewable without network access. Asserting
    that by inspection is worthless -- one `requests.get` in a helper would falsify it silently --
    so the ability to open a socket is removed instead.
    """
    def _refuse(*_args, **_kwargs):
        raise AssertionError("SN22-2 fixtures must never touch the network")

    monkeypatch.setattr(socket, "socket", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)


@pytest.fixture
def world():
    manifest = calibration_manifest()
    snapshot = calibration_snapshot(manifest)
    return manifest, snapshot, tasks_for(manifest)


#: How long the LANE observed each reference submission to take. Fixed per reference so the ladder
#: stays deterministic, and supplied by the harness rather than read from the agent -- latency is a
#: ranked signal, so taking it from the submission would let a candidate win by claiming 0.0.
OBSERVED_SECONDS = {"weak": 3.0, "medium": 2.0, "strong": 1.0, "invalid": 5.0, "malicious": 4.0}


def _attempts(kind: str, tasks, snapshot) -> list[TaskAttempt]:
    """Parse a reference submission into attempts, classifying whatever fails the contract."""
    observed = OBSERVED_SECONDS[kind]
    attempts = []
    for task, raw in zip(tasks, reference_responses(kind, tasks, snapshot), strict=True):
        try:
            attempts.append(TaskAttempt(task=task, output=parse_task_output(raw, task=task),
                                        observed_seconds=observed))
        except ProtocolError as exc:
            attempts.append(TaskAttempt(task=task, error=exc.error_class,
                                        observed_seconds=observed))
    return attempts


def _billed(tasks, variant: str = "candidate", *, calls: int = 1,
            tokens: int = 250) -> UsageManifest:
    """The relay's record. Identical for every reference unless a test varies it, so the ladder is
    decided by answer quality rather than by who happened to be billed more."""
    return UsageManifest(challenge_id="c1", records=tuple(
        UsageRecord(variant=variant, task_id=t.task_id, provider_calls=calls, tokens=tokens,
                    spend_usd=0.002 * calls) for t in tasks))


def _signals(kind: str, tasks, snapshot) -> Signals:
    return score_attempts(_attempts(kind, tasks, snapshot), snapshot=snapshot,
                          usage=_billed(tasks), variant="candidate")


# ---- THE EXIT GATE -------------------------------------------------------------------------------
def test_the_ladder_orders_weak_medium_strong(world):
    manifest, snapshot, tasks = world
    weak = _signals("weak", tasks, snapshot)
    medium = _signals("medium", tasks, snapshot)
    strong = _signals("strong", tasks, snapshot)

    assert compare_signals(medium, weak) == 1
    assert compare_signals(strong, medium) == 1
    assert compare_signals(strong, weak) == 1


def test_the_ladder_orders_identically_in_the_REVERSE_direction(world):
    """The fairness property: which contestant is passed first must not decide the crown."""
    _manifest, snapshot, tasks = world
    weak = _signals("weak", tasks, snapshot)
    medium = _signals("medium", tasks, snapshot)
    strong = _signals("strong", tasks, snapshot)

    assert compare_signals(weak, medium) == -1
    assert compare_signals(medium, strong) == -1
    assert compare_signals(weak, strong) == -1


def test_the_comparator_is_antisymmetric_for_every_pair(world):
    _manifest, snapshot, tasks = world
    ladder = {k: _signals(k, tasks, snapshot) for k in ("weak", "medium", "strong", "malicious")}
    for left in ladder.values():
        for right in ladder.values():
            assert compare_signals(left, right) == -compare_signals(right, left)


def test_the_ladder_is_deterministic_across_repeated_scoring(world):
    """A fixture that drifts cannot serve as a calibration baseline."""
    _manifest, snapshot, tasks = world
    for kind in ("weak", "medium", "strong"):
        runs = [_signals(kind, tasks, snapshot).as_metrics() for _ in range(5)]
        assert all(run == runs[0] for run in runs)


def test_the_ladder_is_identical_under_a_rebuilt_world(world):
    """Same seed, rebuilt from source: same queries, same snapshot, same scores."""
    _manifest, snapshot, tasks = world
    rebuilt_manifest = calibration_manifest()
    rebuilt_snapshot = calibration_snapshot(rebuilt_manifest)
    rebuilt_tasks = tasks_for(rebuilt_manifest)
    assert rebuilt_snapshot.digest() == snapshot.digest()
    for kind in ("weak", "medium", "strong"):
        assert (_signals(kind, rebuilt_tasks, rebuilt_snapshot).as_metrics()
                == _signals(kind, tasks, snapshot).as_metrics())


# ---- the ordered signal schema -------------------------------------------------------------------
def test_the_published_signals_are_exactly_the_planned_seven_in_order():
    assert [name for name, _higher in RANK_SIGNALS] == [
        "sn22_valid_query_rate", "sn22_weighted_quality", "sn22_citation_precision",
        "sn22_coverage", "sn22_invalid_runs", "sn22_cost_units", "sn22_latency_seconds",
    ]
    assert [higher for _name, higher in RANK_SIGNALS] == [True, True, True, True,
                                                          False, False, False]


def test_validity_outranks_quality(world):
    """Priority order, not a weighted sum: a beautiful answer to half the queries does not beat a
    good answer to all of them."""
    _manifest, snapshot, tasks = world
    thorough = Signals(1.0, 0.30, 1.0, 1.0, 0, 1.0, 1.0)
    brilliant_but_partial = Signals(0.5, 0.99, 1.0, 1.0, 0, 1.0, 1.0)
    assert compare_signals(thorough, brilliant_but_partial) == 1
    assert compare_signals(brilliant_but_partial, thorough) == -1


def test_cost_cannot_buy_a_crown(world):
    """Cost ranks BELOW quality, so spending more can win -- but only by being genuinely better.
    Equal quality with higher cost must lose."""
    cheap = Signals(1.0, 0.80, 1.0, 1.0, 0, 5.0, 1.0)
    expensive = Signals(1.0, 0.80, 1.0, 1.0, 0, 500.0, 1.0)
    assert compare_signals(cheap, expensive) == 1
    assert compare_signals(expensive, cheap) == -1


def test_a_margin_defers_to_the_next_signal():
    """Without an indifference band, float noise on a high-priority signal decides a crown alone."""
    a = Signals(1.0, 0.8000001, 1.0, 1.0, 0, 10.0, 1.0)
    b = Signals(1.0, 0.8000000, 1.0, 1.0, 0, 1.0, 1.0)
    assert compare_signals(a, b) == 1                          # no margin: a wins on noise
    assert compare_signals(a, b, margins={"sn22_weighted_quality": 0.01}) == -1  # cost decides
    assert compare_signals(b, a, margins={"sn22_weighted_quality": 0.01}) == 1


def test_a_tie_keeps_the_incumbent(world):
    _manifest, snapshot, tasks = world
    strong = _signals("strong", tasks, snapshot)
    assert compare_signals(strong, strong) == 0
    assert not beats_king(strong, strong)


def test_an_empty_throne_still_requires_a_valid_run(world):
    _manifest, snapshot, tasks = world
    assert beats_king(_signals("weak", tasks, snapshot), None)
    nothing_valid = Signals(0.0, 0.0, 0.0, 0.0, 4, 0.0, 0.0)
    assert not beats_king(nothing_valid, None)


def test_a_non_finite_signal_is_refused_rather_than_compared():
    """NaN compares False against everything, so it would silently win or lose every tie-break."""
    poisoned = Signals(float("nan"), 0.5, 0.5, 0.5, 0, 1.0, 1.0)
    with pytest.raises(ValueError, match="not finite"):
        compare_signals(poisoned, Signals(1.0, 0.5, 0.5, 0.5, 0, 1.0, 1.0))


# ---- invalid and malicious submissions -----------------------------------------------------------
def test_an_invalid_submission_is_classified_not_scored_low(world):
    """A crash and a poor answer must differ, or a broken agent looks merely mediocre."""
    _manifest, snapshot, tasks = world
    attempts = _attempts("invalid", tasks, snapshot)
    assert all(a.error is ErrorClass.INVALID_SCHEMA for a in attempts)
    signals = score_attempts(attempts, snapshot=snapshot, usage=_billed(tasks),
                             variant="candidate")
    assert signals.sn22_valid_query_rate == 0.0
    assert signals.sn22_invalid_runs == len(tasks)


def test_an_invalid_submission_never_outranks_a_valid_one(world):
    _manifest, snapshot, tasks = world
    invalid = _signals("invalid", tasks, snapshot)
    weak = _signals("weak", tasks, snapshot)
    assert compare_signals(weak, invalid) == 1
    assert compare_signals(invalid, weak) == -1


def test_citing_without_retrieving_earns_nothing(world):
    """The malicious fixture cites every relevant document while returning none of them."""
    _manifest, snapshot, tasks = world
    malicious = _signals("malicious", tasks, snapshot)
    assert malicious.sn22_weighted_quality == 0.0
    assert malicious.sn22_coverage == 0.0


def test_a_fabricated_document_is_not_a_supported_citation(world):
    _manifest, snapshot, tasks = world
    assert not snapshot.contains("doc-fabricated-999")
    malicious = _signals("malicious", tasks, snapshot)
    assert malicious.sn22_citation_precision < 1.0


def test_a_malicious_submission_does_not_outrank_an_honest_weak_one(world):
    """The whole point: lying must not pay better than searching badly."""
    _manifest, snapshot, tasks = world
    malicious = _signals("malicious", tasks, snapshot)
    weak = _signals("weak", tasks, snapshot)
    assert compare_signals(weak, malicious) == 1
    assert compare_signals(malicious, weak) == -1
    assert compare_signals(_signals("strong", tasks, snapshot), malicious) == 1


def test_a_prompt_injection_is_carried_as_inert_data(world):
    """Retrieved and submitted text is data. It is stored and compared, never interpreted."""
    _manifest, snapshot, tasks = world
    attempts = _attempts("malicious", tasks, snapshot)
    summaries = [a.output.summary for a in attempts if a.output is not None]
    assert any("IGNORE ALL PREVIOUS INSTRUCTIONS" in s for s in summaries)
    # It reached the scorer verbatim and changed nothing.
    assert _signals("malicious", tasks, snapshot).sn22_weighted_quality == 0.0


def test_under_reported_usage_is_overridden_by_the_relay(world):
    """The malicious fixture claims zero cost. The relay billed it, and the relay wins."""
    _manifest, snapshot, tasks = world
    attempts = _attempts("malicious", tasks, snapshot)
    billed = _billed(tasks, "challenger", calls=3, tokens=900)
    scored = score_attempts(attempts, snapshot=snapshot, usage=billed, variant="challenger")

    # The agent claimed nothing; the relay billed it. The ranked signal follows the relay.
    assert scored.detail["self_reported_calls"] == 0
    assert scored.detail["relay_billed_calls"] == 3 * len(tasks)
    assert scored.sn22_cost_units > 0.0
    assert scored.detail["usage_source"] == "relay"


# ---- the protocol itself -------------------------------------------------------------------------
def test_a_response_for_another_task_is_refused(world):
    _manifest, _snapshot, tasks = world
    payload = json.dumps({"protocol_version": PROTOCOL_VERSION, "task_id": "somewhere-else",
                          "summary": "", "results": [], "citations": [], "usage": {}})
    with pytest.raises(ProtocolError) as caught:
        parse_task_output(payload, task=tasks[0])
    assert caught.value.error_class is ErrorClass.INVALID_SCHEMA


def test_an_unknown_protocol_version_is_refused_not_interpreted(world):
    _manifest, _snapshot, tasks = world
    payload = json.dumps({"protocol_version": 99, "task_id": tasks[0].task_id, "summary": "",
                          "results": [], "citations": [], "usage": {}})
    with pytest.raises(ProtocolError, match="protocol_version"):
        parse_task_output(payload, task=tasks[0])


def test_an_oversized_response_is_refused_before_parsing(world):
    _manifest, _snapshot, tasks = world
    with pytest.raises(ProtocolError) as caught:
        parse_task_output(b"x" * (MAX_OUTPUT_BYTES + 1), task=tasks[0])
    assert caught.value.error_class is ErrorClass.EXCESS_OUTPUT


def test_too_many_results_is_refused(world):
    _manifest, _snapshot, tasks = world
    task = tasks[0]
    payload = json.dumps({
        "protocol_version": PROTOCOL_VERSION, "task_id": task.task_id, "summary": "",
        "results": [{"doc_id": f"doc-{i}", "title": "", "snippet": ""}
                    for i in range(task.limits.max_results + 1)],
        "citations": [], "usage": {}})
    with pytest.raises(ProtocolError) as caught:
        parse_task_output(payload, task=task)
    assert caught.value.error_class is ErrorClass.EXCESS_OUTPUT


def test_a_repeated_document_is_refused(world):
    """Ten copies of one document is one document; counting it ten times would inflate coverage."""
    _manifest, _snapshot, tasks = world
    payload = json.dumps({
        "protocol_version": PROTOCOL_VERSION, "task_id": tasks[0].task_id, "summary": "",
        "results": [{"doc_id": "doc-kata-1", "title": "", "snippet": ""}] * 2,
        "citations": [], "usage": {}})
    with pytest.raises(ProtocolError, match="repeated"):
        parse_task_output(payload, task=tasks[0])


@pytest.mark.parametrize("usage", [
    {"provider_calls": -1},
    {"tokens": "many"},
    {"elapsed_seconds": float("nan")},
    {"elapsed_seconds": float("inf")},
    {"provider_calls": True},
])
def test_unusable_usage_numbers_are_refused(world, usage):
    _manifest, _snapshot, tasks = world
    payload = json.dumps({"protocol_version": PROTOCOL_VERSION, "task_id": tasks[0].task_id,
                          "summary": "", "results": [], "citations": [], "usage": usage})
    with pytest.raises(ProtocolError):
        parse_task_output(payload, task=tasks[0])


def test_a_provider_outage_is_not_the_candidates_fault(world):
    """A shared fault must not count against whichever contestant was running at the time."""
    _manifest, snapshot, tasks = world
    assert not ErrorClass.PROVIDER_UNAVAILABLE.candidate_caused
    assert ErrorClass.TIMEOUT.candidate_caused

    attempts = _attempts("strong", tasks, snapshot)
    degraded = [TaskAttempt(task=attempts[0].task, error=ErrorClass.PROVIDER_UNAVAILABLE),
                *attempts[1:]]
    signals = score_attempts(degraded, snapshot=snapshot, usage=_billed(tasks),
                             variant="candidate")
    assert signals.sn22_valid_query_rate == 1.0        # excluded, not counted as a failure
    assert signals.sn22_invalid_runs == 0
    assert signals.detail["infrastructure_faults"] == 1


@pytest.mark.parametrize("task, match", [
    (Task(task_id="BAD ID", query="q", search_type="ai_search", ai_mode="fast"), "task_id"),
    (Task(task_id="t000", query="  ", search_type="ai_search", ai_mode="fast"), "empty"),
    (Task(task_id="t000", query="q", search_type="telepathy"), "search_type"),
    (Task(task_id="t000", query="q", search_type="ai_search"), "ai_mode"),
    (Task(task_id="t000", query="q", search_type="x_search", ai_mode="fast"), "only to ai_search"),
])
def test_a_malformed_task_is_refused_before_an_agent_sees_it(task, match):
    with pytest.raises(ProtocolError, match=match):
        validate_task(task)


def test_a_valid_task_serializes_the_agent_input(world):
    _manifest, _snapshot, tasks = world
    document = tasks[0].as_input()
    assert document["protocol_version"] == PROTOCOL_VERSION
    assert set(document["limits"]) == {"max_wall_seconds", "max_provider_calls", "max_tokens",
                                       "max_results"}
    # A relay capability, never a provider credential.
    assert set(document["relay"]) == {"endpoint", "capability"}


# ---- manifests -----------------------------------------------------------------------------------
def test_the_query_manifest_is_reproducible_from_source_and_seed():
    first = derive_query_manifest(source_id=QUERY_SOURCE_ID, source_version=QUERY_SOURCE_VERSION,
                                  round_seed=CALIBRATION_SEED, pool=QUERY_POOL, count=4)
    second = derive_query_manifest(source_id=QUERY_SOURCE_ID, source_version=QUERY_SOURCE_VERSION,
                                   round_seed=CALIBRATION_SEED, pool=QUERY_POOL, count=4)
    assert first.entries == second.entries
    assert first.sealed_digest() == second.sealed_digest()


def test_a_different_seed_draws_a_different_manifest():
    a = calibration_manifest(seed="round-a")
    b = calibration_manifest(seed="round-b")
    assert a.sealed_digest() != b.sealed_digest()


def test_the_commitment_reveals_no_query():
    """It travels publicly during the round; a leaked question is a pre-computed answer."""
    manifest = calibration_manifest()
    commitment = json.dumps(manifest.as_commitment())
    for _task_id, query, _search_type, _ai_mode in manifest.entries:
        assert query not in commitment
    assert commitment.count("queries_sha256") == 1
    assert manifest.as_commitment()["query_count"] == len(manifest.entries)


def test_the_commitment_still_verifies_the_sealed_queries():
    manifest = calibration_manifest()
    assert manifest.as_commitment()["queries_sha256"] == manifest.sealed_digest()
    assert manifest.as_sealed_record()["entries"][0][1] == manifest.entries[0][1]


def test_drawing_more_queries_than_the_pool_holds_is_refused():
    with pytest.raises(ManifestError, match="cannot draw"):
        derive_query_manifest(source_id="p", source_version=1, round_seed="s",
                              pool=QUERY_POOL, count=len(QUERY_POOL) + 1)


def test_ground_truth_outside_the_snapshot_is_refused():
    """Truth pointing at a document nobody can retrieve makes a perfect answer unachievable."""
    with pytest.raises(ManifestError, match="absent from the snapshot"):
        SnapshotManifest(snapshot_id="s", documents=(SnapshotDocument("doc-a", "A", "a"),),
                         relevant_by_task={"t000": ("doc-missing",)})


def test_a_duplicate_document_is_refused():
    with pytest.raises(ManifestError, match="duplicate doc_id"):
        SnapshotManifest(snapshot_id="s", documents=(SnapshotDocument("doc-a", "A", "a"),
                                                     SnapshotDocument("doc-a", "A", "a")))


def test_the_snapshot_digest_changes_with_any_byte(world):
    _manifest, snapshot, _tasks = world
    altered = SnapshotManifest(snapshot_id=snapshot.snapshot_id,
                               documents=(*snapshot.documents,
                                          SnapshotDocument("doc-extra", "Extra", "x")),
                               relevant_by_task=snapshot.relevant_by_task)
    assert altered.digest() != snapshot.digest()


def test_the_benchmark_identity_binds_every_load_bearing_pin(world):
    manifest, snapshot, _tasks = world
    base = {
        "query_commitment": manifest.as_commitment(),
        "snapshot_digest": snapshot.digest(),
        "judge_policy_id": "judge-v1",
        "model_identity": "fake-judge-0",
        "upstream_commit": "bea9712f58a5fc01c57ec441ce279499529d8bf6",
        "plugin_revision": "sn22-adapter-1",
    }
    identity = benchmark_identity(**base)
    assert benchmark_identity(**base) == identity          # stable

    for field_name, changed in (
        ("query_commitment", calibration_manifest(seed="other").as_commitment()),
        ("snapshot_digest", "0" * 64),
        ("judge_policy_id", "judge-v2"),
        ("model_identity", "fake-judge-1"),
        ("upstream_commit", "b" * 40),
        ("plugin_revision", "sn22-adapter-2"),
    ):
        assert benchmark_identity(**{**base, field_name: changed}) != identity, field_name


@pytest.mark.parametrize("missing", ["judge_policy_id", "model_identity", "upstream_commit",
                                     "plugin_revision"])
def test_an_unpinned_benchmark_identity_is_refused(world, missing):
    manifest, snapshot, _tasks = world
    base = {"query_commitment": manifest.as_commitment(), "snapshot_digest": snapshot.digest(),
            "judge_policy_id": "j", "model_identity": "m", "upstream_commit": "c",
            "plugin_revision": "p"}
    with pytest.raises(ManifestError, match=missing):
        benchmark_identity(**{**base, missing: ""})


def test_asymmetric_usage_refuses_to_decide_the_challenge():
    """Plan §5.2 item 8: a task served to one side only means the crown would be decided by which
    contestant got served."""
    usage = UsageManifest(challenge_id="c1", records=(
        UsageRecord("king", "t000", 1, 100, 0.002),
        UsageRecord("king", "t001", 1, 100, 0.002),
        UsageRecord("challenger", "t000", 1, 100, 0.002),
    ))
    with pytest.raises(ManifestError, match="asymmetric"):
        usage.assert_symmetric(("king", "challenger"))


def test_symmetric_usage_is_accepted():
    usage = UsageManifest(challenge_id="c1", records=(
        UsageRecord("king", "t000", 1, 100, 0.002),
        UsageRecord("challenger", "t000", 2, 200, 0.004),
    ))
    usage.assert_symmetric(("king", "challenger"))


# ---- the fake relay ------------------------------------------------------------------------------
def test_both_contestants_get_identical_content_for_an_identical_request(world):
    _manifest, snapshot, tasks = world
    relay = FakeRelay(snapshot=snapshot, challenge_id="c1")
    king_cap = relay.grant(variant="king", task_id=tasks[0].task_id, max_calls=4)
    challenger_cap = relay.grant(variant="challenger", task_id=tasks[0].task_id, max_calls=4)
    assert (relay.search(king_cap, tasks[0].query)
            == relay.search(challenger_cap, tasks[0].query))


def test_the_relay_enforces_its_own_call_quota(world):
    _manifest, snapshot, tasks = world
    relay = FakeRelay(snapshot=snapshot, challenge_id="c1")
    capability = relay.grant(variant="king", task_id=tasks[0].task_id, max_calls=2)
    relay.search(capability, tasks[0].query)
    relay.search(capability, tasks[0].query)
    with pytest.raises(RelayDenied, match="quota exhausted"):
        relay.search(capability, tasks[0].query)


def test_a_capability_cannot_be_reused_across_variants(world):
    """Plan §6.2: a candidate reusing another variant's capability must fail."""
    _manifest, snapshot, tasks = world
    relay = FakeRelay(snapshot=snapshot, challenge_id="c1")
    king_cap = relay.grant(variant="king", task_id=tasks[0].task_id, max_calls=4)
    forged = type(king_cap)(lane_id=king_cap.lane_id, challenge_id=king_cap.challenge_id,
                            variant="challenger", task_id=king_cap.task_id, max_calls=99)
    with pytest.raises(RelayDenied, match="unknown or forged"):
        relay.search(forged, tasks[0].query)


def test_a_capability_cannot_raise_its_own_quota(world):
    _manifest, snapshot, tasks = world
    relay = FakeRelay(snapshot=snapshot, challenge_id="c1")
    real = relay.grant(variant="king", task_id=tasks[0].task_id, max_calls=1)
    greedy = type(real)(lane_id=real.lane_id, challenge_id=real.challenge_id, variant=real.variant,
                        task_id=real.task_id, max_calls=1000)
    with pytest.raises(RelayDenied, match="unknown or forged"):
        relay.search(greedy, tasks[0].query)


def test_a_capability_stops_working_when_the_challenge_closes(world):
    """A slow agent must not keep spending after the round it was scored in has ended."""
    _manifest, snapshot, tasks = world
    relay = FakeRelay(snapshot=snapshot, challenge_id="c1")
    capability = relay.grant(variant="king", task_id=tasks[0].task_id, max_calls=4)
    relay.search(capability, tasks[0].query)
    relay.close()
    with pytest.raises(RelayDenied, match="closed"):
        relay.search(capability, tasks[0].query)


def test_an_oversized_relay_request_is_refused(world):
    _manifest, snapshot, tasks = world
    relay = FakeRelay(snapshot=snapshot, challenge_id="c1")
    capability = relay.grant(variant="king", task_id=tasks[0].task_id, max_calls=4)
    with pytest.raises(RelayDenied, match="size limit"):
        relay.search(capability, "x" * 100_000)


def test_the_relay_records_usage_the_candidate_cannot_edit(world):
    _manifest, snapshot, tasks = world
    relay = FakeRelay(snapshot=snapshot, challenge_id="c1")
    capability = relay.grant(variant="king", task_id=tasks[0].task_id, max_calls=4)
    relay.search(capability, tasks[0].query)
    relay.search(capability, tasks[0].query)
    totals = relay.usage_manifest().totals("king")
    assert totals["provider_calls"] == 2
    assert totals["spend_usd"] > 0


def test_the_relay_returns_no_credential(world):
    """Plan §6.1: the candidate environment holds a capability, never a provider key."""
    _manifest, snapshot, tasks = world
    relay = FakeRelay(snapshot=snapshot, challenge_id="c1")
    capability = relay.grant(variant="king", task_id=tasks[0].task_id, max_calls=4)
    serialized = json.dumps([r.as_dict() for r in relay.search(capability, tasks[0].query)])
    for marker in ("api_key", "API_KEY", "sk-", "Bearer", "OPENAI", "APIFY", "SCRAPINGDOG"):
        assert marker not in serialized


# ---- the ladder holds under order reversal in the RUN, not only the compare ----------------------
def test_running_the_contestants_in_either_order_yields_the_same_verdict(world):
    """Plan §5.2 item 5. The relay is shared and stateful (quotas, usage), so 'who went first' is a
    real variable, not a hypothetical one."""
    _manifest, snapshot, tasks = world

    def run(order: tuple[str, str]) -> tuple[Signals, Signals]:
        relay = FakeRelay(snapshot=snapshot, challenge_id="c1")
        scored: dict[str, Signals] = {}
        for variant in order:
            kind = "strong" if variant == "king" else "medium"
            for task in tasks:
                capability = relay.grant(variant=variant, task_id=task.task_id, max_calls=4)
                relay.search(capability, task.query)
            scored[variant] = score_attempts(_attempts(kind, tasks, snapshot), snapshot=snapshot,
                                             usage=relay.usage_manifest(), variant=variant)
        return scored["king"], scored["challenger"]

    king_first = run(("king", "challenger"))
    challenger_first = run(("challenger", "king"))
    assert king_first[0].as_metrics() == challenger_first[0].as_metrics()
    assert king_first[1].as_metrics() == challenger_first[1].as_metrics()
    assert compare_signals(*king_first) == compare_signals(*challenger_first) == 1
    assert not beats_king(king_first[1], king_first[0])


def test_a_full_paired_challenge_produces_a_bound_identity(world):
    """The end-to-end shape SN22-3 will consume: sealed world, both sides scored, one identity."""
    manifest, snapshot, tasks = world
    relay = FakeRelay(snapshot=snapshot, challenge_id="c1")
    for variant in ("king", "challenger"):
        for task in tasks:
            relay.search(relay.grant(variant=variant, task_id=task.task_id, max_calls=4),
                         task.query)
    usage = relay.usage_manifest()
    usage.assert_symmetric(("king", "challenger"))

    identity = benchmark_identity(
        query_commitment=manifest.as_commitment(), snapshot_digest=snapshot.digest(),
        judge_policy_id="judge-v1", model_identity="fake-judge-0",
        upstream_commit="bea9712f58a5fc01c57ec441ce279499529d8bf6",
        plugin_revision="sn22-adapter-1")
    assert len(identity) == 64
    king = score_attempts(_attempts("medium", tasks, snapshot), snapshot=snapshot,
                          usage=usage, variant="king")
    challenger = score_attempts(_attempts("strong", tasks, snapshot), snapshot=snapshot,
                                usage=usage, variant="challenger")
    assert beats_king(challenger, king)
    assert not beats_king(king, challenger)


def test_default_limits_are_explicit_not_silently_permissive():
    """Plan §4: defaults must not silently decide paid work."""
    limits = Limits()
    assert limits.max_provider_calls > 0
    assert limits.max_tokens > 0
    assert limits.max_wall_seconds > 0


# ---- no ranked signal may come from candidate self-report ----------------------------------------
# Each of these three pins a correction the malicious fixture forced. Written after the fact, and
# each fails against the design that preceded it -- which is why they exist rather than a comment.
def test_cost_cannot_be_scored_without_the_relays_record(world):
    """There is deliberately no code path that ranks cost on numbers the candidate chose."""
    _manifest, snapshot, tasks = world
    with pytest.raises(TypeError):
        score_attempts(_attempts("weak", tasks, snapshot), snapshot=snapshot)  # no usage


def test_latency_comes_from_the_lane_not_from_the_submission(world):
    """The malicious fixture reports elapsed_seconds=0.0. Only the lane's clock counts."""
    _manifest, snapshot, tasks = world
    attempts = _attempts("malicious", tasks, snapshot)
    assert all(a.output.usage.elapsed_seconds == 0.0 for a in attempts if a.output)
    scored = score_attempts(attempts, snapshot=snapshot, usage=_billed(tasks), variant="candidate")
    assert scored.sn22_latency_seconds == OBSERVED_SECONDS["malicious"] * len(tasks)


def test_a_citation_must_name_a_document_the_agent_actually_returned(world):
    """The cheapest attack: the relevant ids are guessable from the query, so citing without
    retrieving would otherwise buy full precision for free."""
    _manifest, snapshot, tasks = world
    malicious = _attempts("malicious", tasks, snapshot)
    cited = {c.doc_id for a in malicious if a.output for c in a.output.citations}
    returned = {r.doc_id for a in malicious if a.output for r in a.output.results}
    truth = set().union(*(snapshot.relevant(t.task_id) for t in tasks))
    assert cited & truth, "the fixture must cite genuinely relevant ids to be a real attack"
    assert not (cited & truth & returned), "and must not have retrieved them"
    assert _signals("malicious", tasks, snapshot).sn22_citation_precision == 0.0


def test_citing_nothing_beats_citing_falsely(world):
    """Precision of an empty claim set is 1.0, not 0.0: an honest agent that made no claims has
    none that failed, and must not rank below a liar whose fabrications included a few real ids."""
    _manifest, snapshot, tasks = world
    weak = _signals("weak", tasks, snapshot)
    malicious = _signals("malicious", tasks, snapshot)
    assert weak.detail["citations_made"] == 0
    assert weak.sn22_citation_precision == 1.0
    assert malicious.sn22_citation_precision == 0.0


def test_silence_still_loses_to_a_real_answer(world):
    """The 1.0 default is only safe because quality (priority 2) outranks precision (priority 3)."""
    _manifest, snapshot, tasks = world
    silent = _signals("weak", tasks, snapshot)         # cites nothing, finds nothing
    strong = _signals("strong", tasks, snapshot)
    assert silent.sn22_citation_precision == strong.sn22_citation_precision == 1.0
    assert compare_signals(strong, silent) == 1        # decided at quality, before precision


def test_the_ladder_is_decided_by_quality_not_by_a_tiebreak(world):
    """Guards the ladder's MEANING: weak/medium/strong must separate on the signal the fixtures were
    written to differ on, not incidentally on latency or cost."""
    _manifest, snapshot, tasks = world
    weak = _signals("weak", tasks, snapshot)
    medium = _signals("medium", tasks, snapshot)
    strong = _signals("strong", tasks, snapshot)
    assert weak.sn22_weighted_quality < medium.sn22_weighted_quality < strong.sn22_weighted_quality
    assert weak.sn22_cost_units == medium.sn22_cost_units == strong.sn22_cost_units
