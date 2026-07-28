"""SN22-2: the frozen protocol, the manifests, and the reference ladder.

The exit gate has two halves and both are pinned here:

  "The protocol and fixtures are reviewable **without network access** and deterministically order
   **weak < medium < strong in both comparator directions**."

The second half is the subtle one. "Both directions" is not a restatement of antisymmetry for its
own sake — it is the fairness property from plan §5.2 item 5, that the order contestants are run and
compared in must not decide the crown. Every ladder test below therefore checks the reversal too.

**Offline, but no longer sealed.** The lane now scores LIVE sources: the validator fetches each page
itself and an LLM judges what the miner proved it read (:mod:`kata_sn22.verification`). None of that
happens here — the fixtures supply recorded pages and a scripted judge through the plugin's seams,
so this suite still runs with sockets removed. What is being tested is the SCORING, which is the
same code either way; what a real fetch and a real judge return is not this file's question.
"""
from __future__ import annotations

import json
import socket

import pytest

from kata_sn22.fixtures import (
    CALIBRATION_SEED,
    QUERY_POOL,
    QUERY_SOURCE_ID,
    QUERY_SOURCE_VERSION,
    calibration_manifest,
    recorded_pages,
    recorded_tweets,
    reference_responses,
    scripted_judge,
    tasks_for,
)
from kata_sn22.manifests import (
    ManifestError,
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
    return manifest, tasks_for(manifest)


def _plugin():
    """A plugin wired to the recorded world. Its verification path is the production one -- only
    where the pages and verdicts come from differs."""
    from kata_sn22.fetch import RecordedPages
    from kata_sn22.plugin import Sn22DesearchPlugin

    recorded = recorded_tweets()
    return Sn22DesearchPlugin(
        page_transport=RecordedPages(records=recorded_pages()),
        judge_client=scripted_judge(),
        tweet_scraper=lambda ids: {tid: recorded[tid] for tid in ids if tid in recorded})


#: How long the LANE observed each reference submission to take. Fixed per reference so the ladder
#: stays deterministic, and supplied by the harness rather than read from the agent -- latency is a
#: ranked signal, so taking it from the submission would let a candidate win by claiming 0.0.
OBSERVED_SECONDS = {"weak": 3.0, "medium": 2.0, "strong": 1.0, "invalid": 5.0, "malicious": 4.0}


def _attempts(kind: str, tasks) -> list[TaskAttempt]:
    """Parse a reference submission into attempts, classifying whatever fails the contract."""
    observed = OBSERVED_SECONDS[kind]
    attempts = []
    for task, raw in zip(tasks, reference_responses(kind, tasks), strict=True):
        try:
            attempts.append(TaskAttempt(task=task, output=parse_task_output(raw, task=task),
                                        observed_seconds=observed))
        except ProtocolError as exc:
            attempts.append(TaskAttempt(task=task, error=exc.error_class,
                                        observed_seconds=observed))
    return attempts


def _verified(kind: str, tasks) -> list[TaskAttempt]:
    """Attempts with the VALIDATOR's own findings attached -- fetched pages, evidence checked,
    verdicts judged. Scoring reads only this, never the raw claims."""
    plugin = _plugin()
    return [plugin._verified(attempt) for attempt in _attempts(kind, tasks)]


def _billed(tasks, variant: str = "candidate", *, calls: int = 1,
            tokens: int = 250) -> UsageManifest:
    """The relay's record. Identical for every reference unless a test varies it, so the ladder is
    decided by answer quality rather than by who happened to be billed more."""
    return UsageManifest(challenge_id="c1", records=tuple(
        UsageRecord(variant=variant, task_id=t.task_id, provider_calls=calls, tokens=tokens,
                    spend_usd=0.002 * calls) for t in tasks))


def _signals(kind: str, tasks) -> Signals:
    return score_attempts(_verified(kind, tasks), usage=_billed(tasks), variant="candidate")


# ---- THE EXIT GATE -------------------------------------------------------------------------------
def test_the_ladder_orders_weak_medium_strong(world):
    manifest, tasks = world
    weak = _signals("weak", tasks)
    medium = _signals("medium", tasks)
    strong = _signals("strong", tasks)

    assert compare_signals(medium, weak) == 1
    assert compare_signals(strong, medium) == 1
    assert compare_signals(strong, weak) == 1


def test_the_ladder_orders_identically_in_the_REVERSE_direction(world):
    """The fairness property: which contestant is passed first must not decide the crown."""
    _manifest, tasks = world
    weak = _signals("weak", tasks)
    medium = _signals("medium", tasks)
    strong = _signals("strong", tasks)

    assert compare_signals(weak, medium) == -1
    assert compare_signals(medium, strong) == -1
    assert compare_signals(weak, strong) == -1


def test_the_comparator_is_antisymmetric_for_every_pair(world):
    _manifest, tasks = world
    ladder = {k: _signals(k, tasks) for k in ("weak", "medium", "strong", "malicious")}
    for left in ladder.values():
        for right in ladder.values():
            assert compare_signals(left, right) == -compare_signals(right, left)


def test_the_ladder_is_deterministic_across_repeated_scoring(world):
    """A fixture that drifts cannot serve as a calibration baseline."""
    _manifest, tasks = world
    for kind in ("weak", "medium", "strong"):
        runs = [_signals(kind, tasks).as_metrics() for _ in range(5)]
        assert all(run == runs[0] for run in runs)


def test_the_ladder_is_identical_under_a_rebuilt_world(world):
    """Same seed, rebuilt from source: same queries, same scores."""
    _manifest, tasks = world
    rebuilt_manifest = calibration_manifest()
    rebuilt_tasks = tasks_for(rebuilt_manifest)
    assert [t.task_id for t in rebuilt_tasks] == [t.task_id for t in tasks]
    for kind in ("weak", "medium", "strong"):
        assert (_signals(kind, rebuilt_tasks).as_metrics()
                == _signals(kind, tasks).as_metrics())


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
    _manifest, tasks = world
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
    _manifest, tasks = world
    strong = _signals("strong", tasks)
    assert compare_signals(strong, strong) == 0
    assert not beats_king(strong, strong)


def test_an_empty_throne_still_requires_a_valid_run(world):
    _manifest, tasks = world
    assert beats_king(_signals("weak", tasks), None)
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
    _manifest, tasks = world
    attempts = _attempts("invalid", tasks)
    assert all(a.error is ErrorClass.INVALID_SCHEMA for a in attempts)
    signals = score_attempts(attempts, usage=_billed(tasks),
                             variant="candidate")
    assert signals.sn22_valid_query_rate == 0.0
    assert signals.sn22_invalid_runs == len(tasks)


def test_an_invalid_submission_never_outranks_a_valid_one(world):
    _manifest, tasks = world
    invalid = _signals("invalid", tasks)
    weak = _signals("weak", tasks)
    assert compare_signals(weak, invalid) == 1
    assert compare_signals(invalid, weak) == -1


def test_citing_without_retrieving_earns_nothing(world):
    """The malicious fixture cites every relevant document while returning none of them."""
    _manifest, tasks = world
    malicious = _signals("malicious", tasks)
    assert malicious.sn22_weighted_quality == 0.0
    assert malicious.sn22_coverage == 0.0


def test_a_citation_to_a_source_that_was_never_returned_is_not_supported(world):
    """The malicious reference cites a URL it never put in its results. With no corpus to check ids
    against, THIS is the fabrication that matters: a plausible URL costs nothing to invent."""
    _manifest, tasks = world
    malicious = _signals("malicious", tasks)
    assert malicious.sn22_citation_precision < 1.0


def test_a_malicious_submission_does_not_outrank_an_honest_weak_one(world):
    """The whole point: lying must not pay better than searching badly."""
    _manifest, tasks = world
    malicious = _signals("malicious", tasks)
    weak = _signals("weak", tasks)
    assert compare_signals(weak, malicious) == 1
    assert compare_signals(malicious, weak) == -1
    assert compare_signals(_signals("strong", tasks), malicious) == 1


def test_a_prompt_injection_is_carried_as_inert_data(world):
    """Retrieved and submitted text is data. It is stored and compared, never interpreted."""
    _manifest, tasks = world
    attempts = _attempts("malicious", tasks)
    summaries = [a.output.summary for a in attempts if a.output is not None]
    assert any("IGNORE ALL PREVIOUS INSTRUCTIONS" in s for s in summaries)
    # It reached the scorer verbatim and changed nothing.
    assert _signals("malicious", tasks).sn22_weighted_quality == 0.0


def test_under_reported_usage_is_overridden_by_the_relay(world):
    """The malicious fixture claims zero cost. The relay billed it, and the relay wins."""
    _manifest, tasks = world
    attempts = _attempts("malicious", tasks)
    billed = _billed(tasks, "challenger", calls=3, tokens=900)
    scored = score_attempts(attempts, usage=billed, variant="challenger")

    # The agent claimed nothing; the relay billed it. The ranked signal follows the relay.
    assert scored.detail["self_reported_calls"] == 0
    assert scored.detail["relay_billed_calls"] == 3 * len(tasks)
    assert scored.sn22_cost_units > 0.0
    assert scored.detail["usage_source"] == "relay"


# ---- the protocol itself -------------------------------------------------------------------------
def test_a_response_for_another_task_is_refused(world):
    _manifest, tasks = world
    payload = json.dumps({"protocol_version": PROTOCOL_VERSION, "task_id": "somewhere-else",
                          "summary": "", "results": [], "citations": [], "usage": {}})
    with pytest.raises(ProtocolError) as caught:
        parse_task_output(payload, task=tasks[0])
    assert caught.value.error_class is ErrorClass.INVALID_SCHEMA


def test_an_unknown_protocol_version_is_refused_not_interpreted(world):
    _manifest, tasks = world
    payload = json.dumps({"protocol_version": 99, "task_id": tasks[0].task_id, "summary": "",
                          "results": [], "citations": [], "usage": {}})
    with pytest.raises(ProtocolError, match="protocol_version"):
        parse_task_output(payload, task=tasks[0])


def test_an_oversized_response_is_refused_before_parsing(world):
    _manifest, tasks = world
    with pytest.raises(ProtocolError) as caught:
        parse_task_output(b"x" * (MAX_OUTPUT_BYTES + 1), task=tasks[0])
    assert caught.value.error_class is ErrorClass.EXCESS_OUTPUT


def test_too_many_results_is_refused(world):
    _manifest, tasks = world
    task = tasks[0]
    payload = json.dumps({
        "protocol_version": PROTOCOL_VERSION, "task_id": task.task_id, "summary": "",
        "results": [{"doc_id": f"doc-{i}", "title": "", "snippet": ""}
                    for i in range(task.limits.max_results + 1)],
        "citations": [], "usage": {}})
    with pytest.raises(ProtocolError) as caught:
        parse_task_output(payload, task=task)
    assert caught.value.error_class is ErrorClass.EXCESS_OUTPUT


def test_a_repeated_source_is_refused(world):
    """Ten copies of one page is one page; counting it ten times would inflate coverage."""
    _manifest, tasks = world
    payload = json.dumps({
        "protocol_version": PROTOCOL_VERSION, "task_id": tasks[0].task_id, "summary": "",
        "results": [{"link": "https://a.test/x", "title": "", "snippet": ""}] * 2,
        "citations": [], "usage": {}})
    with pytest.raises(ProtocolError, match="repeats"):
        parse_task_output(payload, task=tasks[0])


def test_a_tracking_parameter_cannot_disguise_a_repeat(world):
    """Sameness is upstream's ``source_key``, so appending ``?utm_source=`` does not buy a second
    slot for the same page."""
    _manifest, tasks = world
    payload = json.dumps({
        "protocol_version": PROTOCOL_VERSION, "task_id": tasks[0].task_id, "summary": "",
        "results": [{"link": "https://a.test/x", "title": "", "snippet": ""},
                    {"link": "https://www.a.test/x?utm_source=q", "title": "", "snippet": ""}],
        "citations": [], "usage": {}})
    with pytest.raises(ProtocolError, match="repeats"):
        parse_task_output(payload, task=tasks[0])


def test_a_source_the_validator_cannot_fetch_is_refused(world):
    """A scheme the validator cannot fetch is a source it cannot verify, and an unverifiable source
    must never reach the judge."""
    _manifest, tasks = world
    for link in ("ftp://a.test/x", "file:///etc/passwd", "javascript:alert(1)", "not a url"):
        payload = json.dumps({
            "protocol_version": PROTOCOL_VERSION, "task_id": tasks[0].task_id, "summary": "",
            "results": [{"link": link, "title": "", "snippet": ""}],
            "citations": [], "usage": {}})
        with pytest.raises(ProtocolError, match="http"):
            parse_task_output(payload, task=tasks[0])


@pytest.mark.parametrize("usage", [
    {"provider_calls": -1},
    {"tokens": "many"},
    {"elapsed_seconds": float("nan")},
    {"elapsed_seconds": float("inf")},
    {"provider_calls": True},
])
def test_unusable_usage_numbers_are_refused(world, usage):
    _manifest, tasks = world
    payload = json.dumps({"protocol_version": PROTOCOL_VERSION, "task_id": tasks[0].task_id,
                          "summary": "", "results": [], "citations": [], "usage": usage})
    with pytest.raises(ProtocolError):
        parse_task_output(payload, task=tasks[0])


def test_a_provider_outage_is_not_the_candidates_fault(world):
    """A shared fault must not count against whichever contestant was running at the time."""
    _manifest, tasks = world
    assert not ErrorClass.PROVIDER_UNAVAILABLE.candidate_caused
    assert ErrorClass.TIMEOUT.candidate_caused

    attempts = _attempts("strong", tasks)
    degraded = [TaskAttempt(task=attempts[0].task, error=ErrorClass.PROVIDER_UNAVAILABLE),
                *attempts[1:]]
    signals = score_attempts(degraded, usage=_billed(tasks), variant="candidate")
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
    _manifest, tasks = world
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


def test_the_benchmark_identity_binds_every_load_bearing_pin(world):
    manifest, _tasks = world
    base = {
        "query_commitment": manifest.as_commitment(),
        "judge_policy_id": "judge-v1",
        "model_identity": "fake-judge-0",
        "upstream_commit": "bea9712f58a5fc01c57ec441ce279499529d8bf6",
        "plugin_revision": "sn22-adapter-1",
    }
    identity = benchmark_identity(**base)
    assert benchmark_identity(**base) == identity          # stable

    for field_name, changed in (
        ("query_commitment", calibration_manifest(seed="other").as_commitment()),
        ("judge_policy_id", "judge-v2"),
        ("model_identity", "fake-judge-1"),
        ("upstream_commit", "b" * 40),
        ("plugin_revision", "sn22-adapter-2"),
    ):
        assert benchmark_identity(**{**base, field_name: changed}) != identity, field_name


@pytest.mark.parametrize("missing", ["judge_policy_id", "model_identity", "upstream_commit",
                                     "plugin_revision"])
def test_an_unpinned_benchmark_identity_is_refused(world, missing):
    manifest, _tasks = world
    base = {"query_commitment": manifest.as_commitment(),
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


# ---- the relay ----------------------------------------------------------------------------------
# The FakeRelay these tests used to drive is gone: it served the sealed corpus, and there is no
# corpus. Its subject -- capability scoping, quota, reservation, redaction, closure -- moved intact
# to the real gateway, and is tested against that in `tests/test_sn22_gateway.py`. The properties
# were not dropped with the fixture; they were tested against a stand-in and are now tested against
# the thing that actually runs.


# ---- the ladder holds under order reversal in the RUN, not only the compare ----------------------
def test_running_the_contestants_in_either_order_yields_the_same_verdict(world):
    """Plan §5.2 item 5. Verification is shared and stateful -- the page fetcher CACHES, so 'who
    went first' decides who triggered the fetch. That must not decide the crown."""
    _manifest, tasks = world

    def run(order: tuple[str, str]) -> tuple[Signals, Signals]:
        plugin = _plugin()          # ONE plugin across both contestants, so they share its cache
        scored: dict[str, Signals] = {}
        for variant in order:
            kind = "strong" if variant == "king" else "medium"
            attempts = [plugin._verified(a) for a in _attempts(kind, tasks)]
            scored[variant] = score_attempts(
                attempts, usage=_billed(tasks, variant), variant=variant)
        return scored["king"], scored["challenger"]

    king_first = run(("king", "challenger"))
    challenger_first = run(("challenger", "king"))
    assert king_first[0].as_metrics() == challenger_first[0].as_metrics()
    assert king_first[1].as_metrics() == challenger_first[1].as_metrics()
    assert compare_signals(*king_first) == compare_signals(*challenger_first) == 1
    assert not beats_king(king_first[1], king_first[0])


def test_a_full_paired_challenge_produces_a_bound_identity(world):
    """The end-to-end shape SN22-3 consumes: both sides verified and scored, one bound identity."""
    manifest, tasks = world
    from kata_sn22.manifests import UsageRecord as _Record

    usage = UsageManifest(challenge_id="c1", records=tuple(
        _Record(variant=variant, task_id=task.task_id, provider_calls=1, tokens=250,
                spend_usd=0.002)
        for variant in ("king", "challenger") for task in tasks))
    usage.assert_symmetric(("king", "challenger"))

    identity = benchmark_identity(
        query_commitment=manifest.as_commitment(),
        judge_policy_id="judge-v1", model_identity="fake-judge-0",
        upstream_commit="bea9712f58a5fc01c57ec441ce279499529d8bf6",
        plugin_revision="sn22-adapter-1")
    assert len(identity) == 64
    plugin = _plugin()
    king = score_attempts([plugin._verified(a) for a in _attempts("medium", tasks)],
                          usage=usage, variant="king")
    challenger = score_attempts([plugin._verified(a) for a in _attempts("strong", tasks)],
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
    _manifest, tasks = world
    with pytest.raises(TypeError):
        score_attempts(_attempts("weak", tasks))  # no usage


def test_latency_comes_from_the_lane_not_from_the_submission(world):
    """The malicious fixture reports elapsed_seconds=0.0. Only the lane's clock counts."""
    _manifest, tasks = world
    attempts = _attempts("malicious", tasks)
    assert all(a.output.usage.elapsed_seconds == 0.0 for a in attempts if a.output)
    scored = score_attempts(attempts, usage=_billed(tasks), variant="candidate")
    assert scored.sn22_latency_seconds == OBSERVED_SECONDS["malicious"] * len(tasks)


def test_a_citation_must_name_a_source_the_agent_actually_RETURNED_and_PROVED(world):
    """Two failures the malicious fixture commits at once, and either alone must cost it precision:

    * it cites a URL that is not in its own results -- inventing a plausible URL costs nothing;
    * the source it DID return carries excerpts that are not on the page, so it never became
      evidence at all.
    """
    _manifest, tasks = world
    malicious = _attempts("malicious", tasks)
    ai = [a for a in malicious if a.output and a.task.search_type == "ai_search"]
    assert ai, "the fixture must exercise the AI-search path"
    cited = {c.link for a in ai for c in a.output.citations}
    returned = {r.link for a in ai for r in a.output.results}
    assert cited - returned, "the fixture must cite at least one source it never returned"
    assert _signals("malicious", tasks).sn22_citation_precision == 0.0


def test_citing_nothing_beats_citing_falsely(world):
    """Precision of an empty claim set is 1.0, not 0.0: an honest agent that made no claims has
    none that failed, and must not rank below a liar whose fabrications included a few real ids."""
    _manifest, tasks = world
    weak = _signals("weak", tasks)
    malicious = _signals("malicious", tasks)
    assert weak.detail["citations_made"] == 0
    assert weak.sn22_citation_precision == 1.0
    assert malicious.sn22_citation_precision == 0.0


def test_silence_still_loses_to_a_real_answer(world):
    """The 1.0 default is only safe because quality (priority 2) outranks precision (priority 3)."""
    _manifest, tasks = world
    silent = _signals("weak", tasks)         # cites nothing, finds nothing
    strong = _signals("strong", tasks)
    assert silent.sn22_citation_precision == strong.sn22_citation_precision == 1.0
    assert compare_signals(strong, silent) == 1        # decided at quality, before precision


def test_the_ladder_is_decided_by_quality_not_by_a_tiebreak(world):
    """Guards the ladder's MEANING: weak/medium/strong must separate on the signal the fixtures were
    written to differ on, not incidentally on latency or cost."""
    _manifest, tasks = world
    weak = _signals("weak", tasks)
    medium = _signals("medium", tasks)
    strong = _signals("strong", tasks)
    assert weak.sn22_weighted_quality < medium.sn22_weighted_quality < strong.sn22_weighted_quality
    assert weak.sn22_cost_units == medium.sn22_cost_units == strong.sn22_cost_units
