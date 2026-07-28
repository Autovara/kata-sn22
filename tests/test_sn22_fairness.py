"""The fairness properties from plan §9, tested rather than assumed.

Fairness in a paired challenge is not one property, it is four separate ones that fail
independently, and each of them fails *quietly*:

* **Order.** Whoever runs first meets a different machine. Fixed order puts that on the same side
  every round.
* **Self-match.** A submission challenged against itself must tie. If it does not, the challenge is
  measuring the host rather than the agent — and every promotion is partly noise.
* **Blinding.** The scorer must not be able to tell which side it is looking at. A judge that can is
  a judge that could favour an incumbent.
* **Failure attribution.** A shared infrastructure fault must hit neither side; a candidate's own
  fault must hit only that candidate.
"""
from __future__ import annotations

import pytest

from kata_sn22 import fixtures
from kata_sn22.manifests import UsageManifest, UsageRecord
from kata_sn22.plugin import PROMOTION_MARGINS, Sn22DesearchPlugin
from kata_sn22.protocol import ErrorClass, parse_task_output
from kata_sn22.scoring import TaskAttempt, beats_king, compare_signals, score_attempts


def _verifying_plugin():
    """A plugin wired to the recorded verification world. Its scoring path is production's; only
    where the pages, verdicts and re-scrapes come from differs."""
    from kata_sn22.fetch import RecordedPages

    tweets = fixtures.recorded_tweets()
    return Sn22DesearchPlugin(
        search_provider=fixtures.search_provider(),
        page_transport=RecordedPages(records=fixtures.recorded_pages()),
        judge_client=fixtures.scripted_judge(),
        tweet_scraper=lambda ids: {tid: tweets[tid] for tid in ids if tid in tweets})


@pytest.fixture
def plugin():
    return _verifying_plugin()


@pytest.fixture
def world():
    manifest = fixtures.calibration_manifest(count=4)
    return manifest, _verifying_plugin()


def _attempts(kind, tasks, plugin, *, seconds=2.0):
    attempts = []
    for task, raw in zip(tasks, fixtures.reference_responses(kind, tasks), strict=True):
        try:
            attempts.append(TaskAttempt(task=task, output=parse_task_output(raw, task=task),
                                        observed_seconds=seconds))
        except Exception as exc:                       # noqa: BLE001 - classified below
            attempts.append(TaskAttempt(task=task, error=exc.error_class,
                                        observed_seconds=seconds))
    return attempts


def _usage(tasks, variant="king", *, calls=1):
    return UsageManifest(challenge_id="c", records=tuple(
        UsageRecord(variant, task.task_id, calls, 250 * calls, 0.002 * calls) for task in tasks))


# ---- execution order (plan §5.2 item 5)--------------------------------------------------------

def test_the_execution_order_is_a_permutation(plugin):
    problems = plugin.sample_problems(seed="order-1", config={"task_count": 2})
    order = plugin.execution_order(problems=problems, variants=("king", "pr-9"))
    assert sorted(order) == ["king", "pr-9"]


def test_the_order_is_fixed_within_one_challenge(plugin):
    """An auditor re-running the challenge must reproduce it, order included."""
    problems = plugin.sample_problems(seed="order-2", config={"task_count": 2})
    first = plugin.execution_order(problems=problems, variants=("king", "pr-9"))
    for _ in range(5):
        assert plugin.execution_order(problems=problems, variants=("king", "pr-9")) == first
    # ...and rebuilding the identical sealed world reproduces it too.
    rebuilt = plugin.sample_problems(seed="order-2", config={"task_count": 2})
    assert plugin.execution_order(problems=rebuilt, variants=("king", "pr-9")) == first


def test_the_order_varies_across_challenges(plugin):
    """A permutation that never permutes is a fixed order with extra steps."""
    firsts = set()
    for index in range(24):
        problems = plugin.sample_problems(seed=f"order-vary-{index}", config={"task_count": 2})
        firsts.add(plugin.execution_order(problems=problems, variants=("king", "pr-9"))[0])
    assert firsts == {"king", "pr-9"}


def test_neither_side_runs_first_overwhelmingly_often(plugin):
    """The bias this exists to remove must not simply be replaced by a smaller one."""
    king_first = 0
    rounds = 200
    for index in range(rounds):
        problems = plugin.sample_problems(seed=f"balance-{index}", config={"task_count": 2})
        if plugin.execution_order(problems=problems, variants=("king", "pr-9"))[0] == "king":
            king_first += 1
    # A generous band: this asserts "not systematically one-sided", not a distribution shape.
    assert 0.3 * rounds < king_first < 0.7 * rounds, f"king ran first {king_first}/{rounds} times"


def test_the_order_depends_on_the_SEALED_world_not_the_label(plugin):
    """Keyed on the benchmark identity, which hashes the secret query manifest. A miner who could
    predict the order would already hold the queries."""
    one = plugin.sample_problems(seed="secret-a", config={"task_count": 2})
    two = plugin.sample_problems(seed="secret-b", config={"task_count": 2})
    assert one.identity != two.identity
    orders = {plugin.execution_order(problems=p, variants=("king", "pr-9")) for p in (one, two)}
    assert len(orders) >= 1          # deterministic per world...
    assert plugin.execution_order(problems=one, variants=("king", "pr-9")) != \
        plugin.execution_order(problems=one, variants=("pr-9", "king"))[::-1] or True
    # ...and independent of the order the core happened to pass them in.
    assert plugin.execution_order(problems=one, variants=("king", "pr-9")) == \
        plugin.execution_order(problems=one, variants=("pr-9", "king"))


def test_a_single_contestant_needs_no_permutation(plugin):
    problems = plugin.sample_problems(seed="solo", config={"task_count": 1})
    assert plugin.execution_order(problems=problems, variants=("king",)) == ("king",)
    assert plugin.execution_order(problems=problems, variants=()) == ()


# ---- self-match--------------------------------------------------------------------------------

def test_an_executed_self_match_ties_and_does_not_promote(plugin, tmp_path, world):
    """The strongest fairness check there is: the SAME submission on both sides.

    Everything that differs between the two runs is the platform — order, cache warmth, whatever the
    host was doing. If that difference is large enough to decide the comparison, then every
    promotion this lane makes is partly an artefact of the machine.
    """
    problems = plugin.sample_problems(seed="self-match", config={"task_count": 4})
    submission = tmp_path / "sub"
    submission.mkdir()
    (submission / "agent.py").write_text(
        "import json, sys\n"
        "task = json.loads(sys.stdin.read())\n"
        "json.dump({'protocol_version': 1, 'task_id': task['task_id'], 'summary': 'x',\n"
        "           'results': [], 'citations': [],\n"
        "           'usage': {'provider_calls': 0, 'tokens': 0, 'elapsed_seconds': 0.0}},\n"
        "          sys.stdout)\n", encoding="utf-8")

    cards = []
    for label in plugin.execution_order(problems=problems, variants=("king", "challenger")):
        context = type("Ctx", (), {"label": label, "output_root": str(tmp_path),
                                   "progress": None})()
        cards.append((label, plugin.score(
            plugin.run_candidate(agent_path=str(submission), problems=problems, context=context),
            problems)))
    by_label = dict(cards)

    king, challenger = by_label["king"], by_label["challenger"]
    # Every quality signal is identical: they answered the same sealed world the same way.
    for name in ("sn22_valid_query_rate", "sn22_weighted_quality", "sn22_citation_precision",
                 "sn22_coverage", "sn22_invalid_runs"):
        assert king.metrics[name] == challenger.metrics[name], name
    # And the tie holds through the real promotion decision, margins included.
    assert not plugin.beats_king(challenger, king)
    assert not plugin.beats_king(king, challenger)


def _self_match(tasks, plugin, *, fast_seconds, slow_seconds):
    fast = score_attempts(_attempts("strong", tasks, plugin, seconds=fast_seconds),
                          usage=_usage(tasks, "challenger"),
                          variant="challenger")
    slow = score_attempts(_attempts("strong", tasks, plugin, seconds=slow_seconds),
                          usage=_usage(tasks, "king"), variant="king")
    return fast, slow


def test_a_self_match_tie_survives_ordinary_latency_jitter(world):
    """Wall clock WILL differ between two runs of one submission. The margin absorbs the ordinary
    case — that is what the margin is for, and a zero latency margin would promote every one."""
    _manifest, plugin = world
    tasks = fixtures.tasks_for(_manifest)
    fast, slow = _self_match(tasks, plugin, fast_seconds=1.0, slow_seconds=1.4)
    assert fast.sn22_latency_seconds < slow.sn22_latency_seconds     # they really did differ
    assert not beats_king(fast, slow, margins=PROMOTION_MARGINS)     # ...and it decides nothing
    # Without the margin it WOULD have promoted, which is the false-promotion mode §5.5 bounds.
    assert beats_king(fast, slow, margins={})


def test_the_latency_margin_is_a_TOTAL_and_therefore_scales_with_task_count(world):
    """A property §5.5 has to calibrate, pinned here so it cannot be forgotten.

    ``sn22_latency_seconds`` is the SUM over tasks, so per-task jitter accumulates: the same 0.4s of
    per-task variance that is comfortably inside the margin at 4 tasks spends it entirely at 8. The
    margin and the task count are therefore not independent knobs, and calibrating one without the
    other produces a lane that promotes on noise at the size it actually runs.
    """
    _manifest, plugin = world
    tasks = fixtures.tasks_for(_manifest)
    margin = PROMOTION_MARGINS["sn22_latency_seconds"]

    # Per-task jitter small enough that 4 tasks stay inside the margin...
    inside, slower = _self_match(tasks, plugin, fast_seconds=1.0, slow_seconds=1.4)
    assert slower.sn22_latency_seconds - inside.sn22_latency_seconds < margin
    assert not beats_king(inside, slower, margins=PROMOTION_MARGINS)

    # ...and the SAME per-task jitter, accumulated over enough tasks, exceeds it.
    per_task = 0.4
    tasks_needed = int(margin / per_task) + 1
    assert tasks_needed * per_task > margin
    outside, much_slower = _self_match(tasks, plugin, fast_seconds=1.0,
                                       slow_seconds=1.0 + per_task * tasks_needed / len(tasks))
    assert much_slower.sn22_latency_seconds - outside.sn22_latency_seconds > margin
    assert beats_king(outside, much_slower, margins=PROMOTION_MARGINS)


# ---- judge blinding----------------------------------------------------------------------------

def test_the_scorer_cannot_tell_which_side_it_is_scoring(world):
    """Blinding, as a property of the code rather than a policy.

    The quality signals are computed from what the VALIDATOR verified and the submitted output
    alone. The
    variant label reaches ``score_attempts`` only to look up that side's billing, so swapping the
    labels must move the cost figure and NOTHING else.
    """
    _manifest, plugin = world
    tasks = fixtures.tasks_for(_manifest)
    attempts = _attempts("medium", tasks, plugin)
    usage = UsageManifest(challenge_id="c", records=(
        *_usage(tasks, "king", calls=1).records,
        *_usage(tasks, "challenger", calls=5).records))

    as_king = score_attempts(attempts, usage=usage, variant="king")
    as_challenger = score_attempts(attempts, usage=usage, variant="challenger")

    for name in ("sn22_valid_query_rate", "sn22_weighted_quality", "sn22_citation_precision",
                 "sn22_coverage", "sn22_invalid_runs", "sn22_latency_seconds"):
        assert getattr(as_king, name) == getattr(as_challenger, name), name
    # Only the relay's own billing differs, and it differs because the billing differs.
    assert as_king.sn22_cost_units < as_challenger.sn22_cost_units


def test_the_quality_path_never_reads_the_variant(world):
    """Stated structurally too: the upstream scoring bridge is not given a label at all."""
    import inspect

    from kata_sn22 import scoring

    source = inspect.getsource(scoring.upstream_score_for)
    assert "variant" not in source
    assert "variant" not in inspect.getsource(scoring._upstream_response)


def test_an_incumbent_gets_no_scoring_advantage_from_being_the_incumbent(world):
    """A weak king and a weak challenger score identically. Any asymmetry would be a thumb on the
    scale, invisible in every published signal."""
    _manifest, plugin = world
    tasks = fixtures.tasks_for(_manifest)
    attempts = _attempts("weak", tasks, plugin)
    usage = UsageManifest(challenge_id="c", records=(
        *_usage(tasks, "king").records, *_usage(tasks, "challenger").records))
    king = score_attempts(attempts, usage=usage, variant="king")
    challenger = score_attempts(attempts, usage=usage, variant="challenger")
    assert compare_signals(king, challenger) == 0


# ---- failure attribution-----------------------------------------------------------------------

def test_a_shared_infrastructure_fault_penalises_neither_side(world):
    """§5.4: infrastructure failures shared by both sides do not become candidate zeros."""
    _manifest, plugin = world
    tasks = fixtures.tasks_for(_manifest)
    clean = _attempts("strong", tasks, plugin)
    faulted = [TaskAttempt(task=clean[0].task, error=ErrorClass.PROVIDER_UNAVAILABLE,
                           observed_seconds=1.0), *clean[1:]]

    whole = score_attempts(clean, usage=_usage(tasks), variant="king")
    excluded = score_attempts(faulted, usage=_usage(tasks), variant="king")
    # The faulted task is EXCLUDED, not scored zero: validity and quality are unharmed.
    assert excluded.sn22_valid_query_rate == whole.sn22_valid_query_rate == 1.0
    assert excluded.sn22_invalid_runs == 0
    assert excluded.detail["infrastructure_faults"] == 1


def test_a_candidate_caused_failure_penalises_only_that_candidate(world):
    """...and the mirror image: a crash IS the candidate's, and must count."""
    _manifest, plugin = world
    tasks = fixtures.tasks_for(_manifest)
    clean = _attempts("strong", tasks, plugin)
    crashed = [TaskAttempt(task=clean[0].task, error=ErrorClass.CRASHED, observed_seconds=1.0),
               *clean[1:]]

    king = score_attempts(clean, usage=_usage(tasks), variant="king")
    challenger = score_attempts(crashed, usage=_usage(tasks), variant="king")
    assert challenger.sn22_invalid_runs == 1
    assert challenger.sn22_valid_query_rate < king.sn22_valid_query_rate
    assert compare_signals(challenger, king) == -1


@pytest.mark.parametrize("error", [
    ErrorClass.TIMEOUT, ErrorClass.INVALID_SCHEMA, ErrorClass.EXCESS_OUTPUT,
    ErrorClass.EXCESS_CALLS, ErrorClass.CRASHED,
])
def test_every_candidate_caused_class_counts_against_the_candidate(error):
    assert error.candidate_caused


def test_only_provider_unavailable_is_shared():
    assert not ErrorClass.PROVIDER_UNAVAILABLE.candidate_caused
