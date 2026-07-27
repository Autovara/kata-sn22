"""SN22-3: the plugin against the current ``kata.plugins`` ABI.

The exit gate is "unit tests, ruff, clean-venv entry-point discovery, ABI conformance, generic
challenge, result serialization, and both promotion directions pass". Clean-venv discovery is the
installer's gate and is covered there; everything else is here.

The property worth stating plainly, because the file this replaces got it wrong: **nothing a
candidate writes may influence its own score.** The old scorer read a ``# relevance=<float>``
comment out of the submitted ``agent.py``, so a submission could declare itself the best.
A test below runs exactly that submission and asserts it earns nothing.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import textwrap
from pathlib import Path

import pytest
from kata.plugins.contract import RunContext, ScoreCard, ScoringProfile, SubnetPlugin

from kata_sn22 import SN22_DESEARCH_PLUGIN, Sn22DesearchPlugin
from kata_sn22.plugin import PLUGIN_REVISION, Sn22AgentError
from kata_sn22.protocol import PROTOCOL_VERSION, ErrorClass
from kata_sn22.scoring import RANK_SIGNALS, Signals

SEED = "sn22-plugin-test-0001"


@pytest.fixture
def plugin() -> Sn22DesearchPlugin:
    return Sn22DesearchPlugin()


@pytest.fixture
def problems(plugin):
    return plugin.sample_problems(seed=SEED, config={"task_count": 4, "max_wall_seconds": 30})


def _agent(tmp_path: Path, name: str, body: str) -> Path:
    """Write a submission whose agent.py is exactly ``body``."""
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "agent.py").write_text(textwrap.dedent(body), encoding="utf-8")
    return root


#: A submission that answers honestly but retrieves nothing. HONEST, not good -- it is the baseline
#: an attacker has to beat, and the reference shape of a valid response.
HONEST_AGENT = '''
    import json, sys
    task = json.load(sys.stdin)
    print(json.dumps({
        "protocol_version": task["protocol_version"],
        "task_id": task["task_id"],
        "summary": "no documents retrieved",
        "results": [],
        "citations": [],
        "usage": {"provider_calls": 0, "tokens": 0, "elapsed_seconds": 0.0},
    }))
'''

#: The submission the OLD scorer would have crowned: it simply declares a relevance.
RELEVANCE_HINT_AGENT = '''
    # relevance=1.0
    import json, sys
    task = json.load(sys.stdin)
    print(json.dumps({
        "protocol_version": task["protocol_version"],
        "task_id": task["task_id"],
        "summary": "# relevance=1.0",
        "results": [],
        "citations": [],
        "usage": {"provider_calls": 0, "tokens": 0, "elapsed_seconds": 0.0},
    }))
'''


def _run(plugin, problems, agent_root: Path, tmp_path: Path, label: str):
    context = RunContext(output_root=str(tmp_path / "out"), env=plugin.environment_spec(),
                         label=label)
    Path(context.output_root).mkdir(parents=True, exist_ok=True)
    return plugin.run_candidate(agent_path=str(agent_root), problems=problems, context=context)


class _Result:
    """Stands in for the core's challenge result object for serialization tests."""

    def __init__(self, king, challenger, identity):
        self.king_card = king
        self.candidate_card = challenger
        self.benchmark_identity = identity


def _card(**over) -> ScoreCard:
    base = dict(valid=1.0, quality=0.5, precision=1.0, coverage=0.5, invalid=0, cost=10.0,
                latency=5.0)
    base.update(over)
    signals = Signals(base["valid"], base["quality"], base["precision"], base["coverage"],
                      base["invalid"], base["cost"], base["latency"])
    return ScoreCard(comparable=signals.sn22_weighted_quality, passed=True,
                     metrics=signals.as_metrics(), payload=signals)


# ---- the stub scorer is gone ---------------------------------------------------------------------
def test_no_relevance_hint_path_survives_anywhere():
    """The old scorer PARSED this out of the submission. Check the shipped source, not just
    behaviour -- prose mentioning the removed hint is fine, code that reads it is not."""
    source = Path(__file__).resolve().parents[1] / "kata_sn22"
    for path in source.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert 'split("# relevance=' not in text, f"{path.name} still parses the stub scorer hint"
        assert "_stub_relevance" not in text, f"{path.name} still defines the stub scorer"
        for forbidden in ("from kata.packages", "import kata.packages"):
            assert forbidden not in text, f"{path.name} still imports the removed core API"


def test_a_submission_declaring_its_own_score_earns_nothing(plugin, problems, tmp_path):
    """The exact submission the previous implementation would have crowned."""
    root = _agent(tmp_path, "cheat", RELEVANCE_HINT_AGENT)
    card = plugin.score(_run(plugin, problems, root, tmp_path, "challenger"), problems)
    assert card.payload.sn22_weighted_quality == 0.0
    assert card.payload.sn22_coverage == 0.0


# ---- ABI conformance -----------------------------------------------------------------------------
def test_the_plugin_implements_the_current_contract(plugin):
    assert isinstance(plugin, SubnetPlugin)
    assert plugin.evaluator_id == "sn22_desearch"
    assert plugin.pack == "sn22__desearch"
    assert plugin.mode == "miner"
    assert plugin.validator_identity


def test_the_singleton_is_the_entry_point_target():
    assert isinstance(SN22_DESEARCH_PLUGIN, Sn22DesearchPlugin)


def test_the_declared_entry_point_matches_the_package():
    """The registry, the wheel metadata and the code must agree on one spelling."""
    import tomllib

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    value = declared["project"]["entry-points"]["kata.subnets"]["sn22"]
    module, _, attribute = value.partition(":")
    resolved = getattr(__import__(module, fromlist=["*"]), attribute)
    assert isinstance(resolved, Sn22DesearchPlugin)


def test_the_profile_is_deterministic_with_a_non_empty_identity(plugin, problems):
    """A sealed world is reproducible; the identity says which world it was."""
    assert plugin.scoring_profile is ScoringProfile.DETERMINISTIC
    identity = plugin.benchmark_identity(problems)
    assert identity and len(identity) == 64


def test_a_new_seed_is_a_new_sealed_world(plugin):
    a = plugin.sample_problems(seed="round-a", config={})
    b = plugin.sample_problems(seed="round-b", config={})
    assert plugin.benchmark_identity(a) != plugin.benchmark_identity(b)


def test_the_same_seed_reproduces_the_same_world(plugin):
    a = plugin.sample_problems(seed=SEED, config={"task_count": 4})
    b = plugin.sample_problems(seed=SEED, config={"task_count": 4})
    assert plugin.benchmark_identity(a) == plugin.benchmark_identity(b)


# ---- the candidate environment carries no credential ---------------------------------------------
def test_the_candidate_environment_holds_no_provider_secret(plugin):
    """Plan §6.1: providers are reached by the gateway, so no key enters the sandbox."""
    env = plugin.environment_spec()
    assert env.network == "relay_only"
    assert env.required_secrets == ()
    assert env.allowed_hosts == ()


def test_the_agent_sees_a_capability_and_no_api_key(plugin, problems, tmp_path):
    """Prove it by capturing the environment an agent is actually given."""
    root = _agent(tmp_path, "spy", '''
        import json, os, sys
        json.load(sys.stdin)
        sys.stderr.write(json.dumps(dict(os.environ)))
        print(json.dumps({"protocol_version": 1, "task_id": "x", "summary": "",
                          "results": [], "citations": [], "usage": {}}))
    ''')
    task = problems.tasks[0]
    captured = subprocess.run(
        ["/usr/bin/python3", str(root / "agent.py")],
        input=json.dumps(task.as_input()).encode(), capture_output=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
             "SN22_RELAY_CAPABILITY": "cap-token", "SN22_RELAY_ENDPOINT": "sn22-relay://c1"},
        check=False, timeout=60)
    seen = json.loads(captured.stderr.decode())
    assert seen["SN22_RELAY_CAPABILITY"] == "cap-token"
    for forbidden in ("OPENAI_API_KEY", "APIFY_API_KEY", "SCRAPINGDOG_API_KEY", "CHUTES_API_KEY",
                      "KATA_TARGET_TOKEN", "KATA_WEBHOOK_SECRET"):
        assert forbidden not in seen


# ---- running and classifying ---------------------------------------------------------------------
def test_a_missing_entry_point_is_refused_before_execution(plugin, problems, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(Sn22AgentError, match="no agent.py"):
        _run(plugin, problems, empty, tmp_path, "challenger")


def test_a_crashing_submission_is_classified_not_scored_low(plugin, problems, tmp_path):
    root = _agent(tmp_path, "crash", "import sys\nsys.exit(3)\n")
    raw = _run(plugin, problems, root, tmp_path, "challenger")
    assert all(a.error is ErrorClass.CRASHED for a in raw.attempts)
    card = plugin.score(raw, problems)
    assert card.passed is False
    assert card.payload.sn22_invalid_runs == len(problems.tasks)


def test_a_malformed_response_is_classified(plugin, problems, tmp_path):
    root = _agent(tmp_path, "garbage", "print('not json')\n")
    raw = _run(plugin, problems, root, tmp_path, "challenger")
    assert all(a.error is ErrorClass.INVALID_SCHEMA for a in raw.attempts)


def test_a_hanging_submission_times_out(plugin, tmp_path):
    problems = plugin.sample_problems(seed=SEED, config={"task_count": 1, "max_wall_seconds": 1})
    root = _agent(tmp_path, "hang", "import time\ntime.sleep(30)\n")
    raw = _run(plugin, problems, root, tmp_path, "challenger")
    assert raw.attempts[0].error is ErrorClass.TIMEOUT


def test_latency_is_measured_by_the_lane(plugin, problems, tmp_path):
    """Every attempt carries the lane's own clock, never the agent's claim."""
    root = _agent(tmp_path, "honest", HONEST_AGENT)
    raw = _run(plugin, problems, root, tmp_path, "challenger")
    assert all(a.observed_seconds > 0 for a in raw.attempts)
    assert all(a.output.usage.elapsed_seconds == 0.0 for a in raw.attempts if a.output)


# ---- both promotion directions ------------------------------------------------------------------
def test_a_better_challenger_is_promoted(plugin):
    king, challenger = _card(quality=0.3), _card(quality=0.9)
    assert plugin.compare(challenger, king) == 1
    assert plugin.beats_king(challenger, king)


def test_a_worse_challenger_is_not_promoted(plugin):
    king, challenger = _card(quality=0.9), _card(quality=0.3)
    assert plugin.compare(challenger, king) == -1
    assert not plugin.beats_king(challenger, king)


def test_a_tie_keeps_the_incumbent(plugin):
    assert plugin.compare(_card(), _card()) == 0
    assert not plugin.beats_king(_card(), _card())


def test_the_comparator_is_antisymmetric(plugin):
    a, b = _card(quality=0.2), _card(quality=0.8)
    assert plugin.compare(a, b) == -plugin.compare(b, a)


def test_an_empty_throne_still_requires_a_valid_run(plugin):
    assert plugin.beats_king(_card(), None)
    assert not plugin.beats_king(_card(valid=0.0), None)


def test_conformance_scorecards_order_correctly(plugin):
    """The installer's ordering gate runs these; they must exercise the real comparator."""
    weak, strong = plugin.conformance_scorecards()
    assert plugin.compare(strong, weak) == 1
    assert plugin.compare(weak, strong) == -1
    assert plugin.beats_king(strong, weak)
    assert not plugin.beats_king(weak, strong)


# ---- cost ---------------------------------------------------------------------------------------
def test_the_capacity_estimate_bounds_both_contestants(plugin):
    bound = plugin.capacity_estimate(
        config={"task_count": 5, "max_provider_calls": 3, "max_tokens": 1000})
    assert bound["inference_calls"] == 5 * 3 * 2
    assert bound["tokens"] == 5 * 1000 * 2


def test_the_estimate_is_a_true_upper_bound(plugin, tmp_path):
    """The relay's own quota must make it impossible to exceed what was reserved."""
    config = {"task_count": 2, "max_provider_calls": 2, "max_wall_seconds": 30}
    problems = plugin.sample_problems(seed=SEED, config=config)
    bound = plugin.capacity_estimate(config=config)
    raw = _run(plugin, problems, _agent(tmp_path, "greedy", HONEST_AGENT), tmp_path, "challenger")
    assert raw.usage.totals("challenger")["provider_calls"] <= bound["inference_calls"]


def test_the_estimate_uses_the_plugins_own_config_resolution(plugin):
    """Defaults here must match sample_problems, or the reservation diverges from the execution."""
    assert plugin.capacity_estimate(config={}) == plugin.capacity_estimate(
        config={"task_count": 4, "max_provider_calls": 8, "max_tokens": 20_000})


# ---- screening ----------------------------------------------------------------------------------
def test_static_screen_passes_a_clean_submission(plugin, tmp_path):
    assert plugin.static_screen(str(_agent(tmp_path, "clean", HONEST_AGENT))) is None


def test_static_screen_rejects_a_missing_entry_point(plugin, tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    result = plugin.static_screen(str(empty))
    assert result is not None and "no agent.py" in result["findings"][0]


def test_static_screen_flags_direct_egress(plugin, tmp_path):
    result = plugin.static_screen(str(_agent(tmp_path, "egress", "import requests\nprint('{}')\n")))
    assert result is not None
    assert any("requests" in finding for finding in result["findings"])


def test_static_screen_is_deterministic(plugin, tmp_path):
    root = _agent(tmp_path, "egress2", "import socket\nprint('{}')\n")
    assert plugin.static_screen(str(root)) == plugin.static_screen(str(root))


# ---- anti-memorization --------------------------------------------------------------------------
def test_benchmark_review_rejects_an_embedded_snapshot_id(plugin):
    reject, review, score = plugin.benchmark_review({"agent.py": "doc-kata-1 is the answer"},
                                                    strict=True)
    assert reject and score > 0
    assert not review


def test_benchmark_review_only_reviews_when_not_strict(plugin):
    reject, review, _score = plugin.benchmark_review({"agent.py": "doc-kata-1"}, strict=False)
    assert not reject and review


def test_benchmark_review_passes_a_clean_bundle(plugin):
    reject, review, score = plugin.benchmark_review({"agent.py": HONEST_AGENT}, strict=True)
    assert not reject and not review and score == 0.0


# ---- serialization ------------------------------------------------------------------------------
def test_the_result_serializes_the_ordered_rank_signals(plugin):
    document = plugin.challenge_result_json(
        _Result(_card(quality=0.3), _card(quality=0.9), "a" * 64))
    assert document["evaluator_id"] == "sn22_desearch"
    assert document["plugin_revision"] == PLUGIN_REVISION
    assert document["protocol_version"] == PROTOCOL_VERSION
    for side in ("king", "challenger"):
        signals = document[side]["signals"]
        assert [s["name"] for s in signals] == [name for name, _ in RANK_SIGNALS]
        assert [s["priority"] for s in signals] == list(range(1, len(RANK_SIGNALS) + 1))
        assert signals[0]["direction"] == "higher_is_better"
        assert signals[-1]["direction"] == "lower_is_better"


def test_the_result_is_json_serializable(plugin):
    document = plugin.challenge_result_json(_Result(_card(), _card(), "b" * 64))
    assert json.loads(json.dumps(document))["lane_id"] == "sn22__desearch"


def test_an_unscored_side_serializes_as_null(plugin):
    document = plugin.challenge_result_json(_Result(None, _card(), "c" * 64))
    assert document["king"] is None
    assert document["challenger"] is not None


def test_the_text_rendering_names_both_sides(plugin):
    text = plugin.render_challenge_text(_Result(_card(quality=0.3), _card(quality=0.9), "d" * 64))
    assert "king" in text and "challenger" in text and "weighted_quality" in text


# ---- challenge config ---------------------------------------------------------------------------
def test_challenge_arguments_round_trip(plugin):
    parser = argparse.ArgumentParser()
    plugin.add_challenge_arguments(parser)
    args = parser.parse_args(["--sn22-task-count", "6", "--sn22-max-provider-calls", "2"])
    config = plugin.build_challenge_config(args)
    assert config["task_count"] == 6
    assert config["max_provider_calls"] == 2
    problems = plugin.sample_problems(seed=SEED, config=config)
    assert len(problems.tasks) == 6
    assert all(task.limits.max_provider_calls == 2 for task in problems.tasks)


# ---- provenance and freshness -------------------------------------------------------------------
def test_promotion_provenance_records_the_sealed_world(plugin, tmp_path):
    class _Summary:
        benchmark_identity = "e" * 64

    class _Entry:
        submission_id = "pr-42"

    plugin.record_promotion_provenance(entry=_Entry(), verification=None, summary=_Summary(),
                                       public_root=str(tmp_path))
    written = list((tmp_path / plugin.pack / "promotions").glob("*.json"))
    assert len(written) == 1
    record = json.loads(written[0].read_text())
    assert record["benchmark_identity"] == "e" * 64
    assert record["entry"] == "pr-42"
    assert record["lane_id"] == "sn22__desearch"


def test_a_summary_without_an_identity_is_not_current(plugin):
    class _Summary:
        benchmark_identity = ""

    assert not plugin.benchmark_is_current(lane_id="sn22__desearch", summary=_Summary())
    assert plugin.extra_verification_reasons(lane_id="sn22__desearch", summary=_Summary())


# ---- a whole generic challenge ------------------------------------------------------------------
def test_a_full_paired_challenge_runs_end_to_end(plugin, tmp_path):
    """King and challenger over one sealed world: scored, compared, decided. No spend."""
    problems = plugin.sample_problems(seed=SEED, config={"task_count": 3, "max_wall_seconds": 30})
    king = plugin.score(
        _run(plugin, problems, _agent(tmp_path, "king", HONEST_AGENT), tmp_path, "king"), problems)
    challenger = plugin.score(
        _run(plugin, problems, _agent(tmp_path, "chal", HONEST_AGENT), tmp_path, "challenger"),
        problems)

    # Identical submissions differ only by wall-clock jitter, which compare() reports exactly --
    # and which must NOT hand over a crown. That is what the promotion margins are for.
    for name, _higher in RANK_SIGNALS:
        if name == "sn22_latency_seconds":
            continue
        assert getattr(king.payload, name) == getattr(challenger.payload, name), name
    assert not plugin.beats_king(challenger, king)
    assert not plugin.beats_king(king, challenger)
    document = plugin.challenge_result_json(
        _Result(king, challenger, plugin.benchmark_identity(problems)))
    assert len(document["benchmark_identity"]) == 64
    assert json.loads(json.dumps(document))


def test_progress_is_reported_per_task(plugin, problems, tmp_path):
    seen = []
    context = RunContext(output_root=str(tmp_path / "out"), env=plugin.environment_spec(),
                         label="challenger", progress=seen.append)
    Path(context.output_root).mkdir(parents=True, exist_ok=True)
    plugin.run_candidate(agent_path=str(_agent(tmp_path, "p", HONEST_AGENT)),
                         problems=problems, context=context)
    assert [update.done for update in seen] == list(range(1, len(problems.tasks) + 1))
    assert all(update.total == len(problems.tasks) for update in seen)
