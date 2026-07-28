"""The production epoch: 60 tasks, four pools, three deep samples each, both sides identical.

The exit gates for this phase are four properties, and each one is a way a duel could look fine and
mean nothing:

* **60 tasks, 15 per pool, exactly 3 deep.** Below three deep samples upstream DROPS a contestant
  from the pool rather than scoring it low — so the deployed ``task_count=8`` would have zeroed
  everyone for a reason unrelated to their answers, and it would have looked like the agents were
  bad.
* **Both sides get the same manifest hash.** Including the same deep-sample ids.
* **No paid query-generation call.** Upstream falls back to an LLM when its dataset is unavailable,
  and that call is the *validator's* to pay for. Kata's validator holds no paid credential, so the
  fallback cannot merely be discouraged — it must not exist.
* **The distribution is upstream's.** Re-derived from the pinned tree rather than restated, so the
  day upstream changes a weight is a test failure and not a silent re-weighting.
"""

from __future__ import annotations

import ast
import json
import random
from collections import Counter

import pytest

from kata_sn22 import epoch_manifest as em
from kata_sn22 import question_pool as qp
from kata_sn22.protocol_v2 import AiSearchTask, ResultType, SearchMode, XSearchTask
from kata_sn22.upstream_snapshot import snapshot_root


@pytest.fixture
def pool():
    return qp.load_pool("development")


@pytest.fixture
def epoch(pool):
    return em.build_epoch(seed="round-0001", pool=pool, production=False)


def _upstream_source(relative: str) -> str:
    return (snapshot_root() / relative).read_text(encoding="utf-8")


def _upstream_constant(relative: str, name: str):
    """Read a module-level constant out of the pinned tree by AST.

    By AST because these modules import ``bittensor`` at module scope, and nothing in this
    repository's test path may need it.
    """
    for node in ast.parse(_upstream_source(relative)).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{relative} no longer defines {name}")


# ---- GATE: every production manifest has 60 tasks ---

def test_an_epoch_is_sixty_tasks(epoch):
    assert len(epoch.tasks) == em.TASKS_PER_EPOCH == 60


def test_every_pool_has_fifteen_tasks_and_exactly_three_deep(epoch):
    """Three is not a target, it is a threshold. ``_pool_raw_scores`` drops a UID with fewer."""
    by_pool = epoch.tasks_by_pool()
    assert sorted(by_pool) == sorted(em.POOLS)
    for pool_name in em.POOLS:
        tasks = by_pool[pool_name]
        assert len(tasks) == 15, pool_name
        deep = [task for task in tasks if task.task_id in epoch.deep_task_ids]
        assert len(deep) == 3, f"{pool_name} has {len(deep)} deep samples"


def test_the_pool_size_is_the_one_the_scorer_policy_derives():
    from kata_sn22.scorer_policy import minimum_tasks_per_pool

    assert em.TASKS_PER_POOL == minimum_tasks_per_pool() == 15


def test_a_manifest_that_is_not_a_full_epoch_is_refused(epoch):
    """Constructed directly, so the guard is on the manifest rather than only on the builder."""
    import dataclasses

    with pytest.raises(em.ManifestError, match="60 tasks"):
        dataclasses.replace(epoch, tasks=epoch.tasks[:59])


def test_a_pool_short_of_deep_samples_is_refused(epoch):
    """The failure this whole size exists to prevent, asserted directly."""
    import dataclasses

    one_short = frozenset(sorted(epoch.deep_task_ids)[1:])
    with pytest.raises(em.ManifestError, match="deep samples"):
        dataclasses.replace(epoch, deep_task_ids=one_short)


def test_task_ids_are_unique(epoch):
    ids = [task.task_id for task in epoch.tasks]
    assert len(set(ids)) == len(ids)


# ---- GATE: both sides receive the same manifest ---

def test_the_same_seed_produces_the_same_epoch(pool):
    first = em.build_epoch(seed="same", pool=pool, production=False)
    second = em.build_epoch(seed="same", pool=pool, production=False)
    assert first.digest() == second.digest()
    assert first.deep_task_ids == second.deep_task_ids


def test_a_different_seed_produces_a_different_epoch(pool):
    first = em.build_epoch(seed="one", pool=pool, production=False)
    second = em.build_epoch(seed="two", pool=pool, production=False)
    assert first.digest() != second.digest()


def test_two_contestants_asked_the_same_questions_pass_the_duel_check(pool):
    king = em.build_epoch(seed="duel", pool=pool, production=False)
    challenger = em.build_epoch(seed="duel", pool=pool, production=False)
    em.duel_manifests_match(king, challenger)


def test_two_contestants_asked_different_questions_are_refused(pool):
    king = em.build_epoch(seed="duel", pool=pool, production=False)
    challenger = em.build_epoch(seed="another", pool=pool, production=False)
    with pytest.raises(em.ManifestError, match="same questions"):
        em.duel_manifests_match(king, challenger)


def test_two_contestants_with_different_deep_samples_are_refused(epoch):
    """One side would be graded on three answers the other was never asked to work hard on."""
    import dataclasses

    swapped = dataclasses.replace(epoch, deep_task_ids=epoch.deep_task_ids)
    object.__setattr__(swapped, "deep_task_ids", frozenset(sorted(epoch.deep_task_ids)[:3]))
    with pytest.raises(em.ManifestError):
        em.duel_manifests_match(epoch, swapped)


@pytest.mark.parametrize("field_name", ["seed", "date_filter", "upstream_commit",
                                        "scorer_policy_hash"])
def test_changing_any_recorded_rule_moves_the_manifest_digest(epoch, field_name):
    """The digest is what an attested report is bound to. If it did not move, a report produced
    under different rules would compare equal to one that was not."""
    import dataclasses

    changed = dataclasses.replace(epoch, **{field_name: "drifted"})
    assert changed.digest() != epoch.digest()


def test_the_question_pool_digest_is_recorded_in_the_manifest(epoch, pool):
    document = epoch.as_document()
    assert document["pool_digest"] == pool.digest
    assert document["pool_kind"] == pool.kind
    assert document["pool_name"] == "development"


def test_the_deep_selection_of_one_pool_does_not_depend_on_another(pool):
    """A separate RNG stream per pool, seeded from the round seed. Otherwise adding a task to one
    pool would silently reshuffle which of the OTHER pools' tasks are deep-scored, and two
    otherwise-identical epochs would stop being comparable."""
    tasks = em.build_epoch(seed="stable", pool=pool, production=False).tasks
    full = em.select_deep_tasks(seed="stable", tasks=tasks)

    ai_fast = [task for task in tasks if em.pool_of(task) == "ai_search:fast"]
    x_only = [task for task in tasks if em.pool_of(task) == "x_search"]
    subset = em.select_deep_tasks(seed="stable", tasks=tuple(ai_fast + x_only))

    assert {task_id for task_id in subset if task_id.startswith("x_search")} == \
        {task_id for task_id in full if task_id.startswith("x_search")}


# ---- GATE: no paid query-generation call ---

def test_no_module_in_the_epoch_path_can_reach_a_network(monkeypatch, pool):
    """The strongest available form of "no paid call": break the network, build an epoch anyway.

    Upstream's generator falls back to an OpenAI call when its dataset is unavailable, and that call
    is the validator's to pay for. Kata's validator holds no paid credential at all, so this is not
    a cost question -- the fallback must not exist.
    """
    import socket

    def _no_network(*_args, **_kwargs):
        raise AssertionError("the epoch builder attempted a network call")

    monkeypatch.setattr(socket, "socket", _no_network)
    monkeypatch.setattr(socket, "create_connection", _no_network)

    built = em.build_epoch(seed="offline", pool=pool, production=False)
    assert len(built.tasks) == 60


def test_the_epoch_modules_import_only_the_standard_library():
    """No ``openai``, no ``huggingface_hub``, no ``datasets``. Those belong to the operator-run
    snapshot tool, which runs on a laptop and never on a duel."""
    import sys
    from pathlib import Path

    for module in (em, qp):
        source = Path(module.__file__).read_text(encoding="utf-8")
        imported: set = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
        third_party = {
            name for name in imported
            if name not in sys.stdlib_module_names and name != "kata_sn22"
        }
        assert not third_party, f"{module.__name__} imports {sorted(third_party)}"


def test_a_missing_question_pool_fails_rather_than_inventing_questions():
    with pytest.raises(qp.PoolError, match="no LLM fallback"):
        qp.load_pool("a-pool-that-does-not-exist")


def test_a_development_pool_is_refused_in_production(pool):
    """The convenient thing -- ship the dev pool, forget to snapshot -- must not be silently
    indistinguishable from the correct one."""
    assert pool.kind == qp.KIND_DEVELOPMENT
    with pytest.raises(qp.PoolError, match="production epoch must be built"):
        em.build_epoch(seed="nope", pool=pool, production=True)


def test_the_refusal_names_the_tool_that_fixes_it(pool):
    with pytest.raises(qp.PoolError, match="snapshot_questions.py"):
        pool.require_production_kind()


def test_a_pool_edited_in_place_is_refused(tmp_path, monkeypatch):
    """The digest is verified on every load, not only when written. A row removed to make a test
    pass would otherwise change what both contestants are asked while the manifest went on
    reporting the old identity."""
    monkeypatch.setattr(qp, "DATASETS_DIR", tmp_path)
    rows = '{"id": "a", "question": "q", "lane": "news"}\n'
    (tmp_path / "edited.jsonl").write_text(rows + '{"id": "b", "question": "extra"}\n')
    (tmp_path / "edited.meta.json").write_text(json.dumps({
        "pool_kind": "development",
        "sha256": __import__("hashlib").sha256(rows.encode()).hexdigest(),
    }))
    with pytest.raises(qp.PoolError, match="edited in place"):
        qp.load_pool("edited")


def test_an_unknown_pool_kind_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(qp, "DATASETS_DIR", tmp_path)
    rows = '{"id": "a", "question": "q", "lane": "news"}\n'
    (tmp_path / "odd.jsonl").write_text(rows)
    (tmp_path / "odd.meta.json").write_text(json.dumps({
        "pool_kind": "whatever",
        "sha256": __import__("hashlib").sha256(rows.encode()).hexdigest(),
    }))
    with pytest.raises(qp.PoolError, match="unknown pool_kind"):
        qp.load_pool("odd")


def test_the_snapshot_tool_is_not_importable_from_the_lane_runtime():
    """It needs ``huggingface_hub`` and ``datasets``. Keeping it in ``tools/`` is what stops those
    becoming lane dependencies -- network access at snapshot time is fine, at duel time it is not.
    """
    from pathlib import Path

    tool = Path(__file__).resolve().parents[1] / "tools" / "snapshot_questions.py"
    assert tool.is_file()
    source = tool.read_text(encoding="utf-8")
    assert "huggingface_hub" in source and "datasets" in source

    lane_sources = " ".join(
        path.read_text(encoding="utf-8")
        for path in (Path(em.__file__).parent).glob("*.py"))
    assert "huggingface_hub" not in lane_sources
    assert "from datasets import" not in lane_sources


# ---- the distribution is upstream's, re-derived ---

def test_the_tool_weights_per_mode_match_the_pinned_upstream():
    """Fast is web-only; balanced and deep are an even split. If upstream re-weighted, the pools
    would stop measuring what their names say."""
    source = _upstream_source("neurons/validators/scoring/synthetic_query_generator.py")
    assert "SearchMode.FAST: {WEB_TOOL: 1.0}" in source
    assert "SearchMode.BALANCED: {WEB_TOOL: 0.50, TWITTER_TOOL: 0.50}" in source
    assert "SearchMode.DEEP: {WEB_TOOL: 0.50, TWITTER_TOOL: 0.50}" in source

    assert em.MODE_TOOL_WEIGHTS[SearchMode.FAST] == {em.WEB_TOOL: 1.0}
    for mode in (SearchMode.BALANCED, SearchMode.DEEP):
        assert em.MODE_TOOL_WEIGHTS[mode] == {em.WEB_TOOL: 0.50, em.TWITTER_TOOL: 0.50}


def test_the_result_type_weights_match_the_pinned_upstream():
    source = _upstream_source("neurons/validators/scoring/synthetic_query_generator.py")
    assert "ResultType.LINKS_WITH_FINAL_SUMMARY: 0.80" in source
    assert "ResultType.ONLY_LINKS: 0.20" in source
    assert em.AI_RESULT_WEIGHTS[ResultType.LINKS_WITH_FINAL_SUMMARY] == 0.80
    assert em.AI_RESULT_WEIGHTS[ResultType.ONLY_LINKS] == 0.20


def test_the_serving_budgets_match_the_pinned_upstream():
    """``max(MODE_BUDGETS[mode], SERVING_FLOOR)``. The performance multiplier and the timeout
    penalty are both measured against this, so it is part of the task and not a Kata convenience."""
    # MODE_BUDGETS is keyed by an enum, so ast.literal_eval cannot read it. Compared against the
    # source text instead, which is still a re-derivation rather than a restatement.
    source = _upstream_source("desearch/utils.py")
    assert "SearchMode.FAST: 5" in source
    assert "SearchMode.BALANCED: 15" in source
    assert "SearchMode.DEEP: 30" in source
    assert em.SERVING_FLOOR == _upstream_constant("desearch/utils.py", "SERVING_FLOOR")
    assert (em.serving_budget(SearchMode.FAST),
            em.serving_budget(SearchMode.BALANCED),
            em.serving_budget(SearchMode.DEEP)) == (15, 15, 30)


def test_the_date_filter_weighting_matches_the_pinned_upstream():
    source = _upstream_source("desearch/dataset/date_filters.py")
    for name, weight in em.DATE_FILTER_WEIGHTS.items():
        assert f"DateFilterType.{name}: {weight}" in source, name
    assert len(em.date_filter_pool()) == sum(em.DATE_FILTER_WEIGHTS.values())


def test_the_x_seed_lists_match_the_pinned_upstream():
    """Packaged rather than parsed at run time, because the room runtime does not read the vendored
    tree. This is what stops the copy drifting from the original."""
    seeds = qp.load_x_seeds()
    source = _upstream_source("desearch/dataset/dataset.py")
    tree = ast.parse(source)

    def module_const(name):
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == name:
                        return ast.literal_eval(node.value)
        raise AssertionError(name)

    def class_const(class_name, name):
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for sub in node.body:
                    if isinstance(sub, ast.Assign):
                        for target in sub.targets:
                            if isinstance(target, ast.Name) and target.id == name:
                                return ast.literal_eval(sub.value)
        raise AssertionError(name)

    assert seeds["topics"] == module_const("TOPICS")
    assert seeds["popular_accounts"] == class_const("BasicQuestionsDataset", "POPULAR_ACCOUNTS")
    assert seeds["popular_crypto_keywords"] == class_const(
        "BasicQuestionsDataset", "POPULAR_CRYPTO_KEYWORDS")


def test_the_requested_result_count_is_upstreams_minimum():
    assert em.RESULT_COUNT == 10


def test_the_generated_distribution_matches_upstream_over_many_epochs(pool):
    """Not a restatement of the weights -- the outcome of actually generating.

    A port whose ``_weighted_counts`` consumed the RNG differently would still declare the right
    weights and produce the wrong epochs.
    """
    tools: Counter = Counter()
    result_types: Counter = Counter()
    for index in range(60):
        built = em.build_epoch(seed=f"dist-{index}", pool=pool, production=False)
        for task in built.tasks:
            if isinstance(task, AiSearchTask):
                tools[(em.pool_of(task), task.tools[0])] += 1
                result_types[task.result_type] += 1

    assert tools[("ai_search:fast", em.TWITTER_TOOL)] == 0
    for pool_name in ("ai_search:balanced", "ai_search:deep"):
        web = tools[(pool_name, em.WEB_TOOL)]
        twitter = tools[(pool_name, em.TWITTER_TOOL)]
        assert abs(web - twitter) / (web + twitter) < 0.05, pool_name

    total = sum(result_types.values())
    assert abs(result_types[ResultType.LINKS_WITH_FINAL_SUMMARY] / total - 0.80) < 0.02
    assert abs(result_types[ResultType.ONLY_LINKS] / total - 0.20) < 0.02


def test_the_twitter_tool_draws_from_the_x_lane(pool):
    """Upstream's rule. The x lane feeds AI search's Twitter TOOL -- not Basic X search, whose
    queries are generated from the seed lists and never drawn from the pool."""
    x_questions = {row["question"] for row in pool.lane("x")}
    web_questions = {row["question"] for row in pool.web()}
    built = em.build_epoch(seed="lanes", pool=pool, production=False)

    for task in built.tasks:
        if isinstance(task, AiSearchTask) and em.TWITTER_TOOL in task.tools:
            assert task.prompt in x_questions
        elif isinstance(task, AiSearchTask):
            assert task.prompt in web_questions


def test_basic_x_queries_are_generated_not_drawn_from_the_pool(pool, epoch):
    pool_questions = {row["question"] for row in pool.rows}
    x_tasks = [task for task in epoch.tasks if isinstance(task, XSearchTask)]
    assert len(x_tasks) == 15
    assert not any(task.query in pool_questions for task in x_tasks)


def test_basic_x_queries_take_upstreams_four_shapes():
    """20% ``from:``, 15% ``$ticker``, 15% ``#crypto``, 50% a plain topic."""
    seeds = qp.load_x_seeds()
    cashtags = ("BTC", "ETH", "SOL")
    shapes: Counter = Counter()
    rng = random.Random("shapes")
    for _ in range(20_000):
        query = em.random_x_query(rng, seeds, cashtags=cashtags)
        if query[0] in "$#":
            shapes[query[0]] += 1
        else:
            shapes["from" if query.startswith("from:") else "topic"] += 1

    total = sum(shapes.values())
    assert abs(shapes["from"] / total - 0.20) < 0.02
    assert abs(shapes["$"] / total - 0.15) < 0.02
    assert abs(shapes["#"] / total - 0.15) < 0.02
    assert abs(shapes["topic"] / total - 0.50) < 0.02


def test_without_a_cashtag_supplier_upstreams_own_except_path_is_taken():
    """``faker`` is a third-party package the room cannot carry. Upstream wraps that call in
    try/except and falls through to the hashtag branch, so a Kata epoch with no supplier takes
    UPSTREAM'S OWN path rather than a substitute of ours.

    The 15% shift is real, and it is recorded in the manifest rather than absorbed.
    """
    source = _upstream_source("desearch/dataset/dataset.py")
    assert "self.faker.cryptocurrency()" in source
    assert "except Exception:\n                pass" in source

    seeds = qp.load_x_seeds()
    rng = random.Random("no-cashtags")
    queries = [em.random_x_query(rng, seeds, cashtags=None) for _ in range(2_000)]
    assert not any(query.startswith("$") for query in queries)
    assert sum(1 for query in queries if query.startswith("#")) / len(queries) > 0.25


def test_the_manifest_records_whether_the_cashtag_path_was_available(epoch):
    assert epoch.as_document()["cashtags_available"] is False


# ---- what the agent sees ---

def test_no_agent_input_carries_the_deep_flag(epoch):
    """An agent that knew which tasks were deep-scored would work hardest on exactly those, and the
    20% sample would stop measuring the other 80%."""
    for agent_input in epoch.agent_inputs():
        assert "deep" not in agent_input
    assert any(task.deep for task in epoch.tasks)


def test_the_manifest_itself_does_record_the_deep_ids(epoch):
    document = epoch.as_document()
    assert len(document["deep_task_ids"]) == 12
    assert sum(1 for task in document["tasks"] if task["deep"]) == 12


def test_every_agent_input_is_a_valid_version_two_task(epoch):
    for agent_input in epoch.agent_inputs():
        assert agent_input["protocol_version"] == 2
        assert agent_input["count"] == 10
        assert agent_input["limits"]["max_execution_time"] in (15, 30)


def test_the_date_filter_is_one_per_epoch_and_reaches_every_ai_task(epoch):
    """Upstream draws one filter per epoch and applies it to every AI task in it."""
    filters = {task.date_filter_type for task in epoch.tasks if isinstance(task, AiSearchTask)}
    assert filters == {epoch.date_filter}


# ---- the production entry point refuses the calibration pool ---

def test_the_production_backend_builds_a_real_epoch(monkeypatch, pool):
    """``sample_problems`` under ``tee`` must produce the 60-task epoch, not the six-query
    calibration pool a miner iterates against."""
    import dataclasses

    from kata_sn22 import plugin as plugin_module
    from kata_sn22.execution import policy as execution_policy

    monkeypatch.setenv(execution_policy.EXECUTION_BACKEND_ENV, "tee")
    monkeypatch.setattr(
        qp, "load_pool",
        lambda _name: dataclasses.replace(pool, kind=qp.KIND_UPSTREAM_SNAPSHOT))

    problems = plugin_module.Sn22DesearchPlugin().sample_problems(seed="prod", config={})

    assert problems.epoch is not None
    assert len(problems.epoch.tasks) == 60


def test_the_sandbox_backend_builds_no_epoch(monkeypatch):
    """Calibration needs thirty-plus paired challenges; requiring a snapshotted pool for those
    would make the calibration this project's plan asks for impossible to run."""
    from kata_sn22 import plugin as plugin_module
    from kata_sn22.execution import policy as execution_policy

    monkeypatch.setenv(execution_policy.EXECUTION_BACKEND_ENV, "sandbox")
    problems = plugin_module.Sn22DesearchPlugin().sample_problems(
        seed="dev", config={"task_count": 2})
    assert problems.epoch is None


def test_production_refuses_a_task_count_that_cannot_be_scored(monkeypatch, pool):
    """8 was the deployed value. It is refused rather than rounded up, because the operator who set
    it believes something false about what it will measure."""
    import dataclasses

    from kata_sn22 import plugin as plugin_module
    from kata_sn22.execution import policy as execution_policy

    monkeypatch.setenv(execution_policy.EXECUTION_BACKEND_ENV, "tee")
    monkeypatch.setattr(
        qp, "load_pool",
        lambda _name: dataclasses.replace(pool, kind=qp.KIND_UPSTREAM_SNAPSHOT))

    with pytest.raises(plugin_module.Sn22AgentError, match="cannot be scored"):
        plugin_module.Sn22DesearchPlugin().sample_problems(
            seed="prod", config={"task_count": 8})


def test_production_refuses_when_the_packaged_rows_are_missing(monkeypatch):
    """Before the duel, while nobody has spent anything."""
    from kata_sn22 import plugin as plugin_module
    from kata_sn22.execution import policy as execution_policy

    monkeypatch.setenv(execution_policy.EXECUTION_BACKEND_ENV, "tee")
    monkeypatch.setattr(plugin_module, "PRODUCTION_QUESTION_POOL", "no-such-pool")

    with pytest.raises(plugin_module.Sn22AgentError, match="cannot build a production epoch"):
        plugin_module.Sn22DesearchPlugin().sample_problems(seed="prod", config={})


def test_production_refuses_the_hand_written_calibration_pool(monkeypatch):
    """The six queries a miner iterates against have nothing to do with the subnet. A King
    defending its crown against them would be defending nothing."""
    from kata_sn22 import fixtures
    from kata_sn22 import plugin as plugin_module
    from kata_sn22.execution import policy as execution_policy

    assert len(fixtures.query_pool()) < 15, "the calibration pool is not an epoch"

    monkeypatch.setenv(execution_policy.EXECUTION_BACKEND_ENV, "tee")
    monkeypatch.setattr(plugin_module, "PRODUCTION_QUESTION_POOL", "development")

    with pytest.raises(plugin_module.Sn22AgentError, match="production epoch must be built"):
        plugin_module.Sn22DesearchPlugin().sample_problems(seed="prod", config={})


def test_the_packaged_rows_ship_inside_the_installed_package():
    """The room installs this repository as a wheel. A dataset directory that lived beside the
    package rather than inside it would pass every test here and then fail at deploy time with
    "no question pool", which reads as a missing snapshot rather than a packaging mistake.
    """
    from pathlib import Path

    package_root = Path(em.__file__).parent
    assert qp.DATASETS_DIR.parent == package_root, \
        "datasets must live INSIDE kata_sn22/ so they are packaged with it"
    assert (qp.DATASETS_DIR / "x_seeds.json").is_file()
    assert qp.pool_path("development").is_file()
    assert qp.meta_path("development").is_file()


def test_the_committed_production_pool_is_a_real_upstream_snapshot():
    """Replaces the earlier "no production pool yet" placeholder, now that one is committed.

    Checks the thing that actually matters: it declares itself an ``upstream-snapshot`` (a
    hand-made stand-in under the production name would be refused by kind), its digest verifies,
    and every lane an epoch draws from is deep enough that rounds do not repeat questions.
    """
    if not qp.pool_path("production").exists():
        pytest.skip("no production pool snapshotted in this checkout")

    pool = qp.load_pool("production")            # raises if the digest does not verify
    assert pool.kind == qp.KIND_UPSTREAM_SNAPSHOT
    assert pool.upstream_commit

    # 45 AI tasks and 15 X queries per round. A lane thinner than this repeats within one round,
    # let alone across rounds.
    assert len(pool.web()) >= 100, f"web lanes hold only {len(pool.web())} rows"
    assert len(pool.lane("x")) >= 100, f"the x lane holds only {len(pool.lane('x'))} rows"


def test_a_production_epoch_builds_from_the_committed_pool():
    """The end of the operator's snapshot step: 60 tasks, four pools, three deep samples each, and
    two rounds that do not ask the same questions."""
    if not qp.pool_path("production").exists():
        pytest.skip("no production pool snapshotted in this checkout")

    pool = qp.load_pool("production")
    first = em.build_epoch(seed="round-001", pool=pool, production=True)
    second = em.build_epoch(seed="round-002", pool=pool, production=True)

    assert len(first.tasks) == 60
    assert len(first.deep_task_ids) == 12
    assert first.as_document()["pool_digest"] == pool.digest

    questions = {task.prompt for task in first.tasks if isinstance(task, AiSearchTask)}
    later = {task.prompt for task in second.tasks if isinstance(task, AiSearchTask)}
    assert not (questions & later), f"two rounds shared {len(questions & later)} questions"


def test_two_rounds_draw_different_questions_from_the_pool():
    """The pool is a bank, not a script. A question set that never changes is one a contestant
    memorises once and answers from cache.

    This failed when written: the builder walked file-ordered rows with a cursor starting at zero,
    so every round drew the same first 45 questions however large the pool was. Upstream shuffles
    before drawing (``HFQuestionPool.sample_lane``, ``SyntheticQueryGenerator._sample_web``); the
    port had dropped it.

    Uses a large synthetic pool because the packaged development pool has fewer rows than one round
    needs, so it necessarily reuses them whatever the order.
    """
    import dataclasses

    rows = tuple({"id": f"r{index}", "question": f"question number {index}",
                  "lane": "news" if index % 4 else "x", "start_date": None, "end_date": None}
                 for index in range(3000))
    big = dataclasses.replace(qp.load_pool("development"), rows=rows)

    def questions(seed: str) -> set:
        built = em.build_epoch(seed=seed, pool=big, production=False)
        return {task.prompt for task in built.tasks if isinstance(task, AiSearchTask)}

    first, second = questions("challenge-001"), questions("challenge-002")
    assert len(first) == 45
    assert not (first & second), (
        f"two rounds shared {len(first & second)} questions out of {len(first)}")


def test_a_round_is_still_exactly_reproducible_from_its_seed():
    """Shuffling must not cost reproducibility: both contestants get one manifest, and a reviewer
    has to be able to rebuild it from the record."""
    pool = qp.load_pool("development")
    first = em.build_epoch(seed="reproduce", pool=pool, production=False)
    second = em.build_epoch(seed="reproduce", pool=pool, production=False)
    assert first.digest() == second.digest()
    # X tasks carry `query`, AI tasks carry `prompt`; compare whichever each one has.
    def _text(task) -> str:
        return getattr(task, "prompt", None) or getattr(task, "query", "")

    assert [_text(task) for task in first.tasks] == [_text(task) for task in second.tasks]


def test_the_runner_image_does_not_ship_the_question_pool():
    """The room never loads it -- each task's descriptor arrives inside the pool job, already drawn
    on the validator host.

    A production snapshot is ~90 MB and every byte of an attested image is covered by the
    measurement. Shipping questions the room cannot read would inflate what an operator has to
    reason about and slow every room deploy, for nothing.
    """
    from pathlib import Path

    build = (Path(em.__file__).parents[1] / "deploy" / "sn22-runner" / "build.sh")
    body = build.read_text(encoding="utf-8")
    assert "datasets" in body and "-name '*.jsonl' -delete" in body, (
        "the runner build no longer strips the question pool from the image context")

    profile = (Path(em.__file__).parents[1] / "deploy" / "sn22-runner" / "tee_profile.py")
    source = profile.read_text(encoding="utf-8")
    assert "load_pool" not in source, (
        "the room now loads the question pool; it would have to ship it, and the pool is the "
        "validator's to draw from")
