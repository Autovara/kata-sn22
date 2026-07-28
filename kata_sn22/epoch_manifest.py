"""One virtual upstream epoch: 60 tasks, four pools, both contestants identical.

**Why 60 and not 8.** Upstream deep-scores 20% of a pool and drops any contestant with fewer than
three deep samples. 15 tasks per pool is the smallest number that produces three, so the deployed
``task_count=8`` could never have scored at all — a contestant would have been zeroed for a reason
that has nothing to do with its answers. See :func:`kata_sn22.scorer_policy.minimum_tasks_per_pool`.

**The distribution is upstream's, ported not invented.** Tools per mode, result-type weights, the
serving budget per mode, the date-filter weighting and the Basic X query shapes are all read off
``neurons/validators/scoring/synthetic_query_generator.py`` and ``desearch/`` at the pin.
``_weighted_counts`` in particular is upstream's exact algorithm rather than something equivalent:
it consumes the RNG in a specific pattern, and a tidier version would produce a different epoch from
the same seed.

**Determinism is the fairness property.** One seed produces one epoch. The King and the Challenger
receive the same manifest object, including the same deep-sample ids — a duel where the two sides
were asked different questions, or where different tasks were deep-scored, is not a duel.

**The deep flag never reaches an agent.** It is recorded here and stripped by
:meth:`~kata_sn22.protocol_v2.AiSearchTask.as_agent_input`. An agent that knew which tasks were
deep-scored would work hardest on exactly those, and the 20% sample would stop measuring the rest.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field

from kata_sn22.protocol_v2 import (
    PROTOCOL_VERSION,
    AiSearchTask,
    Limits,
    ResultType,
    SearchMode,
    XSearchTask,
)
from kata_sn22.question_pool import QuestionPool, load_x_seeds
from kata_sn22.scorer_policy import (
    DEEP_SAMPLE_RATE,
    MIN_DEEP_SAMPLES_PER_POOL,
    TASKS_PER_POOL,
    policy_hash,
)
from kata_sn22.upstream_snapshot import UPSTREAM_COMMIT

MANIFEST_SCHEMA_VERSION = 1

#: Upstream's four lanes, in the order an epoch runs them.
POOLS: tuple[str, ...] = ("ai_search:fast", "ai_search:balanced", "ai_search:deep", "x_search")
TASKS_PER_EPOCH = TASKS_PER_POOL * len(POOLS)

WEB_TOOL = "Web Search"
TWITTER_TOOL = "Twitter Search"

#: ``synthetic_query_generator._MODE_TOOL_WEIGHTS``. Fast is web-only; the other two are an even
#: split, and the Twitter tool draws its row from the ``x`` lane rather than the web lanes.
MODE_TOOL_WEIGHTS: dict = {
    SearchMode.FAST: {WEB_TOOL: 1.0},
    SearchMode.BALANCED: {WEB_TOOL: 0.50, TWITTER_TOOL: 0.50},
    SearchMode.DEEP: {WEB_TOOL: 0.50, TWITTER_TOOL: 0.50},
}

#: ``synthetic_query_generator._AI_RESULT_WEIGHTS``.
AI_RESULT_WEIGHTS: dict = {
    ResultType.LINKS_WITH_FINAL_SUMMARY: 0.80,
    ResultType.ONLY_LINKS: 0.20,
}

#: ``desearch/utils.py``: ``max(MODE_BUDGETS[mode], SERVING_FLOOR)``, floor 15. The performance
#: multiplier and the timeout penalty are both measured against this, so it is part of the task.
MODE_BUDGETS: dict = {SearchMode.FAST: 5, SearchMode.BALANCED: 15, SearchMode.DEEP: 30}
SERVING_FLOOR = 15

#: ``desearch/dataset/date_filters.py::random_date_filters`` -- a Counter expanded to a weighted
#: list. One filter is drawn per epoch and applies to every AI task in it, as upstream does.
DATE_FILTER_WEIGHTS: dict = {
    "PAST_24_HOURS": 4, "PAST_2_DAYS": 5, "PAST_WEEK": 5,
    "PAST_2_WEEKS": 5, "PAST_MONTH": 1, "PAST_YEAR": 1,
}

#: Upstream's requested result count, and its own model's minimum.
RESULT_COUNT = 10


class ManifestError(Exception):
    """An epoch cannot be built, or the one that was built is not usable for a duel."""


def serving_budget(mode: SearchMode) -> int:
    return max(MODE_BUDGETS[mode], SERVING_FLOOR)


def date_filter_pool() -> tuple:
    """Upstream's ``Counter(...).elements()``, expanded in declaration order."""
    return tuple(name for name, count in DATE_FILTER_WEIGHTS.items() for _ in range(count))


# ---- upstream's weighted sampling, ported exactly ------------------------------------------------

def weighted_counts(rng: random.Random, n: int, weights: list) -> list:
    """``synthetic_query_generator._weighted_counts``.

    Ported statement for statement rather than replaced with an equivalent. It draws from the RNG in
    a particular pattern and removes indices as it goes; a tidier version would consume the stream
    differently and produce a different epoch from the same seed, which would be indistinguishable
    from a bug in the pinned upstream.
    """
    counts = [int(n * weight) for weight in weights]
    remainders = [n * weight - count for weight, count in zip(weights, counts)]
    idxs = list(range(len(weights)))
    for _ in range(n - sum(counts)):
        total = sum(remainders[i] for i in idxs)
        pick = rng.random() * total
        acc = 0.0
        for i in idxs:
            acc += remainders[i]
            if pick <= acc:
                counts[i] += 1
                idxs.remove(i)
                break
    return counts


def weighted_list(rng: random.Random, n: int, choices: dict) -> list:
    """``synthetic_query_generator._weighted_list``."""
    values = list(choices)
    out: list = []
    for value, count in zip(values, weighted_counts(rng, n, list(choices.values()))):
        out.extend([value] * count)
    rng.shuffle(out)
    return out


def ai_combos(rng: random.Random, mode: SearchMode, n: int) -> list:
    """``synthetic_query_generator._ai_combos``: ``(tools, result_type)`` per task."""
    if n <= 0:
        return []
    tools = weighted_list(rng, n, MODE_TOOL_WEIGHTS[mode])
    result_types = weighted_list(rng, n, AI_RESULT_WEIGHTS)
    return [([tools[index]], result_types[index]) for index in range(n)]


def random_x_query(rng: random.Random, seeds: dict, *, cashtags: object = None) -> str:
    """``BasicQuestionsDataset.generate_random_x_query``.

    Upstream's four shapes, at upstream's probabilities: 20% ``from:<account>``, 15% ``$<ticker>``,
    15% ``#<crypto>``, 50% a plain topic.

    **The ticker branch needs a supplier.** Upstream gets it from ``faker.cryptocurrency()``, a
    third-party package the room cannot carry -- the agent image ships no installer and the lane
    runtime is standard library only. Upstream wraps that call in ``try/except`` and falls through
    to the hashtag branch when it fails, so a Kata epoch with no supplier takes **upstream's own
    except-path** rather than a substitute of ours. That shifts 15% of X queries from ``$TICKER`` to
    ``#crypto``, and :meth:`EpochManifest.as_document` records which path was taken so the shift is
    visible in the manifest rather than silently absorbed.
    """
    mode = rng.random()
    if mode < 0.20:
        return f"from:{rng.choice(seeds['popular_accounts'])}"
    if mode < 0.35 and cashtags:
        return f"${rng.choice(list(cashtags))}"
    if mode < 0.50:
        return f"#{rng.choice(seeds['popular_crypto_keywords'])}"
    return rng.choice(seeds["topics"])


# ---- the manifest -----------------------------------------------------------------------------

@dataclass(frozen=True)
class EpochManifest:
    """One epoch's questions and the rules they will be scored under.

    Both contestants get this object. Its digest is bound into every attested pool report, so a
    report produced against a different question set is refused rather than compared.
    """

    seed: str
    tasks: tuple
    deep_task_ids: frozenset
    date_filter: str
    pool_provenance: dict
    cashtags_available: bool = False
    upstream_commit: str = UPSTREAM_COMMIT
    scorer_policy_hash: str = field(default_factory=policy_hash)

    def __post_init__(self) -> None:
        if len(self.tasks) != TASKS_PER_EPOCH:
            raise ManifestError(
                f"an epoch is {TASKS_PER_EPOCH} tasks ({TASKS_PER_POOL} per pool), "
                f"got {len(self.tasks)}")
        by_pool = self.tasks_by_pool()
        for pool in POOLS:
            tasks = by_pool.get(pool, ())
            if len(tasks) != TASKS_PER_POOL:
                raise ManifestError(
                    f"pool {pool} has {len(tasks)} tasks, expected {TASKS_PER_POOL}")
            deep = [task for task in tasks if task.task_id in self.deep_task_ids]
            if len(deep) != MIN_DEEP_SAMPLES_PER_POOL:
                raise ManifestError(
                    f"pool {pool} has {len(deep)} deep samples, expected "
                    f"{MIN_DEEP_SAMPLES_PER_POOL}; below the minimum a contestant is DROPPED from "
                    f"the pool rather than scored low")
        ids = [task.task_id for task in self.tasks]
        if len(set(ids)) != len(ids):
            raise ManifestError("task ids are not unique")

    def tasks_by_pool(self) -> dict:
        grouped: dict = {}
        for task in self.tasks:
            grouped.setdefault(pool_of(task), []).append(task)
        return {pool: tuple(tasks) for pool, tasks in grouped.items()}

    def agent_inputs(self) -> tuple:
        """What the agents receive: every task, without the deep flag."""
        return tuple(task.as_agent_input() for task in self.tasks)

    def as_document(self) -> dict:
        """The canonical manifest. Everything a later reviewer needs to say what was asked."""
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "seed": self.seed,
            "upstream_commit": self.upstream_commit,
            "scorer_policy_hash": self.scorer_policy_hash,
            "pools": list(POOLS),
            "tasks_per_pool": TASKS_PER_POOL,
            "deep_sample_rate": DEEP_SAMPLE_RATE,
            "result_count": RESULT_COUNT,
            "date_filter": self.date_filter,
            # False means the ticker branch fell through to upstream's own except-path. Recorded so
            # the distribution shift is visible rather than absorbed. See random_x_query.
            "cashtags_available": self.cashtags_available,
            "deep_task_ids": sorted(self.deep_task_ids),
            "tasks": [
                {**task.as_agent_input(), "deep": task.task_id in self.deep_task_ids,
                 "pool": pool_of(task),
                 "max_execution_time": task.limits.max_execution_time}
                for task in self.tasks
            ],
            **self.pool_provenance,
        }

    def digest(self) -> str:
        """The identity both contestants must share."""
        canonical = json.dumps(self.as_document(), sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def pool_of(task) -> str:
    if isinstance(task, XSearchTask):
        return "x_search"
    return f"ai_search:{task.mode.value}"


# ---- building one ------------------------------------------------------------------------------

def build_epoch(*, seed: str, pool: QuestionPool, production: bool = True,
                cashtags: object = None) -> EpochManifest:
    """Build one epoch's 60 tasks from packaged rows and a seed.

    ``production=True`` refuses a development pool BEFORE anything is built, so the failure lands
    while nobody has spent anything. Passing ``False`` is for this repository's own tests and for a
    miner calibrating locally; it never happens on a duel.
    """
    if production:
        pool.require_production_kind()
    seeds = load_x_seeds()

    rng = random.Random(f"kata-sn22-epoch|{UPSTREAM_COMMIT}|{seed}")
    date_filter = rng.choice(date_filter_pool())

    web_rows = pool.web()
    x_rows = pool.lane("x")
    if not web_rows and not x_rows:
        raise ManifestError(
            "the question pool has no usable rows; a production epoch fails here rather than "
            "asking a model to invent questions at the validator's expense")

    tasks: list = []
    web_cursor = x_cursor = 0
    for mode in (SearchMode.FAST, SearchMode.BALANCED, SearchMode.DEEP):
        for index, (tools, result_type) in enumerate(ai_combos(rng, mode, TASKS_PER_POOL)):
            # Upstream's rule: the Twitter tool draws from the x lane, everything else from web --
            # and either falls back to the other when its own lane is empty.
            if TWITTER_TOOL in tools and x_rows:
                row, x_cursor = x_rows[x_cursor % len(x_rows)], x_cursor + 1
            elif web_rows:
                row, web_cursor = web_rows[web_cursor % len(web_rows)], web_cursor + 1
            elif x_rows:
                row, x_cursor = x_rows[x_cursor % len(x_rows)], x_cursor + 1
            else:
                raise ManifestError(f"no question rows available for {mode.value}")
            tasks.append(AiSearchTask(
                task_id=f"{mode.value}-{index:02d}",
                prompt=str(row["question"]),
                mode=mode,
                result_type=result_type,
                tools=tuple(tools),
                count=RESULT_COUNT,
                start_date=row.get("start_date"),
                end_date=row.get("end_date"),
                date_filter_type=date_filter,
                limits=Limits(max_execution_time=serving_budget(mode)),
            ))

    for index in range(TASKS_PER_POOL):
        tasks.append(XSearchTask(
            task_id=f"x_search-{index:02d}",
            query=random_x_query(rng, seeds, cashtags=cashtags),
            count=RESULT_COUNT,
            limits=Limits(max_execution_time=SERVING_FLOOR),
        ))

    deep = select_deep_tasks(seed=seed, tasks=tuple(tasks))
    tasks = [
        _with_deep(task, task.task_id in deep) for task in tasks
    ]
    return EpochManifest(
        seed=seed, tasks=tuple(tasks), deep_task_ids=deep, date_filter=date_filter,
        pool_provenance=pool.as_provenance(), cashtags_available=bool(cashtags),
    )


def _with_deep(task, deep: bool):
    import dataclasses

    return dataclasses.replace(task, deep=deep)


def select_deep_tasks(*, seed: str, tasks: tuple) -> frozenset:
    """Choose exactly ``MIN_DEEP_SAMPLES_PER_POOL`` deep tasks per pool, deterministically.

    A separate RNG stream from the one that built the tasks, seeded from the same round seed. That
    way adding a task to a pool does not silently reshuffle which of the *other* pools' tasks are
    deep-scored, which would make two otherwise-identical epochs incomparable.

    Both contestants get this same set. If they did not, one would be graded on three answers the
    other was never asked to work hard on.
    """
    by_pool: dict = {}
    for task in tasks:
        by_pool.setdefault(pool_of(task), []).append(task.task_id)

    chosen: set = set()
    for pool in sorted(by_pool):
        ids = sorted(by_pool[pool])
        rng = random.Random(f"kata-sn22-deep|{UPSTREAM_COMMIT}|{seed}|{pool}")
        if len(ids) < MIN_DEEP_SAMPLES_PER_POOL:
            raise ManifestError(
                f"pool {pool} has {len(ids)} tasks, fewer than the "
                f"{MIN_DEEP_SAMPLES_PER_POOL} deep samples upstream requires")
        chosen.update(rng.sample(ids, MIN_DEEP_SAMPLES_PER_POOL))
    return frozenset(chosen)


def duel_manifests_match(king: EpochManifest, challenger: EpochManifest) -> None:
    """Both sides were asked the same thing, or raise.

    The two are normally the same object. This exists for the case where they are not -- a re-run,
    a retry, a manifest rebuilt from a record -- because "the same questions" is the assumption
    every comparison downstream rests on, and it is cheap to check and impossible to notice when
    it is wrong.
    """
    if king.digest() != challenger.digest():
        raise ManifestError("the two contestants were not asked the same questions")
    if king.deep_task_ids != challenger.deep_task_ids:
        raise ManifestError("the two contestants had different tasks deep-scored")
