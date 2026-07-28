"""The production winner path: real upstream validators, real aggregation, one pool tuple out.

Nothing here computes a score. Every number comes from a vendored upstream file, executed:

* ``AdvancedScraperValidator`` / ``XScraperValidator`` — the reward models and the nine/seven
  penalties;
* ``BaseScraperValidator.compute_rewards_and_penalties`` — the weighting, the component floors, the
  performance multiplier, the penalty product;
* ``QueryScheduler._score_one_type`` — cheap/deep aggregation into ``(q_gate, q_weight, volume,
  deep_count)``;
* ``combine_pool_scores`` — the pool shares.

What this module contributes is the four things upstream cannot know about a Kata duel:

1. **The answers already exist.** Upstream queries live miners over a chain. Kata's two contestants
   ran in sealed rooms before scoring began, so this builds upstream's own synapse objects from
   version-2 answers instead of dispatching for them.
2. **Kata chooses the deep samples.** Upstream samples them per-UID at scoring time, which would
   hand the King and the Challenger *different* deep tasks. Kata's manifest fixes them in advance
   and both contestants share the set, which is the whole fairness property.
3. **The judge is routed and pinned.** Upstream's ``call_scoring_llm`` falls back to
   ``gpt-4.1-nano`` when Chutes fails. Kata disables that: one contestant graded by Qwen and another
   by GPT is not one competition. A failed Chutes credential is a *credential failure* that zeroes
   the contestant, not a quietly different grader.
4. **Nothing is written anywhere.** No capacity table, no scoring store, no chain, no metrics.

**Why the scheduler is subclassed rather than reimplemented.** ``_score_one_type`` is where cheap
penalties become a multiplier and deep scores become a weighted mean. Rewriting it "equivalently"
would be a second implementation of the arithmetic that decides the duel. So Kata subclasses it and
overrides exactly two methods — the deep sample and the database write — and inherits the rest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from kata_sn22.credentials_v2 import CredentialReport, CredentialStatus
from kata_sn22.neuron_adapter import CHALLENGER_UID, KING_UID, KataNeuronAdapter
from kata_sn22.report_v2 import PoolResult
from kata_sn22.upstream_runtime import Axon, Dendrite, load, upstream_module

#: Which Kata pool maps to which upstream ``(search_type, mode)`` lane.
POOL_SEARCH_TYPE = {
    "ai_search:fast": "ai_search",
    "ai_search:balanced": "ai_search",
    "ai_search:deep": "ai_search",
    "x_search": "x_search",
}


class ScoringUnavailable(Exception):
    """The pool could not be scored. Never a zero dressed up as a score."""


# ---- the judge, routed through the broker and pinned to one model ---

@dataclass
class JudgeRouter:
    """Sends upstream's own judge prompts to Chutes through the trusted broker.

    ``call_chutes`` and ``call_openai`` in ``desearch/utils.py`` are transports: they build an
    HTTP request and return the text. Everything that decides *what the judge is asked* -- the
    prompt, the temperature, the model -- stays upstream's and stays real. Only the two transports
    are replaced, because Kata's providers are reached through the broker, which holds the
    contestant's key and never hands it out.

    ``call_openai`` is replaced with a refusal rather than a route. It exists in upstream solely as
    the fallback when Chutes fails, and taking it would mean grading one contestant with a
    different model from its opponent.
    """

    #: ``(messages) -> str``. The broker's ``chutes-score`` evaluator operation.
    chutes: object
    #: Provider statuses observed while judging, by provider. Read after scoring to decide whether
    #: this pool is an ``ok`` report or a ``credential_failure`` one.
    statuses: dict = field(default_factory=dict)

    def install(self) -> None:
        """Replace the two transports on the loaded upstream module."""
        utils = upstream_module("desearch.utils")
        utils.call_chutes = self._call_chutes
        utils.call_openai = self._refuse_openai

    async def _call_chutes(self, messages, temperature, model, seed=1234,
                           response_format=None):
        """Upstream's signature exactly, so its own caller is unmodified.

        Returning ``None`` is upstream's documented "Chutes failed" signal, and upstream reacts to
        it by falling back. Kata lets that happen and then refuses the fallback below -- rather than
        raising here, which upstream's ``query_llm`` would swallow into an empty string and score as
        a genuine LOW verdict.
        """
        try:
            answer = await _maybe_await(self.chutes(messages))
        except Exception as exc:  # noqa: BLE001 - classified, never propagated as a verdict
            self._record("chutes", exc)
            return None
        if not isinstance(answer, str) or not answer:
            self._record("chutes", None)
            return None
        self.statuses.setdefault("chutes", CredentialStatus.OK)
        return answer

    async def _refuse_openai(self, messages, model, temperature=1, response_format=None):
        """The disabled fallback.

        Reached only when Chutes has already failed. Recording it here turns "the judge went
        quiet" into "this contestant's Chutes credential failed", which is a zero with a reason
        rather than a silently bad score.
        """
        self.statuses["chutes"] = self.statuses.get("chutes") or CredentialStatus.INVALID
        return None

    def _record(self, provider: str, exc: object) -> None:
        self.statuses[provider] = _classify(exc)

    def credential_report(self, *, required: tuple = ("chutes",)) -> CredentialReport:
        statuses = {name: self.statuses.get(name, CredentialStatus.UNUSED) for name in required}
        return CredentialReport(statuses)


def _classify(exc: object) -> CredentialStatus:
    """A coarse status from the exception TYPE and any HTTP code -- never from its message, which
    can quote a request that carried the key."""
    code = getattr(exc, "code", None) or getattr(exc, "status", None)
    if isinstance(code, int):
        if code == 402:
            return CredentialStatus.PAYMENT_REQUIRED
        if code in (401, 403):
            return CredentialStatus.UNAUTHORIZED
        if code == 429:
            return CredentialStatus.RATE_LIMITED
        if 500 <= code < 600:
            return CredentialStatus.PROVIDER_OUTAGE
    return CredentialStatus.INVALID


async def _maybe_await(value):
    import inspect

    return await value if inspect.isawaitable(value) else value


# ---- the evaluator's own fetch, routed through the broker ---

@dataclass
class EvidenceRouter:
    """Routes upstream's page fetch through the broker's ``web-page-fetch`` operation.

    **This is not a convenience.** Upstream's ``WebSearchContentRelevanceModel`` does not accept
    evidence -- it *gathers* it, during scoring, by fetching every link a contestant returned and
    checking the contestant's own highlights against what came back. That is exactly the mechanism
    that makes a fabricated link score nothing, and it has to run.

    So the seam is the transport, not the evidence. ``BodyFetcher._fetch_and_extract`` is replaced
    with a call to the broker, which spends the contestant's ScrapingDog credential on its own
    verification, and everything above it -- the sampling, the caching, the ordering check, the
    judge prompt -- is upstream's own code, executed.

    An earlier version of this module took an ``evidence=`` argument and pre-populated
    ``validator_links``. It was wrong in a way worth recording: upstream overwrites that field, so
    the supplied evidence was silently discarded and every contestant scored zero on content
    relevance while the code looked correct.
    """

    #: ``(urls) -> {url: {"text": str, "title": str, ...}}``. The broker's evaluator operation.
    fetch_pages: object
    statuses: dict = field(default_factory=dict)

    def install(self) -> None:
        body_fetch = upstream_module("neurons.validators.apify.body_fetch")
        fetcher = body_fetch.get_body_fetcher()
        fetcher._fetch_and_extract = self._fetch_and_extract
        return fetcher

    async def _fetch_and_extract(self, urls: list) -> dict:
        """Upstream's own signature. A URL that cannot be fetched is simply absent from the result,
        which is what upstream's caller already expects and treats as an unverifiable source."""
        try:
            pages = await _maybe_await(self.fetch_pages(list(urls)))
        except Exception as exc:  # noqa: BLE001 - classified, never raised into a verdict
            self.statuses["scrapingdog"] = _classify(exc)
            return {}
        self.statuses.setdefault("scrapingdog", CredentialStatus.OK)
        return {
            url: {"text": str(record.get("text") or ""),
                  "title": str(record.get("title") or ""),
                  "published_date": str(record.get("published_date") or ""),
                  "author": str(record.get("author") or "")}
            for url, record in (pages or {}).items()
            if isinstance(record, dict) and record.get("text")
        }


# ---- streaming granularity, canonicalised on the trusted side ---

#: Upstream's ``MAX_TOKENS_PER_CHUNK``. A chunk above it costs 0.01 per excess token, summed across
#: every chunk and capped at 1.0 -- so a single 300-token chunk is a FULL penalty on its own.
STREAM_TOKENS_PER_CHUNK = 2


def canonical_text_chunks(text_chunks: dict) -> dict:
    """Re-split each role's text the way a token-by-token streaming miner would have emitted it.

    **Why the lane does this rather than the agent.** Upstream's ``text_chunks`` is what a validator
    *observed arriving over a wire*. Kata has no wire: an agent returns one document, so any chunk
    boundaries in it are an unverifiable self-report. Scoring that report would measure which
    contestant guessed the SDK's contract best, and an agent could claim perfect one-token chunks
    whatever it actually did.

    So the lane canonicalises, on the trusted side, where the real ``o200k_base`` tokenizer lives --
    the agent image is standard-library only and cannot carry one. Both contestants are chunked
    identically, which is the same reason the lane owns answer framing at all.

    **What the penalty still measures.** Emitting no summary at all for a task that asked for one is
    a full penalty, and canonicalisation does not touch that: an empty chunk list stays empty. What
    it removes is a penalty for prose style.

    **Nothing else changes.** ``texts`` is upstream's own join of these chunks, so re-splitting is
    exact -- the concatenation is byte-identical and the groundedness judge reads the same summary.
    A character whose own token count exceeds the limit still costs what it costs; no split can
    avoid that, and a real streaming miner would pay it too.
    """
    import tiktoken

    encoding = tiktoken.get_encoding("o200k_base")
    canonical: dict = {}
    for role, chunks in (text_chunks or {}).items():
        text = "".join(chunks or [])
        if not text:
            # Preserved rather than dropped: an empty role is not the same as an absent one, and
            # `not streamed_text_chunks` is the branch that carries the full penalty.
            canonical[str(role)] = list(chunks or [])
            continue
        canonical[str(role)] = _split_tokens(encoding, text)
    return canonical


def _split_tokens(encoding, text: str) -> list:
    """Contiguous groups of at most ``STREAM_TOKENS_PER_CHUNK`` tokens, joining back exactly.

    Grouping is done on token BYTES rather than by decoding each group, because a multi-byte
    character can straddle a token boundary; decoding a partial sequence would substitute a
    replacement character and the rejoined text would no longer be the contestant's answer.
    """
    pieces: list = []
    buffer, held = b"", 0
    for token in encoding.encode(text):
        buffer += encoding.decode_single_token_bytes(token)
        held += 1
        if held < STREAM_TOKENS_PER_CHUNK:
            continue
        try:
            pieces.append(buffer.decode("utf-8"))
        except UnicodeDecodeError:
            continue          # a character straddles the boundary; keep accumulating
        buffer, held = b"", 0
    if buffer:
        pieces.append(buffer.decode("utf-8", errors="replace"))
    return pieces


# ---- building upstream's own synapse objects from version-2 answers ---

def build_ai_synapse(task, answer: dict, *, process_time: float):
    """One ``ScraperStreamingSynapse``, as upstream's scorer expects to receive it.

    ``text_chunks`` rather than ``texts``: upstream derives ``texts`` from it as a property, so
    setting both would let the two disagree -- and the streaming penalty counts the chunks while
    the groundedness judge reads the joined text.
    """
    protocol = upstream_module("desearch.protocol")
    synapse = protocol.ScraperStreamingSynapse(
        prompt=task.prompt,
        model=_ai_model(protocol, task),
        count=task.count,
        tools=list(task.tools),
        result_type=protocol.ResultType(task.result_type.value),
        system_message=task.system_message or None,
        include_domains=list(task.include_domains) or None,
        exclude_domains=list(task.exclude_domains) or None,
        start_date=task.start_date,
        end_date=task.end_date,
        date_filter_type=task.date_filter_type,
        max_execution_time=task.limits.max_execution_time,
        mode=protocol.SearchMode(task.mode.value),
        scoring_model=protocol.ScoringModel(_scoring_model()),
    )
    synapse.completion = answer.get("completion") or ""
    # Canonicalised, not taken as given. See canonical_text_chunks: an agent's own chunk boundaries
    # are an unverifiable claim, and scoring them would measure prose style rather than answers.
    synapse.text_chunks = canonical_text_chunks(answer.get("text_chunks"))
    synapse.search_results = [
        protocol.SearchResultItem(**_search_result_fields(item))
        for item in (answer.get("search_results") or [])
    ]
    synapse.miner_tweets = list(answer.get("miner_tweets") or [])
    # Deliberately EMPTY. Upstream fills these during scoring by fetching every link the contestant
    # returned and re-scraping every tweet -- see EvidenceRouter. Pre-populating them would be
    # overwritten, and the contestant would score zero on content relevance for a reason nothing in
    # this repository would report.
    synapse.validator_links = []
    synapse.validator_tweets = []
    return _with_transport(synapse, process_time=process_time)


def build_x_synapse(task, answer: dict, *, process_time: float):
    """One ``TwitterSearchSynapse``."""
    protocol = upstream_module("desearch.protocol")
    synapse = protocol.TwitterSearchSynapse(
        query=task.query,
        sort=task.sort,
        user=task.user,
        count=task.count,
        start_date=task.start_date,
        end_date=task.end_date,
        lang=task.lang,
        verified=task.verified,
        blue_verified=task.blue_verified,
        is_quote=task.is_quote,
        is_video=task.is_video,
        is_image=task.is_image,
        min_retweets=task.min_retweets,
        min_replies=task.min_replies,
        min_likes=task.min_likes,
        max_execution_time=task.limits.max_execution_time,
    )
    synapse.results = list(answer.get("results") or [])
    synapse.validator_tweets = []          # gathered by upstream during scoring; see EvidenceRouter
    return _with_transport(synapse, process_time=process_time)


def _with_transport(synapse, *, process_time: float):
    """Attach what the room actually measured.

    ``process_time`` is not decoration: the performance reward and the timeout penalty are both
    computed from it, so this is how a measurement Kata took reaches upstream's own formula.
    """
    synapse.dendrite = Dendrite(process_time=float(process_time), status_code=200)
    synapse.axon = Axon()
    return synapse


def _ai_model(protocol, task):
    """Upstream's ``Model`` for a mode. Mapped rather than guessed: the model name drives
    ``get_max_execution_time``, which the timeout penalty measures against."""
    mapping = {"fast": protocol.Model.NOVA, "balanced": protocol.Model.ORBIT,
               "deep": protocol.Model.HORIZON}
    return mapping[task.mode.value]


def _scoring_model() -> str:
    from kata_sn22.protocol_v2 import FIXED_SCORING_MODEL

    return FIXED_SCORING_MODEL.value


def _search_result_fields(item: dict) -> dict:
    """Every field ``SearchResultItem`` declares -- and ALL of them matter.

    An earlier version of this function kept only ``title``, ``link`` and ``snippet``. That looked
    like sensible strictness and was a bug with teeth: ``highlights`` and ``text`` are what
    ``link_meets_evidence`` checks a contestant's claims against, so dropping them made content
    relevance score zero for every contestant, forever, with nothing to say why.

    Unknown keys are still dropped -- upstream's model would reject them, and a contestant should
    not fail validation for a field it invented.
    """
    fields: dict = {
        "title": str(item.get("title") or ""),
        "link": str(item.get("link") or ""),
        "snippet": str(item.get("snippet") or ""),
    }
    highlights = item.get("highlights")
    if isinstance(highlights, list):
        fields["highlights"] = [str(entry) for entry in highlights]
    for name in ("text", "published_date", "author"):
        value = item.get(name)
        if isinstance(value, str) and value:
            fields[name] = value
    return fields


# ---- Kata's scheduler: upstream's aggregation, Kata's deep samples ---

def kata_scheduler(validators: dict, deep_task_ids: frozenset, item_task_ids: dict):
    """A ``QueryScheduler`` with exactly two overrides.

    ``_sample_deep_synth`` — upstream samples 20% per UID at scoring time, which would hand the King
    and the Challenger different deep tasks. Kata's manifest fixed them in advance and both sides
    share the set.

    ``_record_quality`` — upstream writes to a capacity table and a miner database. Kata has
    neither, decides one promotion, and carries nothing between rounds.

    Everything else -- ``_score_one_type``, ``_run_full_scoring``, ``_item_mode`` -- is inherited
    and runs unmodified.
    """
    scheduler_module = upstream_module("neurons.validators.scoring.query_scheduler")

    class _KataQueryScheduler(scheduler_module.QueryScheduler):
        def __init__(self) -> None:  # noqa: D107 - deliberately does not call super()
            # Upstream's __init__ builds a scoring store, a miner database and a capacity table.
            # None exists here, and constructing them would be the chain-and-disk machinery this
            # adapter exists to avoid.
            self.validators = validators

        def _sample_deep_synth(self, synth_items: list) -> set:
            return {
                index for index, item in enumerate(synth_items)
                if item_task_ids.get(id(item["response"])) in deep_task_ids
            }

        def _sample_organic_deep(self, organic_items: list) -> set:
            return set()          # Kata has no organic traffic; every task is synthetic

        async def _record_quality(self, *_args, **_kwargs) -> None:
            return None

    return _KataQueryScheduler()


# ---- the one call the room makes ---

@dataclass(frozen=True)
class PoolScore:
    """One pool, both contestants, plus what the credentials did while it was scored."""

    pool: str
    king: PoolResult
    challenger: PoolResult
    credentials: CredentialReport

    def as_tuples(self) -> dict:
        return {KING_UID: self.king.as_tuple(), CHALLENGER_UID: self.challenger.as_tuple()}


async def score_pool(*, pool: str, tasks: tuple, king_answers: dict, challenger_answers: dict,
                     deep_task_ids: frozenset, judge, fetch_pages,
                     process_times: dict | None = None) -> PoolScore:
    """Score one pool for both contestants with the real upstream validator.

    ``king_answers`` / ``challenger_answers`` map ``task_id`` to a version-2 answer document.
    ``judge`` is ``(messages) -> str`` and ``fetch_pages`` is ``(urls) -> {url: record}`` -- the
    broker's two evaluator operations. Both are transports: upstream decides what to ask and what to
    fetch, and this only decides which credential pays for it.
    """
    load()
    search_type = POOL_SEARCH_TYPE.get(pool)
    if search_type is None:
        raise ScoringUnavailable(f"unknown pool {pool!r}")

    router = JudgeRouter(chutes=judge)
    router.install()
    evidence_router = EvidenceRouter(fetch_pages=fetch_pages)
    evidence_router.install()

    validator = _validator_for(search_type)
    process_times = process_times or {}

    items: list = []
    item_task_ids: dict = {}
    for task in tasks:
        for uid, answers in ((KING_UID, king_answers), (CHALLENGER_UID, challenger_answers)):
            answer = answers.get(task.task_id) or {}
            builder = build_x_synapse if search_type == "x_search" else build_ai_synapse
            synapse = builder(
                task, answer,
                process_time=float(process_times.get((uid, task.task_id), 0.0)))
            items.append({"uid": uid, "response": synapse})
            item_task_ids[id(synapse)] = task.task_id

    scheduler = kata_scheduler({search_type: validator}, deep_task_ids, item_task_ids)
    import datetime

    results_by_mode = await scheduler._score_one_type(
        search_type=search_type,
        synthetics={search_type: items},
        organics={search_type: []},
        time_range_start=datetime.datetime.now(datetime.UTC),
        window_start="",
        allocations_by_lane={},
    )

    by_uid = _single_mode(results_by_mode, pool)
    return PoolScore(
        pool=pool,
        king=_pool_result(by_uid.get(KING_UID)),
        challenger=_pool_result(by_uid.get(CHALLENGER_UID)),
        credentials=_merge_statuses(router, evidence_router),
    )


def _merge_statuses(*routers) -> CredentialReport:
    """One report covering every provider the EVALUATOR spent while scoring this pool.

    ``openai`` and ``apify`` are the agent's; they appear here as ``unused`` because the evaluator
    never touches the first and, for an AI pool, never touches the second.
    """
    statuses: dict = {}
    for router in routers:
        statuses.update(router.statuses)
    return CredentialReport({
        name: statuses.get(name, CredentialStatus.UNUSED)
        for name in ("scrapingdog", "apify", "openai", "chutes")
    })


def _validator_for(search_type: str):
    neuron = KataNeuronAdapter()
    if search_type == "x_search":
        module = upstream_module("neurons.validators.scrapers.x_scraper_validator")
        return module.XScraperValidator(neuron=neuron)
    module = upstream_module("neurons.validators.scrapers.advanced_scraper_validator")
    return module.AdvancedScraperValidator(neuron=neuron)


def _single_mode(results_by_mode: dict, pool: str) -> dict:
    """A Kata pool is one upstream mode, so ``_score_one_type`` returns one bucket.

    More than one would mean tasks from two modes reached the same pool job, and the pool weights
    would be applied to a mixture nobody chose.
    """
    populated = [value for value in results_by_mode.values() if value]
    if not populated:
        return {}
    if len(populated) > 1:
        raise ScoringUnavailable(
            f"pool {pool!r} produced {len(populated)} search modes; a pool job must carry exactly "
            f"one, or the pool share is applied to a mixture")
    return populated[0]


def _pool_result(tuple_or_none) -> PoolResult:
    """Upstream's four numbers, checked before anything downstream believes them."""
    if not tuple_or_none:
        # No deep samples scored: upstream drops the UID rather than scoring it low, and a zero here
        # says the same thing without pretending a number was measured.
        return PoolResult(0.0, 0.0, 0, 0)
    q_gate, q_weight, volume, deep_count = tuple_or_none
    for name, value in (("q_gate", q_gate), ("q_weight", q_weight)):
        if not math.isfinite(float(value)):
            raise ScoringUnavailable(f"upstream returned a non-finite {name}")
    return PoolResult(float(q_gate), float(q_weight), int(volume), int(deep_count))


def pool_share_key(pool: str):
    """Upstream's own ``POOL_SHARES`` key for a Kata pool name.

    Looked up in that dict rather than constructed. ``combine_pool_scores`` indexes by
    ``(SearchType, SearchMode)`` enum members, and enum identity is per-module: building the pair
    from ``desearch.protocol`` produced keys that compared unequal to the ones
    ``scoring.constants`` had used, so every lookup missed and the combined score came back empty --
    a silent zero rather than an error.
    """
    constants = upstream_module("neurons.validators.scoring.constants")
    search_type = POOL_SEARCH_TYPE[pool]
    mode = pool.split(":", 1)[1] if ":" in pool else None
    for key in constants.POOL_SHARES:
        key_type, key_mode = key
        if getattr(key_type, "value", key_type) != search_type:
            continue
        if getattr(key_mode, "value", key_mode) == mode:
            return key
    raise ScoringUnavailable(f"upstream declares no pool share for {pool!r}")


def combine(pool_scores: dict) -> dict:
    """Upstream's ``combine_pool_scores``, called with Kata's four pools.

    Not reimplemented. The pool shares and the within-pool normalisation are the arithmetic that
    turns four tuples into one number per contestant, and it decides the duel.
    """
    scheduler_module = upstream_module("neurons.validators.scoring.query_scheduler")
    qualities = {pool_share_key(pool): score.as_tuples() for pool, score in pool_scores.items()}
    return scheduler_module.combine_pool_scores(qualities)
