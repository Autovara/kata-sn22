"""The production winner path, driven through the REAL vendored upstream.

Every test here needs the pinned upstream to actually execute, so the whole module skips without
the ``upstream`` extra. That is deliberate: a version of these tests that "passed" against the port
would be testing the thing Phase F exists to stop using.

The load-bearing test is :func:`test_the_king_beats_a_fabricator`. It runs the agent
that ships in ``kata/submissions/`` through the real harness, frames it with the real synapse
builder, and scores it with the real ``AdvancedScraperValidator`` against a contestant whose links
do not exist.
Four separate defects were found by writing it, and each one had made every contestant score zero
while every unit test passed:

* ``tiktoken`` classified as infrastructure, so the streaming penalty counted nothing;
* ``highlights`` and ``text`` dropped when building the synapse, so no source could pass evidence;
* the King emitting three enormous chunks, which is a full streaming penalty;
* ``validator_identity`` missing from the neuron adapter, which raised inside
  ``compute_rewards_and_penalties`` and was swallowed into a pool of zeros.
"""

from __future__ import annotations

import ast
import asyncio
import io
import re
from pathlib import Path

import pytest

from kata_sn22 import upstream_runtime
from kata_sn22.neuron_adapter import (
    NEURON_SURFACE,
    REFUSED_CAPABILITIES,
    KataNeuronAdapter,
)
from kata_sn22.upstream_snapshot import snapshot_root

pytestmark = pytest.mark.skipif(
    not upstream_runtime.available(),
    reason="the real upstream needs the 'upstream' extra (uv sync --extra upstream)")

# The reigning King, under kings/ -- NOT submissions/, which holds miners' entries and nothing
# else. The King is also what the protocol tells miners to copy: a separate shipped example would be
# a second agent to keep correct, and miners would be copying one while scored against the other.
KING = (Path(__file__).resolve().parents[2] / "kata" / "kings" / "sn22__desearch" / "miner"
        / "agent.py")

SNIPPET = "The measured figure is 28 percent"
PAGE_BODY = f"Global emissions were measured in 2024. {SNIPPET}, recorded in July 2026."
WEB_RESULTS = [{"link": f"https://source-{index}.test/p", "title": f"Source {index}",
                "snippet": SNIPPET} for index in range(10)]

AI_TASK_INPUT = {
    "protocol_version": 2, "task_id": "t0", "search_type": "ai_search",
    "prompt": "what were 2024 emissions?", "mode": "fast",
    "result_type": "LINKS_WITH_FINAL_SUMMARY", "tools": ["Web Search"], "count": 10,
    "limits": {"max_execution_time": 15},
}


@pytest.fixture
def task():
    from kata_sn22.protocol_v2 import AiSearchTask, Limits, ResultType, SearchMode

    return AiSearchTask(
        task_id="t0", prompt=AI_TASK_INPUT["prompt"], mode=SearchMode.FAST,
        result_type=ResultType.LINKS_WITH_FINAL_SUMMARY, tools=("Web Search",), count=10,
        limits=Limits(max_execution_time=15))


@pytest.fixture
def king_answer(monkeypatch):
    """The shipped reference agent's real output, produced by the real harness."""
    from kata_sn22_sdk import broker as sdk_broker
    from kata_sn22_sdk import harness

    monkeypatch.setenv("SN22_BROKER_URL", "http://broker.internal")
    monkeypatch.setenv("SN22_BROKER_CAPABILITY", "kcap_" + "0" * 32)
    monkeypatch.delenv("SN22_RELAY_ENDPOINT", raising=False)
    monkeypatch.setattr(sdk_broker.BrokerClient, "_call",
                        lambda self, operation, payload: {"results": WEB_RESULTS})
    return harness.run(AI_TASK_INPUT, agent_path=str(KING), stderr=io.StringIO())


async def _fetch_pages(urls):
    """The evaluator's own fetch. A fabricated URL is simply absent, as a real fetch would be."""
    return {url: {"text": PAGE_BODY, "title": "Source"}
            for url in urls if "fabricated" not in url}


async def _judge(_messages):
    return "verdict: HIGH"


# ---- THE integration test ---

def test_the_king_beats_a_fabricator(task, king_answer):
    """Real agent, real harness, real upstream validator, real aggregation.

    The fabricator returns the same shape with links that do not exist, so the evaluator's fetch
    comes back empty for them and no source passes evidence. That is the whole mechanism by which
    inventing sources is worse than returning fewer.
    """
    from kata_sn22 import production_scorer as scorer

    fabricated = {**king_answer, "search_results": [
        {**source, "link": source["link"].replace("source-", "fabricated-")}
        for source in king_answer["search_results"]]}

    score = asyncio.run(scorer.score_pool(
        pool="ai_search:fast", tasks=(task,),
        king_answers={"t0": king_answer}, challenger_answers={"t0": fabricated},
        deep_task_ids=frozenset({"t0"}), judge=_judge, fetch_pages=_fetch_pages,
        process_times={(0, "t0"): 3.0, (1, "t0"): 3.0}))

    assert score.king.q_weight > 0.0, "the seeded King scores nothing"
    assert score.challenger.q_weight == 0.0, "fabricated sources scored"
    assert score.king.q_weight > score.challenger.q_weight
    assert score.credentials.as_dict()["chutes"] == "ok"


def test_no_penalty_fires_on_the_king(task, king_answer):
    """It is the floor every miner copies. A penalty that fires on it makes the floor zero, and a
    miner would have no way to tell their own work from the template's."""
    import numpy as np

    from kata_sn22 import production_scorer as scorer

    scorer.JudgeRouter(chutes=_judge).install()
    scorer.EvidenceRouter(fetch_pages=_fetch_pages).install()
    validator = scorer._validator_for("ai_search")
    synapse = scorer.build_ai_synapse(task, king_answer, process_time=3.0)

    async def _apply(penalty):
        _raw, _adjusted, applied = await penalty.apply_penalties([synapse], np.array([0]), {})
        return float(np.asarray(applied)[0])

    fired = [(p.name, value) for p in validator.penalty_functions
             if (value := asyncio.run(_apply(p))) != 1.0]
    assert not fired, f"penalties fired on the King: {fired}"


def test_both_reward_models_score_the_king(task, king_answer):
    import numpy as np

    from kata_sn22 import production_scorer as scorer

    scorer.JudgeRouter(chutes=_judge).install()
    scorer.EvidenceRouter(fetch_pages=_fetch_pages).install()
    validator = scorer._validator_for("ai_search")
    synapse = scorer.build_ai_synapse(task, king_answer, process_time=3.0)

    async def _apply(reward):
        events, _, _, _ = await reward.apply([synapse], np.array([0]))
        return float(np.asarray(events)[0])

    for reward in validator.reward_functions:
        assert asyncio.run(_apply(reward)) > 0.0, f"{reward.name} scored the King zero"


# ---- fix: the evidence a source needs to be scoreable at all ---

def test_the_king_cites_every_source(king_answer):
    """Without ``highlights`` and ``text`` a source is dropped before it is judged -- it does not
    score badly, it does not score at all."""
    sources = king_answer["search_results"]
    assert sources
    for source in sources:
        assert source.get("highlights"), source
        assert source.get("text"), source


def test_the_synapse_builder_carries_the_evidence_fields(task):
    """It once kept only title/link/snippet. That looked like sensible strictness and made content
    relevance zero for every contestant, forever."""
    from kata_sn22 import production_scorer as scorer

    answer = {"completion": "c", "text_chunks": {"summary": ["c"]}, "search_results": [
        {"title": "T", "link": "https://a.test", "snippet": "s",
         "highlights": [SNIPPET], "text": SNIPPET, "published_date": "2026-07-01"}]}
    synapse = scorer.build_ai_synapse(task, answer, process_time=1.0)
    built = synapse.search_results[0]
    assert built.highlights == [SNIPPET]
    assert built.text == SNIPPET
    assert built.published_date == "2026-07-01"


def test_cite_satisfies_upstreams_own_evidence_check():
    """Checked against the real ``link_meets_evidence`` rather than against a restatement of it."""
    from kata_sn22_sdk import cite

    evidence = upstream_runtime.upstream_module(
        "neurons.validators.reward.search_content_relevance")
    source = cite({"link": "https://a.test", "snippet": SNIPPET}, [SNIPPET])

    assert evidence.link_meets_evidence(source["highlights"], source["text"], PAGE_BODY)
    # ...and a fabricated quote does not, which is the point of the check.
    invented = cite({"link": "https://a.test"}, ["a sentence nobody wrote"])
    assert not evidence.link_meets_evidence(
        invented["highlights"], invented["text"], PAGE_BODY)


def test_cite_drops_empty_quotes():
    from kata_sn22_sdk import cite

    assert cite({}, ["", "  ", SNIPPET])["highlights"] == [SNIPPET]


# ---- fix: streaming granularity ---

def test_canonical_chunks_join_back_to_exactly_the_original(task):
    """The summary the groundedness judge reads is upstream's join of these chunks. A re-split that
    changed a byte would change the answer a contestant is graded on."""
    from kata_sn22 import production_scorer as scorer

    for text in ("**2024**\n\n- [S](https://s.test): the figure is 28 percent.",
                 "Ünïcödé 🌍 排出量の測定 𐀀 mixed with ascii",
                 "x"):
        canonical = scorer.canonical_text_chunks({"summary": [text]})
        assert "".join(canonical["summary"]) == text


def test_canonical_chunks_are_within_upstreams_token_limit():
    """Above two tokens costs 0.01 per excess token, summed and capped at 1.0 -- so one long chunk
    is a full penalty on its own."""
    import tiktoken

    from kata_sn22 import production_scorer as scorer

    encoding = tiktoken.get_encoding("o200k_base")
    text = "**2024 emissions**\n\n" + "\n".join(
        f"- [Source {i}](https://source-{i}.test/p): {SNIPPET}" for i in range(10))
    chunks = scorer.canonical_text_chunks({"summary": [text]})["summary"]
    assert chunks
    assert max(len(encoding.encode(chunk)) for chunk in chunks) <= scorer.STREAM_TOKENS_PER_CHUNK


def test_an_empty_summary_stays_empty_and_still_takes_the_full_penalty(task):
    """Canonicalisation must not manufacture a chunk. Producing no summary for a task that asked
    for one is the one thing the streaming penalty can genuinely observe about a Kata agent."""
    from kata_sn22 import production_scorer as scorer

    assert scorer.canonical_text_chunks({}) == {}
    assert scorer.canonical_text_chunks({"summary": []}) == {"summary": []}


def test_the_streaming_penalty_still_fires_when_nothing_was_streamed(task):
    import numpy as np

    from kata_sn22 import production_scorer as scorer

    validator = scorer._validator_for("ai_search")
    streaming = next(p for p in validator.penalty_functions if "streaming" in p.name)
    silent = scorer.build_ai_synapse(
        task, {"completion": "", "text_chunks": {}, "search_results": []}, process_time=1.0)

    _raw, _adjusted, applied = asyncio.run(
        streaming.apply_penalties([silent], np.array([0]), {}))
    assert float(np.asarray(applied)[0]) == 0.0, "a silent agent escaped the streaming penalty"


# ---- the neuron surface, re-derived from the pinned tree ---

def test_the_neuron_surface_covers_every_attribute_upstream_reads():
    """Derived from BOTH spellings. The first derivation scanned only ``self.neuron.<attr>`` and
    missed ``neurons/validators/clients/``, which calls the same object ``owner``. The two names it
    missed raised inside ``compute_rewards_and_penalties``, were swallowed by ``_score_one_type``,
    and turned every contestant's pool into zeros with nothing in the logs but "Full scoring
    failed".
    """
    read: set = set()
    for package in ("reward", "penalty", "scrapers", "clients", "scoring"):
        directory = snapshot_root() / "neurons" / "validators" / package
        if not directory.is_dir():
            continue
        for path in directory.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            read |= set(re.findall(r"self\.neuron\.([a-z_]+(?:\.[a-z_]+)*)", source))
            read |= set(re.findall(r"\bowner\.([a-z_]+(?:\.[a-z_]+)*)", source))

    # Reached through getattr with a default, or a prefix of something already declared. Listed
    # explicitly so adding one is a decision rather than a widened regex.
    not_required = {"metagraph", "config"}
    # Every read name must be either SUPPLIED or explicitly REFUSED. "Absent" is not an option:
    # that is what produced a silent pool of zeros.
    declared = set(NEURON_SURFACE) | set(REFUSED_CAPABILITIES)
    missing = {
        name for name in read - declared - not_required
        if not any(name.startswith(f"{known}.") or known.startswith(f"{name}.")
                   for known in declared)
    }
    assert not missing, f"upstream reads neuron attribute(s) the adapter does not supply: {missing}"


def test_the_adapter_supplies_every_declared_attribute():
    adapter = KataNeuronAdapter()
    for path in NEURON_SURFACE:
        target = adapter
        for part in path.split("."):
            assert hasattr(target, part), f"the adapter has no {path}"
            target = getattr(target, part)


def test_the_metagraph_is_exactly_two_virtual_uids():
    from kata_sn22.neuron_adapter import VIRTUAL_UIDS

    adapter = KataNeuronAdapter()
    assert len(adapter.metagraph.hotkeys) == len(VIRTUAL_UIDS) == 2
    assert [axon.hotkey for axon in adapter.metagraph.axons] == adapter.metagraph.hotkeys


def test_the_logger_can_resolve_a_hotkey_without_a_chain():
    """Upstream's logger ITERATES ``metagraph.axons`` to find a coldkey. An earlier adapter refused
    that blanketly, which broke scoring entirely -- the querying path is guarded by
    ``get_random_miner``, not by making a bookkeeping read explode."""
    adapter = KataNeuronAdapter()
    coldkeys = [axon.coldkey for axon in adapter.metagraph.axons]
    assert coldkeys and all(coldkeys)


@pytest.mark.parametrize("capability", sorted(REFUSED_CAPABILITIES))
def test_a_disabled_capability_raises_rather_than_returning_none(capability):
    """A stub returning ``None`` lets a path Kata must never take run to completion and produce a
    number nobody notices is wrong. An ABSENT one is worse still: it raises AttributeError deep
    inside upstream, which upstream catches and turns into zeros."""
    from kata_sn22.neuron_adapter import DisabledCapability

    adapter = KataNeuronAdapter()
    with pytest.raises(DisabledCapability):
        attribute = getattr(adapter, capability)         # properties raise here
        asyncio.run(attribute())                          # methods raise here


def test_the_adapter_carries_no_wallet_and_no_chain_client():
    adapter = KataNeuronAdapter()
    assert adapter.wallet is None
    assert adapter.subtensor is None
    assert adapter.utility_api is None          # upstream's own "API not configured" branch
    assert adapter.config.wandb_on is False
    provenance = adapter.as_provenance()
    assert provenance["chain_writes"] is False and provenance["public_api"] is False


# ---- the runtime executes the real thing ---

def test_every_scoring_module_is_the_pinned_upstream_file():
    upstream_runtime.load()
    for name in upstream_runtime.SCORING_MODULES:
        upstream_runtime.upstream_module(name)
    upstream_runtime.assert_scoring_is_real()          # raises if any one is not


def test_no_scoring_module_is_in_the_adapted_set():
    """One extra entry in ``ADAPTED_MODULES`` would replace a penalty with something that always
    returns 1.0, and every score would still look plausible."""
    overlap = set(upstream_runtime.ADAPTED_MODULES) & set(upstream_runtime.SCORING_MODULES)
    assert not overlap, overlap


def test_tiktoken_is_required_rather_than_adapted():
    """It counts the tokens the streaming penalty charges for. Adapting it was a real mistake, and
    the raising adapter caught it on the first import."""
    assert "tiktoken" in upstream_runtime.REQUIRED_REAL_PACKAGES
    assert "tiktoken" not in upstream_runtime.ADAPTED_MODULES


def test_the_disabled_infrastructure_raises_when_touched():
    upstream_runtime.load()
    import wandb
    from bittensor.utils import weight_utils

    for call in (lambda: wandb.log({}), lambda: weight_utils.process_weights()):
        with pytest.raises(upstream_runtime.UpstreamUnavailable):
            call()


# ---- the judge is pinned to one model ---

def test_the_fallback_to_a_different_model_is_refused():
    """Upstream falls back to ``gpt-4.1-nano`` when Chutes fails. One contestant graded by Qwen and
    another by GPT is not one competition, so Kata records a credential failure instead."""
    from kata_sn22 import production_scorer as scorer

    async def _broken(_messages):
        raise RuntimeError("chutes is down")

    router = scorer.JudgeRouter(chutes=_broken)
    router.install()
    utils = upstream_runtime.upstream_module("desearch.utils")

    result = asyncio.run(
        utils.call_scoring_llm(messages=[], model="Qwen/Qwen3.5-397B-A17B-TEE"))

    assert result is None, "the judge fell back to a different model"
    assert router.credential_report().as_dict()["chutes"] != "ok"


def test_a_working_judge_reports_ok():
    from kata_sn22 import production_scorer as scorer

    router = scorer.JudgeRouter(chutes=_judge)
    router.install()
    utils = upstream_runtime.upstream_module("desearch.utils")

    answer = asyncio.run(
        utils.call_scoring_llm(messages=[], model="Qwen/Qwen3.5-397B-A17B-TEE"))
    assert answer == "verdict: HIGH"
    assert router.credential_report().as_dict()["chutes"] == "ok"


# ---- aggregation is upstream's ---

def test_the_pool_share_keys_come_from_upstreams_own_table():
    """Built by lookup, not construction. Enum identity is per-module, so a pair built from
    ``desearch.protocol`` compared unequal to the ones ``scoring.constants`` used -- every lookup
    missed and the combined score came back empty, which reads as "nobody scored"."""
    from kata_sn22 import production_scorer as scorer

    constants = upstream_runtime.upstream_module("neurons.validators.scoring.constants")
    for pool in ("ai_search:fast", "ai_search:balanced", "ai_search:deep", "x_search"):
        assert scorer.pool_share_key(pool) in constants.POOL_SHARES


def test_the_pool_shares_match_the_scorer_policy():
    """Phase A pinned 54/18/18/10 into the policy hash. This is where that claim is checked against
    upstream's own table rather than against itself."""
    from kata_sn22 import production_scorer as scorer
    from kata_sn22.scorer_policy import POOL_WEIGHTS

    constants = upstream_runtime.upstream_module("neurons.validators.scoring.constants")
    for pool, weight in POOL_WEIGHTS.items():
        assert constants.POOL_SHARES[scorer.pool_share_key(pool)] == pytest.approx(weight)


def test_the_scheduler_overrides_only_the_deep_sample_and_the_database_write():
    """``_score_one_type`` is where cheap penalties become a multiplier and deep scores a weighted
    mean. A second implementation of it would be a second arithmetic deciding the duel."""
    from kata_sn22 import production_scorer as scorer

    scheduler = scorer.kata_scheduler({}, frozenset(), {})
    # Dunders vary by interpreter version (3.13 adds __firstlineno__ and __static_attributes__),
    # so compare only what was deliberately defined.
    overridden = {name for name in type(scheduler).__dict__ if not name.startswith("__")}
    assert overridden == {"_sample_deep_synth", "_sample_organic_deep", "_record_quality"}
    # ...and the arithmetic is inherited, not redefined.
    scheduler_module = upstream_runtime.upstream_module(
        "neurons.validators.scoring.query_scheduler")
    for inherited in ("_score_one_type", "_run_full_scoring", "_item_mode"):
        assert getattr(type(scheduler), inherited) is getattr(
            scheduler_module.QueryScheduler, inherited), inherited


def test_production_does_not_import_the_seven_signal_comparator():
    """Kata's own comparator ranks on seven hand-chosen signals. It is calibration machinery, and a
    production winner path that reached for it would be scoring SN22 Kata's way rather than SN22's.
    """
    source = Path(
        __import__("kata_sn22.production_scorer", fromlist=["x"]).__file__
    ).read_text(encoding="utf-8")
    imported: set = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "kata_sn22.scoring" not in imported
    for banned in ("RANK_SIGNALS", "compare_signals", "score_attempts", "Signals"):
        assert banned not in source, banned


# ---- the tweet re-scrape, routed through the broker ---

#: A complete Apify item. Upstream's ``toTwitterScraperTweet`` is strict -- it rejects a thin one --
#: so a fixture that omitted fields would silently produce no tweets and look like a routing bug.
def _tweet_item(tweet_id: str) -> dict:
    return {
        "url": f"https://x.com/a/status/{tweet_id}", "id": tweet_id, "text": "hello",
        "createdAt": "2024-01-01", "likeCount": 1, "retweetCount": 0, "replyCount": 0,
        "quoteCount": 0, "viewCount": 1, "bookmarkCount": 0, "isRetweet": False,
        "isQuote": False, "lang": "en",
        "author": {"id": "u1", "userName": "a", "name": "A", "createdAt": "2020-01-01",
                   "followers": 1, "following": 1, "favouritesCount": 0, "listedCount": 0,
                   "mediaCount": 0, "statusesCount": 1, "isVerified": False,
                   "isBlueVerified": False, "profilePicture": "", "coverPicture": "",
                   "description": "", "location": "", "url": "", "canDm": False},
    }


def test_the_tweet_rescrape_goes_through_the_broker_and_upstreams_own_mapper():
    """The model the field-by-field comparison reads must be built by UPSTREAM's mapper from the
    provider's fields. One built by Kata would be checking a contestant against Kata's idea of a
    tweet rather than the provider's."""
    from kata_sn22 import production_scorer as scorer

    requested: list = []

    async def _rescrape(tweet_ids):
        requested.append(list(tweet_ids))
        return [_tweet_item(tweet_id) for tweet_id in tweet_ids]

    router = scorer.TweetRouter(rescrape=_rescrape)
    router.install()
    actor = upstream_runtime.upstream_module(
        "neurons.validators.apify.twitter_scraper_actor")

    tweets = asyncio.run(
        actor.TwitterScraperActor().get_tweets(urls=["https://x.com/a/status/12345"]))

    assert requested == [["12345"]], "the broker was not asked for the tweet the agent returned"
    assert [tweet.id for tweet in tweets] == ["12345"]
    assert isinstance(tweets[0], actor.TwitterScraperTweet)
    assert router.statuses["apify"].value == "ok"


def test_one_malformed_item_is_skipped_rather_than_failing_the_rescrape():
    """A provider returning one odd row is not a credential failure, and treating it as one would
    defer a duel over a single tweet."""
    from kata_sn22 import production_scorer as scorer

    async def _rescrape(tweet_ids):
        return [{"url": "https://x.com/a/status/1", "id": "1"},      # too thin for the mapper
                _tweet_item("2")]

    router = scorer.TweetRouter(rescrape=_rescrape)
    router.install()
    actor = upstream_runtime.upstream_module(
        "neurons.validators.apify.twitter_scraper_actor")

    tweets = asyncio.run(actor.TwitterScraperActor().get_tweets(
        urls=["https://x.com/a/status/1", "https://x.com/a/status/2"]))

    assert [tweet.id for tweet in tweets] == ["2"]
    assert router.statuses["apify"].value == "ok"


def test_a_failed_rescrape_is_recorded_as_a_credential_status():
    from kata_sn22 import production_scorer as scorer

    async def _broken(_tweet_ids):
        raise RuntimeError("apify is down")

    router = scorer.TweetRouter(rescrape=_broken)
    router.install()
    actor = upstream_runtime.upstream_module(
        "neurons.validators.apify.twitter_scraper_actor")

    assert asyncio.run(
        actor.TwitterScraperActor().get_tweets(urls=["https://x.com/a/status/1"])) == []
    assert router.statuses["apify"].value != "ok"


# ---- GATE: every penalty fires in one case and rests in another ---
#
# A penalty that never fires is one Kata has quietly disabled by building its input wrong -- exactly
# what happened to the streaming penalty, which fired on EVERY contestant including the reference
# agent. A penalty that never rests is one no honest agent can avoid. Both are silent, and both make
# a duel measure something other than the answers.

def _healthy_answer() -> dict:
    """An answer that should rest every AI penalty. The baseline the cases below deviate from."""
    sources = [{"title": f"Source {i}", "link": f"https://source-{i}.test/p",
                "snippet": SNIPPET, "highlights": [SNIPPET], "text": SNIPPET}
               for i in range(10)]
    summary = "**2024 emissions**\n\n" + "\n".join(
        f"- [Source {i}](https://source-{i}.test/p): {SNIPPET}" for i in range(10))
    return {"completion": summary, "text_chunks": {"summary": [summary]},
            "search_results": sources, "miner_tweets": []}


def _applied(penalty, synapse) -> float:
    import numpy as np

    async def _run():
        _raw, _adjusted, applied = await penalty.apply_penalties([synapse], np.array([0]), {})
        return float(np.asarray(applied)[0])

    return asyncio.run(_run())


#: ``name -> (mutate the answer, unused, process_time)``. Each provokes one penalty.
AI_PENALTY_CASES = {
    "streaming_penalty": (
        lambda a: {**a, "text_chunks": {}, "completion": ""}, None, 3.0),
    "timeout_penalty": (lambda a: a, None, 10_000.0),
    "min_realistic_time_penalty": (lambda a: a, None, 0.0001),
    "count_penalty": (
        lambda a: {**a, "search_results": a["search_results"][:1]}, None, 3.0),
    "summary_structure_penalty": (
        lambda a: {**a, "text_chunks": {"summary": ["no markdown links at all"]},
                   "completion": "no markdown links at all"}, None, 3.0),
    "duplicate_results_penalty": (
        lambda a: {**a, "search_results": [a["search_results"][0]] * 10}, None, 3.0),
    # An EMPTY required field, not a malformed URL. ``_is_valid_search_item`` only URL-validates
    # raw dicts; for the model objects a synapse carries it requires title/link/snippet non-empty.
    "result_schema_penalty": (
        lambda a: {**a, "search_results": [{**s, "snippet": ""}
                                           for s in a["search_results"]]}, None, 3.0),
}


@pytest.mark.parametrize("penalty_name", sorted(AI_PENALTY_CASES))
def test_every_ai_penalty_fires_on_a_bad_answer(task, penalty_name):
    from kata_sn22 import production_scorer as scorer

    mutate, _task_mutation, process_time = AI_PENALTY_CASES[penalty_name]
    scorer.JudgeRouter(chutes=_judge).install()
    scorer.EvidenceRouter(fetch_pages=_fetch_pages).install()
    validator = scorer._validator_for("ai_search")
    penalty = next(p for p in validator.penalty_functions if p.name == penalty_name)

    synapse = scorer.build_ai_synapse(task, mutate(_healthy_answer()),
                                      process_time=process_time)
    assert _applied(penalty, synapse) < 1.0, f"{penalty_name} did not fire on a bad answer"


@pytest.mark.parametrize("penalty_name", sorted(AI_PENALTY_CASES))
def test_every_ai_penalty_rests_on_a_healthy_answer(task, penalty_name):
    from kata_sn22 import production_scorer as scorer

    scorer.JudgeRouter(chutes=_judge).install()
    scorer.EvidenceRouter(fetch_pages=_fetch_pages).install()
    validator = scorer._validator_for("ai_search")
    penalty = next(p for p in validator.penalty_functions if p.name == penalty_name)

    synapse = scorer.build_ai_synapse(task, _healthy_answer(), process_time=3.0)
    assert _applied(penalty, synapse) == 1.0, f"{penalty_name} fires on a healthy answer"


def test_the_penalty_matrix_covers_the_penalties_it_can_provoke(task):
    """Names the coverage rather than implying it is complete.

    ``date_range_penalty`` and ``domain_filter_penalty`` need a task carrying a date window or a
    domain filter, which the epoch generator does produce but this fixture does not; they are
    covered by the "rests" half only. Saying so beats a matrix that looks exhaustive and is not.
    """
    from kata_sn22 import production_scorer as scorer

    validator = scorer._validator_for("ai_search")
    declared = {penalty.name for penalty in validator.penalty_functions}
    uncovered = declared - set(AI_PENALTY_CASES)

    assert uncovered == {"date_range_penalty", "domain_filter_penalty"}, uncovered
    # ...and those two still have to REST on a healthy answer, which is what a silent
    # always-firing penalty would break.
    scorer.EvidenceRouter(fetch_pages=_fetch_pages).install()
    synapse = scorer.build_ai_synapse(task, _healthy_answer(), process_time=3.0)
    for name in uncovered:
        penalty = next(p for p in validator.penalty_functions if p.name == name)
        assert _applied(penalty, synapse) == 1.0, f"{name} fires on a healthy answer"


def test_the_seeded_king_emits_its_prose_by_role(king_answer):
    """The King must EMIT rather than return, and emit each graded role.

    The streaming penalty itself is neutralised on the trusted side -- ``canonical_text_chunks``
    re-chunks whatever arrives -- so this is not about chunk counts. What the trusted side cannot
    invent is a role the agent never emitted: a missing ``final_summary`` is the block the
    groundedness judge reads simply not existing, which scores zero rather than badly.
    """
    from kata_sn22_sdk import ScraperTextRole

    # Read the wire names off the enum rather than restating them: FINAL_SUMMARY serialises as
    # "summary", and a test that hardcoded the member name would pass on an agent emitting nothing.
    expected = {role.value for role in
                (ScraperTextRole.INTRO, ScraperTextRole.SEARCH_SUMMARY,
                 ScraperTextRole.FINAL_SUMMARY)}
    chunks = king_answer.get("text_chunks") or {}
    assert set(chunks) == expected, sorted(chunks)
    assert all(any(part.strip() for part in parts) for parts in chunks.values()), (
        "the King emitted an empty role")
