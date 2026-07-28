"""The version-2 agent SDK, the harness, and the reference submission.

The load-bearing test here is :func:`test_the_reference_agent_completes_every_pool`: it runs the
agent that actually ships in ``kata/submissions/`` through the real harness, for all four pools, and
parses each answer with the trusted side's own parser. Nothing else in this repository connects
those three things, and each of them alone can be correct while the set is broken — which is exactly
the shape of every serious defect this project has had.

The broker is faked at the transport, so no test reaches a provider. What is NOT faked is the
harness, the framing, the models or the agent.
"""

from __future__ import annotations

import io
import json
import textwrap
from pathlib import Path

import pytest

from kata_sn22 import protocol_v2 as v2
from kata_sn22_sdk import (
    Agent,
    AiSearchResult,
    AiSearchSynapse,
    BrokerError,
    Emit,
    ScraperTextRole,
    SdkError,
    XSearchResult,
    XSearchSynapse,
    harness,
    synapse_from_input,
)
from kata_sn22_sdk import broker as sdk_broker

# What the protocol tells miners to copy: the reigning King, under kings/. There is no shipped
# example submission -- a second agent to keep correct is one miners would copy while being scored
# against the other.
REFERENCE = (Path(__file__).resolve().parents[2] / "kata" / "kings" / "sn22__desearch" / "miner"
             / "agent.py")

#: The four pools, exactly as a production epoch runs them.
POOL_TASKS = {
    "ai_search:fast": {
        "protocol_version": 2, "task_id": "t-fast", "search_type": "ai_search",
        "prompt": "what were 2024 emissions?", "mode": "fast",
        "result_type": "LINKS_WITH_FINAL_SUMMARY", "tools": ["Web Search"], "count": 10,
        "limits": {"max_execution_time": 30},
    },
    "ai_search:balanced": {
        "protocol_version": 2, "task_id": "t-balanced", "search_type": "ai_search",
        "prompt": "what were 2024 emissions?", "mode": "balanced",
        "result_type": "ONLY_LINKS", "tools": ["Web Search"], "count": 10,
        "limits": {"max_execution_time": 60},
    },
    "ai_search:deep": {
        "protocol_version": 2, "task_id": "t-deep", "search_type": "ai_search",
        "prompt": "what were 2024 emissions?", "mode": "deep",
        "result_type": "LINKS_WITH_FINAL_SUMMARY", "tools": ["Web Search", "X Search"],
        "count": 10, "limits": {"max_execution_time": 120},
    },
    "x_search": {
        "protocol_version": 2, "task_id": "t-x", "search_type": "x_search",
        "query": "emissions 2024", "count": 10, "sort": "Latest",
        "limits": {"max_execution_time": 30},
    },
}

WEB_RESULTS = [{"link": f"https://source-{index}.test/page", "title": f"Source {index}",
                "snippet": f"Reported figure number {index} for the year."} for index in range(12)]
TWEETS = [{"id": str(1000 + index), "text": f"tweet {index}", "created_at": "2024-01-01"}
          for index in range(12)]


@pytest.fixture
def room(monkeypatch):
    """Put the SDK in "sealed room" mode and answer its three operations without a network."""
    monkeypatch.delenv(sdk_broker.RELAY_ENDPOINT_ENV, raising=False)
    monkeypatch.setenv(sdk_broker.BROKER_URL_ENV, "http://broker.internal")
    monkeypatch.setenv(sdk_broker.BROKER_CAPABILITY_ENV, "kcap_" + "0" * 32)

    seen: list = []

    def _fake_call(self, operation, payload):
        seen.append((operation, payload))
        if operation == sdk_broker.OP_WEB_SEARCH:
            return {"results": WEB_RESULTS[:payload.get("count", 10)]}
        if operation == sdk_broker.OP_X_SEARCH:
            return {"results": TWEETS[:payload.get("count", 10)]}
        return {"content": "a summary", "model": "gpt-4.1-nano"}

    monkeypatch.setattr(sdk_broker.BrokerClient, "_call", _fake_call)
    return seen


def _run_reference(document: dict, stderr=None) -> dict:
    return harness.run(document, agent_path=str(REFERENCE),
                       stderr=stderr if stderr is not None else io.StringIO())


# ---- THE exit gate: the reference agent completes all four pools -----------------------------

def test_the_reference_submission_exists_where_miners_are_told_to_copy_it():
    assert REFERENCE.is_file(), f"no reference submission at {REFERENCE}"


@pytest.mark.parametrize("pool", sorted(POOL_TASKS))
def test_the_reference_agent_completes_every_pool(room, pool):
    """All four, through the real harness, parsed by the trusted side's own parser.

    "Parsed by the trusted side's own parser" is the part that matters: an answer the harness is
    happy with but ``kata_sn22.protocol_v2`` rejects is an answer that scores nothing, and the two
    live in different packages precisely so that neither can quietly define the contract alone.
    """
    document = POOL_TASKS[pool]
    answer = _run_reference(document)

    task = _trusted_task(document)
    parsed = v2.parse_answer(answer, task=task)
    assert parsed.task_id == document["task_id"]

    if document["search_type"] == "x_search":
        assert len(parsed.results) == document["count"]
    else:
        assert len(parsed.search_results) == document["count"]


def _trusted_task(document: dict):
    """The same task, built with the TRUSTED side's models."""
    limits = v2.Limits(**document["limits"])
    if document["search_type"] == "x_search":
        return v2.XSearchTask(task_id=document["task_id"], query=document["query"],
                              count=document["count"], sort=document.get("sort"), limits=limits)
    return v2.AiSearchTask(
        task_id=document["task_id"], prompt=document["prompt"],
        mode=v2.SearchMode(document["mode"]), result_type=v2.ResultType(document["result_type"]),
        tools=tuple(document["tools"]), count=document["count"], limits=limits)


def test_the_reference_agent_streams_rather_than_returning_one_block(room):
    """Upstream's streaming penalty counts tokens per emitted chunk. An agent that computed one
    string and returned it takes that penalty for a difference unrelated to answer quality."""
    answer = _run_reference(POOL_TASKS["ai_search:fast"])

    assert len(answer["chunks"]) >= 2
    assert answer["texts"][ScraperTextRole.FINAL_SUMMARY.value]
    # The derived fields must agree with what was streamed, or the penalty was computed against
    # something the scorer never saw.
    assert answer["completion"] == "".join(chunk["text"] for chunk in answer["chunks"])
    for role, texts in answer["text_chunks"].items():
        assert answer["texts"][role] == "".join(texts)


def test_the_reference_agent_writes_no_summary_for_an_only_links_task(room):
    """The AI quality split reweights to (1.0, 0.0), so a summary here is graded by nobody and paid
    for by the contestant."""
    answer = _run_reference(POOL_TASKS["ai_search:balanced"])
    assert answer["chunks"] == []
    assert answer["completion"] == ""
    assert len(answer["search_results"]) == 10


def test_the_reference_agent_returns_tweets_unedited(room):
    """The validator re-scrapes each one and compares field by field."""
    answer = _run_reference(POOL_TASKS["x_search"])
    assert answer["results"] == TWEETS[:10]


def test_the_reference_agent_never_returns_a_duplicate_link(room, monkeypatch):
    """A repeated link takes the duplicate penalty, so padding a short list with copies is worse
    than returning fewer."""
    repeated = [{"link": "https://same.test", "title": "S", "snippet": "s"}] * 12

    def _repeat(self, operation, payload):
        return {"results": repeated}

    monkeypatch.setattr(sdk_broker.BrokerClient, "_call", _repeat)
    answer = _run_reference(POOL_TASKS["ai_search:fast"])
    links = [item["link"] for item in answer["search_results"]]
    assert len(links) == len(set(links))


def test_the_reference_agent_answers_rather_than_crashing_when_the_broker_refuses(
    room, monkeypatch
):
    """A crash is an invalid run and costs more than the pool was worth; an empty answer merely
    loses the pool."""
    def _refuse(self, operation, payload):
        raise BrokerError("the broker refused the request")

    monkeypatch.setattr(sdk_broker.BrokerClient, "_call", _refuse)
    for pool in POOL_TASKS:
        answer = _run_reference(POOL_TASKS[pool])
        assert answer["task_id"] == POOL_TASKS[pool]["task_id"]


# ---- the SDK exposes no credential and no way to ask for one ---

def test_the_sdk_has_no_credential_surface():
    """A method that returned a key, or an attribute that held one, would undo Phase C from inside
    the image that runs untrusted code."""
    import kata_sn22_sdk

    names = " ".join(dir(kata_sn22_sdk) + dir(sdk_broker.BrokerClient)).lower()
    for forbidden in ("api_key", "apikey", "credential", "secret", "token", "bearer"):
        assert forbidden not in names, forbidden


def test_the_sdk_reads_no_key_from_the_environment(monkeypatch):
    """Even if something upstream mistakenly set one, the SDK has no code that would look."""
    source = " ".join(
        path.read_text(encoding="utf-8")
        for path in (Path(sdk_broker.__file__).parent).glob("*.py"))
    for forbidden in ("SN22_INFERENCE_API_KEY", "OPENAI_API_KEY", "CHUTES_API_KEY",
                      "APIFY_API_KEY", "SCRAPINGDOG_API_KEY", "x-inference-api-key"):
        assert forbidden not in source, forbidden


def test_the_sdk_ships_no_scorer_and_no_judge_prompt():
    """The image runs untrusted code. A scorer inside it would let an agent grade itself, and a
    judge prompt inside it would tell it exactly what to write."""
    source = " ".join(
        path.read_text(encoding="utf-8")
        for path in (Path(sdk_broker.__file__).parent).glob("*.py"))
    for forbidden in ("judge_prompts", "scorer_policy", "SYSTEM_", "combine_pool_scores",
                      "policy_hash", "bittensor"):
        assert forbidden not in source, forbidden


def test_the_sdk_can_only_name_the_three_agent_operations():
    assert sdk_broker.AGENT_OPERATIONS == ("web-search", "x-search", "final-summary")
    client = sdk_broker.BrokerClient()
    with pytest.raises(BrokerError, match="unknown operation"):
        client._call("chutes-score", {})


def test_the_sdk_operations_are_the_ones_the_broker_declares():
    """Two files in two packages naming the same three strings. This is what stops them drifting."""
    from kata_sn22.broker_ops import AGENT_OPERATIONS

    assert set(sdk_broker.AGENT_OPERATIONS) == set(AGENT_OPERATIONS)


def test_evaluator_only_operations_are_refused_outside_the_room(monkeypatch):
    monkeypatch.delenv(sdk_broker.BROKER_URL_ENV, raising=False)
    monkeypatch.setenv(sdk_broker.RELAY_ENDPOINT_ENV, "sn22-relay+unix:///tmp/x.sock")
    client = sdk_broker.BrokerClient()
    for call in (lambda: client.x_search("q"), lambda: client.final_summary([])):
        with pytest.raises(BrokerError, match="only available in the sealed room"):
            call()


# ---- the task the agent sees ---

def test_the_agent_never_learns_which_tasks_are_deep_scored():
    """An agent that knew would work hardest on exactly those, and the 20% sample would stop
    measuring the other 80%."""
    task = v2.AiSearchTask(
        task_id="t1", prompt="q", mode=v2.SearchMode.FAST,
        result_type=v2.ResultType.ONLY_LINKS, tools=("Web Search",), deep=True)
    agent_input = task.as_agent_input()
    assert "deep" not in agent_input

    synapse = synapse_from_input(agent_input)
    assert not hasattr(synapse, "deep")


def test_the_sdk_builds_the_same_synapse_the_trusted_side_describes():
    """The two model sets are written in different packages on purpose -- the trusted one also
    carries the parser and the scoring surface. This holds them to the same shape."""
    for document in POOL_TASKS.values():
        synapse = synapse_from_input(document)
        assert synapse.task_id == document["task_id"]
        assert synapse.count == document["count"]
        assert synapse.limits.max_execution_time == document["limits"]["max_execution_time"]


@pytest.mark.parametrize("version", [1, 3, None, "2"])
def test_a_task_of_an_unknown_version_is_refused(version):
    with pytest.raises(SdkError, match="protocol_version"):
        synapse_from_input({**POOL_TASKS["ai_search:fast"], "protocol_version": version})


@pytest.mark.parametrize(("field", "value"), [
    ("search_type", "reddit_search"), ("mode", "instant"), ("result_type", "SUMMARY_ONLY"),
])
def test_an_unknown_enum_value_is_refused(field, value):
    with pytest.raises(SdkError, match=field):
        synapse_from_input({**POOL_TASKS["ai_search:fast"], field: value})


# ---- emit ---

def test_emit_derives_every_field_the_scorer_reads_from_one_call():
    emit = Emit()
    emit(ScraperTextRole.INTRO, "a")
    emit(ScraperTextRole.FINAL_SUMMARY, "b")
    emit(ScraperTextRole.FINAL_SUMMARY, "c")

    assert emit.completion() == "abc"
    assert emit.texts() == {"intro": "a", "summary": "bc"}
    assert emit.text_chunks() == {"intro": ["a"], "summary": ["b", "c"]}
    assert emit.chunks == [{"role": "intro", "text": "a"}, {"role": "summary", "text": "b"},
                           {"role": "summary", "text": "c"}]


def test_emit_accepts_a_plain_string_role_as_well_as_the_enum():
    emit = Emit()
    emit("summary", "x")
    assert emit.texts() == {"summary": "x"}


def test_emit_refuses_a_role_the_scorer_cannot_place():
    """A chunk it cannot place is a chunk it does not count, and the agent would never know."""
    emit = Emit()
    with pytest.raises(SdkError, match="unknown text role"):
        emit("conclusion", "x")


def test_emit_is_bounded():
    """A submission that streams without bound fills the room's memory, and the room is shared with
    the contestant scheduled after it."""
    emit = Emit()
    with pytest.raises(SdkError, match="characters in total"):
        emit(ScraperTextRole.FINAL_SUMMARY, "x" * (Emit.MAX_TOTAL_CHARS + 1))

    counted = Emit()
    for _ in range(Emit.MAX_CHUNKS):
        counted(ScraperTextRole.FINAL_SUMMARY, "x")
    with pytest.raises(SdkError, match="chunks"):
        counted(ScraperTextRole.FINAL_SUMMARY, "x")


def test_the_sdk_roles_are_the_trusted_sides_roles():
    assert ({role.value for role in ScraperTextRole}
            == {role.value for role in v2.ScraperTextRole})


# ---- the harness contains a failing submission rather than breaking ---

def _submission(body: str, tmp_path: Path) -> str:
    path = tmp_path / "agent.py"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(path)


def _run(body: str, tmp_path: Path, document=None) -> tuple:
    stderr = io.StringIO()
    answer = harness.run(document or POOL_TASKS["ai_search:fast"],
                         agent_path=_submission(body, tmp_path), stderr=stderr)
    return answer, stderr.getvalue()


def test_a_submission_that_raises_still_produces_a_parseable_answer(room, tmp_path):
    """A process that exits non-zero reads as a broken ROOM and defers a duel that should simply
    have been lost."""
    answer, stderr = _run("""
        from kata_sn22_sdk import Agent

        class Submission(Agent):
            async def smart_scraper(self, synapse, emit):
                raise ZeroDivisionError("boom")
    """, tmp_path)
    assert answer == harness.empty_answer(synapse_from_input(POOL_TASKS["ai_search:fast"]))
    assert "ZeroDivisionError" in stderr


def test_a_submission_that_never_returns_is_stopped_at_its_execution_budget(room, tmp_path):
    """``max_execution_time`` is upstream's own serving budget for the mode, and the timeout penalty
    is measured against it."""
    document = {**POOL_TASKS["ai_search:fast"], "limits": {"max_execution_time": 1}}
    answer, stderr = _run("""
        import asyncio
        from kata_sn22_sdk import Agent

        class Submission(Agent):
            async def smart_scraper(self, synapse, emit):
                await asyncio.sleep(60)
    """, tmp_path, document)
    assert answer["search_results"] == []
    assert "max_execution_time" in stderr


def test_a_submission_that_will_not_import_produces_an_answer(room, tmp_path):
    answer, stderr = _run("this is not python(", tmp_path)
    assert answer["task_id"] == "t-fast"
    assert "SyntaxError" in stderr


def test_a_submission_with_no_agent_subclass_is_reported_clearly(room, tmp_path):
    _answer, stderr = _run("x = 1", tmp_path)
    assert "no Agent subclass" in stderr


def test_a_submission_with_two_agent_subclasses_is_refused(room, tmp_path):
    """Which one runs would otherwise be answered by definition order, and a submission whose
    behaviour depends on that is one nobody can review."""
    _answer, stderr = _run("""
        from kata_sn22_sdk import Agent

        class First(Agent):
            pass

        class Second(Agent):
            pass
    """, tmp_path)
    assert "2 Agent subclasses" in stderr


def test_a_submission_that_declines_a_pool_loses_only_that_pool(room, tmp_path):
    answer, stderr = _run("""
        from kata_sn22_sdk import Agent

        class Submission(Agent):
            pass
    """, tmp_path, POOL_TASKS["x_search"])
    assert answer == {"protocol_version": 2, "task_id": "t-x", "results": []}
    assert "does not implement" in stderr


def test_a_submission_that_returns_the_wrong_type_is_contained(room, tmp_path):
    answer, stderr = _run("""
        from kata_sn22_sdk import Agent

        class Submission(Agent):
            async def smart_scraper(self, synapse, emit):
                return {"search_results": []}
    """, tmp_path)
    assert answer["search_results"] == []
    assert "AiSearchResult" in stderr


def test_an_enormous_answer_is_bounded(room, tmp_path):
    """Not a scoring rule -- a memory bound on a room shared with the next contestant."""
    answer, _stderr = _run("""
        from kata_sn22_sdk import Agent, AiSearchResult

        class Submission(Agent):
            async def smart_scraper(self, synapse, emit):
                return AiSearchResult(search_results=[
                    {"link": "https://a.test/%d" % i, "title": "x" * 500000}
                    for i in range(5000)])
    """, tmp_path)
    assert len(answer["search_results"]) <= harness.MAX_RESULTS
    assert all(len(item["title"]) <= harness.MAX_FIELD_CHARS
               for item in answer["search_results"])


def test_an_unusable_task_is_the_rooms_fault_and_exits_non_zero(monkeypatch, tmp_path):
    """A task nobody can present is not a contestant's failure, and scoring one for a question it
    was never asked would be worse than refusing."""
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"protocol_version": 99})))
    assert harness.main([str(tmp_path / "agent.py")]) == 2


def test_the_harness_defaults_to_the_bundle_path_the_room_mounts():
    assert harness.DEFAULT_BUNDLE_AGENT == "/bundle/agent.py"


# ---- a real subclass, exercised directly ---

def test_an_agent_written_the_documented_way_works(room):
    class Written(Agent):
        async def smart_scraper(self, synapse: AiSearchSynapse, emit) -> AiSearchResult:
            results = self.broker.web_search(synapse.prompt, count=synapse.count)
            emit(ScraperTextRole.FINAL_SUMMARY, f"Found {len(results)}.")
            return AiSearchResult(search_results=results)

        async def twitter_search(self, synapse: XSearchSynapse) -> XSearchResult:
            return XSearchResult(results=self.broker.x_search(synapse.query, count=synapse.count))

    import asyncio

    agent = Written()
    emit = Emit()
    synapse = synapse_from_input(POOL_TASKS["ai_search:fast"])
    result = asyncio.run(agent.smart_scraper(synapse, emit))
    framed = harness.frame_ai_answer(synapse, result, emit)

    v2.parse_answer(framed, task=_trusted_task(POOL_TASKS["ai_search:fast"]))
    assert framed["texts"]["summary"] == "Found 10."


# ---- the two execution paths have not converged, and that is recorded on purpose ---

def test_the_shipped_submission_is_a_version_two_sdk_agent():
    """What a miner copies must be what the room runs.

    Before Phase D the shipped agent was a version-1 script. It is now an ``Agent`` subclass, and
    anything in the competition repository still speaking version 1 would be teaching miners to
    write something the room cannot execute.
    """
    source = REFERENCE.read_text(encoding="utf-8")
    assert "from kata_sn22_sdk import" in source
    assert "class Submission(Agent)" in source
    assert "async def smart_scraper" in source
    assert "async def twitter_search" in source
    assert "PROTOCOL_VERSION = 1" not in source
    assert "import sn22_relay" not in source


def test_the_version_one_calibration_agent_is_not_shown_to_miners():
    """The sandbox still speaks version 1 and Phase E/F is where the two paths converge. Until then
    the version-1 agent lives in this repository's tests, so nothing a miner is shown speaks it.

    This test is the reminder: when the sandbox moves to version 2, that file has no job and should
    be deleted, and this test with it.
    """
    calibration = Path(__file__).resolve().parent / "agents" / "v1-calibration" / "agent.py"
    assert calibration.is_file(), "the sandbox's end-to-end tests have no agent to drive"
    assert "SANDBOX path only" in calibration.read_text(encoding="utf-8")

    # Both trees a miner ever sees: the King they are told to copy, and any entries already in
    # submissions/. Scanning REFERENCE.parents[1] would now cover only kings/, and a version-1
    # agent sitting in submissions/ -- the tree miners actually write to -- would go unnoticed.
    kata = Path(__file__).resolve().parents[2] / "kata"
    shown = list((kata / "kings").rglob("sn22__desearch/**/agent.py"))
    shown += list((kata / "submissions" / "sn22__desearch").rglob("agent.py"))
    assert shown, "no SN22 agent is shown to miners at all"
    for agent in shown:
        assert "kata_sn22_sdk" in agent.read_text(encoding="utf-8"), agent


def test_the_sdk_imports_only_the_standard_library():
    """The agent image has no package installer, by design. An SDK that grew a third-party import
    would fail at `import kata_sn22_sdk` inside a sealed room, on a duel, with no way to install it
    -- and it would read as a broken submission.
    """
    import ast
    import sys as _sys

    package = Path(sdk_broker.__file__).parent
    third_party: set = set()
    for path in package.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), str(path))):
            if isinstance(node, ast.Import):
                third_party |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                third_party.add(node.module.split(".")[0])
    unknown = {
        name for name in third_party
        if name not in _sys.stdlib_module_names and name != "kata_sn22_sdk"
    }
    assert not unknown, f"the SDK imports {sorted(unknown)}, which the agent image cannot install"
