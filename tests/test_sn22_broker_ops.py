"""The six SN22 operations: the routes are constants, and the roles do not overlap.

``room.broker`` decides whether a caller may invoke an operation. This file is about what an
operation *is* — and that is where "an agent cannot select an arbitrary host or model" is actually
decided, because a host, a model or an actor id that came out of the caller's payload would defeat
every capability check upstream of it.

The tests fall into three groups:

* **the route is not reachable from the payload** — hostile fields are ignored, not honoured;
* **the roles are disjoint** — the agent never touches the judge's credential, the evaluator never
  touches the agent's;
* **input is bounded before a provider is contacted** — so a malformed call costs the contestant
  nothing, and an enormous one cannot be used to bill them.
"""

from __future__ import annotations

import json
from urllib.parse import urlsplit

import pytest

from kata_sn22 import broker_ops as ops
from kata_sn22.credentials_v2 import AGENT_PROVIDERS, EVALUATOR_PROVIDERS, REQUIRED_PROVIDERS

KEY = "provider-secret-key-0123456789"


@pytest.fixture
def captured(monkeypatch):
    """Intercept at the transport, which is the last point before a real network call."""
    seen: dict = {}

    def _capture_post(url, payload, *, headers, timeout):
        seen.update(url=url, payload=payload, headers=headers, method="POST")
        return {"choices": [{"message": {"content": "an answer"}}], "items": []}

    def _capture_get(url, params, *, timeout):
        seen.update(url=url, params=params, method="GET")
        return json.dumps({"organic_results": [{"link": "https://a.test"}]})

    monkeypatch.setattr(ops, "_post_json", _capture_post)
    monkeypatch.setattr(ops, "_get_text", _capture_get)
    return seen


# ---- the route is a constant, not a payload field -------------------------------------------

#: Everything a hostile agent would try in order to redirect the miner's spend.
HOSTILE = {
    "url": "https://evil.test/collect",
    "endpoint": "https://evil.test",
    "host": "evil.test",
    "model": "an-extremely-expensive-model",
    "actor": "someOtherActor",
    "actor_id": "someOtherActor",
    "api_key": "an-attacker-key",
    "token": "an-attacker-token",
    "base_url": "https://evil.test",
    "provider": "chutes",
}


def test_web_search_always_reaches_the_pinned_search_endpoint(captured):
    ops.web_search(KEY, {"query": "emissions", "count": 10, **HOSTILE})
    assert captured["url"] == ops.SCRAPINGDOG_SEARCH_URL
    assert captured["params"]["api_key"] == KEY
    assert "evil.test" not in json.dumps(captured)


def test_x_search_always_reaches_the_pinned_actor(captured):
    ops.x_search(KEY, {"query": "emissions", "count": 10, **HOSTILE})
    assert ops.APIFY_SEARCH_ACTOR in captured["url"]
    assert urlsplit(captured["url"]).netloc == "api.apify.com"
    assert "someOtherActor" not in captured["url"]


def test_the_agent_summary_model_cannot_be_chosen_by_the_agent(captured):
    """An agent that could name the model could name an expensive one and bill the miner -- or a
    different one from its opponent, which would make the two summaries incomparable."""
    messages = [{"role": "user", "content": "sum up"}]
    result = ops.final_summary(KEY, {"messages": messages, **HOSTILE})
    assert captured["url"] == ops.OPENAI_CHAT_URL
    assert captured["payload"]["model"] == ops.AGENT_SUMMARY_MODEL
    assert result["model"] == ops.AGENT_SUMMARY_MODEL


def test_the_judge_model_cannot_be_chosen_either(captured):
    ops.chutes_score(KEY, {"messages": [{"role": "user", "content": "grade"}], **HOSTILE})
    assert captured["url"] == ops.CHUTES_CHAT_URL
    assert captured["payload"]["model"] == ops.JUDGE_MODEL


def test_the_tweet_rescrape_actor_is_upstreams_pinned_one(captured):
    ops.tweet_rescrape(KEY, {"tweet_ids": ["12345"], **HOSTILE})
    assert ops.APIFY_TWEET_ACTOR in captured["url"]
    assert "someOtherActor" not in captured["url"]


def test_the_page_fetch_endpoint_is_fixed_even_though_the_urls_are_not(captured):
    """The URLs ARE the caller's here -- that is the operation: fetch the pages the agent cited.
    What is fixed is the provider it goes through, so the fetch is billed and observed."""
    ops.web_page_fetch(KEY, {"urls": ["https://cited.test/a"], **HOSTILE})
    assert captured["url"] == ops.SCRAPINGDOG_SCRAPE_URL
    assert captured["params"]["url"] == "https://cited.test/a"


def test_every_pinned_route_is_https_and_carries_no_credentials_in_the_url():
    for url in (ops.SCRAPINGDOG_SEARCH_URL, ops.SCRAPINGDOG_SCRAPE_URL,
                ops.OPENAI_CHAT_URL, ops.CHUTES_CHAT_URL, ops.APIFY_ACTOR_BASE):
        parsed = urlsplit(url)
        assert parsed.scheme == "https", url
        assert "@" not in parsed.netloc, url


# ---- the roles are disjoint --------------------------------------------------------------------

def test_the_declared_operations_match_the_credential_roles():
    """The provider split in :mod:`kata_sn22.credentials_v2` and the operation table here are two
    statements of the same rule, written in different files. This is what stops them drifting."""
    agent_providers = {provider for _n, role, provider, *_ in ops.OPERATIONS if role == "agent"}
    evaluator_providers = {
        provider for _n, role, provider, *_ in ops.OPERATIONS if role == "evaluator"
    }
    assert agent_providers <= AGENT_PROVIDERS
    assert evaluator_providers <= EVALUATOR_PROVIDERS


def test_the_agent_never_reaches_the_judges_credential():
    """An agent that could call the judge could grade its own work."""
    assert "chutes" not in {provider for _n, role, provider, *_ in ops.OPERATIONS
                            if role == "agent"}


def test_the_evaluator_never_reaches_the_agents_summary_credential():
    """The evaluator has no reason to hold it, and holding it would let a verification bug bill the
    contestant for work nobody asked for."""
    assert "openai" not in {provider for _n, role, provider, *_ in ops.OPERATIONS
                            if role == "evaluator"}


def test_every_sealed_provider_is_actually_used_by_some_operation():
    """Four keys are demanded at intake. Demanding one nothing spends would be asking a miner for a
    credential for no reason."""
    used = {provider for _n, _r, provider, *_ in ops.OPERATIONS}
    assert used == set(REQUIRED_PROVIDERS)


def test_the_operation_names_are_the_ones_the_relay_client_can_ask_for():
    from kata_sn22 import relay_client

    assert set(ops.AGENT_OPERATIONS) == {
        relay_client.OP_WEB_SEARCH, relay_client.OP_X_SEARCH, relay_client.OP_FINAL_SUMMARY}


# ---- input is bounded before any provider is contacted ---

def _never_called(*_args, **_kwargs):
    raise AssertionError("a provider was contacted for input that should have been refused")


@pytest.fixture
def no_provider(monkeypatch):
    monkeypatch.setattr(ops, "_post_json", _never_called)
    monkeypatch.setattr(ops, "_get_text", _never_called)


@pytest.mark.parametrize("payload", [
    {}, {"query": ""}, {"query": "   "}, {"query": 5},
    {"query": "x" * (ops.MAX_QUERY_CHARS + 1)},
    {"query": "ok", "count": 0}, {"query": "ok", "count": 9}, {"query": "ok", "count": 10_000},
    {"query": "ok", "count": True}, {"query": "ok", "count": "10"},
])
def test_a_bad_search_payload_is_refused_before_the_provider(no_provider, payload):
    """Refused before the call, so a malformed request costs the contestant nothing."""
    with pytest.raises(ops.OperationInputError):
        ops.web_search(KEY, payload)


@pytest.mark.parametrize("urls", [
    None, [], "https://a.test", [5],
    ["file:///etc/passwd"], ["gopher://a.test"], ["//a.test"], ["not a url"],
    ["https://user:pass@a.test"],
    ["https://a.test"] * (ops.MAX_URLS_PER_CALL + 1),
])
def test_a_bad_url_list_is_refused(no_provider, urls):
    """``file://`` in particular: it would make this operation a reader of the room's own filesystem
    on behalf of whoever asked."""
    with pytest.raises(ops.OperationInputError):
        ops.web_page_fetch(KEY, {"urls": urls})


@pytest.mark.parametrize("tweet_ids", [
    None, [], "12345", ["not-a-number"], ["12345; DROP"], ["1" * 33],
    ["12345"] * (ops.MAX_TWEET_IDS_PER_CALL + 1),
])
def test_a_bad_tweet_id_list_is_refused(no_provider, tweet_ids):
    with pytest.raises(ops.OperationInputError):
        ops.tweet_rescrape(KEY, {"tweet_ids": tweet_ids})


@pytest.mark.parametrize("messages", [
    None, [], "hello", [{"role": "user"}], [{"content": "x"}],
    [{"role": "root", "content": "x"}],
    [{"role": "user", "content": 5}],
    [{"role": "user", "content": "x", "name": "y"}],
    [{"role": "user", "content": "x"}] * (ops.MAX_MESSAGES + 1),
    [{"role": "user", "content": "x" * (ops.MAX_MESSAGE_CHARS + 1)}],
])
def test_a_bad_message_list_is_refused(no_provider, messages):
    with pytest.raises(ops.OperationInputError):
        ops.final_summary(KEY, {"messages": messages})


def test_a_page_that_cannot_be_fetched_is_recorded_rather_than_failing_the_call(monkeypatch):
    """One dead link is a fact about that source. Failing the whole verification over it would let
    any contestant citing a flaky page take the round down."""
    import urllib.error

    calls = {"n": 0}

    def _sometimes(url, params, *, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(url, 404, "gone", {}, None)
        return "the page body"

    monkeypatch.setattr(ops, "_get_text", _sometimes)
    result = ops.web_page_fetch(KEY, {"urls": ["https://dead.test", "https://live.test"]})

    assert result["pages"]["https://dead.test"]["error"]
    assert result["pages"]["https://live.test"]["text"] == "the page body"


def test_a_provider_that_answers_for_no_url_at_all_is_an_outage(monkeypatch):
    """The difference between "these links are dead" and "we could not check" decides whether a
    contestant is scored or the duel defers, so the two must not collapse."""
    import urllib.error

    def _unreachable(url, params, *, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(ops, "_get_text", _unreachable)
    with pytest.raises(ops.OperationInputError, match="could not be reached"):
        ops.web_page_fetch(KEY, {"urls": ["https://a.test", "https://b.test"]})


def test_the_judge_temperature_comes_from_the_scoring_policy(captured):
    """Not restated here. A second copy is a second thing that can drift from the policy hash two
    contestants must share."""
    from kata_sn22.scorer_policy import JUDGE_TEMPERATURE

    ops.chutes_score(KEY, {"messages": [{"role": "user", "content": "grade"}]})
    assert captured["payload"]["temperature"] == JUDGE_TEMPERATURE
