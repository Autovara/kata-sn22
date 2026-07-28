"""The live transports, and the wiring that was missing.

**The bug these exist for.** Verification was built as three seams with cassettes behind them, and
nothing was ever put behind them for production. The entry point the engine loads —
``kata_sn22:SN22_DESEARCH_PLUGIN`` — constructed a plugin with no transports at all, so the first
real round would have raised at scoring. The entire test suite passed throughout, because every test
constructs its own plugin and hands it cassettes.

So the first test here is the one that matters: the singleton the engine actually loads arrives
wired.

**What is and is not covered.** These tests drive request construction and response parsing against
fakes. They cannot prove the wire formats match what ScrapingDog, Apify and OpenAI really answer —
the formats were read off the pinned upstream, and only a real call confirms them. What they do
prove is that every failure lands somewhere defined, which is what stops a provider hiccup being
scored as a bad miner.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from kata_sn22 import providers
from kata_sn22.fetch import FetchUnavailable, PageFetcher
from kata_sn22.judge import JudgeUnavailable
from kata_sn22.tweets import ScrapeUnavailable

ARTICLE = "The measured figure is 28 percent, recorded in July 2026. " * 6


# ---- the wiring the engine depends on ---------------------------------------------------------

def test_the_entry_point_singleton_arrives_with_its_transports_wired():
    """THE regression. A plugin built without transports raises on the first thing it scores, and
    no test would notice because tests build their own."""
    from kata_sn22 import SN22_DESEARCH_PLUGIN as plugin

    assert plugin._page_transport is not None
    assert plugin._judge_client is not None
    assert plugin._tweet_scraper is not None


def test_a_missing_credential_names_the_variable_rather_than_failing_obscurely(monkeypatch):
    """An operator reading "no page transport is configured" cannot act on it. One naming
    SCRAPINGDOG_API_KEY can."""
    monkeypatch.delenv(providers.SCRAPINGDOG_KEY_ENV, raising=False)
    transport = providers.transports_from_env().page_transport

    with pytest.raises(FetchUnavailable, match=providers.SCRAPINGDOG_KEY_ENV):
        transport(["https://a.test"])


def test_transports_are_built_even_when_credentials_are_absent(monkeypatch):
    """Constructing must not raise: the three are needed at different moments, and a challenge of
    only web tasks never re-scrapes a tweet. Refusing to build would take a lane down over a
    capability it was not going to use."""
    for name in (providers.SCRAPINGDOG_KEY_ENV, providers.APIFY_KEY_ENV, providers.OPENAI_KEY_ENV):
        monkeypatch.delenv(name, raising=False)
    transports = providers.transports_from_env()
    assert transports.page_transport and transports.judge_client and transports.tweet_scraper


def test_endpoints_can_be_repointed_without_a_code_change(monkeypatch):
    monkeypatch.setenv(providers.OPENAI_URL_ENV, "https://staging.example/v1/chat")
    assert providers.transports_from_env().judge_client.url == "https://staging.example/v1/chat"


# ---- pages ---------------------------------------------------------------------------------------

def _fake_get(monkeypatch, handler):
    class _Response:
        def __init__(self, body): self._body = body
        def read(self, _n): return self._body
        def __enter__(self): return self
        def __exit__(self, *_a): return False

    def _urlopen(request, timeout=None):
        url = request if isinstance(request, str) else request.full_url
        return _Response(handler(url))

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)


def test_a_fetched_page_reaches_the_verifier(monkeypatch):
    _fake_get(monkeypatch, lambda _url: json.dumps(
        {"title": "A Title", "text": ARTICLE, "html": "<p>x</p>"}).encode())
    pages = providers.ScrapingDogPages(api_key="k")(["https://a.test"])

    fetcher = PageFetcher(transport=lambda urls: {u: pages[u] for u in urls})
    page = fetcher.get_many(["https://a.test"])["https://a.test"]
    assert ARTICLE[:40] in page.text
    assert page.title == "A Title"


def test_a_page_returned_as_raw_html_still_yields_a_body(monkeypatch):
    """ScrapingDog answers JSON when it has metadata and raw HTML otherwise. Treating the second as
    a failure would drop every source from a site it has no metadata for."""
    _fake_get(monkeypatch, lambda _url: ARTICLE.encode())
    record = providers.ScrapingDogPages(api_key="k")(["https://a.test"])["https://a.test"]
    assert ARTICLE[:40] in record["text"]


def test_one_dead_link_does_not_take_the_round_down(monkeypatch):
    """A miner citing a flaky page must not be able to fail the whole round -- for itself or for its
    opponent. The dead link becomes an empty body with a reason; the others still fetch."""
    def _handler(url):
        if "dead" in url:
            raise urllib.error.HTTPError(url, 404, "gone", {}, None)
        return json.dumps({"text": ARTICLE}).encode()

    _fake_get(monkeypatch, _handler)
    out = providers.ScrapingDogPages(api_key="k")(["https://dead.test", "https://ok.test"])

    assert out["https://dead.test"]["text"] == ""
    assert "404" in out["https://dead.test"]["error"]
    assert ARTICLE[:20] in out["https://ok.test"]["text"]


def test_an_outage_reaching_the_provider_at_all_stops_the_round(monkeypatch):
    """The other side of the rule above. No page fetched means no contestant was verified against
    independent ground truth, and ranking on that is ranking on the miners' own claims."""
    def _urlopen(_request, timeout=None):
        raise urllib.error.URLError("dns failure")

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    with pytest.raises(FetchUnavailable, match="could not be reached"):
        providers.ScrapingDogPages(api_key="k")(["https://a.test", "https://b.test"])


def test_every_fetch_is_counted_for_the_budget(monkeypatch):
    _fake_get(monkeypatch, lambda _url: json.dumps({"text": ARTICLE}).encode())
    transport = providers.ScrapingDogPages(api_key="k")
    transport(["https://a.test", "https://b.test"])
    assert len(transport.calls) == 2


# ---- the judge ---------------------------------------------------------------------------------

def _fake_post(monkeypatch, handler):
    class _Response:
        def __init__(self, body): self._body = body
        def read(self, _n): return self._body
        def __enter__(self): return self
        def __exit__(self, *_a): return False

    sent = {}

    def _urlopen(request, timeout=None):
        sent["url"] = request.full_url
        sent["headers"] = {k.lower(): v for k, v in request.headers.items()}
        sent["body"] = json.loads(request.data.decode())
        return _Response(handler(sent["body"]))

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    return sent


def test_the_judge_is_called_with_upstreams_model_and_temperature(monkeypatch):
    """Both come from `kata_sn22.judge` rather than being restated here: they are part of the
    scoring policy, and a second copy is a second thing that can drift from it."""
    from kata_sn22 import judge as judge_module

    sent = _fake_post(monkeypatch, lambda _b: json.dumps(
        {"choices": [{"message": {"content": "Verdict: HIGH"}}]}).encode())

    reply = providers.OpenAiJudge(api_key="sk-x")([{"role": "user", "content": "q"}])

    assert reply == "Verdict: HIGH"
    assert sent["body"]["model"] == judge_module.JUDGE_MODEL
    assert sent["body"]["temperature"] == judge_module.JUDGE_TEMPERATURE
    assert sent["headers"]["authorization"] == "Bearer sk-x"


@pytest.mark.parametrize("payload", [
    {},
    {"choices": []},
    {"choices": [{"message": {}}]},
    {"choices": ["not an object"]},
])
def test_an_empty_completion_raises_rather_than_scoring_zero(monkeypatch, payload):
    """The most dangerous failure in the whole file. `verdict_score("")` is 0.0, so an empty
    completion returned as text would score the source LOW -- indistinguishable from a real
    judgement that it was bad. It has to raise."""
    _fake_post(monkeypatch, lambda _b: json.dumps(payload).encode())
    with pytest.raises(JudgeUnavailable):
        providers.OpenAiJudge(api_key="sk-x")([{"role": "user", "content": "q"}])


@pytest.mark.parametrize("code", [401, 429, 500, 503])
def test_a_refused_judge_call_raises(monkeypatch, code):
    def _urlopen(_request, timeout=None):
        raise urllib.error.HTTPError("https://api.openai.com", code, "no", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    with pytest.raises(JudgeUnavailable, match=str(code)):
        providers.OpenAiJudge(api_key="sk-x")([{"role": "user", "content": "q"}])


# ---- tweets ------------------------------------------------------------------------------------

def test_re_scraped_tweets_come_back_keyed_by_id(monkeypatch):
    sent = _fake_post(monkeypatch, lambda _b: json.dumps([
        {"id": "100", "text": "a tweet", "created_at": "Thu Jan 01 00:00:00 +0000 2026"},
        {"id": "101", "text": "another", "created_at": "Thu Jan 01 00:01:00 +0000 2026"},
    ]).encode())

    found = providers.ApifyTweets(api_key="apify-x")(["100", "101"])

    assert set(found) == {"100", "101"}
    assert found["100"]["text"] == "a tweet"
    assert sent["body"]["tweetIDs"] == ["100", "101"]
    assert providers.APIFY_TWEET_ACTOR in sent["url"]


def test_a_tweet_the_platform_no_longer_serves_is_simply_absent(monkeypatch):
    """Not an error. A deleted tweet is a fact about the tweet, and the caller drops it from the
    mean rather than scoring it against the miner."""
    _fake_post(monkeypatch, lambda _b: json.dumps([{"id": "100", "text": "a tweet"}]).encode())
    assert set(providers.ApifyTweets(api_key="apify-x")(["100", "101"])) == {"100"}


def test_a_scraper_outage_raises(monkeypatch):
    def _urlopen(_request, timeout=None):
        raise urllib.error.HTTPError("https://api.apify.com", 502, "bad gateway", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    with pytest.raises(ScrapeUnavailable, match="502"):
        providers.ApifyTweets(api_key="apify-x")(["100"])


def test_a_non_dataset_answer_raises_rather_than_scoring_every_tweet_fabricated(monkeypatch):
    _fake_post(monkeypatch, lambda _b: json.dumps({"error": "nope"}).encode())
    with pytest.raises(ScrapeUnavailable, match="no dataset"):
        providers.ApifyTweets(api_key="apify-x")(["100"])


def test_nothing_to_scrape_costs_nothing(monkeypatch):
    def _urlopen(_request, timeout=None):
        raise AssertionError("must not call the scraper with no tweet ids")

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    assert providers.ApifyTweets(api_key="apify-x")([]) == {}


# ---- what it all spent -------------------------------------------------------------------------

def test_usage_reports_what_each_transport_actually_spent(monkeypatch):
    """The lane meters `data_api_calls`, `inference_calls` and `scrape_units` against this. A
    transport that under-reported would let a round spend past an approved ceiling."""
    _fake_get(monkeypatch, lambda _url: json.dumps({"text": ARTICLE}).encode())
    transports = providers.Transports(
        page_transport=providers.ScrapingDogPages(api_key="k"),
        judge_client=providers.OpenAiJudge(api_key="sk-x"),
        tweet_scraper=providers.ApifyTweets(api_key="apify-x"))
    transports.page_transport(["https://a.test", "https://b.test"])

    _fake_post(monkeypatch, lambda _b: json.dumps(
        {"choices": [{"message": {"content": "Verdict: HIGH"}}]}).encode())
    transports.judge_client([{"role": "user", "content": "q"}])

    _fake_post(monkeypatch, lambda _b: json.dumps([{"id": "100"}]).encode())
    transports.tweet_scraper(["100", "101", "102"])

    assert transports.usage() == {
        "data_api_calls": 2, "inference_calls": 1, "scrape_units": 3}
