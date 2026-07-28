"""The live transports the lane pays for: page fetch, judge, tweet re-scrape.

:mod:`kata_sn22.fetch`, :mod:`kata_sn22.judge` and :mod:`kata_sn22.tweets` define WHAT verification
does and leave HOW to reach a provider as a seam. Until now the only implementations of those seams
were the cassettes, which meant the lane could replay a recorded round and could not run a real one.
These are the real ones.

**Wire formats are upstream's**, read off the pinned tree rather than invented:

* pages — ScrapingDog ``GET /scrape?api_key=&url=``
  (``neurons/validators/apify/scrapingdog_scraper.py``)
* tweets — the Apify twitter actor ``CJdippxWmn9uRfooo``
  (``neurons/validators/apify/twitter_scraper_actor.py``)
* judge — OpenAI ``POST /v1/chat/completions`` with ``gpt-4.1-nano``
  (``desearch/utils.py::call_scoring_llm``)

**Standard library only.** The lane runtime carries no third-party HTTP client and should not start
now: an extra dependency in the process that holds provider credentials is an extra thing to audit.
``urllib`` is enough for three request shapes.

**Every transport fails loudly.** A missing key, a refused request, a malformed answer -- each
raises the seam's own "unavailable" exception rather than returning something empty. That is the
whole reason the seams exist: *failing to verify is not verifying a failure*, and a round that could
not gather evidence must stop rather than rank a contestant on none.

They are configured entirely from the environment (see :func:`transports_from_env`), so a deployment
turns verification on by providing credentials and off by not — with no code change and no silent
half-configured state.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

from kata_sn22.fetch import FetchUnavailable
from kata_sn22.judge import JUDGE_MODEL, JUDGE_TEMPERATURE, JudgeUnavailable
from kata_sn22.tweets import ScrapeUnavailable

#: Credentials, by the names upstream uses. Named here so a deployment configures ONE vocabulary.
SCRAPINGDOG_KEY_ENV = "SCRAPINGDOG_API_KEY"
APIFY_KEY_ENV = "APIFY_API_KEY"
OPENAI_KEY_ENV = "OPENAI_API_KEY"

#: Endpoints. Overridable so a staging deployment can point elsewhere without a code change.
SCRAPINGDOG_URL_ENV = "KATA_SN22_SCRAPINGDOG_URL"
APIFY_URL_ENV = "KATA_SN22_APIFY_URL"
OPENAI_URL_ENV = "KATA_SN22_OPENAI_URL"

DEFAULT_SCRAPINGDOG_URL = "https://api.scrapingdog.com/scrape"
#: Upstream's ``new_actor_id``. Pinned: a different actor returns a different tweet shape, and the
#: field-by-field comparison would start failing honest miners.
APIFY_TWEET_ACTOR = "CJdippxWmn9uRfooo"
DEFAULT_APIFY_URL = f"https://api.apify.com/v2/acts/{APIFY_TWEET_ACTOR}/run-sync-get-dataset-items"
DEFAULT_OPENAI_URL = "https://api.openai.com/v1/chat/completions"

#: One request. Below the round's own deadline, so a hung provider loses its own call rather than
#: the round.
DEFAULT_TIMEOUT_SECONDS = 30.0
#: The judge reasons for longer than a fetch does.
DEFAULT_JUDGE_TIMEOUT_SECONDS = 60.0
#: A tweet re-scrape starts an actor run, which is slower than either.
DEFAULT_SCRAPE_TIMEOUT_SECONDS = 120.0

#: Refuse a response larger than this before buffering it. A provider is not hostile, but it is also
#: not something the lane should let allocate memory without bound.
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


def _post_json(url: str, payload: dict, *, headers: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"content-type": "application/json", **headers})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("response exceeds the size limit")
    return json.loads(raw.decode("utf-8", errors="replace"))


def _get_text(url: str, params: dict, *, timeout: float) -> str:
    full = f"{url}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(full, timeout=timeout) as response:
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("response exceeds the size limit")
    return raw.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------------------------


@dataclass
class ScrapingDogPages:
    """Fetches page bodies through ScrapingDog, one request per URL.

    Serial rather than concurrent, deliberately. Upstream fetches in parallel because it scores a
    whole epoch of miners; a Kata round fetches a handful of pages for two contestants, and a
    thread pool here would add a failure mode (partial results, ordering, shutdown) for a saving
    measured in seconds.

    A URL that fails is recorded as an empty body with a reason, NOT as a fetch failure: one dead
    link is a fact about that source, and failing the whole round over it would let any miner
    citing a flaky page take the round down. Only a failure to reach the provider AT ALL raises.
    """

    api_key: str
    url: str = DEFAULT_SCRAPINGDOG_URL
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    #: Set by the caller to meter `data_api_calls`. One entry per request actually made.
    calls: list = field(default_factory=list)

    def __call__(self, urls: list) -> dict:
        if not self.api_key:
            raise FetchUnavailable(f"{SCRAPINGDOG_KEY_ENV} is not set; the lane cannot verify a "
                                   f"miner's sources without fetching them")
        out: dict = {}
        reached_provider = False
        for target in urls:
            self.calls.append(target)
            try:
                body = _get_text(self.url, {"api_key": self.api_key, "url": target},
                                 timeout=self.timeout)
                reached_provider = True
            except urllib.error.HTTPError as exc:
                # 4xx/5xx for ONE url: the provider answered, so it is reachable. Record the miss.
                reached_provider = True
                out[target] = {"text": "", "title": "", "error": f"scrape failed ({exc.code})"}
                continue
            except (urllib.error.URLError, OSError, ValueError):
                out[target] = {"text": "", "title": "", "error": "scrape failed"}
                continue
            out[target] = _page_record(target, body)
        if urls and not reached_provider:
            # Never reached the provider for ANY url -- that is an outage, not a set of dead links.
            raise FetchUnavailable("the page fetch provider could not be reached for any URL")
        return out


def _page_record(url: str, body: str) -> dict:
    """ScrapingDog returns the page. JSON when it has metadata, raw HTML otherwise.

    ``raw_text`` is populated as well as ``text`` because the fetcher falls back to it when the
    extracted article is unusable -- upstream's own behaviour, and the reason a page whose extractor
    fails is not automatically an empty body.
    """
    try:
        document = json.loads(body)
    except ValueError:
        return {"text": body, "raw_text": body, "title": "", "url": url}
    if not isinstance(document, dict):
        return {"text": body, "raw_text": body, "title": "", "url": url}
    text = str(document.get("text") or document.get("content") or document.get("html") or "")
    return {"text": text, "raw_text": str(document.get("html") or text),
            "title": str(document.get("title") or ""),
            "published_date": str(document.get("published_date") or document.get("date") or ""),
            "author": str(document.get("author") or ""), "url": url}


# ---------------------------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------------------------


@dataclass
class OpenAiJudge:
    """Upstream's scoring model, called with upstream's messages.

    Model and temperature come from :mod:`kata_sn22.judge` rather than being restated: they are part
    of the scoring policy, and a second copy here is a second thing that can drift from it.
    """

    api_key: str
    url: str = DEFAULT_OPENAI_URL
    model: str = JUDGE_MODEL
    temperature: float = JUDGE_TEMPERATURE
    timeout: float = DEFAULT_JUDGE_TIMEOUT_SECONDS
    #: Set by the caller to meter `inference_calls`.
    calls: list = field(default_factory=list)

    def __call__(self, messages: list) -> str:
        if not self.api_key:
            raise JudgeUnavailable(f"{OPENAI_KEY_ENV} is not set; source relevance is decided by "
                                   f"the judge, and there is no judge")
        self.calls.append(len(messages or []))
        try:
            document = _post_json(
                self.url,
                {"model": self.model, "temperature": self.temperature, "messages": messages},
                headers={"authorization": f"Bearer {self.api_key}"}, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            raise JudgeUnavailable(f"the judge refused the request ({exc.code})") from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise JudgeUnavailable(f"the judge could not be reached: {exc}") from exc

        choices = document.get("choices") if isinstance(document, dict) else None
        if not isinstance(choices, list) or not choices:
            # An empty completion is NOT a LOW verdict. `verdict_score` would read it as 0.0 and the
            # source would silently score nothing, which is indistinguishable from a real judgement.
            raise JudgeUnavailable("the judge returned no completion")
        message = (choices[0] or {}).get("message") if isinstance(choices[0], dict) else None
        content = (message or {}).get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise JudgeUnavailable("the judge returned no message content")
        return content


# ---------------------------------------------------------------------------------------------
# Tweets
# ---------------------------------------------------------------------------------------------


@dataclass
class ApifyTweets:
    """Re-scrapes tweets by id through upstream's pinned Apify actor.

    The actor id is pinned for the same reason an image digest is: a different actor returns a
    different tweet shape, and the field-by-field comparison would start scoring honest miners zero
    for a difference that is the validator's.
    """

    api_key: str
    url: str = DEFAULT_APIFY_URL
    timeout: float = DEFAULT_SCRAPE_TIMEOUT_SECONDS
    #: Set by the caller to meter `scrape_units`.
    calls: list = field(default_factory=list)

    def __call__(self, tweet_ids: list) -> dict:
        if not self.api_key:
            raise ScrapeUnavailable(f"{APIFY_KEY_ENV} is not set; X results are verified by "
                                    f"re-scraping them, and there is no scraper")
        wanted = [str(tweet_id) for tweet_id in tweet_ids or []]
        if not wanted:
            return {}
        self.calls.append(len(wanted))
        try:
            items = _post_json(
                f"{self.url}?token={urllib.parse.quote(self.api_key)}",
                {"tweetIDs": wanted, "maxItems": len(wanted)},
                headers={}, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            raise ScrapeUnavailable(f"the tweet scraper refused the request ({exc.code})") from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise ScrapeUnavailable(f"the tweet scraper could not be reached: {exc}") from exc

        if not isinstance(items, list):
            raise ScrapeUnavailable("the tweet scraper returned no dataset")
        # A tweet the platform no longer serves is simply ABSENT from the result. That is a fact
        # about the tweet, handled by the caller; it is not a scraper failure.
        found = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            tweet_id = str(item.get("id") or item.get("id_str") or "")
            if tweet_id:
                found[tweet_id] = item
        return found


# ---------------------------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Transports:
    """The three things a lane needs to verify a round, plus what they spent."""

    page_transport: object
    judge_client: object
    tweet_scraper: object

    def usage(self) -> dict:
        """What was actually spent, for the budget meter."""
        return {
            "data_api_calls": len(getattr(self.page_transport, "calls", []) or []),
            "inference_calls": len(getattr(self.judge_client, "calls", []) or []),
            "scrape_units": sum(getattr(self.tweet_scraper, "calls", []) or []),
        }


def transports_from_env(environ=None) -> Transports:
    """Build the live transports from the deployment environment.

    A missing credential does NOT raise here. It raises when a round actually tries to use that
    transport, with a message naming the variable — because the three are needed at different
    moments (a challenge of only web tasks never re-scrapes a tweet), and refusing to construct
    would take down a lane over a capability it was not going to use.
    """
    env = os.environ if environ is None else environ
    return Transports(
        page_transport=ScrapingDogPages(
            api_key=env.get(SCRAPINGDOG_KEY_ENV, "").strip(),
            url=env.get(SCRAPINGDOG_URL_ENV, "").strip() or DEFAULT_SCRAPINGDOG_URL),
        judge_client=OpenAiJudge(
            api_key=env.get(OPENAI_KEY_ENV, "").strip(),
            url=env.get(OPENAI_URL_ENV, "").strip() or DEFAULT_OPENAI_URL),
        tweet_scraper=ApifyTweets(
            api_key=env.get(APIFY_KEY_ENV, "").strip(),
            url=env.get(APIFY_URL_ENV, "").strip() or DEFAULT_APIFY_URL),
    )
