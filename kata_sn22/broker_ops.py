"""The six SN22 broker operations: what a capability may actually cause to happen.

:mod:`room.broker` in ``kata-tee-runner`` decides *whether* a caller may invoke an operation. This
module decides *what an operation is* -- and that is where "an agent cannot select an arbitrary host
or model" is actually enforced, because the host, the model and the Apify actor id are constants
here and are never read out of the caller's payload.

===================  ==========  ============  =====================================================
Operation            Role        Provider      What it does
===================  ==========  ============  =====================================================
``web-search``       agent       scrapingdog   Google search for the agent
``x-search``         agent       apify         X/Twitter search for the agent
``final-summary``    agent       openai        the agent's own summary, on a FIXED small model
``web-page-fetch``   evaluator   scrapingdog   independently fetch a page the agent cited
``tweet-rescrape``   evaluator   apify         independently re-scrape a tweet the agent returned
``chutes-score``     evaluator   chutes        the fixed judge
===================  ==========  ============  =====================================================

**The asymmetry is the product rule.** The agent never reaches ``chutes``; the evaluator never
reaches ``openai``. An agent that could call the judge could grade its own work, and one that could
spend the evaluator's budget could starve its own verification and then claim it was never checked.

**Why the model is a constant and not a parameter.** If an agent could name the model, it could
name an expensive one and bill the miner -- or a different one from its opponent, which would make
the two summaries incomparable. ``final-summary`` is pinned to the same small model for everyone,
and ``chutes-score`` to the fixed judge. Both identities are part of the scoring policy hash
(:mod:`kata_sn22.scorer_policy`), so changing one changes the identity two contestants must share.

**Standard library only**, for the same reason as :mod:`kata_sn22.providers`: this code runs in the
process that holds four contestants' decrypted credentials, and every dependency there is another
thing to audit.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request

from kata_sn22.protocol_v2 import FIXED_SCORING_MODEL, MAX_RESULT_COUNT, MIN_RESULT_COUNT

# ---- fixed routes. NOT payload fields, NOT environment, NOT miner input -------------------------

SCRAPINGDOG_SEARCH_URL = "https://api.scrapingdog.com/google"
SCRAPINGDOG_SCRAPE_URL = "https://api.scrapingdog.com/scrape"
APIFY_ACTOR_BASE = "https://api.apify.com/v2/acts"
#: Upstream's pinned actors. A different actor returns a different shape, and the field-by-field
#: comparison would start failing honest miners for a difference that is the validator's.
APIFY_TWEET_ACTOR = "CJdippxWmn9uRfooo"
APIFY_SEARCH_ACTOR = "CJdippxWmn9uRfooo"
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
CHUTES_CHAT_URL = "https://llm.chutes.ai/v1/chat/completions"

#: The agent's summary model. Small and fixed: an agent that could name the model could name an
#: expensive one and bill the miner for it.
AGENT_SUMMARY_MODEL = "gpt-4.1-nano"
#: The judge. Same for both contestants or the duel means nothing.
JUDGE_MODEL = FIXED_SCORING_MODEL.value

# ---- bounds ------------------------------------------------------------------------------------

MAX_QUERY_CHARS = 1_000
MAX_URLS_PER_CALL = 50
MAX_TWEET_IDS_PER_CALL = 100
MAX_MESSAGES = 32
MAX_MESSAGE_CHARS = 100_000
MAX_RESPONSE_BYTES = 8 * 1024 * 1024

SEARCH_TIMEOUT_SECONDS = 30.0
FETCH_TIMEOUT_SECONDS = 30.0
SCRAPE_TIMEOUT_SECONDS = 120.0
JUDGE_TIMEOUT_SECONDS = 60.0

_TWEET_ID_RE = re.compile(r"^[0-9]{1,32}$")
_MESSAGE_ROLES = frozenset({"system", "user", "assistant"})


class OperationInputError(ValueError):
    """The payload is not something this operation will act on.

    Raised before any provider is contacted, so a malformed request costs the contestant nothing.
    The broker turns it into its single generic refusal; the detail here is for the room's own
    reasoning, never for the caller.
    """


# ---- validation ---------------------------------------------------------------------------------

def _text(payload: dict, field: str, *, limit: int, required: bool = True) -> str:
    value = payload.get(field)
    if value is None and not required:
        return ""
    if not isinstance(value, str) or not value.strip():
        raise OperationInputError(f"{field} must be a non-empty string")
    if len(value) > limit:
        raise OperationInputError(f"{field} exceeds {limit} characters")
    return value


def _count(payload: dict, field: str = "count") -> int:
    value = payload.get(field, MIN_RESULT_COUNT)
    if isinstance(value, bool) or not isinstance(value, int):
        raise OperationInputError(f"{field} must be an integer")
    if not MIN_RESULT_COUNT <= value <= MAX_RESULT_COUNT:
        raise OperationInputError(f"{field} must be {MIN_RESULT_COUNT}..{MAX_RESULT_COUNT}")
    return value


def _urls(payload: dict) -> list:
    value = payload.get("urls")
    if not isinstance(value, list) or not value:
        raise OperationInputError("urls must be a non-empty list")
    if len(value) > MAX_URLS_PER_CALL:
        raise OperationInputError(f"at most {MAX_URLS_PER_CALL} urls per call")
    out = []
    for item in value:
        if not isinstance(item, str):
            raise OperationInputError("each url must be a string")
        parsed = urllib.parse.urlsplit(item)
        # http(s) only, and no embedded credentials. A `file://` or `gopher://` target would make
        # this operation a reader of the room's own filesystem on behalf of the caller.
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise OperationInputError("each url must be an absolute http(s) URL")
        if "@" in parsed.netloc:
            raise OperationInputError("a url must not carry embedded credentials")
        out.append(item)
    return out


def _tweet_ids(payload: dict) -> list:
    value = payload.get("tweet_ids")
    if not isinstance(value, list) or not value:
        raise OperationInputError("tweet_ids must be a non-empty list")
    if len(value) > MAX_TWEET_IDS_PER_CALL:
        raise OperationInputError(f"at most {MAX_TWEET_IDS_PER_CALL} tweet ids per call")
    out = []
    for item in value:
        text = str(item)
        if not _TWEET_ID_RE.fullmatch(text):
            raise OperationInputError("each tweet id must be numeric")
        out.append(text)
    return out


def _messages(payload: dict) -> list:
    value = payload.get("messages")
    if not isinstance(value, list) or not value:
        raise OperationInputError("messages must be a non-empty list")
    if len(value) > MAX_MESSAGES:
        raise OperationInputError(f"at most {MAX_MESSAGES} messages")
    total = 0
    out = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"role", "content"}:
            raise OperationInputError("each message must be exactly {role, content}")
        role, content = item["role"], item["content"]
        if role not in _MESSAGE_ROLES:
            raise OperationInputError("message role must be system, user or assistant")
        if not isinstance(content, str):
            raise OperationInputError("message content must be a string")
        total += len(content)
        if total > MAX_MESSAGE_CHARS:
            raise OperationInputError(f"messages exceed {MAX_MESSAGE_CHARS} characters in total")
        out.append({"role": role, "content": content})
    return out


# ---- transport -----------------------------------------------------------------------------------

def _post_json(url: str, payload: dict, *, headers: dict, timeout: float) -> object:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"content-type": "application/json", **headers})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise OperationInputError("provider response exceeds the size limit")
    return json.loads(raw.decode("utf-8", errors="replace"))


def _get_text(url: str, params: dict, *, timeout: float) -> str:
    with urllib.request.urlopen(
        f"{url}?{urllib.parse.urlencode(params)}", timeout=timeout
    ) as response:
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise OperationInputError("provider response exceeds the size limit")
    return raw.decode("utf-8", errors="replace")


def _chat_completion(url: str, *, api_key: str, model: str, messages: list, temperature: float,
                     timeout: float) -> str:
    """One chat completion against a FIXED url and a FIXED model.

    ``model`` is a parameter of this helper and a constant at every call site. It is deliberately
    not reachable from ``payload``.
    """
    document = _post_json(
        url,
        {"model": model, "temperature": temperature, "messages": messages},
        headers={"authorization": f"Bearer {api_key}"}, timeout=timeout)
    choices = document.get("choices") if isinstance(document, dict) else None
    if not isinstance(choices, list) or not choices:
        raise OperationInputError("the model returned no completion")
    message = (choices[0] or {}).get("message") if isinstance(choices[0], dict) else None
    content = (message or {}).get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise OperationInputError("the model returned no message content")
    return content


# ---- agent operations ---

def web_search(api_key: str, payload: dict) -> dict:
    """Google search for the agent. The endpoint is fixed; only the query and count are the
    caller's."""
    query = _text(payload, "query", limit=MAX_QUERY_CHARS)
    count = _count(payload)
    body = _get_text(
        SCRAPINGDOG_SEARCH_URL,
        {"api_key": api_key, "query": query, "results": count},
        timeout=SEARCH_TIMEOUT_SECONDS)
    try:
        document = json.loads(body)
    except ValueError:
        return {"results": []}
    results = document.get("organic_results") if isinstance(document, dict) else None
    return {"results": results if isinstance(results, list) else []}


def x_search(api_key: str, payload: dict) -> dict:
    """X/Twitter search for the agent, through the pinned actor."""
    query = _text(payload, "query", limit=MAX_QUERY_CHARS)
    count = _count(payload)
    items = _post_json(
        f"{APIFY_ACTOR_BASE}/{APIFY_SEARCH_ACTOR}/run-sync-get-dataset-items"
        f"?token={urllib.parse.quote(api_key)}",
        {"searchTerms": [query], "maxItems": count},
        headers={}, timeout=SCRAPE_TIMEOUT_SECONDS)
    return {"results": items if isinstance(items, list) else []}


def final_summary(api_key: str, payload: dict) -> dict:
    """The agent's own final summary, on a fixed small model.

    The agent chooses the messages -- it is writing its own answer -- but not the model and not the
    host. Otherwise one contestant could summarise with a frontier model and the other with a small
    one, and the comparison would be measuring budget rather than skill.
    """
    messages = _messages(payload)
    content = _chat_completion(
        OPENAI_CHAT_URL, api_key=api_key, model=AGENT_SUMMARY_MODEL, messages=messages,
        temperature=0.0, timeout=JUDGE_TIMEOUT_SECONDS)
    return {"content": content, "model": AGENT_SUMMARY_MODEL}


# ---- evaluator operations ---

def web_page_fetch(api_key: str, payload: dict) -> dict:
    """Independently fetch the pages the agent cited, so its citations can be checked.

    A URL that fails is recorded as an empty body with a reason rather than failing the call: one
    dead link is a fact about that source, and failing the whole verification over it would let any
    contestant citing a flaky page take the round down.
    """
    urls = _urls(payload)
    pages: dict = {}
    reached = False
    for target in urls:
        try:
            body = _get_text(SCRAPINGDOG_SCRAPE_URL, {"api_key": api_key, "url": target},
                             timeout=FETCH_TIMEOUT_SECONDS)
            reached = True
        except urllib.error.HTTPError as exc:
            reached = True   # the provider answered; this URL is simply dead
            pages[target] = {"text": "", "title": "", "error": f"scrape failed ({exc.code})"}
            continue
        except (urllib.error.URLError, OSError, ValueError):
            pages[target] = {"text": "", "title": "", "error": "scrape failed"}
            continue
        pages[target] = {"text": body, "url": target}
    if not reached:
        # Never reached the provider for ANY url. That is an outage, not a set of dead links, and
        # the difference decides whether a contestant is scored or the duel defers.
        raise OperationInputError("the page fetch provider could not be reached for any URL")
    return {"pages": pages}


def tweet_rescrape(api_key: str, payload: dict) -> dict:
    """Independently re-scrape tweets the agent returned, through the pinned actor."""
    tweet_ids = _tweet_ids(payload)
    items = _post_json(
        f"{APIFY_ACTOR_BASE}/{APIFY_TWEET_ACTOR}/run-sync-get-dataset-items"
        f"?token={urllib.parse.quote(api_key)}",
        {"tweetIDs": tweet_ids, "maxItems": len(tweet_ids)},
        headers={}, timeout=SCRAPE_TIMEOUT_SECONDS)
    return {"tweets": items if isinstance(items, list) else []}


def chutes_score(api_key: str, payload: dict) -> dict:
    """The fixed judge. Never reachable with an agent capability."""
    from kata_sn22.scorer_policy import JUDGE_TEMPERATURE

    messages = _messages(payload)
    content = _chat_completion(
        CHUTES_CHAT_URL, api_key=api_key, model=JUDGE_MODEL, messages=messages,
        temperature=JUDGE_TEMPERATURE, timeout=JUDGE_TIMEOUT_SECONDS)
    return {"content": content, "model": JUDGE_MODEL}


# ---- the declared set ---

#: ``(name, role, provider, handler, max_calls)``. Consumed by the runner profile, which turns each
#: into a ``room.broker.OperationSpec``. Kept as plain data here so this module -- which runs in the
#: sandbox too -- does not import the room package.
OPERATIONS: tuple = (
    ("web-search", "agent", "scrapingdog", web_search, 64),
    ("x-search", "agent", "apify", x_search, 64),
    ("final-summary", "agent", "openai", final_summary, 8),
    ("web-page-fetch", "evaluator", "scrapingdog", web_page_fetch, 256),
    ("tweet-rescrape", "evaluator", "apify", tweet_rescrape, 256),
    ("chutes-score", "evaluator", "chutes", chutes_score, 512),
)

AGENT_OPERATIONS = tuple(name for name, role, *_ in OPERATIONS if role == "agent")
EVALUATOR_OPERATIONS = tuple(name for name, role, *_ in OPERATIONS if role == "evaluator")
