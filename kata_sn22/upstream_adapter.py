"""The pinned SN22 upstream scoring adapter (plan §5.3, SN22-5).

This module is a **dependency-free port** of the pure scoring components of
`Desearch-ai/subnet-22` at the audited commit. The complete upstream tree is vendored beside it
(``upstream/``, pinned by :mod:`kata_sn22.upstream_snapshot`), and :mod:`kata_sn22.parity` executes
the *real* upstream callables against the same recorded inputs to prove this port agrees with them.

**Why a port instead of importing the upstream.** Importing it would drag `bittensor`, `wandb`,
`openai`, `aiohttp`, `apify_client` and a chain of network clients into the resident runtime — a
lane that scores search answers would then carry a chain client and a wallet library. The port
keeps the production runtime at zero third-party dependencies; the vendored tree is what the parity
harness runs, under a shim, at review time. So "we match upstream" is a *tested claim* rather than
an architectural accident.

**What is adapted, and what is deliberately not.**

Adapted (every one is a pure function of a response):

* the pool weights — AI search 0.90 / X search 0.10, and within AI search fast 0.60 /
  balanced 0.20 / deep 0.20;
* the AI quality split — content relevance 0.60 / summary relevance 0.40, with the 0.30 component
  floors, and the ONLY_LINKS reweighting to (1.0, 0.0);
* the performance curve and its per-mode floors;
* nine cheap penalties: count, duplicate results, result schema, domain filter, date range, sort
  order, minimum realistic time, summary structure, timeout;
* the reward-combination arithmetic that turns the above into one number;
* the EVIDENCE layer — `link_meets_evidence` and the ordered-highlight matching it rests on, the
  cited-body collection the groundedness judge reads, and the random-link spot check. This is
  upstream's anti-fabrication rule, and it decides which links are worth paying a judge to score.

NOT adapted, and none of these is an oversight:

* on-chain weight setting, wallet/metagraph operations, PM2, W&B, the public validator API — plan
  §3 puts them outside the lane;
* the pool-relative emission calculation, which depends on miner volume and an hourly scoring
  population. Kata compares exactly two agents on one sealed challenge, so there is no pool;
* `streaming_penalty`, which counts tokens per streamed chunk with `tiktoken`. Kata's protocol is
  one JSON response, not a stream, so the penalty has no input here;
* `summary_rule_penalty`, whose prompt asks a judge to check a summary against style rules Kata's
  protocol already enforces structurally;
* the judge CALL itself and the page/tweet fetchers. Those do I/O and cost money, so they live
  behind their own seams; what this module supplies is the pure decision about *what* is worth
  fetching and judging, and *what* the judge is shown.

Every adapted symbol is registered in :mod:`kata_sn22.parity` against its upstream file and digest,
so the set above cannot drift from what is actually proven.
"""
from __future__ import annotations

import html
import math
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qsl, urlencode, urlparse, urlsplit, urlunsplit

# ---------------------------------------------------------------------------------------------
# Pool weights — upstream/neurons/validators/scoring/constants.py
# ---------------------------------------------------------------------------------------------

SEARCH_TYPE_WEIGHTS: dict[str, float] = {"ai_search": 0.90, "x_search": 0.10}
AI_MODE_WEIGHTS: dict[str, float] = {"fast": 0.60, "balanced": 0.20, "deep": 0.20}

#: The share of the whole pool each (search_type, mode) lane carries. Upstream builds this by
#: multiplying the two tables above; reproduced by the same construction rather than by writing the
#: products out, so the two tables stay the single source of truth.
POOL_SHARES: dict[tuple[str, str | None], float] = {
    ("ai_search", mode): SEARCH_TYPE_WEIGHTS["ai_search"] * weight
    for mode, weight in AI_MODE_WEIGHTS.items()
}
POOL_SHARES[("x_search", None)] = SEARCH_TYPE_WEIGHTS["x_search"]

QUALITY_THRESHOLDS: dict[str, float] = {"ai_search": 0.50, "x_search": 0.60}
QUALITY_EXPONENT = 3.0
VOLUME_EXPONENT = 2.0

# ---------------------------------------------------------------------------------------------
# Quality split — upstream/neurons/validators/scrapers/{advanced,x}_scraper_validator.py
# ---------------------------------------------------------------------------------------------

AI_CONTENT_WEIGHT = 0.60
AI_SUMMARY_WEIGHT = 0.40
#: A component scoring below its floor zeroes the *quality gate* (not the reward). Upstream uses the
#: gate for capacity ramping: a miner can still be paid for a mediocre answer, but it does not earn
#: more concurrency on one.
AI_COMPONENT_FLOORS = (0.30, 0.30)
X_CONTENT_WEIGHT = 1.0

RESULT_TYPE_ONLY_LINKS = "ONLY_LINKS"
RESULT_TYPE_LINKS_WITH_FINAL_SUMMARY = "LINKS_WITH_FINAL_SUMMARY"

# ---------------------------------------------------------------------------------------------
# Performance — upstream/neurons/validators/reward/performance_reward.py
# ---------------------------------------------------------------------------------------------

AI_PERF_FLOOR = 0.50
X_PERF_FLOOR = 0.70
MODE_PERF_FLOORS: dict[str, float] = {"fast": 0.40, "balanced": 0.50, "deep": 0.85}
#: Seconds a mode is budgeted (upstream/desearch/utils.py MODE_BUDGETS).
MODE_BUDGETS: dict[str, int] = {"fast": 5, "balanced": 15, "deep": 30}

FINAL_SUMMARY_ROLE = "summary"
TWITTER_TOOL = "Twitter Search"
SEARCH_SUMMARY_TOOLS = ("Web Search",)
AI_SEARCH_RESULT_FIELDS = ("search_results",)


# ---------------------------------------------------------------------------------------------
# The response shape the ported components read
# ---------------------------------------------------------------------------------------------


@dataclass
class UpstreamResponse:
    """The fields the adapted components actually read off an upstream synapse.

    Upstream's `ScraperStreamingSynapse` and `TwitterSearchSynapse` are pydantic models carrying a
    hundred fields, most of them transport. This carries the scoring surface and nothing else, so
    what the components depend on is visible rather than implied. :mod:`kata_sn22.parity` builds a
    real synapse and one of these from the same case and checks both score identically.

    ``kind`` replaces upstream's `isinstance` dispatch: every ported penalty branches on the synapse
    class, and a string is the honest version of that once the classes are gone.
    """

    kind: str                                     # "ai_search" | "x_search"
    count: int | None = 10
    tools: tuple[str, ...] = ()
    result_type: str = RESULT_TYPE_LINKS_WITH_FINAL_SUMMARY
    mode: str | None = None                       # fast | balanced | deep, AI search only
    sort: str | None = None                       # "Latest" | "Top", X search only

    #: AI search result lists.
    miner_tweets: tuple[dict, ...] = ()
    search_results: tuple[dict, ...] = ()
    #: X search result list.
    results: tuple[dict, ...] = ()

    texts: dict[str, str] = field(default_factory=dict)
    include_domains: tuple[str, ...] = ()
    exclude_domains: tuple[str, ...] = ()
    start_date: str | None = None
    end_date: str | None = None

    #: Timing. ``process_time`` is what the lane measured; the other two bound it.
    process_time: float | None = None
    max_execution_time: float | None = None
    timeout: float | None = None

    #: Whether the response was usable at all. Upstream reads `dendrite.status_code == 200`; Kata
    #: has no dendrite, and its equivalent is "the output parsed and validated by the protocol".
    successful: bool = True

    def __post_init__(self) -> None:
        if self.kind not in ("ai_search", "x_search"):
            raise ValueError(f"unknown response kind {self.kind!r}")
        if self.kind == "ai_search" and self.mode is not None and self.mode not in MODE_BUDGETS:
            raise ValueError(f"unknown ai_search mode {self.mode!r}")


# ---------------------------------------------------------------------------------------------
# response_checks — upstream/neurons/validators/utils/response_checks.py
# ---------------------------------------------------------------------------------------------

MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
HEADER_HASH_PATTERN = re.compile(r"^#{1,6}\s", re.MULTILINE)

_TRACKING_PARAMS = frozenset({
    "fbclid", "gclid", "gclsrc", "dclid", "gbraid", "wbraid", "msclkid", "yclid", "twclid",
    "igshid", "igsh", "mc_cid", "mc_eid", "_ga", "_gl", "mkt_tok", "si", "feature", "spm", "scm",
})


def extract_markdown_links(text: str) -> list[tuple[str, str]]:
    return MARKDOWN_LINK_PATTERN.findall(text or "")


def check_markdown_structure(text: str) -> tuple[bool, list[str]]:
    """Reject ``#`` headers and empty responses; bold headers are no longer required."""
    issues = []
    if HEADER_HASH_PATTERN.search(text or ""):
        issues.append("Uses # headers instead of **")
    if not (text or "").strip():
        issues.append("Empty response")
    return len(issues) == 0, issues


def normalize_source_url(url: str) -> str:
    """Lowercase, drop ``www.``, drop a trailing slash. Scheme and query are KEPT."""
    url = (url or "").strip().lower()
    if url.startswith("https://www."):
        url = "https://" + url[len("https://www."):]
    elif url.startswith("http://www."):
        url = "http://" + url[len("http://www."):]
    return url.removesuffix("/")


def _is_tracking_param(key: str) -> bool:
    key = key.lower()
    return key.startswith("utm_") or key in _TRACKING_PARAMS


def source_key(url: str) -> str:
    """Identity key for a URL: base plus content query params, tracking params dropped."""
    parts = urlsplit(url or "")
    base = normalize_source_url(urlunsplit((parts.scheme, parts.netloc, parts.path, "", "")))
    kept = sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                  if not _is_tracking_param(k))
    return f"{base}?{urlencode(kept)}" if kept else base


def first_duplicate_id(items, key: str = "id", normalize=None):
    """The first repeated key value, or ``None`` when every value is unique."""
    seen: set = set()
    for item in items or []:
        value = item.get(key) if isinstance(item, dict) else getattr(item, key, None)
        if value is None:
            continue
        if normalize is not None:
            value = normalize(value)
        if value in seen:
            return value
        seen.add(value)
    return None


_TWEET_DATE_FORMATS = ("%a %b %d %H:%M:%S %z %Y", "%Y-%m-%d_%H:%M:%S_%Z", "%Y-%m-%dT%H:%M:%SZ")


def parse_tweet_date(value: str) -> datetime | None:
    """Twitter's ``%a %b %d %H:%M:%S %z %Y`` or ISO 8601, defaulting to UTC when naive.

    Upstream defaults with ``pytz.UTC``; ``datetime.timezone.utc`` is the same offset and keeps the
    port dependency-free. The parity harness compares parsed instants, so the substitution is
    checked rather than asserted.
    """
    if not value:
        return None
    for fmt in _TWEET_DATE_FORMATS:
        try:
            parsed = datetime.strptime(value, fmt)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def tweet_date_in_range(created_at: str, start_date: str | None, end_date: str | None) -> bool:
    """Inclusive on both ends; a missing bound skips that side. Unparseable dates are OUT."""
    parsed = parse_tweet_date(created_at)
    if parsed is None:
        return False
    if start_date:
        start = parse_tweet_date(start_date)
        if start and parsed < start:
            return False
    if end_date:
        end = parse_tweet_date(end_date)
        if end and parsed > end:
            return False
    return True


def is_descending_by_created_at(items) -> bool:
    """For X ``sort=Latest``: non-increasing ``created_at``, and every item must carry one."""
    previous: datetime | None = None
    for item in items or []:
        created = item.get("created_at") if isinstance(item, dict) else None
        if not created:
            return False
        parsed = parse_tweet_date(created)
        if parsed is None:
            return False
        if previous is not None and parsed > previous:
            return False
        previous = parsed
    return True


def collect_summary_sources(response: UpstreamResponse) -> set[str]:
    """Every URL the miner returned that a summary link could legitimately reference."""
    sources: set[str] = set()
    for tweet in response.miner_tweets or ():
        if not isinstance(tweet, dict):
            continue
        username = (tweet.get("user") or {}).get("username", "")
        tweet_id = tweet.get("id", "")
        if username and tweet_id:
            sources.add(normalize_source_url(f"https://x.com/{username}/status/{tweet_id}"))
    for field_name in AI_SEARCH_RESULT_FIELDS:
        for result in getattr(response, field_name, ()) or ():
            link = result.get("link") if isinstance(result, dict) else getattr(result, "link", None)
            if link:
                sources.add(normalize_source_url(link))
    return sources


# ---------------------------------------------------------------------------------------------
# web_query_operators — upstream/neurons/validators/utils/web_query_operators.py
# ---------------------------------------------------------------------------------------------

_SITE_RE = re.compile(r"(?i)\bsite:(\S+)")


def _normalize_domain(raw: str) -> str:
    domain = raw.strip().strip("/").lower()
    domain = re.sub(r"^https?://", "", domain)
    domain = domain.split("/", 1)[0]
    return domain.rstrip(".")


def normalize_domains(raw) -> list[str]:
    """Order-preserving dedupe of host names, with scheme/path/trailing dot stripped."""
    seen: list[str] = []
    for item in raw or ():
        domain = _normalize_domain(item)
        if domain and domain not in seen:
            seen.append(domain)
    return seen


def host_in_domains(url: str, domains) -> bool:
    """Exact host match or a subdomain of one of ``domains``. Empty list is always False."""
    if not domains:
        return False
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    if not host:
        return False
    return any(host == domain or host.endswith("." + domain) for domain in domains)


@dataclass
class WebQueryOperators:
    text: str
    sites: list[str] = field(default_factory=list)

    def host_allowed(self, url: str) -> bool:
        if not self.sites:
            return True
        host = (urlparse(url).hostname or "").lower().rstrip(".")
        if not host:
            return False
        return any(host == site or host.endswith("." + site) for site in self.sites)


def parse_web_query(query: str) -> WebQueryOperators:
    """Pull ``site:`` operators out of a raw web query and return the cleaned text."""
    if not query:
        return WebQueryOperators(text="")
    sites = [domain for raw in _SITE_RE.findall(query) if (domain := _normalize_domain(raw))]
    stripped = re.sub(r"\s+", " ", _SITE_RE.sub("", query)).strip()
    return WebQueryOperators(text=stripped, sites=sites)


# ---------------------------------------------------------------------------------------------
# format_text_for_match — upstream/desearch/utils.py
# ---------------------------------------------------------------------------------------------


def format_text_for_match(text: str) -> str:
    """Normalize tweet text for comparison: unescape, strip URLs and leading mentions, drop
    whitespace, truncate to 280 characters."""
    text = html.unescape(text)
    text = re.sub(r"(https?://)?\S+\.\S+\/?(\S+)?", "", text)
    text = re.sub(r"^(@\w+\s*)+", "", text)
    text = re.sub(r"\s+", "", text)
    return text[:280]


# ---------------------------------------------------------------------------------------------
# Result validity — upstream/desearch/utils.py is_valid_tweet / is_valid_web_search_result
#                   plus upstream/neurons/validators/penalty/result_schema_penalty.py
# ---------------------------------------------------------------------------------------------

#: Required (non-Optional, no default) fields of upstream's ``TwitterScraperTweet``. A dict missing
#: any of these fails pydantic construction, which is what `is_valid_tweet` reports.
_TWEET_REQUIRED_FIELDS = ("id", "text", "reply_count", "retweet_count", "like_count",
                          "quote_count", "bookmark_count", "url", "created_at",
                          "is_quote_tweet", "is_retweet")
_TWEET_INT_FIELDS = ("reply_count", "retweet_count", "like_count", "quote_count", "bookmark_count")
#: Required fields of ``WebSearchResult``.
_WEB_RESULT_REQUIRED_FIELDS = ("title", "snippet", "link")


def is_valid_tweet(item) -> bool:
    """Whether upstream's ``TwitterScraperTweet(**item)`` would construct.

    Reproduced field-by-field rather than by carrying pydantic: the model's *required* fields are
    the whole of what the check tests, and the parity harness runs the real model on every case.

    The nested ``user`` is validated too, because it is the one sub-model the relay populates and
    `TwitterScraperUser` has its own required fields — a tweet whose author has no id is rejected
    upstream, and a port that accepted it would score a fabricated author as a real one.
    """
    if not isinstance(item, dict):
        return False
    for name in _TWEET_REQUIRED_FIELDS:
        if name not in item:
            return False
    user = item.get("user")
    if user is not None:
        if not isinstance(user, dict):
            return False
        if not isinstance(user.get("id"), str) or not isinstance(user.get("username"), str):
            return False
    for name in ("id", "text", "created_at"):
        if not isinstance(item.get(name), str):
            return False
    for name in _TWEET_INT_FIELDS:
        value = item.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            return False
    url = item.get("url")
    if url is not None and not isinstance(url, str):
        return False
    for name in ("is_quote_tweet", "is_retweet"):
        value = item.get(name)
        if value is not None and not isinstance(value, bool):
            return False
    return True


def is_valid_web_search_result(item) -> bool:
    if not isinstance(item, dict):
        return False
    return all(isinstance(item.get(name), str) for name in _WEB_RESULT_REQUIRED_FIELDS)


def _tweet_item_ok(item) -> bool:
    """`result_schema_penalty._is_valid_tweet`: valid schema AND non-empty required content."""
    if not is_valid_tweet(item):
        return False
    return all(item.get(name) for name in ("id", "text", "url", "created_at"))


def _search_item_ok(item) -> bool:
    """`result_schema_penalty._is_valid_search_item`."""
    if isinstance(item, dict):
        if not is_valid_web_search_result(item):
            return False
        title, link, snippet = item.get("title"), item.get("link"), item.get("snippet")
    else:
        title = getattr(item, "title", None)
        link = getattr(item, "link", None)
        snippet = getattr(item, "snippet", None)
    return all((title, link, snippet))


# ---------------------------------------------------------------------------------------------
# Performance curve — upstream/neurons/validators/reward/performance_reward.py
# ---------------------------------------------------------------------------------------------


def perf_factor(perf_raw: float, floor: float) -> float:
    """Map a raw performance reward into the multiplier applied to quality."""
    return floor + (1.0 - floor) * perf_raw


def perf_floor_for(response: UpstreamResponse, default: float) -> float:
    return MODE_PERF_FLOORS.get(response.mode, default)


def resolve_scoring_budget(response: UpstreamResponse) -> float:
    """Seconds the response was budgeted: the mode budget when there is a mode, else the declared
    ``max_execution_time``."""
    if response.mode:
        try:
            return float(MODE_BUDGETS[response.mode])
        except (KeyError, ValueError):
            pass
    raw = response.max_execution_time
    try:
        return float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def min_realistic_for_budget(budget: float) -> float:
    """Faster than this is not a search, it is a cache."""
    if not budget or budget <= 0:
        return 1.0
    return min(1.0, 0.3 * budget)


def performance_reward(axon_time: float, budget: float) -> float:
    """Upstream's piecewise speed curve.

    Zero below ``min_realistic`` — implausibly fast means cached, not fast — 1.0 up to 60% of the
    budget, then a straight decline to zero at the budget plus a grace of at most five seconds.
    """
    if budget and budget > 0:
        min_realistic = min_realistic_for_budget(budget)
        target = 0.6 * budget
        zero_point = budget + min(5.0, 0.5 * budget)
    else:
        min_realistic, target = 1.0, 3.0
        zero_point = 8.0

    if axon_time < min_realistic or axon_time >= zero_point:
        return 0.0
    if axon_time <= target:
        return 1.0
    return 1.0 - (axon_time - target) / (zero_point - target)


def response_time_for(response: UpstreamResponse) -> float:
    """The time fed into the curve. An unsuccessful response is pinned to its ceiling, so it
    resolves to reward 0 rather than to whatever it happened to spend before failing."""
    ceiling = float(response.max_execution_time or 0.0)
    if response.successful and response.process_time is not None:
        return float(response.process_time)
    return ceiling


# ---------------------------------------------------------------------------------------------
# Cheap penalties — upstream/neurons/validators/penalty/*.py
# ---------------------------------------------------------------------------------------------

_URL_KEYS = frozenset({"link", "url"})


def count_penalty(response: UpstreamResponse, max_penalty: float = 1.0) -> float:
    """Shortfall against the requested count. AI search takes the worst per-group shortfall."""
    requested = response.count
    if not requested or requested <= 0:
        return 0.0

    if response.kind == "x_search":
        got = len(response.results or ())
        if got >= requested:
            return 0.0
        return min(1 - got / requested, max_penalty)

    tools = set(response.tools or ())
    group_totals: list[int] = []
    if TWITTER_TOOL in tools:
        group_totals.append(len(response.miner_tweets or ()))
    if any(tool in tools for tool in SEARCH_SUMMARY_TOOLS):
        group_totals.append(sum(len(getattr(response, f, ()) or ())
                                for f in AI_SEARCH_RESULT_FIELDS))
    worst = 0.0
    for got in group_totals:
        if got < requested:
            worst = max(worst, 1 - got / requested)
    return min(worst, max_penalty)


def _result_groups(response: UpstreamResponse):
    """``(items, dedup_keys, check_text)`` for every result list a duplicate check applies to."""
    if response.kind == "x_search":
        yield response.results or (), ("id", "url"), True
        return
    if response.miner_tweets:
        yield response.miner_tweets, ("id", "url"), True
    for field_name in AI_SEARCH_RESULT_FIELDS:
        yield getattr(response, field_name, ()) or (), ("link", "url"), False


def _has_duplicate_text(items) -> bool:
    seen: set[str] = set()
    for item in items or ():
        text = item.get("text") if isinstance(item, dict) else getattr(item, "text", "")
        normalized = format_text_for_match(text or "").lower()
        if not normalized:
            continue
        if normalized in seen:
            return True
        seen.add(normalized)
    return False


def duplicate_results_penalty(response: UpstreamResponse, max_penalty: float = 1.0) -> float:
    """All-or-nothing: any repeated id, URL or tweet text is the full penalty.

    Not proportional, unlike the schema penalty, because padding a result list with copies is a
    deliberate act rather than a quality gradient."""
    for items, keys, check_text in _result_groups(response):
        for key in keys:
            normalize = source_key if key in _URL_KEYS else None
            if first_duplicate_id(items, key=key, normalize=normalize) is not None:
                return max_penalty
        if check_text and _has_duplicate_text(items):
            return max_penalty
    return 0.0


def _schema_groups(response: UpstreamResponse):
    if response.kind == "x_search":
        yield response.results or (), _tweet_item_ok
        return
    yield response.miner_tweets or (), _tweet_item_ok
    for field_name in AI_SEARCH_RESULT_FIELDS:
        yield getattr(response, field_name, ()) or (), _search_item_ok


def result_schema_penalty(response: UpstreamResponse, max_penalty: float = 1.0) -> float:
    """Fraction of results that fail their schema or have an empty required content field."""
    total = 0
    invalid = 0
    for items, validator in _schema_groups(response):
        for item in items:
            total += 1
            if not validator(item):
                invalid += 1
    if total == 0:
        return 0.0
    return min(invalid / total, max_penalty)


def domain_filter_penalty(response: UpstreamResponse, max_penalty: float = 1.0) -> float:
    """Fraction of search links violating the requested include/exclude domain filters."""
    if response.kind != "ai_search":
        return 0.0
    include = normalize_domains(response.include_domains)
    exclude = normalize_domains(response.exclude_domains)
    if not include and not exclude:
        return 0.0

    links = []
    for result in response.search_results or ():
        link = result.get("link") if isinstance(result, dict) else getattr(result, "link", None)
        if link:
            links.append(link)
    if not links:
        return 0.0

    def violates(link: str) -> bool:
        if include and not host_in_domains(link, include):
            return True
        if exclude and host_in_domains(link, exclude):
            return True
        return False

    violations = sum(1 for link in links if violates(link))
    return min(violations / len(links), max_penalty)


def date_range_penalty(response: UpstreamResponse, max_penalty: float = 1.0) -> float:
    """Fraction of tweets whose claimed ``created_at`` falls outside the requested window."""
    if response.kind == "x_search":
        tweets = response.results or ()
    else:
        tweets = response.miner_tweets or ()
    start_date, end_date = response.start_date, response.end_date
    if not tweets or (not start_date and not end_date):
        return 0.0

    checked = 0
    out_of_range = 0
    for tweet in tweets:
        created = tweet.get("created_at") if isinstance(tweet, dict) else None
        if not created:
            continue
        checked += 1
        if not tweet_date_in_range(created, start_date, end_date):
            out_of_range += 1
    if checked == 0:
        return 0.0
    return min(out_of_range / checked, max_penalty)


def sort_order_penalty(response: UpstreamResponse, max_penalty: float = 1.0) -> float:
    """X ``sort=Latest`` results that are not in descending ``created_at`` order."""
    if response.kind != "x_search" or response.sort != "Latest":
        return 0.0
    if not is_descending_by_created_at(response.results or ()):
        return max_penalty
    return 0.0


def min_realistic_time_penalty(response: UpstreamResponse, max_penalty: float = 1.0) -> float:
    """Answered faster than a real search for its budget could be: almost certainly cached."""
    process_time = response.process_time
    if process_time is None:
        return 0.0
    if float(process_time) < min_realistic_for_budget(resolve_scoring_budget(response)):
        return max_penalty
    return 0.0


def summary_structure_penalty(response: UpstreamResponse, max_penalty: float = 1.0) -> float:
    """Bad markdown, no links, or a link that is not one of the miner's own returned sources."""
    if response.kind != "ai_search":
        return 0.0
    if response.result_type == RESULT_TYPE_ONLY_LINKS:
        return 0.0
    summary = (response.texts or {}).get(FINAL_SUMMARY_ROLE, "")
    ok_structure, _issues = check_markdown_structure(summary)
    if not ok_structure:
        return max_penalty
    links = [url for _text, url in extract_markdown_links(summary)]
    if not links:
        return max_penalty
    sources = collect_summary_sources(response)
    if any(normalize_source_url(link) not in sources for link in links):
        return max_penalty
    return 0.0


DEFAULT_TIMEOUT_GRACE_SECONDS = 5.0


def _safe_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def timeout_penalty(response: UpstreamResponse, max_penalty: float = 1.0,
                    timeout_grace_seconds: float = DEFAULT_TIMEOUT_GRACE_SECONDS) -> float:
    """Overrun past ``max_execution_time``, charged per whole second of the grace window.

    A missing timing figure is the FULL penalty, not zero: "we could not tell how long this took" is
    not evidence that it was fast.
    """
    process_time = _safe_float(response.process_time)
    max_execution_time = _safe_float(response.max_execution_time)
    if process_time is None or max_execution_time is None:
        return max_penalty
    if process_time <= max_execution_time:
        return 0.0

    declared_timeout = _safe_float(response.timeout)
    if declared_timeout is None or declared_timeout <= max_execution_time:
        window = timeout_grace_seconds
    else:
        window = declared_timeout - max_execution_time
    window = max(window, 1e-6)

    elapsed_seconds = math.ceil(process_time - max_execution_time)
    return min(elapsed_seconds * (max_penalty / window), max_penalty)


#: The penalties each search type applies, in the upstream scraper's declared order. Order does not
#: change the product, but it does change which name a reviewer sees first in a report.
AI_PENALTIES = ("timeout_penalty", "min_realistic_time_penalty", "count_penalty",
                "summary_structure_penalty", "duplicate_results_penalty",
                "result_schema_penalty", "date_range_penalty", "domain_filter_penalty")
X_PENALTIES = ("timeout_penalty", "min_realistic_time_penalty", "count_penalty",
               "duplicate_results_penalty", "result_schema_penalty", "date_range_penalty",
               "sort_order_penalty")

PENALTY_FUNCTIONS = {
    "count_penalty": count_penalty,
    "duplicate_results_penalty": duplicate_results_penalty,
    "result_schema_penalty": result_schema_penalty,
    "domain_filter_penalty": domain_filter_penalty,
    "date_range_penalty": date_range_penalty,
    "sort_order_penalty": sort_order_penalty,
    "min_realistic_time_penalty": min_realistic_time_penalty,
    "summary_structure_penalty": summary_structure_penalty,
    "timeout_penalty": timeout_penalty,
}


def applied_penalty(raw: float, max_penalty: float = 1.0) -> float:
    """Upstream's ``apply_penalties`` tail: clip to [0, 1], clip to max, return the multiplier."""
    adjusted = min(max(raw, 0.0), 1.0)
    adjusted = min(adjusted, max_penalty)
    return 1.0 - adjusted


# ---------------------------------------------------------------------------------------------
# Reward combination — upstream/neurons/validators/scrapers/base_scraper_validator.py
# ---------------------------------------------------------------------------------------------


def reward_weights_for(response: UpstreamResponse) -> tuple[float, ...]:
    """The component weights this response is scored under.

    AI search normally splits content 0.60 / summary 0.40, but an ONLY_LINKS request asked for no
    summary — scoring one it was never asked to write would be charging it for the validator's own
    default. X search has a single component.
    """
    if response.kind == "x_search":
        return (X_CONTENT_WEIGHT,)
    if response.result_type == RESULT_TYPE_ONLY_LINKS:
        return (1.0, 0.0)
    return (AI_CONTENT_WEIGHT, AI_SUMMARY_WEIGHT)


def component_floors_for(response: UpstreamResponse) -> tuple[float, ...]:
    """The per-component quality-gate floors. X search declares none."""
    if response.kind == "x_search":
        return (0.0,)
    return AI_COMPONENT_FLOORS


def perf_floor_default_for(response: UpstreamResponse) -> float:
    return AI_PERF_FLOOR if response.kind == "ai_search" else X_PERF_FLOOR


@dataclass(frozen=True)
class UpstreamScore:
    """One response's upstream-faithful score, with every intermediate a reviewer needs.

    ``reward`` is the paid score. ``quality_gate`` is the same product WITHOUT the performance
    multiplier and zeroed when a component fell below its floor — upstream ramps concurrency on it,
    and Kata reports it because "fast enough to be paid" and "good enough to earn more work" are
    genuinely different questions.
    """

    reward: float
    quality_gate: float
    components: tuple[float, ...]
    weights: tuple[float, ...]
    weighted_quality: float
    perf_raw: float
    perf_multiplier: float
    penalties: dict[str, float]
    penalty_multiplier: float
    pool_share: float

    def as_dict(self) -> dict:
        return {
            "reward": self.reward,
            "quality_gate": self.quality_gate,
            "components": list(self.components),
            "weights": list(self.weights),
            "weighted_quality": self.weighted_quality,
            "perf_raw": self.perf_raw,
            "perf_multiplier": self.perf_multiplier,
            "penalties": dict(sorted(self.penalties.items())),
            "penalty_multiplier": self.penalty_multiplier,
            "pool_share": self.pool_share,
        }


def pool_share_for(response: UpstreamResponse) -> float:
    """This response's share of the upstream pool, from the §5.3 weight tables."""
    if response.kind == "x_search":
        return POOL_SHARES[("x_search", None)]
    return POOL_SHARES.get(("ai_search", response.mode), 0.0)


def score_response(response: UpstreamResponse, components: tuple[float, ...] | list[float],
                   *, penalty_names: tuple[str, ...] | None = None,
                   apply_performance: bool = True) -> UpstreamScore:
    """Combine quality components, the performance curve and the penalties, upstream's way.

    ``components`` are the relevance scores the judge produced — (content, summary) for AI search,
    (twitter_content,) for X search. They are an INPUT because producing them is the paid judge's
    job (§6.1); everything downstream of them is arithmetic, and that arithmetic is what this
    module proves it shares with the pinned upstream.

    The exact order matters and is upstream's:

    1. ``reward = Σ wⱼ·rⱼ``;
    2. ``quality_gate`` starts as a copy of that reward;
    3. a component below its floor (with a non-zero weight) zeroes the GATE, not the reward;
    4. the performance multiplier scales the REWARD only — speed cannot buy capacity;
    5. every penalty multiplies BOTH.

    ``penalty_names`` and ``apply_performance`` default to the full upstream behaviour, which is
    what the parity evidence is recorded against. The Kata lane narrows them (plan §5.3, "retain
    those weights where the corresponding upstream component is INCLUDED") because two upstream
    components have no input in a sealed offline challenge: there is no provider latency to measure,
    and Kata already ranks latency as its own signal. Passing the narrowing in rather than editing
    it into the arithmetic keeps the default path — the one under parity — untouched.
    """
    weights = reward_weights_for(response)
    components = tuple(float(c) for c in components)
    if len(components) != len(weights):
        raise ValueError(f"{response.kind} expects {len(weights)} quality component(s), "
                         f"got {len(components)}")

    reward = sum(weight * component for weight, component in zip(weights, components))
    weighted_quality = reward
    quality_gate = reward

    for index, floor in enumerate(component_floors_for(response)):
        if floor <= 0 or index >= len(components):
            continue
        if components[index] < floor and weights[index] > 0:
            quality_gate = 0.0

    perf_raw = performance_reward(response_time_for(response), resolve_scoring_budget(response))
    if apply_performance:
        perf_multiplier = perf_factor(perf_raw, perf_floor_for(
            response, perf_floor_default_for(response)))
    else:
        perf_multiplier = 1.0
    reward *= perf_multiplier

    default_names = AI_PENALTIES if response.kind == "ai_search" else X_PENALTIES
    if penalty_names is None:
        names = default_names
    else:
        # Intersected with the search type's own list rather than used raw: applying X search's
        # sort-order penalty to an AI response would be a penalty for a field that response does
        # not have, which is a silent zero rather than an error.
        names = tuple(name for name in default_names if name in penalty_names)
    penalties: dict[str, float] = {}
    penalty_multiplier = 1.0
    for name in names:
        raw = PENALTY_FUNCTIONS[name](response)
        penalties[name] = raw
        penalty_multiplier *= applied_penalty(raw)
    reward *= penalty_multiplier
    quality_gate *= penalty_multiplier

    return UpstreamScore(
        reward=reward, quality_gate=quality_gate, components=components, weights=weights,
        weighted_quality=weighted_quality, perf_raw=perf_raw, perf_multiplier=perf_multiplier,
        penalties=penalties, penalty_multiplier=penalty_multiplier,
        pool_share=pool_share_for(response),
    )


def pool_weighted_total(scores) -> float:
    """Combine per-task scores by their pool shares, normalized by the shares actually present.

    Normalizing matters because a Kata challenge draws a handful of queries, not the full hourly
    population: without it, a challenge that happened to draw no `deep` task would score every
    contestant against a denominator including `deep`'s 0.18, and both sides would look worse for a
    reason neither controls. Normalized, the weights say what they mean — the RELATIVE importance of
    the categories that were actually asked.
    """
    total_share = 0.0
    total = 0.0
    for score in scores:
        total_share += score.pool_share
        total += score.pool_share * score.reward
    return total / total_share if total_share > 0 else 0.0


# ---------------------------------------------------------------------------------------------
# Evidence: what a miner must prove before anything it says is judged
# ---------------------------------------------------------------------------------------------
#
# This is upstream's anti-fabrication layer, and it is the part a miner has the most to gain from
# defeating. A miner returns links, ``miner_highlights`` (excerpts it claims are on those pages) and
# ``miner_text``. The validator fetches the page itself and requires the highlights to appear IN
# ORDER in both its own fetched body and the miner's claimed text. Only links that clear that reach
# the paid judge.
#
# Ordering is the load-bearing detail. A membership check ("are these phrases somewhere on the
# page?") is satisfied by a miner that scrapes a page's vocabulary and reassembles it into an
# excerpt that was never written; requiring the excerpts to appear sequentially, non-overlapping,
# means the miner has to have read a contiguous span of the real page.
#
# Matching is deliberately fuzzy in the OTHER direction: casefolded, HTML entities unescaped, and
# every non-word character dropped. A page that renders `it's` as `it&#39;s`, or wraps a sentence
# across markup, must not defeat an honest miner that quoted it correctly.

_NON_WORD = re.compile(r"\W+")

#: Links sampled for body-relevance judging per response, and how many of them may be ones the
#: summary actually cites. The uncited pick is the spot check: a miner that returns ten links and
#: only bothers to make its two cited ones real still gets caught.
MAX_SAMPLED_LINKS = 3
MAX_CITED_SAMPLE = 2

_CITATION_MARKER = re.compile(r"\[\d+\]\((https?://[^)]+)\)")
_TWEET_ID_IN_URL = re.compile(r"/status/(\d+)")


def normalize_for_match(text: str) -> str:
    """Casefold, unescape HTML entities, drop every non-word character (Unicode-aware)."""
    return _NON_WORD.sub("", html.unescape(text or "").casefold())


def highlight_subset_of_body(highlights, body) -> list:
    """The highlights actually present in ``body``, order ignored. Used for reporting what was
    verified, never for deciding evidence -- that is :func:`highlights_in_order`."""
    normalized_body = normalize_for_match(body)
    if not normalized_body:
        return []
    verified = []
    for highlight in highlights or []:
        normalized = normalize_for_match(highlight)
        if normalized and normalized in normalized_body:
            verified.append(highlight)
    return verified


def highlights_in_order(highlights, body) -> bool:
    """True only if every highlight appears in ``body``, in order and non-overlapping.

    The cursor advances past each match, so the same span cannot satisfy two highlights -- a miner
    cannot pass by repeating one real sentence.
    """
    normalized_body = normalize_for_match(body)
    if not normalized_body or not highlights:
        return False
    cursor = 0
    for highlight in highlights:
        normalized = normalize_for_match(highlight)
        if not normalized:
            return False
        index = normalized_body.find(normalized, cursor)
        if index == -1:
            return False
        cursor = index + len(normalized)
    return True


def link_meets_evidence(miner_highlights, miner_text, fetched_body) -> bool:
    """Whether one link has earned the right to be judged.

    Both directions are required, and they catch different lies. Against the VALIDATOR'S fetched
    body: the miner did not invent the excerpt. Against the MINER'S OWN text: the miner actually
    used what it quoted, rather than pasting real excerpts beside an answer written from elsewhere.
    """
    if not miner_highlights or not miner_text:
        return False
    if not highlights_in_order(miner_highlights, fetched_body):
        return False
    if not highlights_in_order(miner_highlights, miner_text):
        return False
    return True


def cited_urls_normalized(summary: str) -> set:
    """The URLs the summary actually cites, as normalized keys."""
    return {normalize_source_url(url) for _, url in extract_markdown_links(summary)}


def sample_cited_and_uncited(urls, cited_norm, max_cited, max_total, *, rng=None):
    """Pick which links to spend judge calls on: some cited, and at least one that is not.

    Judging is paid, so not every link can be judged. Sampling only CITED links would tell a miner
    exactly which links are worth making real, so one uncited link is always drawn when any exists.
    ``rng`` is injectable for tests and parity; the default is the module-level :mod:`random`,
    which is what upstream uses.
    """
    picker = rng or random
    cited = [url for url in urls if normalize_source_url(url) in cited_norm]
    other = [url for url in urls if normalize_source_url(url) not in cited_norm]
    picks = picker.sample(cited, min(max_cited, len(cited)))
    if other:
        picks.append(picker.choice(other))
    while len(picks) < max_total:
        pool = [url for url in urls if url not in picks]
        if not pool:
            break
        picks.append(picker.choice(pool))
    return picks


def dedup_richest(bodies) -> list:
    """One body per source, keeping the longest text. Two fetches of the same page can differ in
    completeness, and the judge should read the fuller one."""
    best: dict = {}
    for body in bodies or []:
        key = source_key(body.get("url", ""))
        if key not in best or len(body.get("text") or "") > len(best[key].get("text") or ""):
            best[key] = body
    return list(best.values())


def align_citation_markers(summary: str, bodies) -> str:
    """Renumber the summary's ``[n]`` markers to match the order the bodies are shown to the judge.

    The groundedness judge is asked whether a value is supported by the source the answer cites FOR
    it. If ``[2]`` in the answer and ``[2]`` in the rendered sources are different pages, the judge
    is answering a different question -- and would fail an honest answer.
    """
    index_by_key = {source_key(body.get("url", "")): i for i, body in enumerate(bodies, 1)}

    def _renumber(match):
        index = index_by_key.get(source_key(match.group(1)))
        return f"[{index}]({match.group(1)})" if index else match.group(0)

    return _CITATION_MARKER.sub(_renumber, summary or "")


def _attr(item, name, default=None):
    """Read ``name`` off a dict or an object. The adapter carries tweets as dicts; upstream carries
    them as pydantic models, and this port must accept the shape it is actually given."""
    if isinstance(item, dict):
        value = item.get(name, default)
    else:
        value = getattr(item, name, default)
    return default if value is None else value


def tweet_relevance_text(tweet) -> str:
    """The text a tweet is judged on: its own, plus any quoted tweet.

    A reply consisting of "this" above a quoted claim is only judgeable with the quote attached.
    """
    text = _attr(tweet, "text", "") or ""
    quote = _attr(tweet, "quote")
    quoted = (_attr(quote, "text", "") or "").strip() if quote else ""
    if not quoted:
        return text
    handle = _attr(_attr(quote, "user"), "username") if quote else None
    header = f"Quoted tweet (@{handle}):" if handle else "Quoted tweet:"
    return f"{text}\n\n{header} {quoted}"


def collect_cited_bodies(*, validator_links, validator_tweets, cited_urls) -> list:
    """Resolve each cited URL to a body the validator ALREADY fetched.

    Never re-fetches, and never falls back to what the miner supplied: the groundedness judge's
    whole purpose is to check the answer against independently obtained text, so a body sourced
    from the miner would let it grade its own homework.
    """
    link_map = {}
    for link in validator_links or []:
        url = link.get("link") or link.get("url") or ""
        text = link.get("body") or ""
        if url and text:
            link_map[source_key(url)] = {"url": url, "title": link.get("title", ""), "text": text}

    tweet_map = {}
    for tweet in validator_tweets or []:
        text = tweet_relevance_text(tweet)
        tweet_id = _attr(tweet, "id")
        if tweet_id and text:
            author = _attr(_attr(tweet, "user"), "username")
            tweet_map[str(tweet_id)] = {
                "url": _attr(tweet, "url", "") or "",
                "title": f"Tweet by @{author}" if author else "Tweet",
                "text": text,
            }

    bodies = []
    for url in cited_urls or []:
        hit = link_map.get(source_key(url))
        if not hit:
            match = _TWEET_ID_IN_URL.search(url)
            if match:
                hit = tweet_map.get(match.group(1))
        if hit and hit.get("text"):
            bodies.append({**hit, "url": url})
    return dedup_richest(bodies)


def render_cited_sources(bodies) -> str:
    """The ``<CitedSources>`` block the groundedness judge reads.

    A source whose fetch failed is shown as explicitly empty rather than omitted: the judge is told
    not to fail an answer over a body it cannot see, and it can only apply that rule if it knows
    the body is missing instead of silently seeing a shorter list.
    """
    blocks = []
    for index, body in enumerate(bodies or [], 1):
        text = (body.get("text") or "").strip() or "[no body could be fetched for this source]"
        blocks.append(
            f"[{index}] {body.get('url', '')}\nTitle: {body.get('title', '')}\nBody: {text}")
    return "\n\n".join(blocks)


def synthetic_created_at(offset_minutes: int = 0) -> str:
    """A ``created_at`` in upstream's Twitter format, for fixtures and the fake relay."""
    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc) - timedelta(minutes=offset_minutes)
    return stamp.strftime("%a %b %d %H:%M:%S %z %Y")


__all__ = [
    "MAX_CITED_SAMPLE",
    "MAX_SAMPLED_LINKS",
    "align_citation_markers",
    "cited_urls_normalized",
    "collect_cited_bodies",
    "dedup_richest",
    "highlight_subset_of_body",
    "highlights_in_order",
    "link_meets_evidence",
    "normalize_for_match",
    "render_cited_sources",
    "sample_cited_and_uncited",
    "tweet_relevance_text",
    "AI_COMPONENT_FLOORS",
    "AI_CONTENT_WEIGHT",
    "AI_MODE_WEIGHTS",
    "AI_PENALTIES",
    "AI_PERF_FLOOR",
    "AI_SUMMARY_WEIGHT",
    "MODE_BUDGETS",
    "MODE_PERF_FLOORS",
    "PENALTY_FUNCTIONS",
    "POOL_SHARES",
    "QUALITY_THRESHOLDS",
    "SEARCH_TYPE_WEIGHTS",
    "UpstreamResponse",
    "UpstreamScore",
    "X_CONTENT_WEIGHT",
    "X_PENALTIES",
    "X_PERF_FLOOR",
    "applied_penalty",
    "check_markdown_structure",
    "collect_summary_sources",
    "count_penalty",
    "date_range_penalty",
    "domain_filter_penalty",
    "duplicate_results_penalty",
    "extract_markdown_links",
    "first_duplicate_id",
    "format_text_for_match",
    "host_in_domains",
    "is_descending_by_created_at",
    "is_valid_tweet",
    "is_valid_web_search_result",
    "min_realistic_for_budget",
    "min_realistic_time_penalty",
    "normalize_domains",
    "normalize_source_url",
    "parse_tweet_date",
    "parse_web_query",
    "perf_factor",
    "perf_floor_for",
    "performance_reward",
    "pool_share_for",
    "pool_weighted_total",
    "resolve_scoring_budget",
    "response_time_for",
    "result_schema_penalty",
    "score_response",
    "sort_order_penalty",
    "source_key",
    "summary_structure_penalty",
    "synthetic_created_at",
    "timeout_penalty",
    "tweet_date_in_range",
]
