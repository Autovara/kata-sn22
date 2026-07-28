"""The scorer policy: everything that decides what a score MEANS, hashed into one identity.

Two contestants are only comparable if they were graded under identical rules. "Identical" has to
be checkable rather than assumed, so every input that could change a score is collected here and
hashed. The result — :func:`policy_hash` — is bound into the attested report, and the host refuses a
duel whose two reports disagree on it.

**What counts as a scoring input.** Anything whose change would move a score:

* the upstream commit — the reward arithmetic itself;
* the judge model and its parameters — a different grader is a different grade;
* the judge prompts, byte for byte — the rubric IS the policy (see :mod:`kata_sn22.judge_prompts`);
* the provider routes and pinned actor ids — a different tweet actor returns a different shape, and
  the field-by-field comparison would start failing honest miners;
* the pool weights, quality exponents and gate thresholds;
* the deep-sample rate and per-pool minimum;
* the requested result count.

**What deliberately does not count.** Timeouts, retry counts, concurrency, image tags. They change
how long a round takes and how it recovers, not what a given answer is worth. Folding them in would
make the hash churn on operational tuning and stop anyone reading a mismatch as meaningful.

**Why a hash rather than a version number.** A version number is a claim someone remembers to
update. A hash over the actual values cannot be forgotten: change a prompt's punctuation and the
identity moves, whether or not anyone meant it to.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from kata_sn22 import judge_prompts
from kata_sn22.protocol_v2 import (
    DEFAULT_RESULT_COUNT,
    FIXED_SCORING_MODEL,
    UPSTREAM_COMMIT,
)

#: Bumped when the SHAPE of the policy document changes, so an old hash is never silently compared
#: against a new one computed over different fields.
POLICY_SCHEMA_VERSION = 1

#: Upstream's judge sampling. Copied rather than rounded: the judge's own variance is part of why
#: SN22 is NOISY, and a tidier 0.0 would be a different grader.
JUDGE_TEMPERATURE = 0.0001

#: Upstream ``neurons/validators/scoring/constants.py`` and its scrapers. Restated here so the
#: policy hash covers them; ``tests/test_sn22_v2_specification.py`` re-reads them out of the
#: vendored tree by AST, so a divergence is a test failure rather than a silent re-weighting.
POOL_WEIGHTS: dict[str, float] = {
    "ai_search:fast": 0.54,
    "ai_search:balanced": 0.18,
    "ai_search:deep": 0.18,
    "x_search": 0.10,
}
QUALITY_THRESHOLDS: dict[str, float] = {"ai_search": 0.50, "x_search": 0.60}
QUALITY_EXPONENT = 3.0
VOLUME_EXPONENT = 2.0
GATE_RAMP = 0.05
AI_CONTENT_WEIGHT = 0.60
AI_SUMMARY_WEIGHT = 0.40
AI_COMPONENT_FLOORS: tuple[float, float] = (0.30, 0.30)

#: Upstream ``DEEP_SAMPLE_RATE`` / ``MIN_DEEP_SAMPLES_PER_POOL``. These two set the pool size: a
#: pool needs at least ``MIN_DEEP / RATE`` = 15 tasks or it cannot produce enough deep samples to
#: score at all, and ``_pool_raw_scores`` drops a UID with fewer.
DEEP_SAMPLE_RATE = 0.20
MIN_DEEP_SAMPLES_PER_POOL = 3
TASKS_PER_POOL = 15

#: Where each credential may be spent. Pinned into the policy because a route change means a
#: different provider answered, and a score produced against a different provider is not comparable.
PROVIDER_ROUTES: dict[str, str] = {
    "agent.web_search": "scrapingdog:google",
    "agent.x_search": "apify:twitter",
    "agent.final_summary": "openai:gpt-4.1-nano",
    "evaluator.web_page_fetch": "scrapingdog:scrape",
    "evaluator.tweet_rescrape": "apify:CJdippxWmn9uRfooo",
    "evaluator.judge": f"chutes:{FIXED_SCORING_MODEL.value}",
}

#: Upstream's pinned tweet actor. A different actor returns a different tweet shape.
APIFY_TWEET_ACTOR = "CJdippxWmn9uRfooo"

#: The judge prompts, by the name each is known by upstream. Values are hashed, not stored, so the
#: policy document stays readable while still covering every byte.
_PROMPT_CONSTANTS: tuple[str, ...] = (
    "SYSTEM_BODY_LINK_RELEVANCE_TEMPLATE",
    "USER_BODY_LINK_RELEVANCE_TEMPLATE",
    "SYSTEM_TWEET_RELEVANCE_TEMPLATE",
    "SYSTEM_SUMMARY_GROUNDEDNESS_TEMPLATE",
    "USER_SUMMARY_GROUNDEDNESS_TEMPLATE",
)


def prompt_digests() -> dict[str, str]:
    """SHA-256 of each judge prompt. Changing one word moves the policy hash."""
    return {
        name: hashlib.sha256(getattr(judge_prompts, name).encode("utf-8")).hexdigest()
        for name in _PROMPT_CONSTANTS
    }


@dataclass(frozen=True)
class ScorerPolicy:
    """Everything that decides what a score means, as one comparable document."""

    upstream_commit: str = UPSTREAM_COMMIT
    scoring_model: str = FIXED_SCORING_MODEL.value
    judge_temperature: float = JUDGE_TEMPERATURE
    #: Kata disables upstream's Chutes -> gpt-4.1-nano fallback. One contestant graded by Qwen and
    #: another by GPT is not one competition, so a failed Chutes credential is a zero instead.
    scorer_fallback_enabled: bool = False
    result_count: int = DEFAULT_RESULT_COUNT
    tasks_per_pool: int = TASKS_PER_POOL
    deep_sample_rate: float = DEEP_SAMPLE_RATE
    min_deep_samples_per_pool: int = MIN_DEEP_SAMPLES_PER_POOL
    pool_weights: dict = field(default_factory=lambda: dict(POOL_WEIGHTS))
    quality_thresholds: dict = field(default_factory=lambda: dict(QUALITY_THRESHOLDS))
    quality_exponent: float = QUALITY_EXPONENT
    volume_exponent: float = VOLUME_EXPONENT
    gate_ramp: float = GATE_RAMP
    ai_content_weight: float = AI_CONTENT_WEIGHT
    ai_summary_weight: float = AI_SUMMARY_WEIGHT
    ai_component_floors: tuple = AI_COMPONENT_FLOORS
    provider_routes: dict = field(default_factory=lambda: dict(PROVIDER_ROUTES))
    apify_tweet_actor: str = APIFY_TWEET_ACTOR

    def as_document(self) -> dict:
        """The canonical document the hash is taken over.

        Prompts appear as digests rather than text: the document is published in every attested
        report, and 10 KB of rubric in each one would bury the fields a reader is checking.
        """
        return {
            "schema_version": POLICY_SCHEMA_VERSION,
            "upstream_commit": self.upstream_commit,
            "scoring_model": self.scoring_model,
            "judge_temperature": self.judge_temperature,
            "scorer_fallback_enabled": self.scorer_fallback_enabled,
            "prompt_digests": prompt_digests(),
            "result_count": self.result_count,
            "tasks_per_pool": self.tasks_per_pool,
            "deep_sample_rate": self.deep_sample_rate,
            "min_deep_samples_per_pool": self.min_deep_samples_per_pool,
            "pool_weights": dict(sorted(self.pool_weights.items())),
            "quality_thresholds": dict(sorted(self.quality_thresholds.items())),
            "quality_exponent": self.quality_exponent,
            "volume_exponent": self.volume_exponent,
            "gate_ramp": self.gate_ramp,
            "ai_content_weight": self.ai_content_weight,
            "ai_summary_weight": self.ai_summary_weight,
            "ai_component_floors": list(self.ai_component_floors),
            "provider_routes": dict(sorted(self.provider_routes.items())),
            "apify_tweet_actor": self.apify_tweet_actor,
        }

    def policy_hash(self) -> str:
        """The identity two reports must agree on before their scores are compared."""
        canonical = json.dumps(self.as_document(), sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


#: The one policy production runs under.
PRODUCTION_POLICY = ScorerPolicy()


def policy_hash() -> str:
    return PRODUCTION_POLICY.policy_hash()


def minimum_tasks_per_pool() -> int:
    """The smallest pool that can be scored at all.

    ``_pool_raw_scores`` drops any UID with fewer than ``MIN_DEEP_SAMPLES_PER_POOL`` deep samples,
    and only ``DEEP_SAMPLE_RATE`` of a pool is deep-scored. Below this, a contestant scores zero for
    a reason that has nothing to do with its answers -- which is why the deployed ``task_count=8``
    could never have worked.
    """
    import math

    return math.ceil(MIN_DEEP_SAMPLES_PER_POOL / DEEP_SAMPLE_RATE)
