"""The recorded evidence of one SN22 challenge (plan §5.2, SN22-2).

A paired challenge is only fair if both contestants faced *the same* challenge. That is not
something you can assert afterwards — the queries are secret and the provider bills per call — so
each is frozen into a manifest, and the manifests are hashed into a single benchmark identity:

* :class:`QueryManifest` — which questions were asked. Derived deterministically from a versioned
  source plus a round seed, so it is reproducible for an auditor but not predictable by a miner who
  does not hold the source.
* :class:`UsageManifest` — what was actually spent, recorded by the RELAY. Both contestants'
  self-reported usage is checked against it; a candidate that under-reports its own cost is caught
  by the party that did the billing.

There was a third: a ``SnapshotManifest`` sealing a corpus both sides retrieved from. It is gone.
Sources are live, and sameness comes from the validator FETCHING the ground truth itself and
verifying both contestants against that one fetch (:mod:`kata_sn22.verification`) — which is
upstream's own model, and does not require pretending the web held still.

**The secrecy rule.** A query manifest travels as a *commitment* — its hash — until the challenge is
over. :meth:`QueryManifest.as_commitment` is what goes into the benchmark identity and the public
proof; the queries themselves stay in the sealed record. Publishing them early would let the next
miner pre-compute answers, which is the whole reason the queries are secret.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass

MANIFEST_SCHEMA_VERSION = 1


class ManifestError(Exception):
    """A manifest is malformed, unsealed, or inconsistent with the challenge. Fail closed."""


def _canonical(document: object) -> bytes:
    """One byte form per document, so a hash means the same thing everywhere it is computed."""
    return json.dumps(document, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _sha256(document: object) -> str:
    return hashlib.sha256(_canonical(document)).hexdigest()


@dataclass(frozen=True)
class QueryManifest:
    """The secret query set for one challenge, plus the versioned source it was drawn from."""

    source_id: str
    source_version: int
    round_seed: str
    #: (task_id, query, search_type, ai_mode) tuples, in the fixed order both contestants receive.
    entries: tuple[tuple[str, str, str, str | None], ...]

    def as_commitment(self) -> dict:
        """The PUBLIC face of the manifest: enough to verify it later, nothing to pre-compute from.

        Names the source and its version, the seed, how many queries there were and their category
        mix — and the digest of the queries themselves. After the challenge an auditor recomputes
        the manifest from source + seed and checks the digest; before it, this reveals no question.
        """
        mix: dict[str, int] = {}
        for _task_id, _query, search_type, ai_mode in self.entries:
            key = f"{search_type}:{ai_mode}" if ai_mode else search_type
            mix[key] = mix.get(key, 0) + 1
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "round_seed": self.round_seed,
            "query_count": len(self.entries),
            "category_mix": dict(sorted(mix.items())),
            "queries_sha256": self.sealed_digest(),
        }

    def sealed_digest(self) -> str:
        """Digest of the queries themselves. Recomputable only by someone holding the source."""
        return _sha256([list(entry) for entry in self.entries])

    def as_sealed_record(self) -> dict:
        """The full record, for the sealed side of the proof. Never published before a challenge."""
        return {**self.as_commitment(),
                "entries": [list(entry) for entry in self.entries]}


def derive_query_manifest(*, source_id: str, source_version: int, round_seed: str,
                          pool: list[dict], count: int) -> QueryManifest:
    """Select ``count`` queries from a versioned pool, deterministically for this seed.

    Deterministic so an auditor with the source can rebuild the exact manifest and check it against
    the published commitment. The selection is keyed by an HMAC of the seed rather than by
    ``random.seed`` — a seeded PRNG's stream is a Python implementation detail and could change
    between versions, which would break every historical proof.
    """
    if count <= 0:
        raise ManifestError("a challenge needs at least one query")
    if count > len(pool):
        raise ManifestError(f"pool holds {len(pool)} queries, cannot draw {count}")
    if not round_seed:
        raise ManifestError("round_seed must not be empty")

    key = f"{source_id}:{source_version}:{round_seed}".encode()
    ranked = sorted(
        pool,
        key=lambda item: hmac.new(key, _canonical(item), hashlib.sha256).hexdigest(),
    )
    entries = []
    for index, item in enumerate(ranked[:count]):
        search_type = item.get("search_type", "ai_search")
        ai_mode = item.get("ai_mode") if search_type == "ai_search" else None
        entries.append((f"t{index:03d}", str(item["query"]), str(search_type), ai_mode))
    return QueryManifest(source_id=source_id, source_version=source_version, round_seed=round_seed,
                         entries=tuple(entries))


@dataclass(frozen=True)
class UsageRecord:
    """What the relay billed for one contestant on one task."""

    variant: str
    task_id: str
    provider_calls: int
    tokens: int
    spend_usd: float

    def as_dict(self) -> dict:
        return {"variant": self.variant, "task_id": self.task_id,
                "provider_calls": self.provider_calls, "tokens": self.tokens,
                "spend_usd": self.spend_usd}


@dataclass(frozen=True)
class UsageManifest:
    """The relay's own record of what a challenge cost, hash-bound to the challenge it belongs to.

    Authoritative over the agent's self-reported ``ToolUsage``: the relay is the party that made the
    calls, and the candidate is the party with a reason to under-report them.
    """

    challenge_id: str
    records: tuple[UsageRecord, ...]

    def totals(self, variant: str) -> dict[str, float]:
        rows = [r for r in self.records if r.variant == variant]
        return {
            "provider_calls": float(sum(r.provider_calls for r in rows)),
            "tokens": float(sum(r.tokens for r in rows)),
            "spend_usd": float(sum(r.spend_usd for r in rows)),
        }

    def as_dict(self) -> dict:
        return {"schema_version": MANIFEST_SCHEMA_VERSION, "challenge_id": self.challenge_id,
                "records": [r.as_dict() for r in self.records]}

    def digest(self) -> str:
        return _sha256(self.as_dict())

    def assert_symmetric(self, variants: tuple[str, str]) -> None:
        """Both contestants must have been given the same shot at the same tasks.

        Plan §5.2 item 8: refuse to decide promotion when the shared infrastructure was incomplete
        for either side. A task the relay served to one contestant but not the other means exactly
        that, and a challenge decided on it would be decided by which side got served.
        """
        left, right = variants
        served = {v: {r.task_id for r in self.records if r.variant == v} for v in (left, right)}
        if served[left] != served[right]:
            only_left = sorted(served[left] - served[right])
            only_right = sorted(served[right] - served[left])
            raise ManifestError(
                f"usage is asymmetric between {left!r} and {right!r}: "
                f"only {left}={only_left}, only {right}={only_right}; refusing to decide promotion")


def benchmark_identity(*, query_commitment: dict, snapshot_digest: str = "",
                       judge_policy_id: str, model_identity: str, upstream_commit: str,
                       plugin_revision: str) -> str:
    """The single hash that identifies "these exact questions, under these exact rules".

    Each element is here because changing it alone would change the result while leaving every
    other input identical: different queries, a different judging policy, a different model, a
    different upstream, or a different adapter. If two challenges share this hash they were scored
    by the same rules on the same questions.

    ``snapshot_digest`` is retained and defaults to empty. SN22 no longer freezes a corpus — sources
    are live and the validator fetches them — so there is nothing to digest, and binding a copy of
    the web would mean no two challenges were ever comparable. The parameter stays because a subnet
    that DOES seal a corpus still needs somewhere to bind it, and silently dropping it from the
    hash would let a sealed-corpus lane's identity stop distinguishing its worlds.

    Note the QUERY COMMITMENT is bound, not the queries: the identity is publishable during the
    round, and the commitment already pins the queries by digest.
    """
    for name, value in (("judge_policy_id", judge_policy_id), ("model_identity", model_identity),
                        ("upstream_commit", upstream_commit), ("plugin_revision", plugin_revision)):
        if not isinstance(value, str) or not value.strip():
            raise ManifestError(f"benchmark identity requires a non-empty {name}")
    if not isinstance(query_commitment, dict) or not query_commitment.get("queries_sha256"):
        raise ManifestError("benchmark identity requires a query commitment with a digest")
    return _sha256({
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "query_commitment": query_commitment,
        "snapshot_digest": snapshot_digest,
        "judge_policy_id": judge_policy_id,
        "model_identity": model_identity,
        "upstream_commit": upstream_commit,
        "plugin_revision": plugin_revision,
    })
