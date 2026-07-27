"""The sealed evidence of one SN22 challenge (plan §5.2, SN22-2).

A paired challenge is only fair if both contestants faced *the same* challenge. That is not
something you can assert afterwards — the queries are secret, the retrieved world changes minute
to minute, and the provider bills per call. So each of those three is frozen into a manifest,
and the manifests are hashed together into a single benchmark identity:

* :class:`QueryManifest` — which questions were asked. Derived deterministically from a versioned
  source plus a round seed, so it is reproducible for an auditor but not predictable by a miner who
  does not hold the source.
* :class:`SnapshotManifest` — the sealed corpus both sides retrieve from. Immutable within a
  challenge, so "an identical relay request made by both contestants resolves to identical content"
  is true by construction rather than by hoping the web held still.
* :class:`UsageManifest` — what was actually spent, recorded by the RELAY. Both contestants'
  self-reported usage is checked against it; a candidate that under-reports its own cost is caught
  by the party that did the billing.

**The secrecy rule.** A query manifest travels as a *commitment* — its hash — until the challenge is
over. :meth:`QueryManifest.as_commitment` is what goes into the benchmark identity and the public
proof; the queries themselves stay in the sealed record. Publishing them early would let the next
miner pre-compute answers, which is the whole reason the queries are secret.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field

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
class SnapshotDocument:
    """One document in the sealed corpus. ``doc_id`` is what citations and results refer to."""

    doc_id: str
    title: str
    text: str
    #: Topic tokens used to decide, offline and deterministically, whether a document is relevant to
    #: a query. Stands in for the live retrieval a production round would do.
    topics: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {"doc_id": self.doc_id, "title": self.title, "text": self.text,
                "topics": list(self.topics)}


@dataclass(frozen=True)
class SnapshotManifest:
    """The round-scoped, immutable corpus both contestants retrieve from."""

    snapshot_id: str
    documents: tuple[SnapshotDocument, ...]
    #: doc_ids that genuinely answer each task_id. The scoring ground truth, sealed with the corpus.
    relevant_by_task: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for document in self.documents:
            if document.doc_id in seen:
                raise ManifestError(f"duplicate doc_id in snapshot: {document.doc_id}")
            seen.add(document.doc_id)
        for task_id, doc_ids in self.relevant_by_task.items():
            missing = sorted(set(doc_ids) - seen)
            if missing:
                # Ground truth pointing outside the corpus would make a perfect answer unachievable.
                raise ManifestError(
                    f"relevant_by_task[{task_id!r}] cites documents absent from the snapshot: "
                    f"{', '.join(missing)}")

    def document(self, doc_id: str) -> SnapshotDocument | None:
        return next((d for d in self.documents if d.doc_id == doc_id), None)

    def contains(self, doc_id: str) -> bool:
        return self.document(doc_id) is not None

    def relevant(self, task_id: str) -> frozenset[str]:
        return frozenset(self.relevant_by_task.get(task_id, ()))

    def as_dict(self) -> dict:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "snapshot_id": self.snapshot_id,
            "documents": [d.as_dict() for d in self.documents],
            "relevant_by_task": {k: sorted(v) for k, v in sorted(self.relevant_by_task.items())},
        }

    def digest(self) -> str:
        """Content identity: any byte change is a different snapshot, so a different challenge."""
        return _sha256(self.as_dict())


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


def benchmark_identity(*, query_commitment: dict, snapshot_digest: str, judge_policy_id: str,
                       model_identity: str, upstream_commit: str, plugin_revision: str) -> str:
    """The single hash that identifies "this exact challenge, under these exact rules".

    Plan §5.2 item 3 enumerates what must be bound, and each element is here because changing it
    alone would change the result while leaving every other input identical: different queries,
    different corpus, different judging policy, a different model, a different upstream, or a
    different adapter. If two challenges share this hash they were genuinely the same challenge.

    Note the QUERY COMMITMENT is bound, not the queries: the identity is publishable during the
    round, and the commitment already pins the queries by digest.
    """
    for name, value in (("judge_policy_id", judge_policy_id), ("model_identity", model_identity),
                        ("upstream_commit", upstream_commit), ("plugin_revision", plugin_revision)):
        if not isinstance(value, str) or not value.strip():
            raise ManifestError(f"benchmark identity requires a non-empty {name}")
    if not isinstance(query_commitment, dict) or not query_commitment.get("queries_sha256"):
        raise ManifestError("benchmark identity requires a query commitment with a digest")
    if not snapshot_digest:
        raise ManifestError("benchmark identity requires a snapshot digest")
    return _sha256({
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "query_commitment": query_commitment,
        "snapshot_digest": snapshot_digest,
        "judge_policy_id": judge_policy_id,
        "model_identity": model_identity,
        "upstream_commit": upstream_commit,
        "plugin_revision": plugin_revision,
    })
