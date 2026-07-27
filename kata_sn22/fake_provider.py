"""A deterministic, offline stand-in for the SN22 relay (plan §5.5 and SN22-2's exit gate).

Calibration needs at least thirty paired challenges before a single real credential is issued, and
the protocol has to be reviewable with no network access at all. Both require a provider that
answers search requests without touching a network, bills deterministically, and refuses exactly the
what real relay refuses.

What it deliberately shares with the real relay (plan §6.1), because these are the properties the
fixtures and calibration depend on:

* it serves only the SEALED snapshot, so an identical request from either contestant resolves to
  identical content;
* it enforces per-task call quotas and records usage itself, rather than trusting the caller;
* it binds every request to a lane/challenge/variant/task capability and rejects a reused or
  cross-variant one;
* it never returns a credential.

What it is not: a judge, a live index, or a cost model. It bills in fixed units so a calibration run
measures the protocol, not a provider's pricing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from kata_sn22.manifests import SnapshotManifest, UsageManifest, UsageRecord
from kata_sn22.protocol import MAX_TEXT_CHARS, SearchResult

#: Fixed synthetic prices, so a calibration measures the protocol rather than a provider's tariff.
COST_PER_CALL_USD = 0.002
TOKENS_PER_CALL = 250


class RelayDenied(Exception):
    """The relay refused the request. Never leaks why in terms of another variant's state."""


@dataclass(frozen=True)
class Capability:
    """A short-lived grant, scoped to exactly one lane/challenge/variant/task."""

    lane_id: str
    challenge_id: str
    variant: str
    task_id: str
    max_calls: int

    @property
    def token(self) -> str:
        return f"{self.lane_id}/{self.challenge_id}/{self.variant}/{self.task_id}"


@dataclass
class FakeRelay:
    """An offline relay over one sealed snapshot."""

    snapshot: SnapshotManifest
    challenge_id: str
    lane_id: str = "sn22__desearch"
    _granted: dict[str, Capability] = field(default_factory=dict, init=False)
    _calls: dict[str, int] = field(default_factory=dict, init=False)
    _spent: list[UsageRecord] = field(default_factory=list, init=False)
    _closed: bool = field(default=False, init=False)

    def grant(self, *, variant: str, task_id: str, max_calls: int) -> Capability:
        """Issue a capability for one contestant on one task."""
        if self._closed:
            raise RelayDenied("the challenge is closed; no further capabilities are issued")
        capability = Capability(lane_id=self.lane_id, challenge_id=self.challenge_id,
                                variant=variant, task_id=task_id, max_calls=max_calls)
        self._granted[capability.token] = capability
        self._calls.setdefault(capability.token, 0)
        return capability

    def close(self) -> None:
        """End the challenge. Every capability becomes unusable, so a slow agent cannot keep
        spending after the round it was scored in has finished."""
        self._closed = True

    def search(self, capability: Capability, query: str, *, limit: int = 10) -> list[SearchResult]:
        """Answer a query from the sealed snapshot, billing one call.

        The same query under the same snapshot always returns the same documents in the same order —
        that is what makes "an identical relay request made by both contestants resolves to
        identical content" true rather than hoped for.
        """
        self._authorize(capability)
        if not isinstance(query, str) or not query.strip():
            raise RelayDenied("empty query")
        if len(query) > MAX_TEXT_CHARS:
            # An oversized request is a cost the relay pays; refuse before doing the work.
            raise RelayDenied("query exceeds the relay's size limit")
        if limit <= 0:
            raise RelayDenied("limit must be positive")

        terms = {word.strip(".,:;!?").lower() for word in query.split() if word.strip(".,:;!?")}
        scored: list[tuple[int, str, SearchResult]] = []
        for document in self.snapshot.documents:
            overlap = len(terms & {t.lower() for t in document.topics})
            if overlap == 0:
                continue
            scored.append((overlap, document.doc_id,
                           SearchResult(doc_id=document.doc_id, title=document.title,
                                        snippet=document.text[:200])))
        # Sort by overlap then by doc_id: a total order with no ties, so the result sequence cannot
        # depend on dict iteration or on which contestant asked.
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [result for _overlap, _doc_id, result in scored[:limit]]

    def _authorize(self, capability: Capability) -> None:
        if self._closed:
            raise RelayDenied("the challenge is closed")
        known = self._granted.get(capability.token)
        # Identity, not just token equality: a forged Capability with a real token must not pass.
        if known is None or known != capability:
            raise RelayDenied("unknown or forged capability")
        used = self._calls[capability.token]
        if used >= capability.max_calls:
            raise RelayDenied(
                f"call quota exhausted for {capability.variant}/{capability.task_id}")
        self._calls[capability.token] = used + 1
        self._spent.append(UsageRecord(
            variant=capability.variant, task_id=capability.task_id,
            provider_calls=1, tokens=TOKENS_PER_CALL, spend_usd=COST_PER_CALL_USD))

    def usage_manifest(self) -> UsageManifest:
        """The relay's own record of what was spent. Authoritative over any self-report."""
        merged: dict[tuple[str, str], UsageRecord] = {}
        for record in self._spent:
            key = (record.variant, record.task_id)
            existing = merged.get(key)
            if existing is None:
                merged[key] = record
            else:
                merged[key] = UsageRecord(
                    variant=record.variant, task_id=record.task_id,
                    provider_calls=existing.provider_calls + record.provider_calls,
                    tokens=existing.tokens + record.tokens,
                    spend_usd=round(existing.spend_usd + record.spend_usd, 10))
        return UsageManifest(challenge_id=self.challenge_id,
                             records=tuple(merged[key] for key in sorted(merged)))
