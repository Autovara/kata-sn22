"""The trusted SN22 provider gateway (plan §6.1, SN22-4).

One sentence states the whole design: **the credential lives here and only here.** A candidate
never holds a provider key; it holds a capability — a short-lived token bound to one lane, one
challenge, one variant and one task — and the gateway makes the provider call on its behalf,
bills it, and hands back data with the credential stripped out.

That inversion is what makes the rest tractable. The candidate is untrusted code written by a
stranger, so any secret it can read is a secret that is gone. Since it never has one, the
interesting attacks (§6.2) reduce to: can it get the gateway to act outside its capability, spend
past its reservation, or return something it should not? Each is a check below, with a test.

What the gateway deliberately does NOT have, because a compromise here would otherwise be total:
no systemd or service control, no GitHub token, no registry write path, no installer capability. It
answers search requests and records what they cost. That is all.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import time
from dataclasses import dataclass, field

from kata_sn22.manifests import UsageManifest, UsageRecord
from kata_sn22.protocol import MAX_TEXT_CHARS

#: Capability tokens are opaque and fixed-shape. A parser that accepted anything else would be a
#: place to smuggle structure into a filesystem path or a log line.
CAPABILITY_RE = re.compile(r"^sn22cap_[0-9a-f]{32}$")

#: Provider credential env names the gateway may hold. Listed so the boundary test can assert that
#: NONE of them appear anywhere in what a candidate receives.
PROVIDER_CREDENTIAL_NAMES = (
    "OPENAI_API_KEY", "APIFY_API_KEY", "SCRAPINGDOG_API_KEY", "CHUTES_API_KEY",
    "TWITTER_BEARER_TOKEN", "EXPECTED_ACCESS_KEY",
)

#: Anything shaped like a secret is scrubbed from every value that crosses back to a candidate or
#: into a log. Broad on purpose: a false positive costs a redacted string, a false negative leaks.
_SECRET_SHAPES = (
    re.compile(r"\bsk-[A-Za-z0-9]{16,}"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}"),
    re.compile(r"\bapify_api_[A-Za-z0-9]{16,}", re.IGNORECASE),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}", re.IGNORECASE),
)


class GatewayDenied(Exception):
    """The request was refused. The message never names another variant's state or a credential."""


def redact(text: str) -> str:
    """Scrub secret-shaped substrings. Applied to everything leaving the gateway."""
    for shape in _SECRET_SHAPES:
        text = shape.sub("[REDACTED]", text)
    return text


@dataclass(frozen=True)
class Capability:
    """A short-lived grant: one lane, one challenge, one variant, one task, N calls, until T."""

    token: str
    lane_id: str
    challenge_id: str
    variant: str
    task_id: str
    max_calls: int
    expires_at: float

    def as_public(self) -> dict:
        """What the candidate is told. Note there is no credential and no provider name here."""
        return {"capability": self.token, "task_id": self.task_id,
                "max_calls": self.max_calls, "expires_at": self.expires_at}


@dataclass
class Sn22Gateway:
    """The trusted broker between untrusted candidates and paid providers.

    ``provider`` is what actually answers a search: ``provider(query, limit) -> list[dict]``. It is
    injected rather than built here, and rather than being a sealed corpus, because who pays for
    search is now a deployment decision — in the TEE backend the miner funds its own calls inside
    its room, and during calibration a recorded provider makes a run free and repeatable. What the
    gateway keeps either way is the part that must not vary: the capability check, the billing, the
    redaction, and the challenge-wide reservation.

    A gateway with no provider REFUSES searches rather than returning nothing. An empty result set
    and "there is no provider configured" would otherwise be the same answer, and only one of them
    is a fact about the query.
    """

    provider: object = None
    challenge_id: str = ""
    lane_id: str = "sn22__desearch"
    #: Total calls the gateway will serve this challenge, across every variant. The RESERVATION.
    #: Even a capability with spare quota is refused once this is reached: the reservation is the
    #: money that was actually approved, and per-task quotas alone do not add up to it.
    reservation_calls: int = 1_000
    #: Wall-clock lifetime of each issued capability.
    capability_ttl_seconds: float = 900.0
    #: Injected so tests can advance time without sleeping.
    clock: object = field(default=None)

    _issued: dict[str, Capability] = field(default_factory=dict, init=False)
    _calls: dict[str, int] = field(default_factory=dict, init=False)
    _records: list[UsageRecord] = field(default_factory=list, init=False)
    _served: int = field(default=0, init=False)
    _closed: bool = field(default=False, init=False)
    _seq: int = field(default=0, init=False)

    def _now(self) -> float:
        return float(self.clock()) if callable(self.clock) else time.monotonic()

    # ---- capability lifecycle --------------------------------------------------------------------
    def issue(self, *, variant: str, task_id: str, max_calls: int) -> Capability:
        """Mint one capability. Deterministic id, unguessable token."""
        if self._closed:
            raise GatewayDenied("the challenge is closed")
        if max_calls <= 0:
            raise GatewayDenied("a capability with no calls is not a capability")
        self._seq += 1
        # Bound to the identity it grants, so two capabilities never collide and a token cannot be
        # transplanted onto a different task by editing a field.
        material = f"{self.lane_id}|{self.challenge_id}|{variant}|{task_id}|{self._seq}"
        token = "sn22cap_" + hashlib.sha256(material.encode()).hexdigest()[:32]
        capability = Capability(
            token=token, lane_id=self.lane_id, challenge_id=self.challenge_id, variant=variant,
            task_id=task_id, max_calls=int(max_calls),
            expires_at=self._now() + self.capability_ttl_seconds,
        )
        self._issued[token] = capability
        self._calls[token] = 0
        return capability

    def close(self) -> None:
        """End the challenge. Every capability dies with it, so nothing spends after scoring."""
        self._closed = True

    def _authorize(self, token: str, *, check_quota: bool = True) -> Capability:
        """Validate a presented token. Every refusal is deliberately uninformative.

        ``check_quota=False`` is for the free ``quota`` read, which must keep working precisely when
        the quota has run out — refusing to report an exhausted quota would make the one answer an
        agent needs the one it cannot get.
        """
        if self._closed:
            raise GatewayDenied("the challenge is closed")
        if not isinstance(token, str) or not CAPABILITY_RE.fullmatch(token):
            raise GatewayDenied("malformed capability")
        capability = self._issued.get(token)
        if capability is None:
            # Same message as a malformed token: a distinct one would let a candidate probe which
            # tokens exist, and existence is another variant's business.
            raise GatewayDenied("malformed capability")
        if self._now() > capability.expires_at:
            raise GatewayDenied("capability expired")
        if not check_quota:
            return capability
        if self._calls[token] >= capability.max_calls:
            raise GatewayDenied("call quota exhausted")
        if self._served >= self.reservation_calls:
            # The reservation is a HARD ceiling. Per-task quotas bound one task; only this
            # bounds the challenge, which is what was actually reserved and approved.
            raise GatewayDenied("challenge reservation exhausted")
        return capability

    # ---- the one operation a candidate may perform -----------------------------------------------
    def search(self, token: str, query: str, *, limit: int = 10) -> list[dict]:
        """Authorize one search, bill it, and return the provider's answer with secrets stripped.

        Returns plain dicts, not internal objects: the candidate gets data, never a handle onto
        gateway state it might mutate.
        """
        capability = self._authorize(token)
        if not isinstance(query, str) or not query.strip():
            raise GatewayDenied("empty query")
        if len(query) > MAX_TEXT_CHARS:
            raise GatewayDenied("query exceeds the relay size limit")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 0 < limit <= 50:
            raise GatewayDenied("limit out of range")

        # Bill BEFORE serving. A crash mid-serve must not produce an unbilled call, because the
        # provider would have charged for it either way.
        self._calls[token] += 1
        self._served += 1
        self._records.append(UsageRecord(
            variant=capability.variant, task_id=capability.task_id,
            provider_calls=1, tokens=250, spend_usd=0.002))

        if self.provider is None:
            raise GatewayDenied("no search provider is configured")
        try:
            results = self.provider(query, limit) or []
        except GatewayDenied:
            raise
        except Exception as exc:  # noqa: BLE001 - a provider fault must not leak its internals
            raise GatewayDenied("the search provider could not answer") from exc

        # Redacted on the way OUT, every field, every result. The provider's response is the one
        # place a credential could travel back to the candidate, and redacting only known fields
        # would be a denylist -- the thing this file avoids everywhere else.
        return [
            {str(key): redact(value) if isinstance(value, str) else value
             for key, value in (item or {}).items()}
            for item in results[:limit] if isinstance(item, dict)
        ]

    def quota(self, token: str) -> tuple[int, int]:
        """``(used, max_calls)`` for a live capability. Free, and it consumes nothing.

        Deliberately NOT billed: an agent that cannot see its own remaining quota either wastes it
        or hoards it, and both make the cost signal measure planning rather than search quality.
        It authorizes through the same path as a search, so an unknown or expired token is refused
        here too — a free operation is still not an oracle.
        """
        capability = self._authorize(token, check_quota=False)
        return self._calls[capability.token], capability.max_calls

    # ---- receipts -------------------------------------------------------------------------------
    def usage_manifest(self) -> UsageManifest:
        """The gateway's own record of spend, merged per (variant, task)."""
        merged: dict[tuple[str, str], UsageRecord] = {}
        for record in self._records:
            key = (record.variant, record.task_id)
            current = merged.get(key)
            merged[key] = record if current is None else UsageRecord(
                variant=record.variant, task_id=record.task_id,
                provider_calls=current.provider_calls + record.provider_calls,
                tokens=current.tokens + record.tokens,
                spend_usd=round(current.spend_usd + record.spend_usd, 10))
        return UsageManifest(challenge_id=self.challenge_id,
                             records=tuple(merged[key] for key in sorted(merged)))

    def usage_receipt(self, *, signing_key: bytes) -> dict:
        """A HASH-BOUND receipt: the usage manifest plus an HMAC over its canonical bytes.

        Hash-bound rather than merely recorded, because a later settlement trusts it.
        Anyone can read it; only the holder of the signing key can produce one that verifies, so an
        edited receipt is detectable rather than merely unlikely.
        """
        manifest = self.usage_manifest()
        body = {
            "schema_version": 1,
            "lane_id": self.lane_id,
            "challenge_id": self.challenge_id,
            "usage": manifest.as_dict(),
            "usage_digest": manifest.digest(),
            "reservation_calls": self.reservation_calls,
            "served_calls": self._served,
        }
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return {**body, "hmac_sha256": hmac.new(signing_key, canonical, hashlib.sha256).hexdigest()}

    @staticmethod
    def verify_receipt(receipt: dict, *, signing_key: bytes) -> bool:
        """Whether a receipt is intact. Constant-time compare: a timing oracle on a MAC is a
        forgery path."""
        if not isinstance(receipt, dict) or "hmac_sha256" not in receipt:
            return False
        body = {k: v for k, v in receipt.items() if k != "hmac_sha256"}
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        expected = hmac.new(signing_key, canonical, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, str(receipt["hmac_sha256"]))
