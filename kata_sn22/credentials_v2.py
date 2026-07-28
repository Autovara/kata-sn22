"""The version-2 sealed credential set, and what a room may say about it.

**The funding inversion.** In version 1 the lane held provider keys and paid for everything. In
version 2 each contestant seals its own four keys to the attested room, and they pay for that
contestant's search, summary, verification and judging. The validator host holds no paid SN22
credential at all — so a contestant that cannot fund its own evaluation scores zero, and there is no
lane key to fall back to. That is a product decision with teeth: the reigning King keeps paying to
defend its crown.

**Four keys, all required.** A production epoch covers all four pools, and each pool needs a
different subset. Requiring the full set at sealing time is what turns "you will fail on pool three
of four, an hour in" into "your submission is rejected at intake".

| Provider | The agent spends it on | The trusted evaluator spends it on |
|---|---|---|
| ScrapingDog | Web search | independently fetching the pages the agent returned |
| Apify | X search | independently re-scraping the tweets the agent returned |
| OpenAI | its final summary (`gpt-4.1-nano`) | nothing |
| Chutes | nothing | the fixed Qwen judge |

Note the asymmetry: the agent never touches Chutes, and the evaluator never touches OpenAI. An agent
that could reach the judge could grade itself; that separation is enforced by the broker, and this
module is where the two roles are *named* so the broker has something to enforce against.

**This module holds no secret values.** It parses, validates and describes a credential set. Every
representation here — repr, logs, errors, the attested report — carries statuses and never key
material. The one function that touches the values, :func:`bundle_binding_matches`, returns a bool.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum

#: Bumped when the sealed payload's shape changes. A v1 payload must never parse as v2: v1 sealed a
#: single key for SN60, and reading it as "one of four provided" would start a round that cannot
#: finish.
CREDENTIAL_VERSION = 2

#: Names the profile so a room can refuse a credential set sealed for a different lane's policy.
CREDENTIAL_PROFILE = "sn22-miner-funded-v1"

#: Exactly these, in this order, no others. Not a minimum — a set.
REQUIRED_PROVIDERS: tuple[str, ...] = ("scrapingdog", "apify", "openai", "chutes")

#: Which role may spend which credential. The broker enforces it; naming it here is what lets a
#: test assert the two agree.
AGENT_PROVIDERS: frozenset[str] = frozenset({"scrapingdog", "apify", "openai"})
EVALUATOR_PROVIDERS: frozenset[str] = frozenset({"scrapingdog", "apify", "chutes"})

#: A key that is shorter than this is a placeholder, not a credential. A key longer than this is
#: something else pasted by mistake, and refusing it early keeps it out of memory.
MIN_KEY_CHARS = 16
MAX_KEY_CHARS = 512

#: Printable ASCII without whitespace. Every provider here issues keys in this alphabet, and a key
#: carrying a newline would break every log line and header it ever reaches.
_KEY_RE = re.compile(r"^[!-~]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CredentialError(ValueError):
    """A sealed credential set is malformed, incomplete, or not bound to this bundle.

    Deliberately carries no value, no provider response and no key fragment: this exception is
    logged, attested and shown to a miner.
    """


class CredentialStatus(str, Enum):
    """What a room may report about one credential, after trying to use it.

    These are the only outcomes. Each maps to exactly one product rule, which is why "it did not
    work" is not one of them -- ``credential_failure`` scores the contestant zero, while
    ``provider_outage`` defers the whole duel, and confusing the two either punishes an innocent
    miner or hands a free pass to a broken one.
    """

    OK = "ok"
    #: The contestant's fault, and its score is zero: absent, undecryptable, not bound to the
    #: bundle, rejected (401/403), out of credit (402), or still limited after bounded retries.
    MISSING = "missing"
    INVALID = "invalid"
    UNAUTHORIZED = "unauthorized"
    PAYMENT_REQUIRED = "payment_required"
    RATE_LIMITED = "rate_limited"
    EXPIRED = "expired"
    INSUFFICIENT = "insufficient"
    #: NOT the contestant's fault, and the duel defers: the provider itself is down.
    PROVIDER_OUTAGE = "provider_outage"
    #: Never used: this credential was not needed for the pools that ran.
    UNUSED = "unused"


#: The statuses that zero a contestant. Everything else either passed or was nobody's fault.
CONTESTANT_FAULT_STATUSES: frozenset[CredentialStatus] = frozenset({
    CredentialStatus.MISSING, CredentialStatus.INVALID, CredentialStatus.UNAUTHORIZED,
    CredentialStatus.PAYMENT_REQUIRED, CredentialStatus.RATE_LIMITED,
    CredentialStatus.EXPIRED, CredentialStatus.INSUFFICIENT,
})

#: The statuses that defer the duel rather than deciding it.
DEFER_STATUSES: frozenset[CredentialStatus] = frozenset({CredentialStatus.PROVIDER_OUTAGE})


@dataclass(frozen=True)
class CredentialSet:
    """One contestant's four decrypted provider keys, bound to its exact bundle.

    Constructed only inside the trusted runner. It never leaves: what leaves is a
    :class:`CredentialReport`, which carries statuses.
    """

    keys: dict
    bundle_binding: str
    profile: str = CREDENTIAL_PROFILE
    version: int = CREDENTIAL_VERSION

    def __repr__(self) -> str:
        # Overridden because the default would print the keys. A dataclass holding secrets that
        # renders them in a traceback is a secret in every log that ever catches an exception.
        return (f"CredentialSet(providers={sorted(self.keys)!r}, "
                f"bundle_binding={self.bundle_binding[:8]!r}..., profile={self.profile!r})")

    __str__ = __repr__

    def key(self, provider: str) -> str:
        """The key for one provider. Callers are trusted-runner code only."""
        if provider not in self.keys:
            raise CredentialError(f"no credential for provider {provider!r}")
        return self.keys[provider]

    def for_role(self, role: str) -> frozenset[str]:
        """Which providers a role may spend. ``role`` is ``"agent"`` or ``"evaluator"``."""
        if role == "agent":
            return AGENT_PROVIDERS
        if role == "evaluator":
            return EVALUATOR_PROVIDERS
        raise CredentialError(f"unknown role {role!r}")


def parse_credential_payload(raw: bytes | str | dict) -> CredentialSet:
    """Parse the decrypted sealed payload. Raises :class:`CredentialError` on anything unexpected.

    Strict about the whole document, not just the parts it needs: an extra top-level field means the
    miner sealed something this room does not understand, and proceeding would run a contestant
    under rules it did not agree to.
    """
    if isinstance(raw, dict):
        document = raw
    else:
        payload = raw.encode("utf-8") if isinstance(raw, str) else raw
        try:
            document = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            # The message names no content: this text was encrypted, and echoing a fragment of a
            # failed decrypt into a log is how a partially-correct key gets recovered.
            raise CredentialError("sealed payload is not valid JSON") from exc
    if not isinstance(document, dict):
        raise CredentialError("sealed payload is not a JSON object")

    # VERSION FIRST. A version-1 payload also fails the unknown-field check below, but on a
    # message ("unknown field(s): api_key, provider") that does not tell a miner what they actually
    # did -- which was seal with the SN60 tool.
    version = document.get("version")
    if version != CREDENTIAL_VERSION:
        raise CredentialError(
            f"sealed payload version {version!r} is not {CREDENTIAL_VERSION}; a version-1 "
            f"single-key credential cannot be used for SN22. Reseal with kata_seal_sn22")

    unknown = sorted(set(document) - {"version", "credential_profile", "credentials",
                                      "bundle_binding"})
    if unknown:
        raise CredentialError(f"sealed payload has unknown field(s): {', '.join(unknown)}")
    profile = document.get("credential_profile")
    if profile != CREDENTIAL_PROFILE:
        raise CredentialError(f"credential_profile {profile!r} is not {CREDENTIAL_PROFILE!r}")

    binding = document.get("bundle_binding")
    if not isinstance(binding, str) or not _SHA256_RE.fullmatch(binding):
        raise CredentialError("bundle_binding must be a 64-character lowercase SHA-256")

    credentials = document.get("credentials")
    if not isinstance(credentials, dict):
        raise CredentialError("credentials must be an object")
    present = set(credentials)
    missing = [name for name in REQUIRED_PROVIDERS if name not in present]
    if missing:
        raise CredentialError(
            f"sealed payload is missing required credential(s): {', '.join(missing)}. All of "
            f"{', '.join(REQUIRED_PROVIDERS)} are required because a production epoch covers all "
            f"four pools")
    extra = sorted(present - set(REQUIRED_PROVIDERS))
    if extra:
        raise CredentialError(f"sealed payload has unknown provider(s): {', '.join(extra)}")

    keys: dict = {}
    for provider in REQUIRED_PROVIDERS:
        entry = credentials[provider]
        if not isinstance(entry, dict) or set(entry) != {"api_key"}:
            raise CredentialError(f"credentials.{provider} must be exactly {{\"api_key\": ...}}")
        value = entry["api_key"]
        if not isinstance(value, str):
            raise CredentialError(f"credentials.{provider}.api_key must be a string")
        # Length and alphabet only. The message never contains the value or its length in a way
        # that narrows it, because this error reaches the miner.
        if not MIN_KEY_CHARS <= len(value) <= MAX_KEY_CHARS:
            raise CredentialError(
                f"credentials.{provider}.api_key must be {MIN_KEY_CHARS}..{MAX_KEY_CHARS} "
                f"characters")
        if not _KEY_RE.fullmatch(value):
            raise CredentialError(
                f"credentials.{provider}.api_key must be printable ASCII with no whitespace")
        keys[provider] = value

    return CredentialSet(keys=keys, bundle_binding=binding, profile=profile, version=version)


def bundle_binding_matches(credential_set: CredentialSet, bundle_sha256: str) -> bool:
    """Whether this credential set was sealed for THIS bundle.

    The binding is what stops a validator replaying a miner's public ciphertext against a different
    agent to make the miner's keys pay for someone else's work -- or to probe them. Compared in
    constant time: the comparison is against attacker-influenced input, and an early-exit compare
    leaks the matching prefix one byte at a time.
    """
    import hmac

    return hmac.compare_digest(credential_set.bundle_binding, (bundle_sha256 or "").lower())


def compute_bundle_binding(files: dict) -> str:
    """The binding a miner seals: SHA-256 over every bundle file except the ciphertext itself.

    ``files`` maps a bundle-relative path to its bytes. The ciphertext is excluded because it cannot
    commit to itself; everything else is included, so editing ``agent.py``, a helper or a manifest
    invalidates the seal and forces a reseal. That is the point -- a credential bound to code the
    miner has since changed is a credential paying for code nobody reviewed.
    """
    digest = hashlib.sha256()
    for path in sorted(files):
        if path == SEALED_FILENAME:
            continue
        digest.update(path.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(hashlib.sha256(files[path]).digest())
    return digest.hexdigest()


#: The bundle file holding the ciphertext. Excluded from its own binding.
SEALED_FILENAME = "sealed_inference_key"


@dataclass(frozen=True)
class CredentialReport:
    """What a room says about a contestant's credentials. Statuses only, never values."""

    statuses: dict

    def as_dict(self) -> dict:
        return {provider: status.value for provider, status in sorted(self.statuses.items())}

    @property
    def contestant_at_fault(self) -> bool:
        """True when the contestant's score is zero under the product rule."""
        return any(status in CONTESTANT_FAULT_STATUSES for status in self.statuses.values())

    @property
    def defer(self) -> bool:
        """True when the duel defers rather than being decided.

        Checked AFTER ``contestant_at_fault`` by the caller, but the two are not mutually
        exclusive: a contestant can present a bad key while another provider is also down. A duel
        that defers can be re-run; a contestant zeroed on someone else's outage cannot be
        un-zeroed, so the safe reading of an ambiguous round is to defer it.
        """
        return any(status in DEFER_STATUSES for status in self.statuses.values())

    @classmethod
    def all_ok(cls) -> "CredentialReport":
        return cls({provider: CredentialStatus.OK for provider in REQUIRED_PROVIDERS})
