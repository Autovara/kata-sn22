"""Validator-side client for the attested SN22 sealed room.

The room protocol is intentionally the same narrow protocol used by SN60: send a bounded candidate
bundle and one task, authenticate the exact request bytes, then accept the answer only when a
cryptographically verified TDX quote binds the nonce, task, bundle digest, answer and provenance to
an operator-approved room measurement.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import http.client
import inspect
import io
import json
import os
import random
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

ROOM_AUTH_SECRET_ENV = "KATA_ROOM_AUTH_SECRET"
ROOM_MEASUREMENTS_ENV = "KATA_SN22_ROOM_MEASUREMENTS"
ROOM_HTTP_TIMEOUT_ENV = "KATA_SN22_ROOM_HTTP_TIMEOUT_SECONDS"
ROOM_REQUEST_LIFETIME_ENV = "KATA_SN22_ROOM_REQUEST_LIFETIME_SECONDS"
ROOM_MAX_ATTEMPTS_ENV = "KATA_SN22_ROOM_MAX_ATTEMPTS"
ROOM_RETRY_BASE_SECONDS_ENV = "KATA_SN22_ROOM_RETRY_BASE_SECONDS"
PCCS_URL_ENV = "KATA_SN22_PCCS_URL"
ALLOW_INSECURE_ROOM_URL_ENV = "KATA_SN22_ALLOW_INSECURE_ROOM_URL"
ROOM_SIGNATURE_HEADER = "X-Kata-Signature"
DEFAULT_ROOM_HTTP_TIMEOUT_SECONDS = 180.0
MAX_ROOM_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_BUNDLE_BYTES = 256 * 1024
MAX_BUNDLE_FILES = 16
SEALED_CREDENTIAL_FILENAME = "sealed_inference_key"
_BUNDLE_BINDING_DOMAIN = b"kata-miner-credential-bundle-v1\0"
_RETRYABLE_HTTP_STATUS = frozenset({502, 503, 504})


class RoomTransportError(RuntimeError):
    """A transient connection failure for which a fresh-nonce retry is safe."""


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _open_room(request, *, timeout: float):
    return urllib.request.build_opener(_RejectRedirects).open(request, timeout=timeout)


@dataclass(frozen=True)
class VerifiedQuote:
    ok: bool
    report_data: bytes
    measurement: str
    detail: str = ""


class QuoteVerifier(Protocol):
    def verify(self, quote_hex: str) -> VerifiedQuote: ...


@dataclass(frozen=True)
class RoomPolicy:
    approved_measurements: frozenset[str]


@dataclass(frozen=True)
class RoomResult:
    report: object
    quote_hex: str
    bundle_sha256: str
    provenance: dict[str, object]


@dataclass(frozen=True)
class CandidateOutcome:
    accepted: bool
    report: object | None
    reason: str
    provenance: dict[str, object] | None = None


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def room_signature(body: bytes) -> str:
    secret = os.environ.get(ROOM_AUTH_SECRET_ENV, "").strip().encode()
    if not secret:
        raise RuntimeError(f"{ROOM_AUTH_SECRET_ENV} is required to authenticate sealed-room runs")
    return hmac.new(secret, body, hashlib.sha256).hexdigest()


def resolve_room_policy() -> RoomPolicy:
    raw = os.environ.get(ROOM_MEASUREMENTS_ENV, "")
    measurements = frozenset(value.strip() for value in raw.split(",") if value.strip())
    if not measurements:
        raise RuntimeError(f"{ROOM_MEASUREMENTS_ENV} must list approved room measurements")
    invalid = sorted(value for value in measurements if not _is_sha256(value))
    if invalid:
        raise RuntimeError(f"{ROOM_MEASUREMENTS_ENV} values must be 64 lowercase hexadecimal bytes")
    return RoomPolicy(measurements)


def validate_room_url(value: str) -> str:
    url = value.strip().rstrip("/")
    parsed = urllib.parse.urlparse(url)
    if (
        parsed.scheme not in {"https", "http"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError("KATA_SN22_ROOM_URL must be an absolute HTTPS URL without credentials")
    allow_insecure = os.environ.get(ALLOW_INSECURE_ROOM_URL_ENV, "").strip().lower()
    if parsed.scheme != "https" and allow_insecure not in {"1", "true", "yes", "on"}:
        raise RuntimeError(
            f"KATA_SN22_ROOM_URL must use HTTPS; set {ALLOW_INSECURE_ROOM_URL_ENV}=1 "
            "only for a local test room"
        )
    return url


def verify_room_identity(
    base_url: str,
    *,
    policy: RoomPolicy,
    verifier: QuoteVerifier,
    timeout: float = 15.0,
) -> None:
    """Check public health and prove that ``/pubkey`` belongs to an approved TDX room."""
    url = validate_room_url(base_url)
    try:
        with _open_room(f"{url}/health", timeout=timeout) as response:
            health = json.loads(response.read().decode())
        if not isinstance(health, dict) or health.get("ok") is not True:
            raise RuntimeError("room /health did not report ok")
        with _open_room(f"{url}/pubkey", timeout=timeout) as response:
            document = json.loads(response.read().decode())
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RuntimeError(f"cannot reach the SN22 room health endpoints: {exc}") from exc
    if not isinstance(document, dict):
        raise RuntimeError("room /pubkey returned a non-object response")
    public_key = document.get("pubkey")
    quote_hex = document.get("quote")
    if (
        not isinstance(public_key, str)
        or len(public_key) != 66
        or not all(char in "0123456789abcdef" for char in public_key)
        or not isinstance(quote_hex, str)
    ):
        raise RuntimeError("room /pubkey returned an invalid key or quote")
    quote = verifier.verify(quote_hex)
    if not quote.ok:
        raise RuntimeError(f"room /pubkey quote was not verified: {quote.detail}")
    if quote.measurement not in policy.approved_measurements:
        raise RuntimeError(f"room /pubkey measurement is not approved: {quote.measurement}")
    expected = hashlib.sha256(b"kata-sealing-pubkey:" + bytes.fromhex(public_key)).digest()
    if not hmac.compare_digest(quote.report_data[:32], expected):
        raise RuntimeError("room /pubkey quote does not bind the published sealing key")


def resolve_room_http_timeout_seconds() -> float:
    raw = os.environ.get(ROOM_HTTP_TIMEOUT_ENV, "").strip()
    if not raw:
        return DEFAULT_ROOM_HTTP_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{ROOM_HTTP_TIMEOUT_ENV} must be a positive number") from exc
    if value <= 0:
        raise RuntimeError(f"{ROOM_HTTP_TIMEOUT_ENV} must be a positive number")
    return value


def resolve_room_max_attempts() -> int:
    raw = os.environ.get(ROOM_MAX_ATTEMPTS_ENV, "3").strip()
    try:
        value = int(raw)
    except ValueError:
        return 3
    return value if 1 <= value <= 5 else 3


class DcapQvlVerifier:
    """Verify a TDX quote and derive the stable dstack compose measurement.

    dcap-qvl 0.5.x exposes TDX report fields through ``parse_quote().report`` and returns uppercase
    verification statuses. Older spelling is normalized only for compatibility; configuration or
    out-of-date statuses remain rejected.
    """

    ACCEPTED_STATUSES = frozenset({"OK", "SW_HARDENING_NEEDED"})
    _STATUS_ALIASES = {
        "UpToDate": "OK",
        "SWHardeningNeeded": "SW_HARDENING_NEEDED",
    }

    def verify(self, quote_hex: str) -> VerifiedQuote:
        try:
            import dcap_qvl
        except ImportError:
            return VerifiedQuote(False, b"", "", "dcap-qvl 0.5.x is not installed")
        try:
            raw = bytes.fromhex(quote_hex)
            parsed = dcap_qvl.parse_quote(raw)
            if hasattr(parsed, "is_tdx") and not parsed.is_tdx():
                return VerifiedQuote(False, b"", "", "the room quote is not a TDX quote")
            report = parsed.report
            report_data = bytes(report.report_data)
            mr_config_id = bytes(report.mr_config_id)
            if len(report_data) < 32 or len(mr_config_id) < 33:
                return VerifiedQuote(False, b"", "", "the TDX quote report is incomplete")
            measurement = mr_config_id[1:33].hex()
            pccs = os.environ.get(PCCS_URL_ENV, "").strip() or dcap_qvl.PHALA_PCCS_URL

            async def _verify():
                collateral = dcap_qvl.get_collateral(pccs, raw)
                if inspect.isawaitable(collateral):
                    collateral = await collateral
                verified = dcap_qvl.verify(raw, collateral, int(time.time()))
                if inspect.isawaitable(verified):
                    verified = await verified
                return verified

            verified = asyncio.run(_verify())
            raw_status = str(getattr(verified, "status", ""))
            status = self._STATUS_ALIASES.get(raw_status, raw_status)
            if status not in self.ACCEPTED_STATUSES:
                return VerifiedQuote(False, report_data, measurement, f"TCB status {raw_status}")
            return VerifiedQuote(True, report_data, measurement, status)
        except Exception as exc:  # noqa: BLE001 - a malformed/unverifiable quote is one refusal
            return VerifiedQuote(False, b"", "", f"dcap-qvl verification failed: {exc}")


def verify_room_run(
    *,
    result: RoomResult,
    nonce: bytes,
    project_key: str,
    expected_bundle_sha256: str,
    policy: RoomPolicy,
    verifier: QuoteVerifier,
    seen_nonces: set[bytes] | None = None,
) -> str | None:
    if result.bundle_sha256 != expected_bundle_sha256:
        return "room returned a different candidate bundle hash"
    quote = verifier.verify(result.quote_hex)
    if not quote.ok:
        return f"quote not verified: {quote.detail}"
    if quote.measurement not in policy.approved_measurements:
        return f"runner image measurement is not approved: {quote.measurement}"
    binding = {
        "report": result.report,
        "bundle_sha256": result.bundle_sha256,
        "provenance": result.provenance,
    }
    binding_hash = hashlib.sha256(canonical(binding)).digest()
    expected = hashlib.sha256(nonce + project_key.encode() + binding_hash).digest()
    if not hmac.compare_digest(quote.report_data[:32], expected):
        return "quote does not bind this task, bundle and answer"
    if seen_nonces is not None:
        if nonce in seen_nonces:
            return "nonce already accepted"
        seen_nonces.add(nonce)
    return None


class HttpRoomLauncher:
    def __init__(self, base_url: str, timeout: float | None = None):
        self.base_url = validate_room_url(base_url)
        self.timeout = resolve_room_http_timeout_seconds() if timeout is None else timeout

    def launch_and_run(
        self,
        *,
        agent_ref: str,
        project_key: str,
        nonce: bytes,
        sealed_key_ref: str,
        bundle_sha256: str,
    ) -> RoomResult:
        issued_at = int(time.time())
        lifetime = _request_lifetime_seconds()
        payload = json.dumps(
            {
                "nonce": nonce.hex(),
                "project_key": project_key,
                "sealed_key": sealed_key_ref,
                "bundle": _bundle_tar_b64(agent_ref),
                "bundle_sha256": bundle_sha256,
                "issued_at": issued_at,
                "expires_at": issued_at + lifetime,
            },
            separators=(",", ":"),
        ).encode()
        request = urllib.request.Request(
            f"{self.base_url}/run",
            data=payload,
            headers={
                "Content-Type": "application/json",
                ROOM_SIGNATURE_HEADER: room_signature(payload),
            },
            method="POST",
        )
        try:
            with _open_room(request, timeout=self.timeout) as response:
                raw_response = response.read(MAX_ROOM_RESPONSE_BYTES + 1)
            if len(raw_response) > MAX_ROOM_RESPONSE_BYTES:
                raise RuntimeError("room response exceeds the 4 MiB safety limit")
            document = json.loads(raw_response.decode())
        except urllib.error.HTTPError as exc:
            body = exc.read(401).decode(errors="replace")[:400]
            if exc.code in _RETRYABLE_HTTP_STATUS:
                raise RoomTransportError(f"room HTTP {exc.code}: {body}") from exc
            raise RuntimeError(f"room HTTP {exc.code}: {body}") from exc
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            ConnectionError,
            TimeoutError,
        ) as exc:
            reason = getattr(exc, "reason", exc)
            raise RoomTransportError(f"could not reach room: {reason}") from exc
        if not isinstance(document, dict):
            raise RuntimeError("room returned a non-object response")
        if (
            "report" not in document
            or "quote" not in document
            or not isinstance(document.get("provenance"), dict)
        ):
            raise RuntimeError(f"room error: {document.get('error', document)}")
        return RoomResult(
            report=document["report"],
            quote_hex=str(document["quote"]),
            bundle_sha256=str(document.get("bundle_sha256", "")),
            provenance=document["provenance"],
        )


def evaluate_candidate_in_room(
    *,
    agent_ref: str,
    project_key: str,
    sealed_key_ref: str,
    bundle_sha256: str,
    policy: RoomPolicy,
    launcher: HttpRoomLauncher,
    verifier: QuoteVerifier,
    mint_nonce: Callable[[], bytes] = lambda: os.urandom(20),
    seen_nonces: set[bytes] | None = None,
    max_attempts: int | None = None,
) -> CandidateOutcome:
    attempts = resolve_room_max_attempts() if max_attempts is None else max(1, max_attempts)
    reason = "sealed room did not answer"
    for attempt in range(1, attempts + 1):
        nonce = mint_nonce()
        try:
            result = launcher.launch_and_run(
                agent_ref=agent_ref,
                project_key=project_key,
                nonce=nonce,
                sealed_key_ref=sealed_key_ref,
                bundle_sha256=bundle_sha256,
            )
        except RoomTransportError as exc:
            reason = str(exc)
            if attempt < attempts:
                time.sleep(_retry_backoff_seconds(attempt))
                continue
            return CandidateOutcome(
                False,
                None,
                f"room unreachable after {attempts} attempts: {reason}",
            )
        except Exception as exc:  # noqa: BLE001 - non-transport room faults are not retryable
            return CandidateOutcome(False, None, f"room run failed: {exc}")
        rejection = verify_room_run(
            result=result,
            nonce=nonce,
            project_key=project_key,
            expected_bundle_sha256=bundle_sha256,
            policy=policy,
            verifier=verifier,
            seen_nonces=seen_nonces,
        )
        if rejection:
            return CandidateOutcome(False, None, rejection)
        return CandidateOutcome(True, result.report, "ok", result.provenance)
    return CandidateOutcome(False, None, reason)


def sealed_key_for_bundle(bundle_root: str | Path) -> str:
    path = Path(bundle_root).expanduser().resolve() / "sealed_inference_key"
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError("TEE submissions must include sealed_inference_key") from exc
    try:
        encrypted = bytes.fromhex(value)
    except ValueError as exc:
        raise RuntimeError("sealed_inference_key must be hexadecimal ciphertext") from exc
    if len(encrypted) < 32:
        raise RuntimeError("sealed_inference_key is too short to be valid ciphertext")
    return value


def hash_bundle(bundle_root: str | Path) -> str:
    root = Path(bundle_root).expanduser().resolve()
    digest = hashlib.sha256(_BUNDLE_BINDING_DOMAIN)
    files = _bundle_files(root)
    if not files:
        raise RuntimeError("candidate bundle is empty")
    for path in files:
        if path.relative_to(root).as_posix() == SEALED_CREDENTIAL_FILENAME:
            continue
        relative = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _bundle_tar_b64(bundle_root: str | Path) -> str:
    root = Path(bundle_root).expanduser().resolve()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in _bundle_files(root):
            archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    return base64.b64encode(buffer.getvalue()).decode()


def _bundle_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise RuntimeError(f"candidate bundle does not exist: {root}")
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in {".git", "__pycache__"} for part in relative.parts):
            continue
        if path.is_symlink():
            raise RuntimeError(f"candidate bundle contains a symlink: {relative}")
        if path.is_file() and path.suffix not in {".pyc", ".pyo"}:
            files.append(path)
            if len(files) > MAX_BUNDLE_FILES:
                raise RuntimeError(
                    f"candidate bundle exceeds the {MAX_BUNDLE_FILES}-file room policy"
                )
    total_bytes = sum(path.stat().st_size for path in files)
    if total_bytes > MAX_BUNDLE_BYTES:
        raise RuntimeError(
            f"candidate bundle exceeds the {MAX_BUNDLE_BYTES}-byte room policy"
        )
    return files


def _request_lifetime_seconds() -> int:
    raw = os.environ.get(ROOM_REQUEST_LIFETIME_ENV, "180").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{ROOM_REQUEST_LIFETIME_ENV} must be an integer") from exc
    if not 1 <= value <= 1_200:
        raise RuntimeError(f"{ROOM_REQUEST_LIFETIME_ENV} must be 1..1200")
    return value


def _retry_backoff_seconds(attempt: int) -> float:
    try:
        base = float(os.environ.get(ROOM_RETRY_BASE_SECONDS_ENV, "2") or "2")
    except ValueError:
        base = 2.0
    delay = min(15.0, max(0.0, base) * (2 ** (attempt - 1)))
    return delay + random.uniform(0.0, delay * 0.25)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)
