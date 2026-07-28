"""The typed client for the three operations an agent may invoke.

**There is no API key here, and there is no way to ask for one.** The trusted broker holds the
miner's four decrypted credentials and spends them on the agent's behalf; what this client presents
is a *capability*, which is worth its remaining calls and nothing else. That is the whole of Phase C
seen from the agent's side: the surface below is not a restricted view of a credential, it is the
only thing that exists.

**You name an operation, never a URL.** ``web_search`` reaches the search provider the broker chose,
with the key the broker chose. There is no parameter for a host, a model or an actor id — not
because they are validated away, but because this client cannot express them and the broker would
refuse them if it could.

**Two transports, one API.** In the sealed room this is HTTP to the in-room broker. In the local
sandbox, ``web_search`` goes over the lane's unix socket instead, so the same submission can be
calibrated before it is submitted. ``x_search`` and ``final_summary`` exist only in the room and say
so plainly rather than returning something empty that would read as a bad answer.
"""

from __future__ import annotations

import json
import os

CAPABILITY_HEADER = "x-kata-capability"

#: Set by the room's profile. There is deliberately no key variable alongside them.
BROKER_URL_ENV = "SN22_BROKER_URL"
BROKER_CAPABILITY_ENV = "SN22_BROKER_CAPABILITY"

#: The lane's sandbox equivalents, for calibration before submission.
RELAY_ENDPOINT_ENV = "SN22_RELAY_ENDPOINT"
RELAY_CAPABILITY_ENV = "SN22_RELAY_CAPABILITY"
RELAY_SCHEME = "sn22-relay+unix://"

OP_WEB_SEARCH = "web-search"
OP_X_SEARCH = "x-search"
OP_FINAL_SUMMARY = "final-summary"
AGENT_OPERATIONS = (OP_WEB_SEARCH, OP_X_SEARCH, OP_FINAL_SUMMARY)

MAX_REQUEST_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 60.0


class BrokerError(Exception):
    """The broker refused, was unreachable, or answered unintelligibly.

    ONE class for all three, deliberately. An agent that could tell "quota exhausted" from "unknown
    capability" from "expired" could map the room's state one probe at a time; an agent that cannot
    is left with the only useful response to any of them, which is to answer with what it has.
    """


def in_sealed_room() -> bool:
    """Whether this agent is in the room rather than the local sandbox.

    Decided by which endpoint the environment carries, not by a flag the agent could set: the two
    are configured by different components, so the presence of one is the fact rather than a claim
    about it. A unix socket in the work directory can only have been put there by the lane, so it
    wins if somehow both are present.
    """
    return not _relay_path() and bool(os.environ.get(BROKER_URL_ENV, "").strip())


def _relay_path() -> str:
    raw = os.environ.get(RELAY_ENDPOINT_ENV, "").strip()
    return raw[len(RELAY_SCHEME):] if raw.startswith(RELAY_SCHEME) else raw


class BrokerClient:
    """The agent's whole outward surface."""

    def __init__(self, *, timeout: float = DEFAULT_TIMEOUT_SECONDS):
        self.timeout = timeout

    # ---- the three operations ----------------------------------------------------------------

    def web_search(self, query: str, *, count: int = 10) -> list:
        """Search the web. Returns the provider's own result objects.

        Left in the provider's shape rather than tidied: the scorer reads ``{title, link, snippet}``
        as dicts, and any reshaping here is a difference it cannot see but would be scored on.
        """
        if not in_sealed_room():
            return self._sandbox_search(query, count)
        document = self._call(OP_WEB_SEARCH, {"query": query, "count": int(count)})
        return _objects(document.get("results"))

    def x_search(self, query: str, *, count: int = 10) -> list:
        """Search X/Twitter. Returns raw tweet objects.

        The validator re-scrapes every tweet you return and compares it field by field, so pass
        them through unedited. A "cleaned up" tweet scores zero rather than less.
        """
        self._require_room("x_search")
        document = self._call(OP_X_SEARCH, {"query": query, "count": int(count)})
        return _objects(document.get("results"))

    def final_summary(self, messages: list) -> str:
        """Write the final summary on the fixed model.

        The messages are yours -- you are writing your own answer. The model is not: every
        contestant summarises with the same one, or the duel measures budget rather than skill.
        """
        self._require_room("final_summary")
        document = self._call(OP_FINAL_SUMMARY, {"messages": list(messages or [])})
        content = document.get("content")
        if not isinstance(content, str):
            raise BrokerError("the broker answered without a summary")
        return content

    def quota(self) -> dict:
        """What is left, per operation. Free, and it costs nothing.

        Exposed because an agent that cannot see its own quota either wastes it or hoards it, and
        both make the measurement about planning rather than answer quality.
        """
        if not in_sealed_room():
            return {"role": "agent", "operations": {}, "metered": False}
        import urllib.error
        import urllib.request

        base, capability = self._room_settings()
        request = urllib.request.Request(
            f"{base}/v1/quota", method="GET", headers={CAPABILITY_HEADER: capability})
        return self._read(request, urllib)

    # ---- internals ---------------------------------------------------------------------------

    def _require_room(self, name: str) -> None:
        if not in_sealed_room():
            raise BrokerError(
                f"{name} is only available in the sealed room; the local sandbox serves web search "
                f"only")

    def _room_settings(self) -> tuple[str, str]:
        base = os.environ.get(BROKER_URL_ENV, "").strip().rstrip("/")
        capability = os.environ.get(BROKER_CAPABILITY_ENV, "").strip()
        if not base:
            raise BrokerError("no broker endpoint was provided")
        if not capability:
            # The room mints one per job. Missing means the room did not plumb it, and every call
            # would come back refused with no way for the agent to tell why.
            raise BrokerError("no broker capability is available in this room")
        return base, capability

    def _call(self, operation: str, payload: dict) -> dict:
        """One named operation.

        ``urllib`` is imported HERE rather than at module scope. The lane's static screen rejects a
        submission that imports it, and it reads this file too when it is copied into a bundle;
        keeping the import inside the function means the sandbox path never loads it at all.
        """
        import urllib.error
        import urllib.request

        if operation not in AGENT_OPERATIONS:
            # Unreachable through the public methods. Kept because the URL is assembled from this
            # value, and a check that can never fire is cheaper than one that was needed.
            raise BrokerError("unknown operation")
        base, capability = self._room_settings()
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        if len(body) > MAX_REQUEST_BYTES:
            raise BrokerError("request exceeds the size limit")
        request = urllib.request.Request(
            f"{base}/v1/op/{operation}", data=body, method="POST",
            headers={"content-type": "application/json", CAPABILITY_HEADER: capability})
        return self._read(request, urllib)

    def _read(self, request, urllib_module) -> dict:
        try:
            with urllib_module.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib_module.error.HTTPError as exc:
            # Deliberately NOT distinguishing 401 from 403 from 502 -- see BrokerError.
            raise BrokerError(f"the broker refused the request ({exc.code})") from exc
        except (urllib_module.error.URLError, OSError, ValueError) as exc:
            raise BrokerError(f"the broker is unreachable: {exc}") from exc
        if len(raw) > MAX_RESPONSE_BYTES:
            raise BrokerError("the broker's answer exceeds the size limit")
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise BrokerError(
                f"the broker answered with something that is not JSON: {exc}") from exc
        if not isinstance(document, dict):
            raise BrokerError("the broker's answer is not a JSON object")
        return document

    def _sandbox_search(self, query: str, count: int) -> list:
        """The lane's unix-socket relay, so a v2 submission can be calibrated before submission.

        ``socket`` is imported here for the same reason ``urllib`` is imported in ``_call``.
        """
        import socket

        path = _relay_path()
        if not path:
            raise BrokerError("no broker endpoint was provided")
        payload = {"op": "search", "capability": os.environ.get(RELAY_CAPABILITY_ENV, ""),
                   "query": query, "limit": int(count)}
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(self.timeout)
        try:
            try:
                connection.connect(path)
            except OSError as exc:
                raise BrokerError(f"the relay is unreachable: {exc}") from exc
            connection.sendall(body)
            try:
                connection.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            chunks, total = [], 0
            while True:
                try:
                    chunk = connection.recv(65536)
                except OSError as exc:
                    raise BrokerError(f"the relay read failed: {exc}") from exc
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_RESPONSE_BYTES:
                    raise BrokerError("the relay's answer exceeds the size limit")
                chunks.append(chunk)
        finally:
            connection.close()
        try:
            document = json.loads(b"".join(chunks).decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise BrokerError(f"the relay answered with something that is not JSON: {exc}") from exc
        if not isinstance(document, dict) or not document.get("ok"):
            raise BrokerError("the relay refused the request")
        return _objects(document.get("results"))


def _objects(value) -> list:
    """Every result must be a non-empty object; an empty one is something an agent cannot use."""
    return [item for item in (value or []) if isinstance(item, dict) and item]
