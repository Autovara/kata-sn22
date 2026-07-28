"""The relay client a submission imports: ONE API, two transports (SN22-7, Phase F).

A submission has to be able to search. It also has no network namespace, no provider key, and a
static screen that rejects `import socket`, `import requests` and `urllib.request` outright. Those
facts look contradictory until you notice what resolves them: **the candidate does not implement the
transport at all.** The lane provides this module and the agent imports it.

That inversion buys three things a "call this HTTP endpoint" contract does not:

* **The ban can stay absolute.** No submission has a legitimate reason to open a socket, so the
  screen never has to distinguish a relay call from an exfiltration attempt — a distinction it could
  not make reliably anyway, since both are `socket.connect`.
* **There is one transport implementation, and it is reviewed.** Framing, size limits, timeouts and
  error classification are the lane's, identical for every contestant. A per-agent HTTP client would
  make "the relay returned the same content to both sides" depend on two strangers' retry loops.
* **It works under `--unshare-net`.** A unix socket is a filesystem object, so it survives into a
  sandbox with no network at all.

**Two places, two transports, one API.** The agent calls ``search(...)`` and never knows which:

===============  ==============================  ==============================================
                 Sandbox (calibration)           Sealed room (production)
===============  ==============================  ==============================================
Reached by       AF_UNIX socket in the workdir   HTTP to the in-room trusted broker
Selected by      ``SN22_RELAY_ENDPOINT``         ``SN22_BROKER_URL``
Who pays         the LANE, metered by capability the MINER, from keys it never sees
Authority        a capability                    a capability
Quota            enforced per capability         enforced per capability, per operation
===============  ==============================  ==============================================

Having both behind one function is not tidiness. It is what makes calibration mean anything: a
number measured in the sandbox only predicts a number measured in the room if the SAME ``agent.py``
runs in both, unchanged. Before this, the room provided no ``sn22_relay`` at all and every
submission would have died on ``import sn22_relay`` in its first real round.

**The agent never holds a provider key.** An earlier version of this module read the miner's
decrypted key out of ``SN22_INFERENCE_API_KEY`` and sent it upstream itself. That put a real
credential in the environment of code written by a stranger, where ``os.environ``,
``/proc/self/environ``, a crash dump or a stray ``print`` would have been enough to take it. The key
now stays in the trusted runner and the agent presents a capability instead: a token that is worth
its remaining calls and nothing else, and that is dead the moment the job ends.
"""
from __future__ import annotations

import json
import os
import socket

#: Wire framing. One newline-terminated JSON object each way, so a partial write is detectable
#: rather than being reassembled into something plausible.
ENCODING = "utf-8"
#: Ceilings on ONE message. Applied by both ends: the server refuses an oversized request before
#: parsing it, and the client refuses an oversized response before buffering it.
MAX_REQUEST_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 30.0

#: Where the lane tells the agent to find the socket, and the capability to present.
ENDPOINT_ENV = "SN22_RELAY_ENDPOINT"
CAPABILITY_ENV = "SN22_RELAY_CAPABILITY"

#: The sealed room's equivalents. The broker URL points at the in-room trusted broker, which holds
#: the miner's decrypted keys and never hands one out. There is deliberately NO key variable here:
#: see the module docstring.
BROKER_URL_ENV = "SN22_BROKER_URL"
BROKER_CAPABILITY_ENV = "SN22_BROKER_CAPABILITY"
CAPABILITY_HEADER = "x-kata-capability"

#: The broker operations this client knows how to ask for. Names, not URLs: the agent cannot name a
#: host, and an operation it has not been granted is refused by the broker rather than by this file.
OP_WEB_SEARCH = "web-search"
OP_X_SEARCH = "x-search"
OP_FINAL_SUMMARY = "final-summary"
#: The scheme the endpoint carries. A path, not a URL: there is nothing to resolve and nothing to
#: route, which is the point.
ENDPOINT_SCHEME = "sn22-relay+unix://"


class RelayError(Exception):
    """The relay refused, was unreachable, or answered unintelligibly.

    ONE class for all three on purpose. An agent that could distinguish "quota exhausted" from
    "unknown capability" from "expired" could probe the lane's state; an agent that cannot is left
    with the only useful response to any of them, which is to answer with what it already has.
    """


def endpoint_path(endpoint: str | None = None) -> str:
    """The socket path from an endpoint string, or from the environment."""
    raw = endpoint if endpoint is not None else os.environ.get(ENDPOINT_ENV, "")
    raw = (raw or "").strip()
    if raw.startswith(ENDPOINT_SCHEME):
        return raw[len(ENDPOINT_SCHEME):]
    return raw


def in_sealed_room() -> bool:
    """Whether this agent is running in the sealed room rather than the local sandbox.

    Decided by which endpoint the environment carries, not by a flag the agent could set: the two
    are configured by different components (the lane's sandbox vs the room's profile), so the
    presence of one is the fact rather than a claim about it. The unix socket wins if somehow both
    are set — a socket in the workdir can only have been put there by the lane.
    """
    return not endpoint_path() and bool(os.environ.get(BROKER_URL_ENV, "").strip())


#: Maps the sandbox's ``op`` to the room's broker operation. The sandbox relay has one search; the
#: room's broker has a separate operation per provider, because it -- not the agent -- decides which
#: credential a call spends.
_SANDBOX_OP_TO_BROKER_OP = {"search": OP_WEB_SEARCH}


def _request(payload: dict, *, endpoint: str | None = None,
             timeout: float = DEFAULT_TIMEOUT_SECONDS) -> dict:
    if endpoint is None and in_sealed_room():
        operation = _SANDBOX_OP_TO_BROKER_OP.get(str(payload.get("op") or ""), "")
        if not operation:
            raise RelayError("relay refused the request")
        body = {key: value for key, value in payload.items() if key not in {"op", "capability"}}
        if "limit" in body:
            # The sandbox calls it `limit`; the broker uses upstream's own field name, and enforces
            # upstream's own range on it.
            body["count"] = body.pop("limit")
        return _room_request(operation, body, timeout=timeout)
    path = endpoint_path(endpoint)
    if not path:
        raise RelayError("no relay endpoint was provided")
    body = json.dumps(payload, separators=(",", ":")).encode(ENCODING) + b"\n"
    if len(body) > MAX_REQUEST_BYTES:
        raise RelayError("request exceeds the relay size limit")

    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(timeout)
    try:
        try:
            connection.connect(path)
        except OSError as exc:
            raise RelayError(f"relay is unreachable: {exc}") from exc
        connection.sendall(body)
        # Half-close so the server sees EOF on the request without waiting for a timeout. The read
        # direction stays open for the answer.
        try:
            connection.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        chunks: list[bytes] = []
        total = 0
        while True:
            try:
                chunk = connection.recv(65536)
            except OSError as exc:
                raise RelayError(f"relay read failed: {exc}") from exc
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_RESPONSE_BYTES:
                raise RelayError("relay response exceeds the size limit")
            chunks.append(chunk)
    finally:
        connection.close()

    try:
        document = json.loads(b"".join(chunks).decode(ENCODING))
    except (UnicodeDecodeError, ValueError) as exc:
        raise RelayError(f"relay answered with something that is not JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise RelayError("relay answer is not a JSON object")
    if not document.get("ok"):
        raise RelayError(str(document.get("error") or "relay refused the request"))
    return document


def _broker_url(operation: str) -> str:
    """The broker URL for one named operation.

    Built from the base the room supplies plus a validated operation NAME. The agent never supplies
    a URL, and an operation name that is not one of the three below cannot be assembled here at all
    — so the "pick your own host" attack has nowhere to enter, quite apart from the broker refusing
    it on the other end.
    """
    base = os.environ.get(BROKER_URL_ENV, "").strip().rstrip("/")
    if not base:
        raise RelayError("no relay endpoint was provided")
    if operation not in {OP_WEB_SEARCH, OP_X_SEARCH, OP_FINAL_SUMMARY}:
        raise RelayError("relay refused the request")
    return f"{base}/v1/op/{operation}"


def _room_request(operation: str, payload: dict, *,
                  timeout: float = DEFAULT_TIMEOUT_SECONDS) -> dict:
    """One named operation, performed by the in-room broker with the miner's key.

    ``urllib`` is imported HERE rather than at module scope on purpose. The static screen rejects a
    submission that imports it, and this module is read by that screen too when it is copied into a
    bundle; keeping the import inside the one function that needs it means the sandbox path never
    loads it at all.
    """
    import urllib.error
    import urllib.request

    url = _broker_url(operation)
    capability = os.environ.get(BROKER_CAPABILITY_ENV, "").strip()
    if not capability:
        # The room mints one per job and puts it here. Missing means the room did not plumb it, and
        # every call would come back refused with no way for the agent to tell why.
        raise RelayError("no broker capability is available in this room")

    body = json.dumps(payload, separators=(",", ":")).encode(ENCODING)
    if len(body) > MAX_REQUEST_BYTES:
        raise RelayError("request exceeds the relay size limit")
    request = urllib.request.Request(
        url, data=body, method="POST",
        headers={"content-type": "application/json", CAPABILITY_HEADER: capability})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        # Deliberately NOT distinguishing 401 from 403 from 502 to the agent, for the same reason
        # the unix transport collapses its refusals: a caller that could tell them apart could map
        # the room's state one request at a time.
        raise RelayError(f"relay refused the request ({exc.code})") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RelayError(f"relay is unreachable: {exc}") from exc
    if len(raw) > MAX_RESPONSE_BYTES:
        raise RelayError("relay response exceeds the size limit")

    try:
        document = json.loads(raw.decode(ENCODING))
    except (UnicodeDecodeError, ValueError) as exc:
        raise RelayError(f"relay answered with something that is not JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise RelayError("relay answer is not a JSON object")
    return document


def search(query: str, *, limit: int = 10, capability: str | None = None,
           endpoint: str | None = None, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> list[dict]:
    """Search through the lane's gateway. Returns whatever the provider returned, minus secrets.

    Results are the PROVIDER's own shape -- a web search returns ``{link, title, snippet}``, an X
    search returns tweet fields -- because a relay that reformatted them would be deciding what a
    search returns, which is not its job. Every result is required to be an object with at least one
    field; an empty object is dropped, since an agent can do nothing with it.

    The capability defaults to the one the lane put in the environment, so the common call is
    ``sn22_relay.search("my query", limit=5)`` and an agent never handles a token at all.
    """
    token = capability if capability is not None else os.environ.get(CAPABILITY_ENV, "")
    document = _request({"op": "search", "capability": token, "query": query, "limit": int(limit)},
                        endpoint=endpoint, timeout=timeout)
    results = document.get("results")
    return [item for item in (results or []) if isinstance(item, dict) and item]


def quota(capability: str | None = None, *, endpoint: str | None = None,
          timeout: float = DEFAULT_TIMEOUT_SECONDS) -> dict:
    """How many calls this capability has left. Free, and it does not consume one.

    Exposed because an agent that cannot see its own quota either wastes it or hoards it, and both
    make the cost signal measure planning rather than search quality.
    """
    token = capability if capability is not None else os.environ.get(CAPABILITY_ENV, "")
    if endpoint is None and in_sealed_room():
        return _room_quota(timeout=timeout)
    document = _request({"op": "quota", "capability": token}, endpoint=endpoint, timeout=timeout)
    return {"used": int(document.get("used", 0)), "max_calls": int(document.get("max_calls", 0)),
            "remaining": int(document.get("remaining", 0)), "metered": True}


def _room_quota(*, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> dict:
    """The broker's per-operation quota for this capability.

    This used to return ``metered: False`` with three zeroes, because the old gateway imposed no
    limit — the miner's own budget was the only ceiling. That is no longer true and reporting it
    would be a lie an agent paces itself against: the broker enforces an equal per-operation quota
    for every contestant, so the numbers are real and both sides get the same ones.
    """
    import urllib.error
    import urllib.request

    base = os.environ.get(BROKER_URL_ENV, "").strip().rstrip("/")
    capability = os.environ.get(BROKER_CAPABILITY_ENV, "").strip()
    if not base or not capability:
        raise RelayError("no relay endpoint was provided")
    request = urllib.request.Request(
        f"{base}/v1/quota", method="GET", headers={CAPABILITY_HEADER: capability})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise RelayError(f"relay refused the request ({exc.code})") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RelayError(f"relay is unreachable: {exc}") from exc
    try:
        document = json.loads(raw.decode(ENCODING))
    except (UnicodeDecodeError, ValueError) as exc:
        raise RelayError(f"relay answered with something that is not JSON: {exc}") from exc
    operations = document.get("operations") if isinstance(document, dict) else None
    operations = operations if isinstance(operations, dict) else {}
    search_quota = operations.get(OP_WEB_SEARCH) or {}
    return {
        # The flat keys keep one shape across both transports, so an agent written against the
        # sandbox does not have to branch. `operations` carries the detail the room can offer.
        "used": int(search_quota.get("used", 0)),
        "max_calls": int(search_quota.get("max_calls", 0)),
        "remaining": int(search_quota.get("remaining", 0)),
        "metered": True,
        "operations": operations,
    }


def x_search(query: str, *, count: int = 10, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> list:
    """Search X/Twitter through the broker. Room only — the sandbox has no X provider.

    The actor is pinned by the broker, not chosen here: a different actor returns a different tweet
    shape, and the evaluator's field-by-field comparison would start failing honest contestants.
    """
    if not in_sealed_room():
        raise RelayError("x_search is only available in the sealed room")
    document = _room_request(OP_X_SEARCH, {"query": query, "count": int(count)}, timeout=timeout)
    results = document.get("results")
    return [item for item in (results or []) if isinstance(item, dict) and item]


def final_summary(messages: list, *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> str:
    """Write the agent's final summary through the broker, on the fixed model.

    The messages are the agent's -- it is writing its own answer. The model is not: every contestant
    summarises with the same one, or the comparison measures budget rather than skill.
    """
    if not in_sealed_room():
        raise RelayError("final_summary is only available in the sealed room")
    document = _room_request(OP_FINAL_SUMMARY, {"messages": list(messages or [])}, timeout=timeout)
    content = document.get("content")
    if not isinstance(content, str):
        raise RelayError("relay answered without a summary")
    return content


__all__ = ["BROKER_CAPABILITY_ENV", "BROKER_URL_ENV", "CAPABILITY_ENV", "ENDPOINT_ENV",
           "ENDPOINT_SCHEME", "MAX_REQUEST_BYTES", "MAX_RESPONSE_BYTES", "RelayError",
           "endpoint_path", "final_summary", "quota", "search", "x_search"]
