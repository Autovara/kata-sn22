"""The trusted side of the relay socket (SN22-4/§6.1, completed in SN22-7).

:mod:`kata_sn22.gateway` decides what a capability may do and bills it. This is the only thing that
lets an untrusted process ask — a unix-socket server, one per challenge, that accepts a request,
hands it to the gateway, and writes back a bounded answer.

The trust boundary runs exactly here, so the rules are narrow on purpose:

* **Two operations, both read-only to the candidate.** ``search`` and ``quota``. There is no
  operation that mints, extends or transfers a capability — those are the lane's, taken before the
  candidate starts.
* **Every refusal says the same thing.** A caller that could tell "unknown capability" from "quota
  exhausted" from "expired" could map the lane's state one probe at a time. It cannot, so a refusal
  is a refusal.
* **Bounded before parsed.** A request is size-capped on BYTES before ``json.loads`` sees it,
  because decoding a 500 MB document is a denial of service while it is still being decoded.
* **One thread per connection, and the challenge closes.** When the round ends the socket is
  removed and the gateway is closed, so a slow agent cannot keep spending after it was scored.

The gateway is guarded by a lock rather than being made thread-safe internally: billing is a
read-modify-write over shared counters, and one lock around the whole operation is the version of
that which is obviously correct.
"""
from __future__ import annotations

import json
import os
import socketserver
import stat
import threading
from pathlib import Path

from kata_sn22.gateway import GatewayDenied, Sn22Gateway, redact
from kata_sn22.relay_client import ENCODING, ENDPOINT_SCHEME, MAX_REQUEST_BYTES

#: The single refusal message. See the module docstring: a specific one is a probe oracle.
DENIED = "relay refused the request"

SOCKET_NAME = ".sn22-relay.sock"


def _error(message: str) -> bytes:
    return json.dumps({"ok": False, "error": redact(message)}, separators=(",", ":")).encode(
        ENCODING) + b"\n"


def _ok(payload: dict) -> bytes:
    return json.dumps({"ok": True, **payload}, separators=(",", ":")).encode(ENCODING) + b"\n"


class _Handler(socketserver.BaseRequestHandler):
    """One request, one answer, one connection. No session, no state between calls."""

    def handle(self) -> None:
        server: "RelayServer" = self.server           # type: ignore[assignment]
        self.request.settimeout(server.request_timeout)
        try:
            payload = self._read_request()
        except (OSError, ValueError):
            return   # a peer that cannot frame a request is answered with nothing
        try:
            answer = server.dispatch(payload)
        except GatewayDenied:
            answer = _error(DENIED)
        except Exception:                             # noqa: BLE001 - never leak a traceback outward
            answer = _error(DENIED)
        try:
            self.request.sendall(answer)
        except OSError:
            pass

    def _read_request(self) -> dict:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = self.request.recv(8192)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_REQUEST_BYTES:
                raise ValueError("request too large")
            chunks.append(chunk)
            if b"\n" in chunk:
                break
        document = json.loads(b"".join(chunks).split(b"\n", 1)[0].decode(ENCODING))
        if not isinstance(document, dict):
            raise ValueError("request is not an object")
        return document


class _ThreadedUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


class RelayServer:
    """Serves one challenge's gateway over a unix socket. Start it, run the agents, close it."""

    def __init__(self, gateway: Sn22Gateway, socket_dir: str | os.PathLike, *,
                 request_timeout: float = 30.0):
        self.gateway = gateway
        self.request_timeout = request_timeout
        self._lock = threading.Lock()
        self.socket_path = Path(socket_dir).expanduser().resolve() / SOCKET_NAME
        self._server: _ThreadedUnixServer | None = None
        self._thread: threading.Thread | None = None

    # ---- lifecycle -------------------------------------------------------------------------------
    def start(self) -> "RelayServer":
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        # A stale socket from a killed previous run would make bind() fail. Removing it is safe
        # because the path is inside the round's own disposable work directory.
        if self.socket_path.exists() or self.socket_path.is_symlink():
            self.socket_path.unlink()
        server = _ThreadedUnixServer(str(self.socket_path), _Handler)
        server.dispatch = self.dispatch          # type: ignore[attr-defined]
        server.request_timeout = self.request_timeout   # type: ignore[attr-defined]
        # The candidate runs as uid 65534 and owns nothing; without this it cannot connect at all.
        # Nothing else can reach the path: it is inside the sandbox's private work directory.
        os.chmod(self.socket_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP
                 | stat.S_IROTH | stat.S_IWOTH)
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, daemon=True,
                                        name="sn22-relay")
        self._thread.start()
        return self

    def close(self) -> None:
        """End the challenge. The gateway closes FIRST, so a request already in flight is refused
        rather than served after scoring has begun."""
        self.gateway.close()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        try:
            self.socket_path.unlink()
        except OSError:
            pass

    def __enter__(self) -> "RelayServer":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.close()

    @property
    def endpoint(self) -> str:
        """What the agent is told. A path with a scheme, not a URL."""
        return f"{ENDPOINT_SCHEME}{self.socket_path}"

    # ---- dispatch --------------------------------------------------------------------------------
    def dispatch(self, payload: dict) -> bytes:
        operation = payload.get("op")
        capability = payload.get("capability")
        if not isinstance(capability, str):
            return _error(DENIED)
        with self._lock:                              # billing is a read-modify-write; serialize it
            if operation == "search":
                query = payload.get("query")
                limit = payload.get("limit", 10)
                if not isinstance(query, str) or isinstance(limit, bool) or not isinstance(
                        limit, int):
                    return _error(DENIED)
                results = self.gateway.search(capability, query, limit=limit)
                return _ok({"results": results})
            if operation == "quota":
                used, max_calls = self.gateway.quota(capability)
                return _ok({"used": used, "max_calls": max_calls,
                            "remaining": max(0, max_calls - used)})
        return _error(DENIED)


def install_client(workdir: str | os.PathLike) -> Path:
    """Copy the relay client into the run directory as ``sn22_relay.py``.

    Copied rather than bind-mounted from the installed package: the sandbox's read-only mounts are
    system directories, and binding a path out of the lane's own virtualenv would hand a candidate a
    view of the lane's code. A copy of one reviewed file is the whole surface.
    """
    from kata_sn22 import relay_client

    source = Path(relay_client.__file__).resolve()
    target = Path(workdir).expanduser().resolve() / "sn22_relay.py"
    target.write_bytes(source.read_bytes())
    target.chmod(0o644)
    return target


__all__ = ["DENIED", "SOCKET_NAME", "RelayServer", "install_client"]
