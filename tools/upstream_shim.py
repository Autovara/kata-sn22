"""Import the pinned upstream for parity, with its infrastructure stubbed out (SN22-5).

The upstream validator is a running daemon: importing `neurons.validators.penalty.count_penalty`
pulls in `bittensor`, `wandb`, `openai`, `aiohttp`, `apify_client` and a chain client, because a
penalty module sits in a package whose siblings talk to all of them. None of that is needed to
compute a penalty — it is needed to *be* a validator.

So this installs a stub for each infrastructure dependency before importing, and imports the real
upstream files for everything that actually scores. The rule it follows:

* **Stub the transport, never the arithmetic.** `bittensor`, `wandb`, `apify_client` and the HTTP
  clients are replaced. Every module whose numbers we compare — the penalties, the performance
  curve, the weight tables, `response_checks`, `format_text_for_match` — is the upstream file,
  executed. `pydantic` and `numpy` are the *real* packages, so the protocol models validate exactly
  as they do in production.
* **This never runs in the lane.** It mutates ``sys.modules``, so it lives in `tools/` and is
  imported only by the parity recorder and the executed-parity test. The resident runtime imports
  :mod:`kata_sn22.upstream_adapter`, which has no third-party dependency at all.

`tiktoken` is stubbed rather than installed, and that is safe for a specific reason: the only thing
that uses it is `streaming_penalty`, which counts tokens per streamed chunk. Kata's protocol returns
one JSON document rather than a stream, so that penalty is out of the adapted set and is never
executed here — the stub exists purely so `advanced_scraper_validator` can be imported for the
weight tables that sit beside it. If it were ever executed, its tokenizer would return a `_Stub`
and the comparison would fail loudly rather than agree on a wrong number.
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

#: Third-party packages the parity harness genuinely needs, and cannot stub, because the upstream's
#: own validation semantics depend on them.
REQUIRED_REAL_PACKAGES = ("pydantic", "pytz", "numpy")


class ShimUnavailable(Exception):
    """A real package the parity harness cannot stub is missing. Skip, never fake."""


def _missing_real_packages() -> list[str]:
    import importlib.util

    return [name for name in REQUIRED_REAL_PACKAGES if importlib.util.find_spec(name) is None]


class _Stub:
    """A permissive placeholder: every attribute and call returns another one.

    Only ever reached by transport code. If a scoring path ever touched one of these, the result
    would be a ``_Stub`` rather than a number, and the comparison would fail loudly instead of
    quietly agreeing.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __getattr__(self, name):
        return _Stub()

    def __call__(self, *args, **kwargs):
        return _Stub()


@dataclass
class Dendrite:
    """The two fields upstream scoring reads off a dendrite."""

    process_time: float | None = None
    status_code: int = 200


@dataclass
class Axon:
    hotkey: str = "sn22-parity-hotkey"


def _module(name: str, **attributes) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def install(upstream_root: str | Path) -> Path:
    """Install the stubs and put the pinned tree on ``sys.path``. Returns the root used."""
    missing = _missing_real_packages()
    if missing:
        raise ShimUnavailable(
            "parity execution needs the real " + ", ".join(missing)
            + "; install the 'parity' extra (uv sync --extra parity)")

    root = Path(upstream_root).expanduser().resolve()
    if not (root / "desearch").is_dir() or not (root / "neurons").is_dir():
        raise ShimUnavailable(f"{root} does not look like the pinned upstream tree")

    import pydantic

    class Synapse(pydantic.BaseModel):
        """Stands in for ``bittensor.Synapse``: a pydantic base plus the transport fields.

        Upstream's protocol models inherit from this and add every field that scoring reads, so the
        FIELDS UNDER TEST are still declared and validated by the real `desearch/protocol.py`.
        """

        model_config = {"arbitrary_types_allowed": True}

        name: str = "Synapse"
        timeout: float = 12.0
        axon: object | None = None
        dendrite: object | None = None

        def get_required_fields(self):
            return []

    class StreamingSynapse(Synapse):
        pass

    bittensor = _module("bittensor")
    bittensor.__path__ = []          # importable as a package, for `bittensor.core.metagraph`
    bittensor.Synapse = Synapse
    bittensor.StreamingSynapse = StreamingSynapse
    bittensor.logging = _Stub()
    bittensor.Wallet = _Stub
    bittensor.AsyncSubtensor = _Stub
    bittensor.axon = _Stub
    bittensor.dendrite = _Stub

    _module("bittensor.core")
    _module("bittensor.core.metagraph", AsyncMetagraph=object)
    _module("bittensor.utils")
    _module("bittensor.utils.weight_utils", process_weights=lambda **kwargs: (None, None))
    _module("wandb", init=lambda **kwargs: _Stub(), log=lambda *a, **k: None, config=_Stub())
    _module("aiohttp", ClientResponse=object, ClientSession=_Stub)
    _module("starlette")
    _module("starlette.responses", StreamingResponse=object)
    _module("openai", AsyncOpenAI=_Stub, OpenAI=_Stub)
    _module("apify_client", ApifyClientAsync=_Stub, ApifyClient=_Stub)
    _module("jsonpickle", encode=lambda obj, **k: "", decode=lambda text, **k: None)
    _module("faker", Faker=_Stub)
    _module("aiosqlite", connect=_Stub, Connection=_Stub, Row=object)
    _module("tiktoken", get_encoding=lambda name: _Stub())
    _module("redis")
    _module("redis.asyncio", Redis=_Stub)

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def available() -> bool:
    return not _missing_real_packages()


__all__ = ["Axon", "Dendrite", "ShimUnavailable", "available", "install"]
