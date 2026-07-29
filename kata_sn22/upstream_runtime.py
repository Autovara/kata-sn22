"""Load the pinned upstream validator for production, with infrastructure adapted and nothing else.

This is the module that makes the difference between "Kata scores like SN22" and "Kata scores
SN22's way". Everything that decides a number — the reward models, the penalties, the performance
curve, the aggregation — is the **real vendored file**, executed. What is replaced is the machinery
a validator needs in order to *be a validator*: a chain client, a wallet, a metrics sink, an HTTP
framework.

**Why anything has to be replaced at all.** ``neurons.validators.penalty.count_penalty`` sits in a
package whose siblings import ``bittensor``, ``wandb``, ``aiohttp``, ``openai`` and
``apify_client``. None of that is needed to compute a penalty. Importing it would mean shipping a
chain client into an attested room, where every byte of the image is covered by the measurement, in
order to run arithmetic that never touches the chain.

**The rule, and it is the whole design.** *Adapt the transport, never the arithmetic.* Every module
named in :data:`ADAPTED_MODULES` is infrastructure. Every module that produces a score is imported
from the pinned tree and run. :func:`assert_scoring_is_real` checks that after loading, so a future
edit that adapts one module too many fails a test rather than quietly returning a stub where a
number should be.

**What this supersedes.** ``tools/upstream_shim.py`` did the same thing for the parity harness and
was explicitly development-only. Phase F needs the real upstream on the *production* path, so the
capability moves here — reviewed, checked, and recorded in the attested report. The shim stays where
it is: it serves the parity evidence, which must be able to diverge from production in order to be
evidence about it.

**This is a deliberate reversal of a Phase A decision.**
the README's "Why bittensor is not in the room" recorded that the room's plugin closure was
standard-library-only. That is still true of the AGENT image and of everything the agent can reach.
It is no longer true of the trusted runner, which now carries ``pydantic``, ``numpy``, ``pytz`` and
``tiktoken`` — the four packages upstream's own scoring semantics depend on — plus the vendored
tree. ``bittensor`` is still absent, and so is every other transport.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

from kata_sn22.upstream_snapshot import UPSTREAM_COMMIT, snapshot_root

#: Packages that CANNOT be adapted, because upstream's own validation semantics are theirs.
#:
#: * ``pydantic`` validates the protocol models the scorer reads;
#: * ``numpy`` IS the arithmetic;
#: * ``pytz`` is how the date-range penalty decides what "within range" means;
#: * ``tiktoken`` counts tokens per streamed chunk, and the streaming penalty is 0.01 per token
#:   over two per chunk. A different tokenizer is a different penalty, so this one is not a
#:   transport however much it looks like one. It was classified as infrastructure when this
#:   module was first written, and the raising adapter below caught it on the very first import --
#:   which is the whole argument for refusing rather than returning something plausible.
REQUIRED_REAL_PACKAGES = ("pydantic", "numpy", "pytz", "tiktoken")

#: Infrastructure, and only infrastructure. Adapted so the scoring code can be imported and run
#: without a chain, a wallet, a metrics sink or an HTTP server. Every entry is a thing a validator
#: needs to BE a validator, never a thing that decides a score.
ADAPTED_MODULES = (
    "bittensor",             # chain client, wallet, axon/dendrite transport
    "bittensor.core",
    "bittensor.core.metagraph",
    "bittensor.utils",
    "bittensor.utils.weight_utils",
    "wandb",                 # metrics sink
    "aiohttp",               # HTTP client used by the organic-query path
    "starlette",             # the public validator API
    "starlette.responses",
    "openai",                # provider SDKs -- Kata reaches providers through the broker instead
    "apify_client",
    "jsonpickle",            # logging serialisation
    "faker",                 # question generation, which Kata does from packaged rows
    "aiosqlite",             # the validator's own miner database
    "redis",
    "redis.asyncio",
)

#: Modules that must be the REAL vendored file after loading. Every one of them decides a number.
#: Checked by :func:`assert_scoring_is_real` rather than assumed.
SCORING_MODULES = (
    "desearch.protocol",
    "desearch.utils",
    "neurons.validators.reward.reward",
    "neurons.validators.reward.content_relevance",
    "neurons.validators.reward.summary_relevance",
    "neurons.validators.reward.performance_reward",
    "neurons.validators.penalty.penalty",
    "neurons.validators.penalty.streaming_penalty",
    "neurons.validators.penalty.count_penalty",
    "neurons.validators.penalty.timeout_penalty",
    "neurons.validators.penalty.min_realistic_time_penalty",
    "neurons.validators.penalty.summary_structure_penalty",
    "neurons.validators.penalty.duplicate_results_penalty",
    "neurons.validators.penalty.result_schema_penalty",
    "neurons.validators.penalty.date_range_penalty",
    "neurons.validators.penalty.domain_filter_penalty",
    "neurons.validators.penalty.sort_order_penalty",
    "neurons.validators.scrapers.base_scraper_validator",
    "neurons.validators.scrapers.advanced_scraper_validator",
    "neurons.validators.scrapers.x_scraper_validator",
    "neurons.validators.scoring.query_scheduler",
    "neurons.validators.scoring.constants",
)


class UpstreamUnavailable(Exception):
    """The pinned upstream cannot be executed here. Never silently degraded into a port."""


@dataclass(frozen=True)
class RuntimeProvenance:
    """What the attested report records about how the score was produced."""

    upstream_commit: str
    upstream_root: str
    adapted_modules: tuple
    real_packages: dict

    def as_dict(self) -> dict:
        return {
            "upstream_commit": self.upstream_commit,
            "upstream_root": self.upstream_root,
            "adapted_modules": list(self.adapted_modules),
            "real_packages": dict(sorted(self.real_packages.items())),
        }


class _Adapter:
    """A permissive placeholder for one piece of infrastructure.

    Every attribute and call returns another one. Only ever reached by transport code: if a scoring
    path touched one, the result would be an ``_Adapter`` rather than a number, and the arithmetic
    would fail loudly instead of quietly agreeing on something wrong.
    """

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def __getattr__(self, _name):
        return _Adapter()

    def __call__(self, *_args, **_kwargs):
        return _Adapter()

    def __repr__(self) -> str:
        return "<kata infrastructure adapter>"


@dataclass
class Dendrite:
    """The two fields upstream's scoring reads off a dendrite.

    Real values, supplied by Kata: ``process_time`` is what the room measured, and the performance
    reward and the timeout penalty are both computed from it. This is not a stub -- it is how a
    measurement Kata actually took reaches upstream's own formula.
    """

    process_time: float | None = None
    status_code: int = 200


@dataclass
class Axon:
    """Identifies a contestant to upstream's logging. Carries no key and no network address."""

    hotkey: str = "kata-contestant"


def _module(name: str, **attributes) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def missing_real_packages() -> list:
    import importlib.util

    return [name for name in REQUIRED_REAL_PACKAGES if importlib.util.find_spec(name) is None]


def available() -> bool:
    return not missing_real_packages()


_loaded: RuntimeProvenance | None = None


def load(*, root: str | Path | None = None) -> RuntimeProvenance:
    """Install the infrastructure adapters and put the pinned tree on the path. Idempotent.

    Raises rather than degrading. A round that cannot execute the real upstream must stop:
    scoring with the port instead would produce numbers that look right and are not upstream's,
    and nothing downstream could tell the difference.
    """
    global _loaded
    if _loaded is not None:
        return _loaded

    missing = missing_real_packages()
    if missing:
        raise UpstreamUnavailable(
            f"the pinned upstream needs the real {', '.join(missing)}; these carry upstream's own "
            f"validation semantics and cannot be adapted. Install the 'upstream' extra")

    upstream = Path(root).expanduser().resolve() if root is not None else snapshot_root()
    if not (upstream / "desearch").is_dir() or not (upstream / "neurons").is_dir():
        raise UpstreamUnavailable(f"{upstream} does not look like the pinned upstream tree")

    _install_adapters()

    if str(upstream) not in sys.path:
        # Appended, not inserted: this repository's own modules must keep winning. Inserting would
        # let a vendored module named like one of ours shadow it.
        sys.path.append(str(upstream))

    _loaded = RuntimeProvenance(
        upstream_commit=UPSTREAM_COMMIT,
        upstream_root=str(upstream),
        adapted_modules=ADAPTED_MODULES,
        real_packages=_real_package_versions(),
    )
    return _loaded


def _real_package_versions() -> dict:
    import importlib.metadata

    versions = {}
    for name in REQUIRED_REAL_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover - checked above
            versions[name] = "unknown"
    return versions


def _install_adapters() -> None:
    """Replace each piece of infrastructure with something that cannot decide a score."""
    import pydantic

    class Synapse(pydantic.BaseModel):
        """Stands in for ``bittensor.Synapse``: a pydantic base plus the transport fields.

        Upstream's protocol models inherit from this and declare every field the scorer reads, so
        the fields under test are still declared and validated by the REAL ``desearch/protocol.py``.
        Only the chain transport is ours.
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
    bittensor.logging = _Adapter()
    bittensor.Wallet = _Adapter
    bittensor.AsyncSubtensor = _Adapter
    bittensor.axon = _Adapter
    bittensor.dendrite = _Adapter

    _module("bittensor.core")
    _module("bittensor.core.metagraph", AsyncMetagraph=object)
    _module("bittensor.utils")
    # Chain weight submission. Kata never writes weights: it decides one promotion, on one host,
    # and a validator that could set weights from inside a duel would be a validator with a stake
    # in its outcome.
    _module("bittensor.utils.weight_utils", process_weights=_refuse("chain weight submission"))
    _module("wandb", init=_refuse("W&B"), log=_refuse("W&B"), config=_Adapter())
    _module("aiohttp", ClientResponse=object, ClientSession=_Adapter)
    _module("starlette")
    _module("starlette.responses", StreamingResponse=object)
    _module("openai", AsyncOpenAI=_Adapter, OpenAI=_Adapter)
    _module("apify_client", ApifyClientAsync=_Adapter, ApifyClient=_Adapter)
    _module("jsonpickle", encode=lambda obj, **k: "", decode=lambda text, **k: None)
    _module("faker", Faker=_Adapter)
    _module("aiosqlite", connect=_Adapter, Connection=_Adapter, Row=object)
    _module("redis")
    _module("redis.asyncio", Redis=_Adapter)


def _refuse(what: str):
    """A disabled capability that RAISES rather than returning something plausible.

    The difference matters. A stub that returns ``None`` lets a code path Kata must never take run
    to completion and produce a number nobody notices is wrong. Raising means the first time
    anything reaches for a chain write, a metrics sink or a token encoder, the round stops and says
    so.
    """
    def _refused(*_args, **_kwargs):
        raise UpstreamUnavailable(
            f"{what} is disabled in Kata's runtime and must not be reached from the scoring path")

    return _refused


def assert_scoring_is_real() -> None:
    """Every module that decides a number is the real vendored file. Raise if one is not.

    The guard on the guard. ``ADAPTED_MODULES`` growing by one entry is a one-line change that would
    replace a penalty with something that always returns 1.0, and every score would still look
    plausible.
    """
    upstream = str(snapshot_root())
    for name in SCORING_MODULES:
        module = sys.modules.get(name)
        if module is None:
            continue          # not imported yet by this caller; nothing to check
        origin = getattr(module, "__file__", None)
        if not origin or not str(Path(origin).resolve()).startswith(upstream):
            raise UpstreamUnavailable(
                f"{name} decides a score but is not the pinned upstream file (loaded from "
                f"{origin!r}); the adapted set must never include a scoring module")


def upstream_module(name: str):
    """Import one pinned upstream module, loading the runtime first if needed."""
    import importlib

    load()
    module = importlib.import_module(name)
    assert_scoring_is_real()
    return module
