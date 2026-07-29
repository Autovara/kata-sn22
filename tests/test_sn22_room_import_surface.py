"""What the sealed room is allowed to import, enforced rather than documented.

Phase A's open question was whether ``bittensor`` has to go into the attested room image. It does
not — but "does not" was only ever true because nobody had added it yet. This file makes it a
property CI checks.

Two things are pinned, and both fail in the same nasty way if they drift: the failure appears
**inside a sealed TEE room**, on a duel, as an ImportError with no debugger attached. That is the
worst place in this system to discover a dependency.

1. The runtime import closure of the plugin entry point is standard library plus ``kata`` (the
   engine ABI). Not "small" — closed. Adding ``pydantic`` to a scoring module would pass every
   other test in this repo and fail only in the room.
2. ``kata_sn22.parity`` is not on that closure. It reads the 2.4 MB vendored ``upstream/`` tree and
   imports through ``tools/upstream_shim``, which mutates ``sys.modules``. It is a development and
   evidence tool. If it were ever reachable from the entry point, the room image would have to ship
   the vendored tree and the shim's stubs would be one import away from live scoring.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "kata_sn22"

#: The only non-stdlib import the room may carry: the engine ABI it is loaded through.
ALLOWED_THIRD_PARTY = {"kata"}

#: Reached only by tests and ``tools/record_parity.py``. See the module docstring.
DEVELOPMENT_ONLY_MODULES = {"kata_sn22.parity"}

#: TRUSTED-RUNNER modules. These deliberately import ``pydantic``, ``numpy``, ``pytz`` and
#: ``tiktoken`` — the four packages upstream's own scoring semantics depend on — because since
#: Phase F production executes the real vendored SN22 validator rather than a port of it.
#:
#: They ARE reachable from the plugin entry point, and that is correct: the plugin runs on the
#: validator host and inside the trusted runner, both of which carry those packages. They are
#: exempt from the standard-library rule below. What must stay standard-library-only is the AGENT
#: image, which is a different artifact — see ``test_the_agent_image_ships_only_the_sdk...`` and
#: the README's "Why bittensor is not in the room" as amended.
TRUSTED_RUNNER_MODULES = {
    "kata_sn22.upstream_runtime",
    "kata_sn22.neuron_adapter",
    "kata_sn22.production_scorer",
}


def _imports(path: Path) -> set[str]:
    """Top-level package name of every import in a file, including function-local ones.

    Function-local imports count. A deferred ``import numpy`` inside the scoring path is still a
    dependency of the room -- it just fails later, on a duel, instead of at start-up.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), str(path))):
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def _local_imports(path: Path) -> set[str]:
    """The ``kata_sn22.*`` modules this file imports, for walking the closure."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), str(path))):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.startswith("kata_sn22"):
                found.add(node.module if node.module != "kata_sn22"
                          else "kata_sn22")
                # `from kata_sn22 import providers` names the submodule in `names`, not `module`.
                if node.module == "kata_sn22":
                    found |= {f"kata_sn22.{alias.name}" for alias in node.names}
        elif isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names if alias.name.startswith("kata_sn22")}
    return found


def _runtime_closure() -> set[str]:
    """Every ``kata_sn22`` module reachable from the plugin entry point."""
    reachable: set[str] = set()
    pending = ["kata_sn22"]
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        path = PACKAGE / ("__init__.py" if name == "kata_sn22" else f"{name.split('.')[-1]}.py")
        if not path.exists():
            continue
        reachable.add(name)
        pending.extend(_local_imports(path) - reachable)
    return reachable


def test_the_entry_point_resolves_without_any_third_party_package():
    """The realest version of this check: actually build the plugin."""
    import kata_sn22

    assert kata_sn22.SN22_DESEARCH_PLUGIN is not None


def test_the_room_never_imports_the_parity_harness():
    """It reads the vendored tree and imports through a sys.modules-mutating shim."""
    leaked = _runtime_closure() & DEVELOPMENT_ONLY_MODULES
    assert not leaked, (
        f"{sorted(leaked)} is reachable from the plugin entry point. It reads the vendored "
        f"upstream tree, so the room image would have to ship it -- and the shim's stubs would be "
        f"one import away from live scoring.")


#: What the AGENT image actually contains. This -- not the plugin closure -- is the surface that
#: must be standard library only, because that image ships no package installer.
AGENT_IMAGE_COPIES = ("kata_sn22_sdk", "sn22_relay.py")


def test_the_agent_image_ships_only_the_sdk_and_the_relay_client():
    """The load-bearing check, and it replaced a weaker one.

    This used to assert that ``kata_sn22``'s plugin closure was standard-library-only. That held
    while SN22 scored with a dependency-free PORT of the upstream reward code -- and it stopped
    holding the moment production began executing the real vendored upstream, which needs pydantic,
    numpy, pytz and tiktoken.

    The old guard was aimed at the wrong artifact. ``kata_sn22.plugin`` never enters the agent
    image: it runs on the validator host and inside the trusted runner, both of which carry those
    four packages deliberately. What the agent image contains is the two things below, and THAT is
    what has no installer to fix a missing dependency with.
    """
    dockerfile = (PACKAGE.parent / "deploy" / "sn22-agent" / "Dockerfile").read_text(
        encoding="utf-8")
    copied = {line.split()[1] for line in dockerfile.splitlines()
              if line.startswith("COPY ")}
    assert copied == set(AGENT_IMAGE_COPIES), (
        f"the agent image now copies {sorted(copied)}. Anything beyond the SDK and the relay "
        f"client has to be standard library only, and has no installer to fix it with.")
    assert "kata_sn22 " not in dockerfile, (
        "the agent image copies the whole lane package, which imports the real upstream scorer")


def test_the_relay_client_the_agent_image_ships_imports_only_the_standard_library():
    """``kata_sn22_sdk`` is checked in ``tests/test_sn22_sdk.py``; this covers the other file."""
    import sys

    path = PACKAGE / "relay_client.py"
    third_party = {
        name for name in _imports(path)
        if name not in sys.stdlib_module_names and name != "kata_sn22"
    }
    assert not third_party, f"the relay client imports {sorted(third_party)}"


def test_the_trusted_runner_modules_still_exist():
    """A guard on the guard: renaming one would make the check above pass vacuously."""
    for module in TRUSTED_RUNNER_MODULES:
        assert (PACKAGE / f"{module.split('.')[-1]}.py").exists(), module


@pytest.mark.parametrize("module", sorted(_runtime_closure() - TRUSTED_RUNNER_MODULES))
def test_every_runtime_module_imports_only_the_standard_library(module):
    """bittensor, pydantic, numpy, aiohttp, apify_client, openai: none of them belong in the room.

    This is the Phase A decision, enforced. The four provider integrations in ``providers.py`` use
    stdlib ``urllib`` for exactly this reason -- an SDK would drag a dependency tree into an
    attested image whose measurement covers every byte of it.
    """
    path = PACKAGE / ("__init__.py" if module == "kata_sn22" else f"{module.split('.')[-1]}.py")
    third_party = {
        name for name in _imports(path)
        if name not in sys.stdlib_module_names
        and name != "kata_sn22"
        and name not in ALLOWED_THIRD_PARTY
    }
    assert not third_party, (
        f"{module} imports {sorted(third_party)}, which is not in the standard library. A sealed "
        f"room's attested measurement covers its whole image; adding a package here changes the "
        f"measurement and can only fail on a live duel.")


def test_the_development_only_modules_still_exist():
    """A guard on the guard: if `parity` were renamed, the closure check above would pass
    vacuously and stop protecting anything."""
    for module in DEVELOPMENT_ONLY_MODULES:
        assert (PACKAGE / f"{module.split('.')[-1]}.py").exists(), module
