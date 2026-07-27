"""The SN22 (Desearch) subnet package.

Two layers live here, and importing one must not drag in the other:

* **The evaluation protocol** (``protocol``, ``manifests``, ``scoring``, ``fake_provider``,
  ``fixtures``) — SN22-2. Self-contained, offline, and independent of the Kata core, so the contract
  can be reviewed and calibrated before any lane exists to run it.
* **The plugin** (``plugin``) — the adapter the core resolves by evaluator id. Still the
  pre-existing skeleton, written against a Kata core API (``kata.packages.registry``) that no
  longer exists;
  SN22-3 replaces it.

The plugin is therefore imported LAZILY. Doing it eagerly would make the whole package unimportable
on the current core, which is exactly what it did before: every test module in this repo failed to
collect because ``import kata_sn22`` reached for a module that is gone.
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "SN22_DESEARCH_PLUGIN",
    "Sn22DesearchPlugin",
    "Sn22Problems",
    "Sn22RawRun",
]


def __getattr__(name: str) -> Any:
    """Resolve plugin symbols on first use, so the protocol layer imports on any core."""
    if name in __all__:
        from kata_sn22 import plugin as _plugin

        if name == "SN22_DESEARCH_PLUGIN":
            singleton = _plugin.Sn22DesearchPlugin()
            globals()["SN22_DESEARCH_PLUGIN"] = singleton
            return singleton
        return getattr(_plugin, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
