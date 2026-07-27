"""The SN22 (Desearch) subnet package.

Two layers live here, and importing one must not drag in the other:

* **The evaluation protocol** (``protocol``, ``manifests``, ``scoring``, ``fake_provider``,
  ``fixtures``) — SN22-2. Self-contained, offline, and independent of the Kata core, so the contract
  can be reviewed and calibrated before any lane exists to run it.
* **The plugin** (``plugin``) — the adapter the core resolves by evaluator id, rewritten in SN22-3
  against the current ``kata.plugins`` ABI on top of that protocol.

The plugin is still imported LAZILY, so the protocol layer stays importable on a host that has no
Kata core installed at all — which is what makes it reviewable and calibratable on its own.
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
