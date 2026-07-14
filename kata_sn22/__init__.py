"""The SN22 (Desearch) subnet plugin -- skeleton.

Importing this package registers the SN22 plugin. It is the second tenant of the
multi-subnet platform, proving a subnet is added as a plugin with no core edits.
"""

from __future__ import annotations

from kata.packages.registry import register_plugin

from .plugin import Sn22DesearchPlugin, Sn22Problems, Sn22RawRun

#: The singleton SN22 plugin instance the core resolves by evaluator id.
SN22_DESEARCH_PLUGIN = Sn22DesearchPlugin()

register_plugin(SN22_DESEARCH_PLUGIN)

__all__ = [
    "SN22_DESEARCH_PLUGIN",
    "Sn22DesearchPlugin",
    "Sn22Problems",
    "Sn22RawRun",
]
