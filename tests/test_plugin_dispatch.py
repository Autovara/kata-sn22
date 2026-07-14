"""Tests: the platform discovers and resolves this subnet's plugin via its entry point.

Installing kata-sn22 registers ``sn22_desearch`` in the ``kata.subnets`` group; these check the
platform's ``load_builtin_plugins`` / ``plugin_for_evaluator`` find it with no core edits.
"""

from __future__ import annotations

import pytest
from kata.packages.dispatch import load_builtin_plugins, plugin_for_evaluator
from kata.packages.registry import clear_registry, get_plugin_or_none


@pytest.fixture(autouse=True)
def _keep_builtins_registered():
    yield
    load_builtin_plugins()  # leave the registry in its normal (discovered) state


def test_plugin_for_evaluator_resolves_registered() -> None:
    plugin = plugin_for_evaluator("sn22_desearch")
    assert plugin is not None
    assert plugin.evaluator_id == "sn22_desearch"


def test_plugin_for_evaluator_unknown_or_blank() -> None:
    assert plugin_for_evaluator("does-not-exist") is None
    assert plugin_for_evaluator(None) is None
    assert plugin_for_evaluator("") is None


def test_load_builtin_plugins_repairs_cleared_registry() -> None:
    load_builtin_plugins()
    assert get_plugin_or_none("sn22_desearch") is not None
    clear_registry()
    assert get_plugin_or_none("sn22_desearch") is None
    load_builtin_plugins()  # defensively re-registers even after the module import cache
    assert get_plugin_or_none("sn22_desearch") is not None
