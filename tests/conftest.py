"""Test-suite defaults for the SN22 plugin.

The production execution backend is the attested sealed room (`KATA_SN22_EXECUTION_BACKEND`
defaults to ``tee``), and `run_candidate` REFUSES to run when it is asked for a room and given
none — a lane that declared a TEE and quietly executed locally would be the worst possible
outcome, so that refusal is deliberate.

The test suite is the development and §5.5 calibration path, which is precisely the case the
``sandbox`` backend exists for. Selecting it here, once, keeps every test honest about which mode it
is exercising: a test that wants the TEE behaviour sets the variable itself and gets the refusal.

Set at import time rather than in a fixture because `environment_spec()` is read during plugin
construction in some tests, which happens before any fixture runs.
"""
from __future__ import annotations

import os

import pytest

from kata_sn22.execution import policy as execution_policy

os.environ.setdefault(execution_policy.EXECUTION_BACKEND_ENV, "sandbox")


@pytest.fixture(autouse=True)
def _development_backend(monkeypatch):
    """Re-assert the development backend per test, so one test's override cannot leak into the
    next. Tests that want the production default clear or override it themselves."""
    monkeypatch.setenv(execution_policy.EXECUTION_BACKEND_ENV, "sandbox")
