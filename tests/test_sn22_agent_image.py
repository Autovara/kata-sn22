"""The agent image: what it must contain, and what it must not.

Two layers, because they fail at different times.

**Static.** The Dockerfile and build script are read and asserted against. These always run, they
are fast, and they catch the edit that removes a guarantee months before anyone rebuilds.

**Real.** If ``KATA_SN22_AGENT_IMAGE`` names a built image, the same properties are checked against
the actual thing — non-root, no installer, no egress, and the reference agent completing all four
pools inside it. Gated rather than always-on because building needs a registry pull, and a test
suite that cannot run offline is a test suite people learn to skip.

    cd deploy/sn22-agent
    PYTHON_BASE=python@sha256:<digest> IMAGE=kata-sn22-agent:local ./build.sh local
    KATA_SN22_AGENT_IMAGE=kata-sn22-agent:local uv run pytest tests/test_sn22_agent_image.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

AGENT_DIR = Path(__file__).resolve().parents[1] / "deploy" / "sn22-agent"
DOCKERFILE = (AGENT_DIR / "Dockerfile").read_text(encoding="utf-8")
BUILD_SH = (AGENT_DIR / "build.sh").read_text(encoding="utf-8")
# The reigning King's bundle. It lives under kings/, not submissions/ -- submissions/ holds
# miners' entries only.
REFERENCE = (Path(__file__).resolve().parents[2] / "kata" / "kings" / "sn22__desearch" / "miner")

IMAGE = os.environ.get("KATA_SN22_AGENT_IMAGE", "").strip()

needs_image = pytest.mark.skipif(
    not IMAGE or not shutil.which("docker"),
    reason="set KATA_SN22_AGENT_IMAGE to a built image (and have docker) to run the real checks")


# ---- static: the image is built to contain nothing worth reaching -----------------------------

def test_the_base_image_is_pinned_by_digest():
    """A tag is a pointer somebody else can move, and this image executes code from a stranger."""
    assert "ARG PYTHON_BASE=python@sha256:" in DOCKERFILE
    assert "PYTHON_BASE must be an immutable image digest" in BUILD_SH


@pytest.mark.parametrize("tool", ["pip", "pip3", "apt-get", "dpkg", "curl", "wget", "git", "ssh"])
def test_the_dockerfile_removes_every_installer_and_network_tool(tool: str):
    """An untrusted agent with a package manager is one network path away from running code that
    was never reviewed -- and the attested measurement would still be the approved one."""
    assert tool in DOCKERFILE


def test_the_build_fails_if_the_toolchain_survives():
    """The deletions are only worth anything if their failure is loud. A base image that moved
    site-packages would otherwise produce an image that looks stripped and is not."""
    for guard in ("! command -v pip", "! command -v apt-get", "! command -v curl",
                  "! python -c 'import pip'", "! python -c 'import ensurepip'"):
        assert guard in DOCKERFILE, guard


def test_the_image_runs_as_a_non_root_user():
    assert "USER 10001" in DOCKERFILE
    assert DOCKERFILE.index("RUN set -eux") < DOCKERFILE.index("USER 10001"), \
        "the stripping must happen while still root, before the image drops privileges"


def test_the_entry_point_is_the_harness_not_the_submission():
    """A submission is IMPORTED by reviewed code rather than executed as a program, so every
    contestant's answer is framed by the same function."""
    assert 'ENTRYPOINT ["python", "-m", "kata_sn22_sdk.harness"]' in DOCKERFILE
    assert 'CMD ["/bundle/agent.py"]' in DOCKERFILE


def test_the_image_carries_the_sdk_and_the_relay_client():
    assert "COPY kata_sn22_sdk /opt/kata/kata_sn22_sdk" in DOCKERFILE
    assert "COPY sn22_relay.py /opt/kata/sn22_relay.py" in DOCKERFILE
    assert "PYTHONPATH=/opt/kata" in DOCKERFILE


def test_the_sdk_is_copied_from_this_repository_at_build_time():
    """Vendored beside the Dockerfile it would drift, and a submission calibrated in the sandbox
    would behave differently in a duel with nothing to say why."""
    assert 'SDK="${REPO}/kata_sn22_sdk"' in BUILD_SH
    assert 'CLIENT="${REPO}/kata_sn22/relay_client.py"' in BUILD_SH
    assert 'cp -R "$SDK"' in BUILD_SH


def test_the_build_stages_are_cleaned_up_even_on_failure():
    assert "trap 'rm -rf" in BUILD_SH


def test_the_build_produces_an_sbom_and_a_vulnerability_report():
    """Asked after an incident, when rebuilding the image to find out is exactly the thing you
    cannot do."""
    assert "syft" in BUILD_SH and "spdx-json" in BUILD_SH
    assert "grype" in BUILD_SH


def test_a_missing_scanner_is_reported_rather_than_silently_skipped():
    """A missing scanner must not become a missing report nobody noticed."""
    assert "WARNING: syft not installed" in BUILD_SH
    assert "WARNING: grype not installed" in BUILD_SH
    assert '"sbom": "${SBOM_STATUS}"' in BUILD_SH


def test_the_staged_build_context_is_never_committed():
    ignored = (AGENT_DIR / ".gitignore").read_text(encoding="utf-8")
    for name in ("sn22_relay.py", "kata_sn22_sdk/", "artifacts/"):
        assert name in ignored, name


# ---- real: the same properties, against a built image -------------------------------------------

def _run(*args: str, image_args=(), stdin: str | None = None, timeout: int = 120):
    argv = ["docker", "run", "--rm", *(["-i"] if stdin is not None else []), *args, IMAGE,
            *image_args]
    return subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=timeout,
                          check=False)


@needs_image
def test_the_built_image_runs_as_uid_10001():
    result = _run("--entrypoint", "python", image_args=("-c", "import os; print(os.getuid())"))
    assert result.stdout.strip() == "10001", result.stderr


@needs_image
@pytest.mark.parametrize("tool", ["pip", "pip3", "apt-get", "apt", "dpkg", "curl", "wget", "git"])
def test_the_built_image_has_no_installer_or_network_tool(tool: str):
    result = _run("--entrypoint", "sh", image_args=("-c", f"command -v {tool}"))
    assert result.returncode != 0, f"{tool} is present at {result.stdout.strip()}"


@needs_image
@pytest.mark.parametrize("module", ["pip", "ensurepip", "setuptools", "pkg_resources"])
def test_the_built_image_cannot_import_the_installer(module: str):
    result = _run("--entrypoint", "python", image_args=("-c", f"import {module}"))
    assert result.returncode != 0, f"{module} is importable"


@needs_image
def test_the_built_image_carries_the_sdk():
    result = _run("--entrypoint", "python",
                  image_args=("-c", "import kata_sn22_sdk, sn22_relay;"
                                    "print(kata_sn22_sdk.PROTOCOL_VERSION)"))
    assert result.stdout.strip() == "2", result.stderr


@needs_image
def test_the_built_image_has_no_egress_on_an_internal_network():
    """The agent carries a capability and whatever it scraped. A network that can reach the
    internet is still an exfiltration path even with no key on it."""
    subprocess.run(["docker", "network", "create", "--internal", "kata-sn22-egress-test"],
                   capture_output=True, check=False)
    try:
        result = _run(
            "--network", "kata-sn22-egress-test", "--entrypoint", "python",
            image_args=("-c", "import socket; socket.create_connection(('1.1.1.1', 443), 4)"))
        assert result.returncode != 0, "the agent container reached the public internet"
    finally:
        subprocess.run(["docker", "network", "rm", "kata-sn22-egress-test"],
                       capture_output=True, check=False)


@needs_image
def test_the_reference_agent_runs_in_the_image_under_production_restrictions():
    """Every restriction the room applies, at once, with no broker reachable.

    With nothing to search, the answer is empty -- and that is the assertion: it degrades to a
    well-formed empty answer rather than crashing, so a contestant loses a pool instead of the room
    failing a duel.
    """
    task = json.dumps({
        "protocol_version": 2, "task_id": "t-fast", "search_type": "ai_search",
        "prompt": "2024 emissions", "mode": "fast",
        "result_type": "LINKS_WITH_FINAL_SUMMARY", "tools": ["Web Search"], "count": 10,
        "limits": {"max_execution_time": 30}})
    result = _run(
        "--network", "none", "--read-only", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--user", "10001:10001",
        "--memory", "512m", "--cpus", "1", "--pids-limit", "64",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
        "--mount", f"type=bind,source={REFERENCE},target=/bundle,readonly",
        image_args=("/bundle/agent.py",), stdin=task)

    assert result.returncode == 0, result.stderr
    answer = json.loads(result.stdout)
    assert answer["task_id"] == "t-fast"
    assert answer["search_results"] == []

    from kata_sn22 import protocol_v2 as v2

    v2.parse_answer(answer, task=v2.AiSearchTask(
        task_id="t-fast", prompt="2024 emissions", mode=v2.SearchMode.FAST,
        result_type=v2.ResultType.LINKS_WITH_FINAL_SUMMARY, tools=("Web Search",), count=10))
