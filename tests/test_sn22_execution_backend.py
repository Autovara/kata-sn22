"""Which backend SN22 executes a submission on, and what happens when the two disagree.

SN22 runs a stranger's code against paid providers. Where that runs is a security decision with two
very different credential stories:

* **``tee``** — the attested sealed room. The MINER's credential is sealed to its exact bundle and
  opened only inside the room, so the miner funds its own search and inference and no validator
  credential is ever in reach of candidate code.
* **``sandbox``** — local `bwrap`, the lane's own relay, no provider call at all. Development and
  §5.5 calibration.

The failure this file exists to prevent is the quiet one: a lane that DECLARES a TEE and then runs
the agent locally. An operator reading `EnvSpec` would believe untrusted code was confined in
attested hardware funding itself, while in fact it was on the validator host. So the declaration and
the execution must agree, and disagreement is a refusal rather than a fallback.
"""
from __future__ import annotations

import pytest

from kata_sn22.execution import policy as execution_policy
from kata_sn22.plugin import ROOM_ENDPOINT_ENV, Sn22AgentError, Sn22DesearchPlugin


@pytest.fixture
def plugin():
    return Sn22DesearchPlugin()


def _submission(tmp_path):
    root = tmp_path / "sub"
    root.mkdir()
    (root / "agent.py").write_text("print('{}')\n", encoding="utf-8")
    return root


def _context(tmp_path, label="king"):
    return type("Ctx", (), {"label": label, "output_root": str(tmp_path), "progress": None})()


# ---- the policy--------------------------------------------------------------------------------

def test_the_production_default_is_the_sealed_room(monkeypatch):
    """An accidental fallback to local execution is the failure that looks like success, so the
    default is the safe one and the unsafe one has to be asked for."""
    monkeypatch.delenv(execution_policy.EXECUTION_BACKEND_ENV, raising=False)
    assert execution_policy.resolve_execution_backend() == "tee"
    assert execution_policy.tee_execution_enabled()


def test_the_development_backend_must_be_selected_explicitly(monkeypatch):
    monkeypatch.setenv(execution_policy.EXECUTION_BACKEND_ENV, "sandbox")
    assert execution_policy.resolve_execution_backend() == "sandbox"
    assert not execution_policy.tee_execution_enabled()


@pytest.mark.parametrize("value", ["TEE", " tee ", "Sandbox"])
def test_case_and_whitespace_are_accepted(monkeypatch, value):
    monkeypatch.setenv(execution_policy.EXECUTION_BACKEND_ENV, value)
    assert execution_policy.resolve_execution_backend() in ("tee", "sandbox")


@pytest.mark.parametrize("value", ["local", "docker", "room", "1", "true"])
def test_an_unknown_backend_is_refused_not_defaulted(monkeypatch, value):
    """A typo falling back to 'tee' would be harmless; falling back to 'sandbox' would run an
    untrusted agent outside the room while the deployment believed otherwise. So neither."""
    monkeypatch.setenv(execution_policy.EXECUTION_BACKEND_ENV, value)
    with pytest.raises(ValueError, match=execution_policy.EXECUTION_BACKEND_ENV):
        execution_policy.resolve_execution_backend()


# ---- what the lane declares--------------------------------------------------------------------

def test_the_env_spec_reports_the_selected_backend(plugin, monkeypatch):
    monkeypatch.setenv(execution_policy.EXECUTION_BACKEND_ENV, "tee")
    assert plugin.environment_spec().execution == "tee"
    monkeypatch.setenv(execution_policy.EXECUTION_BACKEND_ENV, "sandbox")
    assert plugin.environment_spec().execution == "sandbox"


@pytest.mark.parametrize("backend", ["tee", "sandbox"])
def test_no_validator_credential_is_ever_declared(plugin, monkeypatch, backend):
    """Under BOTH backends. Under `tee` the miner's key is a sealed bundle artifact the room
    opens, never an environment variable the platform hands out; under `sandbox` there is no
    provider call to make. A declared secret would put a validator credential in candidate reach."""
    monkeypatch.setenv(execution_policy.EXECUTION_BACKEND_ENV, backend)
    spec = plugin.environment_spec()
    assert spec.required_secrets == ()
    assert spec.network == "relay_only"


# ---- declaration and execution must agree------------------------------------------------------

@pytest.fixture
def epoch_from_development_rows(monkeypatch):
    """Let a tee `sample_problems` build its epoch from the development rows.

    Since Phase E the production backend refuses a development pool by kind and refuses any
    ``task_count`` that is not a full 60-task epoch -- both correct, and both beside the point in
    this module, which is about whether a declared TEE actually reaches a room. Relabelling the
    packaged rows here keeps each test about one thing; the refusals themselves are covered in
    ``tests/test_sn22_epoch_manifest.py``.
    """
    import dataclasses

    from kata_sn22 import plugin as plugin_module
    from kata_sn22 import question_pool

    real = question_pool.load_pool("development")
    relabelled = dataclasses.replace(real, kind=question_pool.KIND_UPSTREAM_SNAPSHOT)
    monkeypatch.setattr(plugin_module, "PRODUCTION_QUESTION_POOL", "development")
    monkeypatch.setattr(question_pool, "load_pool", lambda _name: relabelled)
    return relabelled


def test_declaring_tee_without_a_room_refuses_rather_than_running_locally(
    plugin, monkeypatch, tmp_path, epoch_from_development_rows
):
    """The whole point of this module."""
    monkeypatch.setenv(execution_policy.EXECUTION_BACKEND_ENV, "tee")
    monkeypatch.delenv(ROOM_ENDPOINT_ENV, raising=False)
    problems = plugin.sample_problems(seed="tee-no-room", config={})

    with pytest.raises(Sn22AgentError, match="no sealed room is configured"):
        plugin.run_candidate(agent_path=str(_submission(tmp_path)), problems=problems,
                             context=_context(tmp_path))


def test_the_refusal_names_both_ways_out(plugin, monkeypatch, tmp_path,
                                        epoch_from_development_rows):
    """An operator hitting this must not have to read the source to resolve it."""
    monkeypatch.setenv(execution_policy.EXECUTION_BACKEND_ENV, "tee")
    monkeypatch.delenv(ROOM_ENDPOINT_ENV, raising=False)
    problems = plugin.sample_problems(seed="tee-msg", config={})
    with pytest.raises(Sn22AgentError) as caught:
        plugin.run_candidate(agent_path=str(_submission(tmp_path)), problems=problems,
                             context=_context(tmp_path))
    message = str(caught.value)
    assert ROOM_ENDPOINT_ENV in message
    assert execution_policy.EXECUTION_BACKEND_ENV in message


def test_the_development_backend_runs_without_a_room(plugin, monkeypatch, tmp_path):
    """Calibration needs thirty-plus paired challenges; requiring a room for those would make §5.5
    impossible to run."""
    monkeypatch.setenv(execution_policy.EXECUTION_BACKEND_ENV, "sandbox")
    monkeypatch.delenv(ROOM_ENDPOINT_ENV, raising=False)
    problems = plugin.sample_problems(seed="dev-runs", config={"task_count": 1})
    raw = plugin.run_candidate(agent_path=str(_submission(tmp_path)), problems=problems,
                               context=_context(tmp_path))
    assert raw.variant == "king"


def test_a_configured_room_satisfies_the_tee_declaration(plugin, monkeypatch, tmp_path,
                                                        epoch_from_development_rows):
    """A configured TEE selects the remote client and never falls through to local execution."""
    from kata_sn22 import plugin as plugin_module

    monkeypatch.setenv(execution_policy.EXECUTION_BACKEND_ENV, "tee")
    monkeypatch.setenv(ROOM_ENDPOINT_ENV, "https://room.example:8443")
    problems = plugin.sample_problems(seed="tee-room", config={})
    remote_result = object()
    monkeypatch.setattr(
        plugin,
        "_run_candidate_in_tee",
        lambda **_kwargs: remote_result,
    )
    monkeypatch.setattr(
        plugin_module,
        "_execute",
        lambda *_args, **_kwargs: pytest.fail("TEE execution fell through to the local runner"),
    )

    assert plugin.run_candidate(
        agent_path=str(_submission(tmp_path)),
        problems=problems,
        context=_context(tmp_path),
    ) is remote_result


# ---- the room profile is the subnet's only piece of the room-----------------------------------

def test_the_room_profile_implements_the_generic_seam():
    """`kata-tee-runner` is the subnet-blind base: a subnet reuses it by shipping a profile. This
    asserts SN22's profile matches that contract by NAME, so a rename upstream is caught here rather
    than at deployment."""
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parent.parent
              / "deploy" / "sn22-runner" / "tee_profile.py")
    assert source.is_file(), "SN22 ships no room profile"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    classes = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert "Sn22TeeProfile" in classes
    methods = {node.name for node in ast.walk(classes["Sn22TeeProfile"])
               if isinstance(node, ast.FunctionDef)}
    assert "run" in methods
    fields = {target.id for node in ast.walk(classes["Sn22TeeProfile"])
              if isinstance(node, ast.Assign) for target in node.targets
              if isinstance(target, ast.Name)}
    assert "fixture_project" in fields


def test_the_room_profile_refuses_a_mutable_image_tag():
    """It executes code from a stranger. A tag is a pointer somebody else can move."""
    import sys
    from pathlib import Path
    from unittest import mock

    profile_dir = Path(__file__).resolve().parent.parent / "deploy" / "sn22-runner"
    room_stub = mock.MagicMock()
    with mock.patch.dict(sys.modules, {
        "room": room_stub, "room.inference_network": room_stub, "room.profile": room_stub,
        # The profile now also builds the trusted broker at construction. Stubbed for the same
        # reason as the rest: this test is about the image digest, and pulling in the real room
        # would make it a test of two repositories being checked out together.
        "room.broker": room_stub,
    }):
        sys.path.insert(0, str(profile_dir))
        try:
            import importlib

            module = importlib.import_module("tee_profile")
            importlib.reload(module)
            profile = module.Sn22TeeProfile()
            with mock.patch.dict("os.environ", {module.AGENT_IMAGE_ENV: "sn22-agent:latest"}):
                with pytest.raises(RuntimeError, match="immutable digest"):
                    profile.agent_image()
            with mock.patch.dict("os.environ", {module.AGENT_IMAGE_ENV: ""}):
                with pytest.raises(RuntimeError, match="is not set"):
                    profile.agent_image()
        finally:
            sys.path.remove(str(profile_dir))
            sys.modules.pop("tee_profile", None)


# ---- preflight matches the funding model ---

def test_preflight_requires_no_validator_provider_credential(monkeypatch):
    """It used to require three, and that was left behind when the funding inverted.

    A correctly configured lane has no provider key anywhere by design, so the old check failed it
    with three errors telling the operator to add exactly the credentials that must not exist.
    """
    from kata_sn22.plugin import Sn22DesearchPlugin

    for name in ("SCRAPINGDOG_API_KEY", "APIFY_API_KEY", "OPENAI_API_KEY", "CHUTES_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(execution_policy.EXECUTION_BACKEND_ENV, "tee")
    monkeypatch.setenv("KATA_SN22_VERIFICATION_MODE", "live")
    monkeypatch.delenv(ROOM_ENDPOINT_ENV, raising=False)

    issues = Sn22DesearchPlugin().preflight()
    messages = " ".join(issue["message"] for issue in issues)

    for name in ("SCRAPINGDOG_API_KEY", "APIFY_API_KEY", "OPENAI_API_KEY"):
        assert name not in messages, f"preflight still asks the operator for {name}"
    # ...and it still catches what IS the operator's to supply.
    assert ROOM_ENDPOINT_ENV in messages
