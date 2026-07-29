"""The documentation says what the code does, and stops saying what it used to.

Documentation drifts silently and expensively: a miner reading a stale page seals the wrong thing,
or writes to an interface that no longer exists, and finds out on a scored duel. So the things that
would mislead someone into losing money are checked here rather than trusted to review.

Two kinds of check, and both matter:

* **Nothing claims a removed behaviour.** A sealed corpus, a five-result default, validator-paid
  verification, an eight-task round, promotion margins.
* **Everything a miner or operator must do is written down**, and the commands are the real ones —
  every flag is checked against the tool that implements it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
KATA = REPO.parent / "kata"
BOT = REPO.parent / "kata-bot"

README = REPO / "README.md"
OPERATOR_GUIDE = REPO / "SN22-OPERATOR-GUIDE.md"
#: This lane's wire protocol. It lived in `kata/docs/` until the engine repo was made
#: subnet-agnostic; a subnet's own protocol is the subnet's to document.
PROTOCOL = REPO / "docs" / "SN22-PROTOCOL.md"
ENV_EXAMPLE = BOT / ".env.example"


def _text(path: Path) -> str:
    if not path.is_file():
        pytest.skip(f"{path} is not checked out beside this repository")
    return path.read_text(encoding="utf-8")


#: Every miner- or operator-facing document this phase is responsible for.
DOCUMENTS = {
    "kata-sn22/README.md": README,
    "kata-sn22/SN22-OPERATOR-GUIDE.md": OPERATOR_GUIDE,
    "docs/SN22-PROTOCOL.md": PROTOCOL,
}


# ---- GATE: no document mentions a removed behaviour ---

#: Claims that were true once and would now cost a reader real money.
#:
#: ``max_results`` of 5 is the subtle one: it was BELOW upstream's own minimum of 10, so a miner
#: reading it would build an agent upstream's model rejects.
REMOVED_CLAIMS = {
    "sealed corpus": "the corpus is gone; both contestants search live sources",
    "snapshot digest": "there is no frozen snapshot to bind an identity to",
    "validator-paid": "the validator holds no paid SN22 credential",
    "validator pays": "the validator holds no paid SN22 credential",
    "KATA_PROMOTE_MARGINS": "production ranks on one upstream-derived score",
    "seven ordered rank signals": "production ranks on one score; the seven are calibration only",
    "sn22_valid_query_rate": "a calibration signal, never a production ranking",
}


@pytest.mark.parametrize("name", sorted(DOCUMENTS))
@pytest.mark.parametrize("claim", sorted(REMOVED_CLAIMS))
def test_no_document_repeats_a_removed_claim(name, claim):
    body = _text(DOCUMENTS[name])
    # A document may still NAME a removed thing in order to say it is gone. What it may not do is
    # describe it as current, so an explicit "gone"/"no longer"/"used to" nearby is allowed.
    for line in body.splitlines():
        if claim.lower() not in line.lower():
            continue
        explained = any(marker in line.lower() for marker in
                        ("gone", "no longer", "used to", "belonged", "gets none", "gets its",
                         "removed", "not ", "never", "calibration", "would be"))
        assert explained, f"{name} still presents {claim!r} as current: {line.strip()!r}"


def test_no_document_states_the_old_five_result_default():
    """It was below upstream's own minimum of 10, so a miner following it would build an agent
    upstream's model rejects."""
    for name, path in DOCUMENTS.items():
        body = _text(path)
        for bad in ('"max_results": 5', "max_results=5", "max_results` is 5"):
            assert bad not in body, f"{name} states the old five-result default"


def test_no_document_states_the_old_eight_task_round():
    """Eight tasks could never be scored at all: upstream drops a contestant with fewer than three
    deep samples, so every contestant would have been zeroed for a reason unrelated to answers."""
    for name, path in DOCUMENTS.items():
        body = _text(path)
        for bad in ('"task_count": 8', "task_count=8", "8 tasks per"):
            assert bad not in body, f"{name} states the old eight-task round"


def test_the_bot_env_example_offers_no_sn22_provider_key():
    """Setting one would create the fallback the sealed credential set exists to prevent."""
    body = _text(ENV_EXAMPLE)
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for key in ("SCRAPINGDOG_API_KEY", "APIFY_API_KEY", "OPENAI_API_KEY"):
            assert not stripped.startswith(f"{key}="), f".env.example still offers {key}"


# ---- GATE: everything a miner must know is written down ---

MINER_FACTS = {
    "four required credentials": ("scrapingdog", "apify", "openai", "chutes"),
    "the exact sealing command": ("kata_seal_multi.py", "--credential-profile", "--providers"),
    "why the agent image exists": ("no package installer", "kata_sn22_sdk"),
    "all four task pools": ("ai_search:fast", "ai_search:balanced", "ai_search:deep", "x_search"),
    "a bad key means zero": ("scores you ZERO",),
    "the King keeps paying": ("defend it",),
    "there is no validator fallback": ("The validator holds none",),
    "how to read a failure": ("credential_payment_required", "infrastructure"),
}


@pytest.mark.parametrize("fact", sorted(MINER_FACTS))
def test_the_protocol_document_states_every_fact_a_miner_needs(fact):
    body = _text(PROTOCOL)
    for phrase in MINER_FACTS[fact]:
        assert phrase.lower() in body.lower(), f"the protocol does not state {fact}: {phrase!r}"


OPERATOR_FACTS = {
    "the five operator-supplied values": (
        "KATA_SN22_TEE_AGENT_IMAGE", "KATA_SN22_TEE_RUNNER_IMAGE",
        "KATA_SN22_ROOM_MEASUREMENT", "KATA_SN22_SCORER_POLICY_HASH",
        "KATA_SN22_ROUTE_POLICY_HASH"),
    "how to capture the question pool": ("snapshot_questions.py",),
    "how to rebuild and re-attest": ("build.sh", "measurement"),
    "that a rebuild invalidates sealed credentials": ("reseal",),
    "how to read a failure": ("failure_category", "infrastructure"),
    "why there are no provider budgets": ("BUDGET_INFERENCE_CALLS",),
    "local testing": ("uv run pytest",),
}


@pytest.mark.parametrize("fact", sorted(OPERATOR_FACTS))
def test_the_operator_guide_states_every_fact_an_operator_needs(fact):
    body = _text(OPERATOR_GUIDE)
    for phrase in OPERATOR_FACTS[fact]:
        assert phrase.lower() in body.lower(), (
            f"the operator guide does not state {fact}: {phrase!r}")


# ---- GATE: every documented command is real ---

def test_the_documented_sealing_flags_are_the_ones_the_tool_accepts():
    """A documented flag the tool does not have is a miner stuck at the one step where they are
    handling real credentials."""
    sealer = REPO.parent / "kata-tee-runner" / "kata_seal_multi.py"
    if not sealer.is_file():
        pytest.skip("kata-tee-runner is not checked out beside this repository")

    source = sealer.read_text(encoding="utf-8")
    body = _text(PROTOCOL)
    documented = {
        token.split("=")[0] for line in body.splitlines() for token in line.split()
        # A real flag, not a markdown horizontal rule or a bare `--` separator.
        if token.startswith("--") and len(token) > 2 and token[2].isalpha()
    }
    # `uv`'s own flags appear in the same command line and belong to uv, not the sealer.
    uv_flags = {"--extra", "--with", "--python", "--directory", "--no-project", "--"}
    for flag in sorted(documented - uv_flags):
        assert f'"{flag}"' in source, (
            f"the protocol documents {flag}, which kata_seal_multi.py does not accept")


def test_the_documented_snapshot_command_matches_the_tool():
    tool = (REPO / "tools" / "snapshot_questions.py").read_text(encoding="utf-8")
    for guide in (OPERATOR_GUIDE, README):
        body = _text(guide)
        if "snapshot_questions.py" not in body:
            continue
        assert "--out" in tool
        assert "--extra snapshot" in body, "the snapshot command omits the extra it needs"


def test_the_documented_pool_shares_match_the_scorer_policy():
    """A miner optimising the wrong pool because a table was stale is a real cost."""
    from kata_sn22.scorer_policy import POOL_WEIGHTS

    body = _text(PROTOCOL)
    for pool, weight in POOL_WEIGHTS.items():
        assert pool in body, f"the protocol does not name the {pool} pool"
        assert f"{weight:.2f}" in body, f"the protocol does not state {pool}'s share {weight}"


def test_the_documented_credential_profile_is_the_one_the_room_requires():
    from kata_sn22.credentials_v2 import CREDENTIAL_PROFILE, REQUIRED_PROVIDERS

    body = _text(PROTOCOL)
    assert CREDENTIAL_PROFILE in body
    assert ",".join(REQUIRED_PROVIDERS) in body, (
        "the documented --providers list is not the set the room requires, in its order")


def test_the_documented_failure_categories_are_the_ones_the_bot_publishes():
    """A category a miner is told to look for and never sees is worse than none.

    Two vocabularies are legitimate and live in different layers, so this test knows both:

    * ``credential_failure`` is a sealed room's REPORT STATUS -- what one pool job returned;
    * ``credential_payment_required`` and its siblings are the BOT's published categories -- what a
      miner reads in a result file.

    A name in neither set is a documented string nobody will ever produce.
    """
    from kata_sn22.report_v2 import ReportStatus

    bot_continuous = BOT / "src" / "kata_bot" / "continuous.py"
    if not bot_continuous.is_file():
        pytest.skip("kata-bot is not checked out beside this repository")

    source = bot_continuous.read_text(encoding="utf-8")
    room_statuses = {status.value for status in ReportStatus}

    for document in (PROTOCOL, OPERATOR_GUIDE):
        body = _text(document)
        for line in body.splitlines():
            for raw in line.replace("`", " ").split():
                token = raw.strip(".,()")
                if not token.startswith("credential_"):
                    continue
                assert token in room_statuses or f'"{token}"' in source, (
                    f"{document.name} names {token}, which is neither a room report status nor a "
                    f"category the bot publishes")


def test_the_agent_the_protocol_points_at_is_the_reigning_king():
    """Miners are told to copy the King itself, not a separate shipped example.

    Two agents would be two things to keep correct, and they drift: a miner would copy the example
    while being scored against the King. Pointing at the King makes that impossible by construction.
    ``submissions/`` holds miners' entries and nothing else.
    """
    body = _text(PROTOCOL)
    referenced = "kings/sn22__desearch/miner/agent.py"
    assert referenced in body, "the protocol does not point miners at the King"
    assert (KATA / referenced).is_file(), "the protocol points at a King that does not exist"
    assert "submissions/sn22__desearch/miner/example" not in body, (
        "the protocol still points at the removed example submission")
