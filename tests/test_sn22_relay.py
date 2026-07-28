"""The relay socket: the one thing that crosses the candidate boundary (SN22-7).

The sandbox has no network namespace, the static screen rejects `socket`/`requests`/`urllib`, and a
submission still has to be able to search. This module tests the seam that resolves those three: a
unix socket the lane opens, a client module the lane installs, and a gateway that authorizes and
bills every request behind it.

The end-to-end test at the bottom is the one that matters. It runs a *real* submission — the
reference agent shipped in the competition repository — through the plugin's own `run_candidate`,
against the real gateway, over the real socket. Everything else here is an attack on that seam.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from kata_sn22 import fixtures, relay_client, relay_server, sandbox
from kata_sn22.gateway import GatewayDenied, Sn22Gateway
from kata_sn22.plugin import Sn22DesearchPlugin


@pytest.fixture
def world():
    return fixtures.calibration_manifest(count=4), _provider()


def _verifying_plugin(provider):
    """A plugin wired end to end: a real relay+gateway in front of ``provider``, and the RECORDED
    verification world behind it. The relay half is what this file tests; the verification half is
    supplied so a scored run is possible at all without a network."""
    from kata_sn22.fetch import RecordedPages
    from kata_sn22.plugin import Sn22DesearchPlugin

    tweets = fixtures.recorded_tweets()
    return Sn22DesearchPlugin(
        search_provider=provider,
        page_transport=RecordedPages(records=fixtures.recorded_pages()),
        judge_client=fixtures.scripted_judge(),
        tweet_scraper=lambda ids: {tid: tweets[tid] for tid in ids if tid in tweets})


def _provider(results=None):
    """A stand-in search provider. The relay is a TRANSPORT and the gateway is a BROKER: what they
    must get right is the capability check, the billing and the redaction, none of which depend on
    who answers the query."""
    def _search(query, limit):
        if results is not None:
            return list(results)
        return [{"link": f"https://example.test/{index}", "title": f"Result {index} for {query}",
                 "snippet": "a snippet"} for index in range(limit)]

    return _search


@pytest.fixture
def relay(world, tmp_path):
    _manifest, provider = world
    gateway = Sn22Gateway(provider=provider, challenge_id="c1", reservation_calls=50)
    workdir = tmp_path / "work"
    workdir.mkdir()
    server = relay_server.RelayServer(gateway, workdir).start()
    try:
        yield server, gateway
    finally:
        server.close()


# ---- the transport-----------------------------------------------------------------------------

def test_a_capability_can_search_over_the_socket(relay):
    server, gateway = relay
    capability = gateway.issue(variant="king", task_id="t000", max_calls=3)
    results = relay_client.search("bittensor subnet emissions schedule", limit=5,
                                  capability=capability.token, endpoint=server.endpoint)
    assert results
    # The PROVIDER's own fields reach the agent unchanged: a relay that reformatted results
    # would be deciding what a search returns.
    assert all(set(item) == {"link", "title", "snippet"} for item in results)


def test_the_endpoint_is_a_path_not_a_url(relay):
    server, _gateway = relay
    assert server.endpoint.startswith(relay_client.ENDPOINT_SCHEME)
    assert relay_client.endpoint_path(server.endpoint) == str(server.socket_path)
    # A unix socket, so it survives a namespace with no network at all — which is why the sandbox
    # can keep `--unshare-net` and the agent can still search.
    assert Path(relay_client.endpoint_path(server.endpoint)).exists()


def test_both_contestants_get_identical_content(relay):
    """§5.2: an identical relay request from either side must resolve to identical content."""
    server, gateway = relay
    king = gateway.issue(variant="king", task_id="t000", max_calls=3)
    challenger = gateway.issue(variant="challenger", task_id="t000", max_calls=3)
    query = "desearch decentralized search architecture"
    assert relay_client.search(query, capability=king.token, endpoint=server.endpoint) == \
        relay_client.search(query, capability=challenger.token, endpoint=server.endpoint)


def test_quota_is_free_and_readable_even_when_exhausted(relay):
    """An agent that cannot see its remaining quota either wastes it or hoards it."""
    server, gateway = relay
    capability = gateway.issue(variant="king", task_id="t000", max_calls=1)
    assert relay_client.quota(capability.token, endpoint=server.endpoint) == {
        "used": 0, "max_calls": 1, "remaining": 1}
    relay_client.search("emissions", capability=capability.token, endpoint=server.endpoint)
    # Exhausted, and STILL readable: refusing to report an exhausted quota would withhold the one
    # answer the agent needs at exactly the moment it needs it.
    assert relay_client.quota(capability.token, endpoint=server.endpoint)["remaining"] == 0
    with pytest.raises(relay_client.RelayError):
        relay_client.search("emissions", capability=capability.token, endpoint=server.endpoint)


# ---- §6.2 attacks------------------------------------------------------------------------------

def test_a_forged_capability_is_refused(relay):
    server, _gateway = relay
    with pytest.raises(relay_client.RelayError):
        relay_client.search("x", capability="sn22cap_" + "0" * 32, endpoint=server.endpoint)


def test_a_malformed_capability_is_refused_the_same_way(relay):
    """Identical refusals: a distinct message would let a candidate probe which tokens exist."""
    server, gateway = relay
    gateway.issue(variant="king", task_id="t000", max_calls=3)
    messages = set()
    for token in ("", "not-a-capability", "sn22cap_" + "f" * 32):
        with pytest.raises(relay_client.RelayError) as caught:
            relay_client.search("x", capability=token, endpoint=server.endpoint)
        messages.add(str(caught.value))
    assert len(messages) == 1


def test_a_capability_dies_with_the_challenge(relay):
    server, gateway = relay
    capability = gateway.issue(variant="king", task_id="t000", max_calls=3)
    server.close()
    with pytest.raises(relay_client.RelayError):
        relay_client.search("x", capability=capability.token, endpoint=server.endpoint)


def test_spending_past_the_challenge_reservation_is_refused(world, tmp_path):
    """Per-task quotas bound one task; only the reservation bounds the challenge."""
    _manifest, provider = world
    gateway = Sn22Gateway(provider=provider, challenge_id="c1", reservation_calls=2)
    workdir = tmp_path / "work"
    workdir.mkdir()
    with relay_server.RelayServer(gateway, workdir) as server:
        capability = gateway.issue(variant="king", task_id="t000", max_calls=99)
        for _ in range(2):
            relay_client.search("emissions", capability=capability.token, endpoint=server.endpoint)
        with pytest.raises(relay_client.RelayError):
            relay_client.search("emissions", capability=capability.token, endpoint=server.endpoint)


def test_an_oversized_request_is_refused_before_it_is_parsed(relay):
    server, gateway = relay
    capability = gateway.issue(variant="king", task_id="t000", max_calls=3)
    with pytest.raises(relay_client.RelayError):
        relay_client.search("x" * (relay_client.MAX_REQUEST_BYTES + 1),
                            capability=capability.token, endpoint=server.endpoint)


def test_an_unknown_operation_is_refused(relay):
    server, gateway = relay
    capability = gateway.issue(variant="king", task_id="t000", max_calls=3)
    with pytest.raises(relay_client.RelayError):
        relay_client._request({"op": "issue", "capability": capability.token,
                               "variant": "king", "task_id": "t000", "max_calls": 999},
                              endpoint=server.endpoint)


def test_there_is_no_operation_that_mints_a_capability(relay):
    """The dangerous verbs are simply absent from the wire protocol, not merely guarded."""
    server, gateway = relay
    capability = gateway.issue(variant="king", task_id="t000", max_calls=1)
    for operation in ("issue", "grant", "extend", "close", "usage_receipt", "search "):
        with pytest.raises(relay_client.RelayError):
            relay_client._request({"op": operation, "capability": capability.token},
                                  endpoint=server.endpoint)
    assert relay_client.quota(capability.token, endpoint=server.endpoint)["used"] == 0


def test_garbage_on_the_socket_does_not_crash_the_relay(relay):
    """A peer that cannot frame a request must not take the relay down for the other contestant."""
    import socket as _socket

    server, gateway = relay
    connection = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    connection.connect(str(server.socket_path))
    connection.sendall(b"\x00\xffnot json at all\n")
    connection.close()

    capability = gateway.issue(variant="king", task_id="t000", max_calls=3)
    assert relay_client.search("emissions", capability=capability.token,
                               endpoint=server.endpoint)


def test_the_relay_never_returns_a_credential_shaped_string(world, tmp_path):
    """Whatever the PROVIDER returns, secret-shaped text is scrubbed on the way out.

    The provider is now live rather than a corpus we wrote, which makes this stronger, not weaker:
    a real search result can contain anything, including a key someone leaked onto a web page.
    """
    gateway = Sn22Gateway(provider=_provider([
        {"link": "https://leak.example/x", "title": "Leaked key sk-abcdefghijklmnopqrstuvwx",
         "snippet": "token sk-abcdefghijklmnopqrstuvwx here"}]), challenge_id="c1")
    workdir = tmp_path / "work"
    workdir.mkdir()
    with relay_server.RelayServer(gateway, workdir) as server:
        capability = gateway.issue(variant="king", task_id="t000", max_calls=3)
        results = relay_client.search("emissions", capability=capability.token,
                                      endpoint=server.endpoint)
    assert results
    assert "sk-abcdefghijklmnopqrstuvwx" not in json.dumps(results)
    assert "[REDACTED]" in json.dumps(results)


# ---- the installed client----------------------------------------------------------------------

def test_the_client_is_installed_into_the_run_directory(tmp_path):
    target = relay_server.install_client(tmp_path)
    assert target.name == "sn22_relay.py"
    assert "def search(" in target.read_text(encoding="utf-8")


def test_the_candidate_environment_points_python_at_the_client(tmp_path):
    """Without this the agent could not import the client — and would have to open a socket."""
    env = sandbox.candidate_env(task_input={}, relay_endpoint="e", capability="c",
                                workdir=str(tmp_path))
    assert env["PYTHONPATH"] == str(tmp_path)


def test_an_agent_can_import_and_use_the_installed_client(relay, tmp_path):
    """The whole seam, from an agent's point of view: import, search, print."""
    server, gateway = relay
    workdir = server.socket_path.parent
    relay_server.install_client(workdir)
    capability = gateway.issue(variant="king", task_id="t000", max_calls=3)

    agent = tmp_path / "agent.py"
    agent.write_text(textwrap.dedent("""
        import json, sys
        import sn22_relay
        task = json.loads(sys.stdin.read())
        results = sn22_relay.search(task["query"], limit=3)
        json.dump({"count": len(results), "links": [r["link"] for r in results]}, sys.stdout)
    """), encoding="utf-8")

    env = sandbox.candidate_env(
        task_input={"protocol_version": 1, "task_id": "t000"},
        relay_endpoint=server.endpoint, capability=capability.token, workdir=str(workdir))
    completed = subprocess.run(
        [sys.executable, str(agent)],
        input=json.dumps({"query": "bittensor subnet emissions schedule"}).encode(),
        capture_output=True, cwd=str(workdir), env=env, timeout=60, check=False)
    assert completed.returncode == 0, completed.stderr.decode()
    answer = json.loads(completed.stdout)
    assert answer["count"] > 0
    assert all(link.startswith("https://") for link in answer["links"])


def test_the_client_carries_no_provider_credential(tmp_path):
    """It is copied into an untrusted directory, so it must contain nothing worth stealing."""
    text = relay_server.install_client(tmp_path).read_text(encoding="utf-8")
    for name in ("OPENAI_API_KEY", "APIFY_API_KEY", "SCRAPINGDOG_API_KEY", "sk-"):
        assert name not in text


# ---- end to end--------------------------------------------------------------------------------

#: Miners submit to the ONE competition repo, under their subnet's tree. There is no separate
#: per-subnet submissions repository any more.
REFERENCE_AGENT = (Path(__file__).resolve().parents[2] / "kata" / "submissions"
                   / "sn22__desearch" / "miner" / "example-20260727-01")


@pytest.mark.skipif(not REFERENCE_AGENT.is_dir(),
                    reason="the kata competition repository is not checked out beside this one")
def test_the_reference_submission_scores_through_the_real_relay(tmp_path):
    """The shipped reference agent, run by the plugin, against the real gateway and socket.

    This is the SN22-4 exit gate in one test: a fake-provider end-to-end challenge that a real
    submission actually completes. If the relay seam were wrong — wrong transport, missing client,
    unreachable socket — every task would come back as an invalid run and the valid rate would be 0.
    """
    plugin = _verifying_plugin(fixtures.search_provider())
    problems = plugin.sample_problems(seed="relay-e2e", config={"task_count": 4})

    class _Context:
        label = "challenger"
        output_root = str(tmp_path)
        progress = None

    raw = plugin.run_candidate(agent_path=str(REFERENCE_AGENT), problems=problems,
                               context=_Context())
    card = plugin.score(raw, problems)

    assert card.metrics["sn22_valid_query_rate"] == 1.0, [a.error for a in raw.attempts]
    assert card.metrics["sn22_invalid_runs"] == 0
    # It actually searched: the relay billed it, and the billing is the relay's own record.
    assert card.metrics["sn22_cost_units"] > 0
    # It found real documents rather than fabricating ids: every citation points at something the
    # sealed corpus holds and the agent actually returned.
    assert card.metrics["sn22_citation_precision"] > 0
    assert card.metrics["sn22_coverage"] > 0
    # ...but NOT perfect precision, and that is the reference agent being honest about its own
    # limits: it cites everything it retrieved, so the results that were merely returned rather
    # than genuinely relevant cost it. Selecting what to cite is the first thing a real submission
    # improves, and the fixture would be a poor starting point if it hid that.
    assert card.metrics["sn22_citation_precision"] < 1.0


@pytest.mark.skipif(not REFERENCE_AGENT.is_dir(), reason="competition repository not present")
def test_the_reference_submission_passes_the_static_screen():
    """It must be a submission the lane would actually accept, not one only the tests tolerate."""
    assert Sn22DesearchPlugin().static_screen(str(REFERENCE_AGENT)) is None


@pytest.mark.skipif(not REFERENCE_AGENT.is_dir(), reason="competition repository not present")
def test_the_reference_submission_does_not_ship_the_answers():
    """Anti-memorization: a bundle carrying the query pool verbatim has not earned its answers."""
    files = {p.name: p.read_text(encoding="utf-8")
             for p in REFERENCE_AGENT.rglob("*") if p.is_file()}
    reject, review, score = Sn22DesearchPlugin().benchmark_review(files, strict=True)
    assert not reject and not review and score == 0.0


def test_the_run_records_whether_it_was_isolated(tmp_path):
    """A calibration run and a confined run must never be indistinguishable afterwards."""
    plugin = Sn22DesearchPlugin()
    problems = plugin.sample_problems(seed="isolation-record", config={"task_count": 1})
    agent_dir = tmp_path / "sub"
    agent_dir.mkdir()
    (agent_dir / "agent.py").write_text("import sys; sys.exit(0)\n", encoding="utf-8")

    class _Context:
        label = "king"
        output_root = str(tmp_path)
        progress = None

    raw = plugin.run_candidate(agent_path=str(agent_dir), problems=problems, context=_Context())
    assert raw.isolated is sandbox.available()


def test_production_refuses_to_run_unconfined(tmp_path, monkeypatch):
    """Sandbox absent + the production flag set: the run is an error, not a quiet pass."""
    monkeypatch.setattr(sandbox, "available", lambda: False)
    monkeypatch.setenv("KATA_SN22_REQUIRE_SANDBOX", "1")
    plugin = Sn22DesearchPlugin()
    problems = plugin.sample_problems(seed="require-sandbox", config={"task_count": 1})
    agent_dir = tmp_path / "sub"
    agent_dir.mkdir()
    (agent_dir / "agent.py").write_text("print('{}')\n", encoding="utf-8")

    class _Context:
        label = "king"
        output_root = str(tmp_path)
        progress = None

    from kata_sn22.plugin import Sn22AgentError

    with pytest.raises(Sn22AgentError, match="refusing to run an untrusted submission unconfined"):
        plugin.run_candidate(agent_path=str(agent_dir), problems=problems, context=_Context())


def test_the_gateway_reservation_is_derived_from_the_issued_tasks(tmp_path):
    """The challenge ceiling must come from the tasks the runner issues, not a separate default."""
    plugin = Sn22DesearchPlugin()
    problems = plugin.sample_problems(seed="reservation", config={"task_count": 3,
                                                                 "max_provider_calls": 7})
    assert plugin._reservation_calls(problems) == 21


def test_a_stale_socket_from_a_killed_run_does_not_block_the_next_one(world, tmp_path):
    _manifest, provider = world
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / relay_server.SOCKET_NAME).write_text("left over", encoding="utf-8")
    gateway = Sn22Gateway(provider=provider, challenge_id="c1")
    with relay_server.RelayServer(gateway, workdir) as server:
        capability = gateway.issue(variant="king", task_id="t000", max_calls=1)
        assert relay_client.search("emissions", capability=capability.token,
                                   endpoint=server.endpoint) is not None


def test_closing_removes_the_socket(world, tmp_path):
    _manifest, provider = world
    workdir = tmp_path / "work"
    workdir.mkdir()
    server = relay_server.RelayServer(Sn22Gateway(provider=provider, challenge_id="c1"), workdir)
    server.start()
    assert os.path.exists(server.socket_path)
    server.close()
    assert not os.path.exists(server.socket_path)


def test_the_gateway_quota_read_still_refuses_an_expired_capability(world, tmp_path):
    """A free operation is not an oracle: it authorizes through the same path as a paid one."""
    _manifest, provider = world
    clock = {"t": 0.0}
    gateway = Sn22Gateway(provider=provider, challenge_id="c1", capability_ttl_seconds=10,
                          clock=lambda: clock["t"])
    capability = gateway.issue(variant="king", task_id="t000", max_calls=3)
    assert gateway.quota(capability.token) == (0, 3)
    clock["t"] = 100.0
    with pytest.raises(GatewayDenied):
        gateway.quota(capability.token)


def test_the_published_result_records_whether_the_challenge_was_isolated(tmp_path):
    """A canary must be able to check it, so it has to reach the published result.

    Both sides, or nothing: a per-side flag would let a challenge where only one contestant was
    confined read as a confined challenge — and the unconfined side is the one that matters.
    """
    plugin = Sn22DesearchPlugin()
    problems = plugin.sample_problems(seed="isolation-json", config={"task_count": 1})
    agent_dir = tmp_path / "sub"
    agent_dir.mkdir()
    (agent_dir / "agent.py").write_text("print('{}')\n", encoding="utf-8")

    class _Context:
        label = "king"
        output_root = str(tmp_path)
        progress = None

    card = plugin.score(plugin.run_candidate(agent_path=str(agent_dir), problems=problems,
                                             context=_Context()), problems)
    assert card.metrics["isolated"] is sandbox.available()

    from kata.core.challenge import ChallengeOutcome, GenericChallengeResult, ScoredVariant

    variant = ScoredVariant("king", str(agent_dir), card)
    document = plugin.challenge_result_json(GenericChallengeResult(
        run_id=problems.challenge_id, output_root=str(tmp_path),
        outcome=ChallengeOutcome(problems=problems, benchmark_identity=problems.identity,
                                 scoring_profile=plugin.scoring_profile, king=variant,
                                 ranked=[ScoredVariant("pr-7", str(agent_dir), card)],
                                 winner=None)))
    assert document["isolated"] is sandbox.available()
    assert document["challenge_id"] == problems.challenge_id
    assert document["king"]["isolated"] is sandbox.available()


def test_a_result_with_no_cards_is_not_reported_as_isolated():
    """`all()` over an empty set is True, which would make a challenge that ran nothing look
    confined. Stated as its own test because it is the exact shape of that mistake."""
    plugin = Sn22DesearchPlugin()

    from kata.core.challenge import ChallengeOutcome, GenericChallengeResult

    empty = GenericChallengeResult(
        run_id="r", output_root="/tmp",
        outcome=ChallengeOutcome(problems=None, benchmark_identity="x" * 64,
                                 scoring_profile=plugin.scoring_profile, king=None, ranked=[],
                                 winner=None))
    assert plugin.challenge_result_json(empty)["isolated"] is False


# ---- quota cannot be bypassed by concurrency or retry (plan §9 "cost and recovery") -------------

def test_parallel_requests_cannot_exceed_the_challenge_reservation(world, tmp_path):
    """The reservation is a HARD ceiling, and concurrency is how a check-then-act ceiling fails.

    Billing is a read-modify-write over shared counters. Without serialization, N threads all read
    "served = 9" against a reservation of 10 and all proceed — the classic overspend, and on a
    metered lane every extra call is real money. Threads rather than a mocked lock because the race
    is the thing being tested.
    """
    import threading

    _manifest, provider = world
    reservation = 10
    gateway = Sn22Gateway(provider=provider, challenge_id="c1", reservation_calls=reservation)
    workdir = tmp_path / "work"
    workdir.mkdir()

    with relay_server.RelayServer(gateway, workdir) as server:
        # Per-task quotas deliberately far above the reservation, so ONLY the reservation can bound
        # this. If the challenge ceiling were not enforced, all 40 calls would land.
        capabilities = [gateway.issue(variant=f"v{i}", task_id="t000", max_calls=100)
                        for i in range(4)]
        served = []
        errors = []

        def _hammer(capability):
            for _ in range(10):
                try:
                    relay_client.search("emissions", capability=capability.token,
                                        endpoint=server.endpoint)
                    served.append(1)
                except relay_client.RelayError:
                    errors.append(1)

        threads = [threading.Thread(target=_hammer, args=(c,)) for c in capabilities]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        assert len(served) == reservation, f"served {len(served)} against a cap of {reservation}"
        assert errors, "some requests must have been refused"
        # The gateway's OWN billing record agrees — the ceiling bounded what was billed, not just
        # what was returned.
        totals = sum(record.provider_calls
                     for record in gateway.usage_manifest().records)
        assert totals == reservation


def test_parallel_requests_cannot_exceed_a_PER_TASK_quota(world, tmp_path):
    """The same race one level down: a single capability hammered from many threads."""
    import threading

    _manifest, provider = world
    gateway = Sn22Gateway(provider=provider, challenge_id="c1", reservation_calls=1000)
    workdir = tmp_path / "work"
    workdir.mkdir()

    with relay_server.RelayServer(gateway, workdir) as server:
        capability = gateway.issue(variant="king", task_id="t000", max_calls=5)
        served = []

        def _hammer():
            for _ in range(10):
                try:
                    relay_client.search("emissions", capability=capability.token,
                                        endpoint=server.endpoint)
                    served.append(1)
                except relay_client.RelayError:
                    pass

        threads = [threading.Thread(target=_hammer) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

    assert len(served) == 5


def test_retrying_an_exhausted_capability_never_grants_another_call(world, tmp_path):
    """Retry is the other bypass: a refusal must be terminal, not a rate limit to wait out."""
    _manifest, provider = world
    gateway = Sn22Gateway(provider=provider, challenge_id="c1", reservation_calls=1000)
    workdir = tmp_path / "work"
    workdir.mkdir()

    with relay_server.RelayServer(gateway, workdir) as server:
        capability = gateway.issue(variant="king", task_id="t000", max_calls=2)
        for _ in range(2):
            relay_client.search("emissions", capability=capability.token,
                                endpoint=server.endpoint)
        for _ in range(20):
            with pytest.raises(relay_client.RelayError):
                relay_client.search("emissions", capability=capability.token,
                                    endpoint=server.endpoint)
        assert relay_client.quota(capability.token, endpoint=server.endpoint)["used"] == 2

    # A refused call is not billed: the ledger must not grow with attempts.
    assert sum(r.provider_calls for r in gateway.usage_manifest().records) == 2


def test_a_second_capability_for_the_same_task_does_not_multiply_the_quota(world, tmp_path):
    """Per-task quotas would be meaningless if a contestant could simply be issued another one —
    which is why minting is the LANE's operation and absent from the wire protocol entirely."""
    _manifest, provider = world
    gateway = Sn22Gateway(provider=provider, challenge_id="c1", reservation_calls=3)
    workdir = tmp_path / "work"
    workdir.mkdir()
    with relay_server.RelayServer(gateway, workdir) as server:
        first = gateway.issue(variant="king", task_id="t000", max_calls=2)
        second = gateway.issue(variant="king", task_id="t000", max_calls=2)
        served = 0
        for capability in (first, second):
            for _ in range(2):
                try:
                    relay_client.search("emissions", capability=capability.token,
                                        endpoint=server.endpoint)
                    served += 1
                except relay_client.RelayError:
                    pass
    # Four calls were attempted across two capabilities; the CHALLENGE reservation still bound them.
    assert served == 3
