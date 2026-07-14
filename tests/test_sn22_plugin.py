"""Phase 8: the SN22 skeleton proves a second subnet runs with zero core edits.

A live/noisy subnet (allowlist env, non-cacheable king, stub scorer) runs a full
King-of-the-Hill round through the *same* generic orchestrator SN60 uses, producing a
winner (the new king) -- without any subnet-specific code in the core.
"""

from __future__ import annotations

from pathlib import Path

from kata.core.round import RoundOutcome
from kata.packages import ScoringProfile, get_plugin

from kata_sn22 import SN22_DESEARCH_PLUGIN, Sn22DesearchPlugin


def _write_agent(root: Path, relevance: float) -> str:
    root.mkdir(parents=True, exist_ok=True)
    (root / "agent.py").write_text(
        f"# relevance={relevance}\n"
        "def agent_main(query=None, data_api=None):\n"
        "    return {'results': []}\n",
        encoding="utf-8",
    )
    return str(root)


def test_sn22_plugin_registers_and_declares_noisy_live_profile() -> None:
    assert get_plugin("sn22_desearch") is SN22_DESEARCH_PLUGIN
    plugin = Sn22DesearchPlugin()
    assert plugin.pack == "sn22__desearch"
    assert plugin.scoring_profile is ScoringProfile.NOISY
    env = plugin.environment_spec()
    assert env.network == "allowlist"
    assert "api.desearch.ai" in env.allowed_hosts
    assert env.required_secrets == ("SN22_DATA_API_KEY",)
    # Live subnet: the king is not cacheable across rounds.
    assert plugin.benchmark_identity(plugin.sample_problems(seed="r", config={})) == ""


def test_sn22_runs_a_full_round_through_the_generic_orchestrator(tmp_path: Path) -> None:
    plugin = Sn22DesearchPlugin()
    king = _write_agent(tmp_path / "king", 0.5)
    candidates = [
        ("miner-low", _write_agent(tmp_path / "low", 0.2)),
        ("miner-high", _write_agent(tmp_path / "high", 0.8)),
    ]

    # The default (unoverridden) run_round drives the generic orchestrator.
    outcome = plugin.run_round(
        king_agent_path=king,
        candidates=candidates,
        config={},
        output_root=str(tmp_path / "runs"),
        run_id="round-1",
    )

    assert isinstance(outcome, RoundOutcome)
    assert outcome.scoring_profile is ScoringProfile.NOISY
    assert outcome.king is not None and outcome.king.card.comparable == 0.5
    # The higher-relevance miner beats the king and becomes the new king.
    assert outcome.winner is not None and outcome.winner.label == "miner-high"
    assert [v.label for v in outcome.ranked] == ["miner-high", "miner-low"]
    assert outcome.winner.card.metrics["relevance"] == 0.8


def test_sn22_round_has_no_winner_when_king_unbeaten(tmp_path: Path) -> None:
    plugin = Sn22DesearchPlugin()
    king = _write_agent(tmp_path / "king", 0.9)
    outcome = plugin.run_round(
        king_agent_path=king,
        candidates=[("weak", _write_agent(tmp_path / "weak", 0.3))],
        config={},
        output_root=str(tmp_path / "runs"),
        run_id="round-2",
    )
    assert outcome.winner is None
