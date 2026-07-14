"""SN22 (Desearch) subnet plugin -- SKELETON (Phase 8 of the multi-subnet refactor).

This proves the multi-subnet seam end to end: a *second* subnet running through the
generic King-of-the-Hill orchestrator with a ``NOISY`` scoring profile and an
``allowlist`` network environment -- with **no edits to the core**.

The scorer here is a STUB. Real Desearch validation (live X/Twitter + web retrieval,
LLM-judged relevance against live data) is a follow-up; it plugs in behind exactly
these methods without touching the core. See docs/notes on SN22 being a live/noisy
subnet whose king is not cacheable across rounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kata.packages.plugin import (
    EnvSpec,
    RunContext,
    ScoreCard,
    ScoringProfile,
    SubnetPlugin,
)


@dataclass(frozen=True)
class Sn22Problems:
    """The round's queries (a real run would sample live/organic queries)."""

    queries: list[str]


@dataclass(frozen=True)
class Sn22RawRun:
    agent_path: str
    relevance: float


def _stub_relevance(agent_path: str) -> float:
    """STUB: read a ``# relevance=<float>`` hint from the agent bundle.

    Stands in for real Desearch scoring (live retrieval + LLM-judged relevance) so
    the skeleton is deterministic and testable end to end.
    """
    agent_py = Path(agent_path).expanduser().resolve() / "agent.py"
    if not agent_py.exists():
        return 0.0
    for line in agent_py.read_text(encoding="utf-8").splitlines():
        if "# relevance=" in line:
            try:
                return float(line.split("# relevance=")[1].strip())
            except ValueError:
                return 0.0
    return 0.0


class Sn22DesearchPlugin(SubnetPlugin):
    """Desearch (Bittensor SN22) skeleton plugin: live search-quality competition."""

    evaluator_id = "sn22_desearch"
    pack = "sn22__desearch"
    mode = "miner"
    # Live + LLM-judged: scores drift, so the core averages repeats and re-scores the
    # king every round (see benchmark_identity below).
    scoring_profile = ScoringProfile.NOISY
    validator_identity = "sn22-desearch-stub-v0"

    def environment_spec(self) -> EnvSpec:
        # A live subnet: the agent reaches allowlisted data providers using a
        # validator-injected key (never a sealed relay-only sandbox like SN60).
        return EnvSpec(
            network="allowlist",
            allowed_hosts=("api.twitter.com", "api.x.com", "api.desearch.ai"),
            required_secrets=("SN22_DATA_API_KEY",),
        )

    def sample_problems(self, *, seed: str, config: dict[str, Any]) -> Sn22Problems:
        queries = config.get("queries") or [f"query::{seed}::1", f"query::{seed}::2"]
        return Sn22Problems(queries=list(queries))

    def benchmark_identity(self, problems: Sn22Problems) -> str:
        # Empty == not cacheable: live results drift, so the king is re-scored every
        # round rather than reused (the NOISY contract).
        return ""

    def run_candidate(
        self, *, agent_path: str, problems: Sn22Problems, context: RunContext
    ) -> Sn22RawRun:
        # STUB: a real run would query the allowlisted providers for each problem.
        return Sn22RawRun(agent_path=str(agent_path), relevance=_stub_relevance(agent_path))

    def score(self, raw: Sn22RawRun, problems: Sn22Problems) -> ScoreCard:
        # STUB: a real scorer would LLM-judge the returned results against live data.
        return ScoreCard(
            comparable=raw.relevance,
            passed=True,
            metrics={"relevance": raw.relevance},
        )

    def compare(self, a: ScoreCard, b: ScoreCard) -> int:
        return (a.comparable > b.comparable) - (a.comparable < b.comparable)

    def beats_king(self, candidate: ScoreCard, king: ScoreCard | None) -> bool:
        if king is None:
            return candidate.comparable > 0.0
        return candidate.comparable > king.comparable
