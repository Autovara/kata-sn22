"""``KataNeuronAdapter``: everything upstream's validators read off a neuron, and nothing more.

Upstream's scoring classes take a ``neuron`` and use it for two kinds of thing: a handful of config
flags and a metagraph they index by UID, and — on paths Kata never takes — a chain client that
queries live miners. This adapter supplies the first and **refuses** the second.

**The surface is derived, not guessed.** Scanning every ``self.neuron.<attr>`` access in the reward,
penalty and scraper packages at the pinned commit yields exactly seven names. ``tests`` re-derives
that list against the vendored tree, so an upstream that starts reading an eighth is a test failure
rather than an ``AttributeError`` on a duel.

**Two virtual UIDs, and that is the whole metagraph.** A Kata duel is a pair. UID 0 is the King and
UID 1 the Challenger, fixed, so ``scores[uid]`` indexes into a two-element array and upstream's own
aggregation works unmodified.

**Disabled means raises, not returns None.** ``get_random_miner`` and ``metagraph.axons`` belong to
upstream's chain-querying path — the one where a validator picks a miner off the network and sends
it a synapse. Kata never does that: the contestants are two sealed rooms it already has answers
from. If anything ever reached for them it would mean the round had fallen into that path, and a
stub returning ``None`` would let it keep going and produce a number. So they raise.

The one exception is ``utility_api``, which is ``None`` on purpose: upstream's own log submitter
checks for exactly that and skips, so the public validator API is disabled through upstream's
documented "not configured" branch rather than by anything of ours.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kata_sn22.protocol_v2 import FIXED_SCORING_MODEL

#: A Kata duel is a pair, so the metagraph is two entries. Fixed rather than allocated: upstream
#: writes ``scores[uid]`` and reads ``metagraph.hotkeys`` to size that array.
KING_UID = 0
CHALLENGER_UID = 1
VIRTUAL_UIDS = (KING_UID, CHALLENGER_UID)

#: Every attribute upstream's scoring path reads off a neuron, at the pinned commit. Re-derived by
#: ``tests/test_sn22_production_scorer.py`` so a new upstream read is a test failure.
NEURON_SURFACE = (
    "config.neuron.disable_log_rewards",
    "config.neuron.scoring_model",
    "config.wandb_on",
    "dendrites",
    "get_random_miner",
    "metagraph.axons",
    "metagraph.hotkeys",
)


class DisabledCapability(RuntimeError):
    """Something reached for a validator capability Kata deliberately does not have."""


def _refuse(what: str, why: str):
    def _refused(*_args, **_kwargs):
        raise DisabledCapability(
            f"{what} is not available in Kata's runtime: {why}. Reaching it means the round has "
            f"fallen into upstream's chain-querying path, which Kata never takes")

    return _refused


class _RefusingList:
    """Indexable in form only. Reading an entry raises rather than returning a plausible object."""

    def __init__(self, what: str, why: str) -> None:
        self._refuse = _refuse(what, why)

    def __getitem__(self, _index):
        self._refuse()

    def __len__(self) -> int:
        return len(VIRTUAL_UIDS)

    def __iter__(self):
        self._refuse()


@dataclass(frozen=True)
class _NeuronConfig:
    """``config.neuron``. Two flags, both read by ``compute_rewards_and_penalties``."""

    #: True, so upstream skips assembling its per-reward event dictionary. That dictionary is a
    #: logging artifact; skipping it changes no number and keeps miner content out of the log.
    disable_log_rewards: bool = True
    #: The judge every contestant is graded by. Fixed for the whole duel -- one contestant graded
    #: by Qwen and another by GPT is not one competition. Upstream's own fallback to gpt-4.1-nano is
    #: disabled in Kata's scorer policy, so a failed Chutes credential is a zero rather than a
    #: different grader.
    scoring_model: str = FIXED_SCORING_MODEL.value


@dataclass(frozen=True)
class _Config:
    neuron: _NeuronConfig = field(default_factory=_NeuronConfig)
    #: There is no metrics sink in a sealed room, and upstream checks this flag before every
    #: ``wandb.log``. Disabled through upstream's own branch, not by replacing the call.
    wandb_on: bool = False


class _Metagraph:
    """Two entries. No chain, no synchronisation, no axon anybody can reach."""

    def __init__(self, hotkeys: tuple) -> None:
        self.hotkeys = list(hotkeys)
        # Indexed only by upstream's organic/querying path. See the module docstring.
        self.axons = _RefusingList(
            "metagraph.axons",
            "Kata has no chain metagraph; its two contestants are sealed rooms it already holds "
            "answers from")

    def sync(self, *_args, **_kwargs):
        raise DisabledCapability(
            "metagraph synchronisation is not available: Kata reads no chain state and writes none")


@dataclass
class KataNeuronAdapter:
    """The infrastructure half of a validator, and none of the authority.

    What it deliberately does not have, because a compromise or a bug here would otherwise reach the
    network: no wallet, no subtensor, no chain writes, no W&B, no public API, no process manager, no
    metagraph synchronisation. It answers config questions and holds two hotkeys.
    """

    king_hotkey: str = "kata-king"
    challenger_hotkey: str = "kata-challenger"
    config: _Config = field(default_factory=_Config)

    def __post_init__(self) -> None:
        self.metagraph = _Metagraph((self.king_hotkey, self.challenger_hotkey))
        #: Upstream iterates these to send synapses. Kata sends none: the answers already exist.
        self.dendrites = _RefusingList(
            "dendrites",
            "Kata does not query miners; both contestants' answers were produced in sealed rooms "
            "before scoring began")
        #: ``None`` on purpose. Upstream's ``submit_logs`` checks for exactly this and skips, so the
        #: public validator API is disabled through its own documented branch.
        self.utility_api = None
        self.wallet = None
        self.subtensor = None

    # ---- capabilities Kata does not have -------------------------------------------------------

    async def get_random_miner(self, *_args, **_kwargs):
        raise DisabledCapability(
            "get_random_miner is not available: Kata scores two named contestants, not a miner "
            "drawn from the network")

    async def update_moving_averaged_scores(self, *_args, **_kwargs):
        raise DisabledCapability(
            "moving-average scores are not available: a Kata duel decides one promotion from one "
            "epoch, and carrying state between rounds would let an old score defend a crown")

    async def set_weights(self, *_args, **_kwargs):
        raise DisabledCapability(
            "weight submission is not available: Kata writes nothing to the chain")

    async def initialize(self, *_args, **_kwargs):
        raise DisabledCapability(
            "there is nothing to initialise: no wallet, no subtensor, no metagraph to sync")

    async def check_uid(self, *_args, **_kwargs):
        raise DisabledCapability("there are no live UIDs to check")

    # ---- what a report records ------------------------------------------------------------------

    def as_provenance(self) -> dict:
        """The infrastructure facts an attested report carries. No key, no address, no hotkey that
        corresponds to anything on a chain."""
        return {
            "neuron": "kata",
            "virtual_uids": list(VIRTUAL_UIDS),
            "scoring_model": self.config.neuron.scoring_model,
            "wandb": False,
            "chain_writes": False,
            "public_api": False,
            "wallet": False,
        }
