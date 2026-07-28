"""``KataNeuronAdapter``: everything upstream's validators read off a neuron, and nothing more.

Upstream's scoring classes take a ``neuron`` and use it for two kinds of thing: a handful of config
flags and a metagraph they index by UID, and — on paths Kata never takes — a chain client that
queries live miners. This adapter supplies the first and **refuses** the second.

**The surface is derived, not guessed — and the first derivation was wrong.** Scanning
``self.neuron.<attr>`` across the reward, penalty and scraper packages yields seven names, and that
is what this module originally supplied. It missed ``neurons/validators/clients/``, which calls the
same object ``owner`` and reads two more off it. The result was not an obvious failure: the
``AttributeError`` was raised deep inside ``compute_rewards_and_penalties``, caught by
``_score_one_type``'s ``except Exception``, and turned into a pool of zeros. Every contestant scored
nothing and the logs said only "Full scoring failed".

So the derivation now covers both spellings, and ``tests`` re-derives it against the vendored tree.

**Two virtual UIDs, and that is the whole metagraph.** A Kata duel is a pair. UID 0 is the King and
UID 1 the Challenger, fixed, so ``scores[uid]`` indexes into a two-element array and upstream's own
aggregation works unmodified.

**Disabled means raises, not returns None.** ``get_random_miner`` belongs to upstream's
chain-querying path — the one where a validator picks a miner off the network and sends it a
synapse. Kata never does that: the contestants are two sealed rooms it already has answers from. If
anything reached for it, the round would have fallen into that path, and a stub returning ``None``
would let it keep going and produce a number. So it raises.

``metagraph.axons`` is deliberately NOT refused, and the distinction is worth stating. Upstream's
own logger iterates it to resolve a hotkey to a coldkey — that is bookkeeping, not querying. An
earlier version of this adapter refused it blanketly and broke scoring entirely. What actually
guards the querying path is ``get_random_miner``; the axons here carry two hotkeys and no network
address, so nothing can be dispatched to them.

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
#: Attributes the adapter SUPPLIES with real values.
NEURON_SURFACE = (
    "config.netuid",
    "config.neuron.disable_log_rewards",
    "config.neuron.scoring_model",
    "config.wandb_on",
    "dendrites",
    "get_random_miner",
    "metagraph.axons",
    "metagraph.hotkeys",
    "validator_identity",
)

#: Attributes upstream reads on paths Kata never takes. Present and RAISING rather than absent: an
#: absent one is an ``AttributeError`` raised deep inside ``compute_rewards_and_penalties``, which
#: ``_score_one_type`` catches and turns into a pool of zeros. That is exactly how
#: ``validator_identity`` went missing for a whole phase -- every contestant scored nothing and the
#: only trace was a log line saying "Full scoring failed".
REFUSED_CAPABILITIES = (
    "available_uids",
    "check_uid",
    "get_random_miner",
    "initialize",
    "set_weights",
    "update_moving_averaged_scores",
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
class _Axon:
    """What upstream's logger reads to attribute a response. No address, nothing to dispatch to."""

    hotkey: str
    coldkey: str = "kata"


@dataclass(frozen=True)
class _Config:
    neuron: _NeuronConfig = field(default_factory=_NeuronConfig)
    #: There is no metrics sink in a sealed room, and upstream checks this flag before every
    #: ``wandb.log``. Disabled through upstream's own branch, not by replacing the call.
    wandb_on: bool = False
    #: Read by the logger for its record. Zero because Kata is on no subnet: it decides one
    #: promotion on one host and has no netuid of its own.
    netuid: int = 0


class _Metagraph:
    """Two entries. No chain, no synchronisation, no axon anybody can reach."""

    def __init__(self, hotkeys: tuple) -> None:
        self.hotkeys = list(hotkeys)
        # Real entries, because upstream's LOGGER iterates them to resolve a hotkey to a coldkey.
        # They carry no address: what stops a query is get_random_miner refusing, not this.
        self.axons = [_Axon(hotkey=hotkey) for hotkey in self.hotkeys]

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
        #: What upstream's logger stamps on each record. Names Kata, carries no chain identity, and
        #: deliberately no hotkey that corresponds to anything anyone could pay.
        self.validator_identity = {"validator": "kata", "netuid": 0, "hotkey": "", "uid": None}

    # ---- capabilities Kata does not have -------------------------------------------------------

    @property
    def available_uids(self):
        """Read by the scheduler's own epoch loop, which Kata does not run -- it calls
        ``_score_one_type`` directly with two contestants it already has answers from."""
        raise DisabledCapability(
            "available_uids is not available: Kata scores a fixed pair, not whichever miners "
            "happened to be reachable on the network this hour")

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
