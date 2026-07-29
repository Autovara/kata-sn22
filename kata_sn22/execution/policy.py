"""SN22 execution-backend policy.

SN22 runs a stranger's agent against paid providers, so where that agent executes is a security
decision, not a deployment convenience. Two backends, and the default is the safe one:

* **``tee``** — the attested sealed room (`kata-tee-runner`). The miner's own provider credential is
  sealed to its exact submission bundle and decrypted only inside the room, so the miner funds its
  own search and inference and *no validator credential is ever in reach of candidate code*. The
  room returns a hardware attestation binding the answer to the challenge.
* **``sandbox``** — local `bwrap` isolation with the lane's own relay. An explicit development and
  calibration mode: §5.5 needs thirty-plus paired challenges before a single real credential is
  issued, and doing those in a TEE would be slow and pointless.

TEE is the default for the same reason SN60's is: an accidental production fallback to local
execution is the failure that looks like success. Selecting the sandbox is something a deployment
has to *say*, and the choice travels into the challenge result so no run is ambiguous afterwards.
"""

from __future__ import annotations

from kata.core.execution_backend import ExecutionBackendPolicy

EXECUTION_BACKEND_ENV = "KATA_SN22_EXECUTION_BACKEND"
_BACKENDS = frozenset({"tee", "sandbox"})

#: The rule itself lives in core, subnet-neutral. SN22 supplies only what is genuinely its own:
#: the variable name, the permitted values, and which one is safe when nothing is configured.
#:
#: This replaced a byte-identical copy of the same logic in the other subnet. A fix to one -- for
#: instance tightening what counts as an acceptable value -- silently did not reach the other.
_POLICY = ExecutionBackendPolicy(EXECUTION_BACKEND_ENV, _BACKENDS, "tee")


def resolve_execution_backend() -> str:
    """Return ``tee`` by default, or an explicitly selected development backend."""
    return _POLICY.resolve()


def tee_execution_enabled() -> bool:
    return _POLICY.is_selected("tee")
