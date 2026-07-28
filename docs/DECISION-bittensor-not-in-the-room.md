# Decision: `bittensor` does not go into the attested room image

**Status:** decided Phase A; **amended in Phase F — read the amendment first.**
**Enforced by:** `tests/test_sn22_room_import_surface.py`.

---

## Amendment (Phase F): this decision now applies to the AGENT image, not to the trusted runner

The claim below — "the runtime import closure is standard library only" — was true of everything in
Phase A, when Kata scored SN22 with its own port of the reward arithmetic.

Phase F changed the requirement, not the reasoning. Production must execute the **real vendored
upstream validator**, because a port that computes something close to upstream's number is a port
that decides duels on a number nobody upstream would recognise. That means the trusted runner now
carries four packages upstream's own scoring semantics depend on:

| Package | Why it cannot be adapted away |
|---|---|
| `pydantic` | validates the protocol models the scorer reads |
| `numpy` | *is* the arithmetic |
| `pytz` | decides what "within date range" means for the date-range penalty |
| `tiktoken` | counts the tokens the streaming penalty charges for |

**`bittensor` is still absent, and so is every other transport.** `wandb`, `aiohttp`, `openai`,
`apify_client`, `aiosqlite`, `redis` and the chain client are all replaced by
`kata_sn22.upstream_runtime`, which follows one rule: *adapt the transport, never the arithmetic*.
`assert_scoring_is_real()` checks after loading that all 22 scoring modules are the real vendored
files, so an edit that adapts one module too many fails a test rather than silently returning a stub
where a number should be.

**What is unchanged:** the **agent image** is still standard library plus the SDK, with no installer.
Everything an untrusted agent can reach is exactly as described below. That is the part of this
decision that was protecting something, and it still holds.

One thing worth recording, because it is the argument for the whole design: `tiktoken` was
originally classified as infrastructure and adapted. The adapter *raises* rather than returning
something plausible, and it fired on the very first import — before any score was computed. A
permissive stub would have returned zero tokens forever and the streaming penalty would have
measured nothing, silently.

---

## The question

The pinned upstream is a Bittensor validator. Importing almost anything from it —
`neurons.validators.penalty.count_penalty`, say — pulls in `bittensor`, `wandb`, `openai`,
`aiohttp`, `apify_client` and a chain client, because a penalty module sits in a package whose
siblings talk to all of them. None of that is needed to *compute* a penalty; it is needed to *be* a
validator.

So: does the sealed room have to carry that dependency tree in order to score the way upstream
scores?

## The answer (Phase A, still true of the agent image)

No. And it is not a preference — it is already true, and now checked.

The runtime import closure of the plugin entry point is **18 modules, standard library only**, plus
`kata` (the engine ABI it is loaded through). Verified two ways: by walking the closure statically
(including function-local imports, which are still room dependencies — they just fail later, on a
duel, instead of at start-up), and by building `SN22_DESEARCH_PLUGIN` for real.

The port is what makes this possible. `kata_sn22.upstream_adapter` reimplements the arithmetic with
no third-party dependency at all; `kata_sn22.providers` reaches the four provider APIs over stdlib
`urllib` rather than four SDKs.

## Where the real upstream still runs

`kata_sn22.parity` executes the actual vendored files through `tools/upstream_shim`, which stubs the
transport (`bittensor`, `wandb`, `apify_client`, the HTTP clients) and keeps the arithmetic real —
`pydantic`, `pytz` and `numpy` are the genuine packages there, so the protocol models validate
exactly as they do in production. That is how parity is *evidenced* rather than asserted.

It is a development and evidence tool, and it is **not** reachable from the entry point. The guard
checks that too, because if it ever were:

- the room image would have to ship the 2.4 MB vendored `upstream/` tree, and the attested
  measurement covers every byte of the image;
- `upstream_shim` mutates `sys.modules`, so its stubs would be one import away from live scoring —
  a scoring path that reached one would produce a `_Stub` instead of a number.

## Why this is worth a file and a test

The failure mode is the expensive one. A `pydantic` added to a scoring module passes every other
test in this repo, changes the image measurement, and surfaces **inside a sealed TEE room, on a
duel, as an ImportError with no debugger attached**.

This is the recurring shape named in `KATA-ARCHITECTURE.md` §10 rule 5: two components each correct
in isolation, never checked against each other. The runtime is correct. The parity harness is
correct. Nothing checked that the first had not quietly acquired the second's dependencies.
