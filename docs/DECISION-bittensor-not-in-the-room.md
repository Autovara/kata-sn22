# Decision: `bittensor` does not go into the attested room image

**Status:** decided, Phase A. **Enforced by:** `tests/test_sn22_room_import_surface.py`.

## The question

The pinned upstream is a Bittensor validator. Importing almost anything from it —
`neurons.validators.penalty.count_penalty`, say — pulls in `bittensor`, `wandb`, `openai`,
`aiohttp`, `apify_client` and a chain client, because a penalty module sits in a package whose
siblings talk to all of them. None of that is needed to *compute* a penalty; it is needed to *be* a
validator.

So: does the sealed room have to carry that dependency tree in order to score the way upstream
scores?

## The answer

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
