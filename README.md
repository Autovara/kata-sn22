# kata-sn22

The **SN22 (Desearch)** subnet plugin for the [Kata](../kata) competition platform: a sealed, paired
king-versus-challenger search-quality lane whose scoring components are a proven port of a pinned
upstream commit.

See `../KATA-SN22-ACTIVATION-PLAN.md` for the design and the phase gates this repo implements.

## What SN22 (Desearch) is

Desearch is a search-quality subnet. An agent is given a query and answers it by retrieving from the
web and X, and its answer is scored on how relevant and complete the results are. Upstream scores a
whole population of miners against the live web every hour. Kata does something narrower and
therefore fairer to reason about: two agents, one sealed challenge, the same secret queries and the
same frozen corpus.

## Layout

```
kata_sn22/
  protocol.py            the frozen submission contract: task/output schema, limits, error classes
  manifests.py           the sealed query / snapshot / usage manifests and the benchmark identity
  scoring.py             the seven ordered rank signals and the promotion comparator
  upstream_adapter.py    the dependency-free port of the pinned upstream scoring components
  upstream_snapshot.py   identity and integrity of the vendored upstream tree
  parity.py              the parity contract: components, recorded cases, evidence checks
  parity_expectations.json   what the REAL upstream computed, recorded by a reviewer
  gateway.py             the trusted provider gateway: capabilities, quotas, signed receipts
  sandbox.py             the candidate execution jail (bwrap, no network, constructed environment)
  fake_provider.py       an offline relay over the sealed snapshot, with quotas and its own billing
  fixtures.py            the fixed weak/medium/strong/invalid/malicious reference submissions
  plugin.py              the SubnetPlugin implementation
upstream/                the complete pinned upstream tree + UPSTREAM_MANIFEST.json
tools/
  vendor_upstream.py     regenerate or verify the upstream manifest
  upstream_shim.py       import the pinned upstream with its infrastructure stubbed (dev only)
  record_parity.py       execute the pinned upstream and record what it computes
```

## Three properties worth knowing before reading the code

- **A challenge is sealed.** Queries are drawn deterministically from a versioned pool by an HMAC of
  the round seed and travel publicly as a *commitment* — a digest plus the category mix — so nobody
  can pre-compute answers. The corpus is frozen for the round, so an identical relay request from
  either contestant returns identical content. Both are hashed into one benchmark identity alongside
  the judge policy, model identity, upstream commit and plugin revision.
- **No ranked signal comes from the candidate.** Cost is taken from the relay's usage manifest and
  latency from the lane's own clock, because a candidate reporting its own spend has every reason to
  report zero. A citation counts only if the snapshot holds that document, it genuinely answers that
  query, *and* the agent actually returned it.
- **Promotion is lexicographic, not a weighted sum.** Validity, then quality, then citation
  precision, coverage, invalid runs, cost, latency. A weighted sum would let a candidate buy a
  quality win with unlimited spend.

## The upstream parity gate (SN22-5)

`sn22_weighted_quality` is not a Kata-shaped approximation of the upstream score — it *is* the
upstream reward: the AI content/summary split, the ONLY_LINKS reweighting, the component floors, the
applicable penalties and the pool shares. That claim is earned rather than asserted:

- `upstream/` holds the complete tree at `bea9712f58a5fc01c57ec441ce279499529d8bf6`, produced by
  `git archive` and pinned by a manifest of 195 per-file digests plus one tree digest.
- `kata_sn22/upstream_adapter.py` is a **dependency-free port**. The lane runtime carries no
  `bittensor`, no `pydantic`, no HTTP client.
- `tools/record_parity.py` imports the **real** upstream under a stub shim and runs it over 18
  recorded response cases and 48 scalar probes. Every one of the nine adapted penalties fires in at
  least one case and rests in another.
- `kata_sn22/parity.py` checks the adapter against that recording, and checks the recording still
  describes the tree on disk. **Change one upstream byte and the tree digest moves, so the evidence
  no longer matches the code it claims to come from** — which is the SN22-5 exit gate.

One component is pinned by source digest but not executed: the reward-combination arithmetic in
`base_scraper_validator.compute_rewards_and_penalties`, because upstream's method is a live
validator step that logs to W&B and writes a metagraph-sized array. Every input it combines *is*
executed. The parity report says so rather than leaving a reader to find out.

Two upstream components are deliberately excluded from the Kata quality signal, and the challenge
result declares it: `timeout_penalty` and `min_realistic_time_penalty` measure live provider latency,
which a sealed offline snapshot does not have — and Kata already ranks latency as its own signal, so
folding it into quality would let a fast agent outrank a better one.

```bash
uv run pytest                                        # everything except the executed-parity half
uv sync --extra parity                               # adds pydantic/pytz/numpy for the harness
uv run --extra parity pytest                         # ...including the live upstream comparison
uv run --extra parity python tools/record_parity.py --check   # evidence == a fresh recording?
uv run python tools/vendor_upstream.py verify        # tree still matches its manifest?
```

Re-vendoring at a newer upstream commit is a reviewed act, not a build step: `git archive` the new
commit into `upstream/`, run `tools/vendor_upstream.py write`, re-record parity, and **read the
diff**. A changed number there is an upstream behaviour change and has to be understood before the
adapter is taught to agree with it.

## Credential and execution boundary (SN22-4)

A candidate never holds a provider key. It holds a capability — a short-lived token bound to one
lane, one challenge, one variant and one task — and `gateway.py` makes the call on its behalf, bills
it, and returns data with anything secret-shaped scrubbed. `sandbox.py` runs the submission under
`bwrap` as uid 65534 with no network namespace and an environment *constructed from nothing* rather
than filtered, and **refuses to run at all** where it cannot isolate: a submission that could not be
confined has not been evaluated, and scoring it anyway would throw away the point.
