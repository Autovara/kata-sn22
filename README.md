# kata-sn22

An early scaffold of the **SN22 (Desearch)** subnet plugin for the [Kata](../kata) competition
platform. It shows how a second subnet plugs into Kata. It is not production, and it does not yet run
against the current engine.

## What SN22 (Desearch) is

Desearch is a search-quality subnet. An agent is given a query and answers it by retrieving from the
live web and X (Twitter). Its answer is scored on how relevant and complete the results are, judged
against live data by an LLM. Because the web keeps changing, the same agent can score differently
from one round to the next.

## What this repo is today

A skeleton. It implements the `SubnetPlugin` methods (see `../kata` for that contract and the agent
submission format) so you can read how a subnet declares itself to Kata:

- `NOISY` scoring: results drift, so the king is re-scored every round instead of being cached.
  `benchmark_identity` returns an empty string to say "not cacheable".
- A network allowlist (`api.twitter.com`, `api.x.com`, `api.desearch.ai`) plus a required
  `SN22_DATA_API_KEY` secret, declared through `environment_spec`.

The scorer is a stub. `_stub_relevance` just reads a `# relevance=<float>` hint from the agent's
`agent.py` so the skeleton is deterministic. It does no retrieval and no LLM judging. Real Desearch
scoring plugs in behind these same methods later, with no change to Kata's core.

```
kata_sn22/
  protocol.py       the frozen submission contract: task/output schema, limits, error classes
  manifests.py      the sealed query / snapshot / usage manifests and the benchmark identity
  scoring.py        the seven ordered rank signals and the promotion comparator
  fake_provider.py  an offline relay over the sealed snapshot, with quotas and its own billing
  fixtures.py       the fixed weak/medium/strong/invalid/malicious reference submissions
  plugin.py         the SubnetPlugin methods (stub; replaced in SN22-3)
  __init__.py       lazily exposes the plugin so the protocol layer imports on any core
```

## The evaluation protocol (SN22-2)

Everything except `plugin.py` is the frozen evaluation contract, and it is deliberately independent
of the Kata core: it imports nothing from `kata`, touches no network, and can be reviewed and
calibrated before a lane exists to run it. See `KATA-SN22-ACTIVATION-PLAN.md` §5 for the design.

Three properties are worth knowing before reading the code:

- **A challenge is sealed.** The queries are drawn deterministically from a versioned pool by an
  HMAC of the round seed, and travel publicly as a *commitment* (a digest plus the category mix) so
  nobody can pre-compute answers. The corpus is frozen for the round, so an identical relay request
  from either contestant returns identical content. Both are hashed into one benchmark identity
  alongside the judge policy, model identity, upstream commit and plugin revision.
- **No ranked signal comes from the candidate.** Cost is taken from the relay's usage manifest and
  latency from the lane's own clock, because a candidate reporting its own spend has every reason to
  report zero. A citation counts only if the snapshot holds that document, it genuinely answers that
  query, *and* the agent actually returned it.
- **Promotion is lexicographic, not a weighted sum.** Validity, then quality, then citation
  precision, coverage, invalid runs, cost, latency. A weighted sum would let a candidate buy a
  quality win with unlimited spend.

## Not yet wired

`plugin.py` does not load against the latest engine — its imports (`kata.packages.plugin`,
`kata.packages.registry`) target a module layout the current `../kata` no longer ships. The package
therefore resolves plugin symbols lazily, so importing the protocol layer works regardless. SN22-3
rewrites the plugin against the current core and the protocol above; its two test modules are
skipped until then. The `pyproject.toml` already declares the `kata.subnets` entry point.
