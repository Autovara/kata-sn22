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
  plugin.py    the SubnetPlugin methods (allowlist env, NOISY/non-cacheable king, stub scorer)
  __init__.py  builds and registers the SN22_DESEARCH_PLUGIN singleton
```

## Not yet wired

This plugin does not currently load against the latest engine. Its imports
(`kata.packages.plugin`, `kata.packages.registry`) target an older module layout that the current
`../kata` no longer ships. Treat this as a reference for the plugin shape, not as a working lane. The
`pyproject.toml` already declares the `kata.subnets` entry point, so once the imports are updated to
the current engine the platform can discover it with no core change.
