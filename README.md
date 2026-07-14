# kata-sn22

The **SN22 (Desearch)** subnet plugin for the [Kata](../kata) competition platform — a live
search-quality competition. This is a **skeleton**: the plugin wiring is complete (it runs a full
King-of-the-Hill round through the generic orchestrator), but the scorer is a stub. Real Desearch
validation (live X/Twitter + web retrieval, LLM-judged relevance) plugs in behind these same
methods later, with no platform changes.

It plugs into the platform via the `kata.subnets` entry point (`pyproject.toml`); the Kata engine
discovers and loads it with no code change. Install it into the engine's environment
(`uv pip install -e .`) and the `sn22__desearch` lane becomes available.

```
kata_sn22/
  plugin.py    implements SubnetPlugin (allowlist env, NOISY/non-cacheable king, stub scorer)
  __init__.py  builds + registers the SN22_DESEARCH_PLUGIN singleton
```

Depends on `kata` for the `SubnetPlugin` contract and the registry. See `../KATA-REDESIGN-PLAN.md`.
