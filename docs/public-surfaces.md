# `kata-sn22` public surfaces

A subnet plugin. Its consumers are the engine (which loads it by entry point), the operator (who
configures it), the sealed room (which runs its scorer), and every miner (who writes to its SDK).

## Entry point

`kata.subnets` → `sn22 = "kata_sn22:SN22_DESEARCH_PLUGIN"`

The distribution name (`kata-sn22`), the entry-point name (`sn22`) and the object path are all
declared in `kata-subnets-deploy`'s registry and asserted by `verify-resident-env`. Changing any of
the three requires a registry change and a reinstall.

## Plugin methods the engine calls

`sample_problems`, `run_challenge`, `beats_king`, `preflight`, `capacity_estimate`,
`environment_spec`, `static_screen`, `challenge_result_json`, `scoring_profile`

## Miner-facing SDK (`kata_sn22_sdk`)

Exported: `Agent`, `AiSearchResult`, `AiSearchSynapse`, `XSearchResult`, `XSearchSynapse`,
`Synapse`, `BrokerClient`, `BrokerError`, `SdkError`, `Emit`, `Limits`, `ResultType`, `SearchMode`,
`SearchType`, `ScraperTextRole`, `cite`, `in_sealed_room`, `synapse_from_input`,
`AGENT_OPERATIONS`, `PROTOCOL_VERSION`

Every miner writes against this. It is the widest-blast-radius surface in the repository, and it
ships inside an image with **no package installer**, so it may import only the standard library.

## Scoring identity

`scorer_policy.policy_hash()` and `scorer_policy.route_policy_hash()` are declared by the operator as
`KATA_SN22_SCORER_POLICY_HASH` / `KATA_SN22_ROUTE_POLICY_HASH`, and `preflight()` refuses a value
this checkout cannot reproduce.

These must **not** move into this repository's own settings: a hash the plugin both declares and
verifies compares the plugin to itself, always passes, and proves nothing. The check exists because
the operator approved a specific policy.

## Operator-supplied values (`kata-subnets-deploy` refuses to render a unit without all five)

`KATA_SN22_TEE_AGENT_IMAGE`, `KATA_SN22_TEE_RUNNER_IMAGE`, `KATA_SN22_ROOM_MEASUREMENT`,
`KATA_SN22_SCORER_POLICY_HASH`, `KATA_SN22_ROUTE_POLICY_HASH`

## Environment variables

Backend: `KATA_SN22_EXECUTION_BACKEND`, `KATA_SN22_VERIFICATION_MODE`, `KATA_SN22_REQUIRE_SANDBOX`,
`KATA_SN22_BWRAP`, `KATA_SN22_SUDO`

Room: `KATA_SN22_ROOM_URL`, `_MEASUREMENTS`, `_HTTP_TIMEOUT_SECONDS`, `_REQUEST_LIFETIME_SECONDS`,
`_MAX_ATTEMPTS`, `_RETRY_BASE_SECONDS`, `KATA_ROOM_AUTH_SECRET`, `KATA_SN22_PCCS_URL`,
`KATA_SN22_ALLOW_INSECURE_ROOM_URL`

Agent container: `KATA_SN22_TEE_AGENT_IMAGE`, `_CPUS`, `_MEMORY`

Inside the room only: `SN22_BROKER_URL`, `SN22_BROKER_CAPABILITY`, `SN22_TASK_ID`,
`SN22_PROTOCOL_VERSION`, `SN22_RELAY_ENDPOINT`, `SN22_RELAY_CAPABILITY`

**No validator-side provider credential exists.** `SCRAPINGDOG_API_KEY`, `APIFY_API_KEY`,
`OPENAI_API_KEY` and `CHUTES_API_KEY` appear only as sealed, contestant-supplied names.

## Subnet-owned settings

`deploy/settings.json` — `challenge_config` (`task_count` 60, `max_results` 10), `lane_env` budgets,
`unit_params` (`timeout_start_sec`, `round_gap_sec`, `requires_docker`). Read by the installer, which
validates every value against an allowlist and ceilings.

## Data and formats

`kata_sn22/datasets/` (question pool, `x_seeds.json`), `upstream/` (pinned upstream tree, digest- and
parity-gated), the version-2 credential/report schemas, and the challenge result JSON the bot parses.
