# kata-sn22

The **SN22 (Desearch)** subnet plugin for the [Kata](../kata) competition platform: a paired
king-versus-challenger search lane that scores with the **real pinned upstream validator**, executed.

Two facts do most of the explaining, and everything else follows from them.

**The miner funds its own evaluation.** Every paid SN22 call — the agent's web search, its X search,
its final summary, and the evaluator's page fetches, tweet re-scrapes and judge calls — is made
inside a sealed room with credentials the *contestant* sealed to its own bundle. The validator holds
no paid SN22 credential at all, so there is no fallback: a contestant that cannot fund its own
evaluation scores **zero**. The reigning King keeps paying to defend its crown.

**The scorer is upstream's, not ours.** `AdvancedScraperValidator`, `XScraperValidator`,
`compute_rewards_and_penalties`, `QueryScheduler._score_one_type` and `combine_pool_scores` are the
vendored files at the pinned commit, run as-is. Kata supplies the questions, the two contestants and
the transports; it computes no score of its own.

> [!TIP]
> **Values you need to seal your credentials:**
> - **Room URL** — `https://fdb462709fa90d9fcb5659d51d21a36cc9600e0c-8080.dstack-pha-prod9.phala.network`
> - **Measurement** — `acc309c703fecdff898553b7fd3e838044a1d06e0fc528cc8ba051e6badabe8f`
> - **Credential profile** — `sn22-miner-funded-v1`
> - **Providers you must fund** — `scrapingdog`, `apify`, `openai`, `chutes`
>
> Verify the measurement before sealing: it is the single value standing between you and sealing
> keys to a room somebody else controls. A room redeploy changes both its sealing key and,
> potentially, its approved measurement, so do not reuse values from an earlier deployment.

## Reading order

| Start here | For |
|---|---|
| [The SN22 submission protocol](#the-sn22-submission-protocol-version-2) | writing a submission |
| [`SN22-OPERATOR-GUIDE.md`](SN22-OPERATOR-GUIDE.md) | running the lane |
| [Why bittensor is not in the room](#why-bittensor-is-not-in-the-room) | why the images contain what they contain |

## What a round is

One **epoch**: 60 tasks per contestant, 15 in each of four pools.

| Pool | Tools | Share | Serving budget |
|---|---|---|---|
| `ai_search:fast` | Web Search | 0.54 | 15s |
| `ai_search:balanced` | 50/50 Web / Twitter | 0.18 | 15s |
| `ai_search:deep` | 50/50 Web / Twitter | 0.18 | 30s |
| `x_search` | Basic X search | 0.10 | 15s |

**15 is not a preference.** Upstream deep-scores 20% of a pool and *drops* a contestant with fewer
than three deep samples, so 15 is the smallest pool that can be scored at all. The lane refuses any
other task count in production rather than rounding it up.

Both contestants receive the same manifest, including the same deep-sample ids. The agent is never
told which those are — one that knew would work hardest on exactly those, and the 20% sample would
stop measuring the other 80%.

The duel runs as **eight attested pool jobs**: four per contestant, one contestant's four before the
other's, in a deterministic randomised order. Sixty tasks behind one request is one timeout away
from losing every answer already paid for.

## How a promotion is decided

One number: **`sn22_combined_score`**, from upstream's own `combine_pool_scores` over both
contestants' four pool tuples. Higher wins; a tie keeps the King.

There are no tie-breakers, no indifference bands and no seven-signal ordering. Those were
calibration machinery — see the banner at the top of `kata_sn22/scoring.py`, which is still what the
local sandbox ranks on while a miner iterates, and which cannot decide a promotion.

`combine_pool_scores` normalises each pool across **both** contestants, so the call is made once
with both. Scoring each side alone would give each of them nearly the pool's full share and the
comparison would collapse.

## Failure, and what each kind means

| What happened | Outcome |
|---|---|
| A contestant's own credential is missing, invalid, unauthorized, out of credit, rate-limited or expired | That contestant scores **zero**. The duel still decides. |
| Timeout, malformed output, bad schema, too few results, duplicates | Upstream's own penalties apply. Not automatically zero. |
| Quote verification fails, a room is unreachable, a provider has a confirmed outage, a report is missing | The whole duel **defers**. Nothing is promoted. |

The distinction is the point: a zeroed contestant cannot be un-zeroed, while a deferred duel can be
re-run. Anything ambiguous defers.

A room that cannot run a contestant returns a **quote-bound `credential_failure` report**, not an
HTTP error. A plain 4xx is not evidence — it could come from anywhere on the path, including a host
that would rather one side lost.

## Layout

```
kata_sn22/
  protocol_v2.py         the version-2 task/answer contract and the scoring surface
  epoch_manifest.py      the 60-task epoch: upstream's distribution, Kata's deep samples
  question_pool.py       packaged question rows, and the refusal when they are not real
  scorer_policy.py       every input that decides what a score MEANS, hashed into one identity
  upstream_runtime.py    load the pinned upstream with INFRASTRUCTURE adapted and nothing else
  neuron_adapter.py      the seven-attribute surface upstream reads off a validator
  production_scorer.py   the winner path: real validators, real aggregation, one pool tuple
  paired_scoring.py      eight attested reports in, one promotion decision out
  production_challenge.py  splits an epoch into eight pool jobs and runs the duel
  broker_ops.py          the six reviewed provider operations and their FIXED routes
  credentials_v2.py      the four-provider sealed set and the agent/evaluator split
  report_v2.py           what a sealed room says, and how the host refuses a mismatched duel
  execution/tee_room.py  remote room client and TDX attestation verification
  scoring.py             CALIBRATION ONLY: the old seven signals, for the local sandbox
  upstream_adapter.py    the dependency-free port -- now test/reference evidence, not the winner
kata_sn22_sdk/           what a SUBMISSION imports. Standard library only, no credentials.
upstream/                the complete pinned upstream tree + UPSTREAM_MANIFEST.json
deploy/sn22-agent/       the container one submission runs in
deploy/sn22-runner/      the trusted runner: room server + SN22 profile + the real scorer
tools/
  snapshot_questions.py  operator-run, once: capture the upstream question rows
  vendor_upstream.py     regenerate or verify the upstream manifest
  record_parity.py       execute the pinned upstream and record what it computes
```

## Two images, and why they differ

| | Agent image | Trusted runner |
|---|---|---|
| Runs | the miner's `agent.py` | the room server and the real scorer |
| Carries | Python, `kata_sn22_sdk`, the harness | the lane package, the pinned upstream tree |
| Third-party packages | **none** | `pydantic`, `numpy`, `pytz`, `tiktoken` — and nothing else |
| Package installer | **removed at build time** | present |
| Credentials | **none, and no way to ask for one** | the contestant's four, in memory only |

The agent image has `pip`, `apt`, `curl`, `wget` and `ensurepip` deleted, and the build fails if any
survive. An untrusted agent with a package manager is one network path away from running code that
was never reviewed — and the attested measurement would still be the approved one.

The four packages in the runner are there because upstream's own scoring semantics depend on them:
`pydantic` validates the protocol models, `numpy` *is* the arithmetic, `pytz` decides what "within
date range" means, and `tiktoken` counts the tokens the streaming penalty charges for. Everything
else upstream imports — `bittensor`, `wandb`, `aiohttp`, the provider SDKs — is infrastructure and is
replaced. See [Why bittensor is not in the room](#why-bittensor-is-not-in-the-room).

## Testing locally, with no provider and no room

```bash
uv sync --extra dev --extra upstream
uv run pytest -q
```

The suite runs offline. Provider calls are faked at the transport, the room is injected, and the
question pool is a labelled `development` one that production refuses by kind.

To run the tests that exercise the real vendored upstream you need the `upstream` extra above;
without it those modules skip rather than silently falling back to the port — a version of them that
"passed" against the port would be testing the thing this lane exists to stop using.

To exercise the real agent image:

```bash
cd deploy/sn22-agent
PYTHON_BASE=python@sha256:<digest> IMAGE=kata-sn22-agent:local ./build.sh local
KATA_SN22_AGENT_IMAGE=kata-sn22-agent:local uv run pytest tests/test_sn22_agent_image.py
```

## Before this lane can run a production round

`kata_sn22/datasets/` ships only the labelled `development` question rows, and a production epoch
refuses them by kind. Capture the real ones once, on a machine with network:

```bash
uv run --extra snapshot python tools/snapshot_questions.py --out production
git add kata_sn22/datasets/production.jsonl kata_sn22/datasets/production.meta.json
```

Every manifest records that pool's SHA-256, so "what were the two contestants asked" has an answer
that does not depend on when you look. Re-snapshot only deliberately.

## The SN22 submission protocol, version 2

This is the complete contract between your agent and the lane.

Start by copying the reigning King:
[`kings/sn22__desearch/miner/agent.py`](../kata/kings/sn22__desearch/miner/agent.py).
It answers all four pools, it is documented line by line, and every place it is deliberately lazy is
marked. Beating it should be easy — and it is literally the thing you have to beat, so there is no
separate example that might drift away from what you are actually scored against.

```python
from kata_sn22_sdk import Agent, AiSearchResult, ScraperTextRole, XSearchResult, cite


class Submission(Agent):
    async def smart_scraper(self, synapse, emit):
        results = self.broker.web_search(synapse.prompt, count=synapse.count)
        sources = [cite(r, [r.get("snippet", "")]) for r in results]
        emit(ScraperTextRole.FINAL_SUMMARY, "...")
        return AiSearchResult(search_results=sources)

    async def twitter_search(self, synapse):
        return XSearchResult(results=self.broker.x_search(synapse.query, count=synapse.count))
```

Exactly one `Agent` subclass per submission. Two would make "which one runs" a question answered by
definition order.

---

### 1. You fund your own evaluation

**You seal four provider credentials to your bundle. The validator holds none.**

| Provider | Your agent spends it on | The validator spends it on |
|---|---|---|
| ScrapingDog | web search | independently fetching the pages you returned |
| Apify | X search | independently re-scraping the tweets you returned |
| OpenAI | your final summary (`gpt-4.1-nano`) | nothing |
| Chutes | nothing | the fixed judge that grades you |

All four are required. A production epoch covers all four pools, and demanding the full set at
sealing time is what turns *"you fail on pool three, an hour in"* into *"your submission is rejected
at intake"*.

**An invalid, unauthorized, out-of-credit, expired or rate-limited key scores you ZERO.** Not "less"
— zero, for the whole four-pool epoch. There is no validator key to fall back on, because the
validator does not have one. This applies to the King too: whoever holds the crown keeps paying to
defend it.

Note the asymmetry in the table. Your agent never reaches Chutes — an agent that could call the
judge could grade its own work. The validator never reaches your OpenAI key.

### Sealing your credentials

```bash
uv run --extra seal python kata_seal.py \
  --room https://fdb462709fa90d9fcb5659d51d21a36cc9600e0c-8080.dstack-pha-prod9.phala.network \
  --credential-profile sn22-miner-funded-v1 \
  --providers scrapingdog,apify,openai,chutes \
  --key-env scrapingdog=SCRAPINGDOG_API_KEY \
  --key-env apify=APIFY_API_KEY \
  --key-env openai=OPENAI_API_KEY \
  --key-env chutes=CHUTES_API_KEY \
  --bundle ./my-submission \
  --measurement acc309c703fecdff898553b7fd3e838044a1d06e0fc528cc8ba051e6badabe8f
```

The room URL and measurement above are this lane's current approved values. The tool:

1. fetches the room's public key and **verifies its attestation is a genuine TEE** matching
   `--measurement`, so you cannot be tricked into sealing to a room somebody else controls;
2. binds the credential set to your bundle;
3. seals it and writes `sealed_inference_key` into the bundle, atomically, at `0600`.

Add that file to your PR.

**No key is ever a command-line value** — only the name of an environment variable, a `0600` file, or
a hidden prompt. Command lines are world-readable in the process list.

**The seal covers your whole bundle.** Editing `agent.py` after sealing invalidates it, and the room
will refuse the credential. Reseal after any change.

---

### 2. What you are asked: four pools, 60 tasks

| Pool | Tools | Share of the score | Time budget |
|---|---|---|---|
| `ai_search:fast` | Web Search | 0.54 | 15s |
| `ai_search:balanced` | Web or Twitter | 0.18 | 15s |
| `ai_search:deep` | Web or Twitter | 0.18 | 30s |
| `x_search` | Basic X search | 0.10 | 15s |

15 tasks each. `ai_search:fast` is more than half the score — a submission that only improves the
deep pool is optimising 18% of its result.

`smart_scraper` answers the three AI pools; `twitter_search` answers Basic X. A submission that
implements only one still runs: the other returns an empty answer and loses that pool. That is a
decision, not a saving — every pool is weighted.

Some tasks are **deep-scored** and you are not told which. Working hardest on the ones you could
identify is exactly what the sample exists to prevent.

---

### 3. What your agent can do

`self.broker` is the whole outward surface. Three operations, and **there is no API key on it and no
method that returns one**:

```python
self.broker.web_search(query, count=10)   # -> [ {title, link, snippet}, ... ]
self.broker.x_search(query, count=10)     # -> [ raw tweet objects ]
self.broker.final_summary(messages)       # -> str
self.broker.quota()                       # -> what is left, per operation
```

You name an **operation**, never a URL. There is no parameter for a host, a model or an Apify actor —
not because they are validated away, but because the client cannot express them. The model for
`final_summary` is fixed for everyone: one contestant summarising with a frontier model and another
with a small one would be a comparison of budgets.

Every contestant gets the same per-operation quota. `quota()` is free and does not consume one; an
agent that cannot see its own quota either wastes it or hoards it.

Anything the broker refuses raises `BrokerError` — one class for refused, unreachable and
unintelligible alike. An agent that could tell them apart could map the room's state one probe at a
time, and none of them changes the useful response, which is to answer with what you have.

---

### 4. How a source earns anything: you must cite it

The validator **fetches every link you return, itself**, and then checks two things:

1. every string in `highlights` appears **in order** in its own copy of the page;
2. every string in `highlights` appears **in order** in your own `text`.

A source that fails either is **dropped before it is judged** — it does not score badly, it does not
score at all. `cite()` attaches that evidence:

```python
sources = [cite(result, [result["snippet"]]) for result in results]
```

So:

- **quote contiguous, real spans of the page.** Reassembling a page's vocabulary into a sentence
  nobody wrote fails the ordering check, which is precisely what it is for;
- **say the same thing in `text`.** Pasting real excerpts beside an answer written from somewhere
  else fails the second direction;
- only what survives is sent to the judge, and it is judged on the **excerpts**. You are graded on
  what you proved you read.

A returned source with no evidence is worth nothing, however good it is.

**Tweets are compared field by field** against the validator's own re-scrape. Return them exactly as
the provider gave them; an "improved" tweet scores zero rather than less.

---

### 5. Emit your prose; do not return it

Upstream miners stream, and the streaming penalty counts tokens per emitted chunk.

```python
emit(ScraperTextRole.INTRO, "...")
emit(ScraperTextRole.FINAL_SUMMARY, "...")
```

`texts`, `text_chunks` and `completion` are all derived from what you emitted. `texts["summary"]` is
what the groundedness judge reads, so that is the one worth getting right.

**Emitting nothing for a task that asked for a summary is a full penalty.** Emitting nothing for an
`ONLY_LINKS` task is correct — there is no summary to judge, and writing one spends your money on
something nobody grades.

Write the summary with **bold headers, not `#`** (a `#` header takes the full structure penalty), and
with markdown links to sources you actually returned. The judge follows those links to decide which
source to check each claim against, so a claim cited to the wrong source fails even when it is true.

---

### 6. Return exactly `synapse.count` results

Fewer takes the count penalty. Duplicates take the duplicate penalty, so padding a short list with
copies of what you have is worse than returning less. Malformed results — a missing title, link or
snippet — take the schema penalty in proportion.

For `sort="Latest"` X tasks, results not in descending time order are an **immediate zero** for that
task. The reference agent trusts the provider's order; checking it is the first thing worth adding.

---

### 7. How you win

One number decides the duel: **`sn22_combined_score`**, produced by the pinned upstream's own
aggregation over both contestants' four pool tuples. Higher wins.

**A tie keeps the King.** There are no tie-breakers and no indifference band — matching the King is
not beating it.

Because the score is normalised across both contestants, your number depends on who you are up
against. A strong King is genuinely harder to beat than a weak one; that is the competition working.

---

### 8. Testing before you submit

The local sandbox serves `web_search` over a unix socket, so you can iterate offline. `x_search` and
`final_summary` exist only in the room and say so plainly rather than returning something empty that
would read as a bad answer.

Run the lane's own suite against your bundle to check it loads, frames a valid answer for all four
pools, and survives every upstream penalty — see `kata-sn22/README.md`, *Testing locally*.

---

### 9. Reading a failure

| What you see | What it means | What to do |
|---|---|---|
| `credential_missing` / `credential_invalid` / `credential_unauthorized` | your sealed key is absent, unreadable, or rejected | reseal with a working key |
| `credential_payment_required` / `credential_insufficient` | out of credit at that provider | top up and resubmit |
| `credential_rate_limited` / `credential_expired` | still limited after bounded retries, or expired mid-round | wait or rotate, then resubmit |
| `infrastructure` | not your fault — a room, a quote or a provider outage | nothing; the duel is deferred and re-run |

The first two groups score you **zero**. The last does not score you at all.

You will never see a provider's own error text. A provider rejecting a request quotes that request,
and the request carried your key.

---

### 10. What the image gives you, and what it does not

Your agent runs in a container with:

- Python, `kata_sn22_sdk`, and the harness that imports your `agent.py`;
- **no package installer** — `pip`, `apt`, `curl` and `wget` are deleted at build time;
- **no network except the broker**, on an egress-blocked internal network;
- a read-only bundle, a non-root user, all Linux capabilities dropped, and memory/CPU/PID limits;
- **no provider key anywhere** — not in the environment, argv, `/proc` or any file.

So: standard library plus `kata_sn22_sdk`. Nothing is installed at run time and there is nothing to
install it with. A dependency you need has to be code you ship in your bundle.

If your agent raises, times out or returns the wrong type, the harness writes a well-formed empty
answer and a diagnostic to stderr. You lose that task; you do not break the round. Your stderr is
kept for the operator and is never scored — an agent that could influence its score by what it
printed would be scoring itself.

## Why bittensor is not in the room

**Status:** decided Phase A; **amended in Phase F — read the amendment first.**
**Enforced by:** `tests/test_sn22_room_import_surface.py`.

---

### Amendment (Phase F): this decision now applies to the AGENT image, not to the trusted runner

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

### The question

The pinned upstream is a Bittensor validator. Importing almost anything from it —
`neurons.validators.penalty.count_penalty`, say — pulls in `bittensor`, `wandb`, `openai`,
`aiohttp`, `apify_client` and a chain client, because a penalty module sits in a package whose
siblings talk to all of them. None of that is needed to *compute* a penalty; it is needed to *be* a
validator.

So: does the sealed room have to carry that dependency tree in order to score the way upstream
scores?

### The answer (Phase A, still true of the agent image)

No. And it is not a preference — it is already true, and now checked.

The runtime import closure of the plugin entry point is **18 modules, standard library only**, plus
`kata` (the engine ABI it is loaded through). Verified two ways: by walking the closure statically
(including function-local imports, which are still room dependencies — they just fail later, on a
duel, instead of at start-up), and by building `SN22_DESEARCH_PLUGIN` for real.

The port is what makes this possible. `kata_sn22.upstream_adapter` reimplements the arithmetic with
no third-party dependency at all; `kata_sn22.providers` reaches the four provider APIs over stdlib
`urllib` rather than four SDKs.

### Where the real upstream still runs

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

### Why this is worth a file and a test

The failure mode is the expensive one. A `pydantic` added to a scoring module passes every other
test in this repo, changes the image measurement, and surfaces **inside a sealed TEE room, on a
duel, as an ImportError with no debugger attached**.

This is the recurring shape named in `KATA-ARCHITECTURE.md` §10 rule 5: two components each correct
in isolation, never checked against each other. The runtime is correct. The parity harness is
correct. Nothing checked that the first had not quietly acquired the second's dependencies.

## Public surfaces

A subnet plugin. Its consumers are the engine (which loads it by entry point), the operator (who
configures it), the sealed room (which runs its scorer), and every miner (who writes to its SDK).

### Entry point

`kata.subnets` → `sn22 = "kata_sn22:SN22_DESEARCH_PLUGIN"`

The distribution name (`kata-sn22`), the entry-point name (`sn22`) and the object path are all
declared in `kata-subnets-deploy`'s registry and asserted by `verify-resident-env`. Changing any of
the three requires a registry change and a reinstall.

### Plugin methods the engine calls

`sample_problems`, `run_challenge`, `beats_king`, `preflight`, `capacity_estimate`,
`environment_spec`, `static_screen`, `challenge_result_json`, `scoring_profile`

### Miner-facing SDK (`kata_sn22_sdk`)

Exported: `Agent`, `AiSearchResult`, `AiSearchSynapse`, `XSearchResult`, `XSearchSynapse`,
`Synapse`, `BrokerClient`, `BrokerError`, `SdkError`, `Emit`, `Limits`, `ResultType`, `SearchMode`,
`SearchType`, `ScraperTextRole`, `cite`, `in_sealed_room`, `synapse_from_input`,
`AGENT_OPERATIONS`, `PROTOCOL_VERSION`

Every miner writes against this. It is the widest-blast-radius surface in the repository, and it
ships inside an image with **no package installer**, so it may import only the standard library.

### Scoring identity

`scorer_policy.policy_hash()` and `scorer_policy.route_policy_hash()` are declared by the operator as
`KATA_SN22_SCORER_POLICY_HASH` / `KATA_SN22_ROUTE_POLICY_HASH`, and `preflight()` refuses a value
this checkout cannot reproduce.

These must **not** move into this repository's own settings: a hash the plugin both declares and
verifies compares the plugin to itself, always passes, and proves nothing. The check exists because
the operator approved a specific policy.

### Operator-supplied values (`kata-subnets-deploy` refuses to render a unit without all five)

`KATA_SN22_TEE_AGENT_IMAGE`, `KATA_SN22_TEE_RUNNER_IMAGE`, `KATA_SN22_ROOM_MEASUREMENT`,
`KATA_SN22_SCORER_POLICY_HASH`, `KATA_SN22_ROUTE_POLICY_HASH`

### Environment variables

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

### Subnet-owned settings

`deploy/settings.json` — `challenge_config` (`task_count` 60, `max_results` 10), `lane_env` budgets,
`unit_params` (`timeout_start_sec`, `round_gap_sec`, `requires_docker`). Read by the installer, which
validates every value against an allowlist and ceilings.

### Data and formats

`kata_sn22/datasets/` (question pool, `x_seeds.json`), `upstream/` (pinned upstream tree, digest- and
parity-gated), the version-2 credential/report schemas, and the challenge result JSON the bot parses.
