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

## Reading order

| Start here | For |
|---|---|
| [`kata/docs/SN22-PROTOCOL.md`](../kata/docs/SN22-PROTOCOL.md) | writing a submission |
| [`SN22-OPERATOR-GUIDE.md`](SN22-OPERATOR-GUIDE.md) | running the lane |
| [`docs/DECISION-bittensor-not-in-the-room.md`](docs/DECISION-bittensor-not-in-the-room.md) | why the images contain what they contain |
| `../KATA-ARCHITECTURE.md` §6.3 | how this lane fits the platform |

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
replaced. See `docs/DECISION-bittensor-not-in-the-room.md`.

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
