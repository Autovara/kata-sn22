# Running the SN22 lane

Everything an operator has to do that the repository cannot do for itself, and why each step exists.

If you read nothing else: **the validator holds no paid SN22 credential.** Every provider call in a
round is funded by the contestant whose answers it produces. An operator who provisions an SN22
provider key has not helped the lane — they have created a fallback that lets a contestant who
stopped paying keep being evaluated at the validator's expense, which is the one thing the sealed
credential set exists to prevent.

---

## 1. What you must supply, once

Five values, none of which a subnet's repository can know. `kata-subnets-deploy` refuses to render a
unit without all five, and validates the shape of each.

| Value | What it is | Shape |
|---|---|---|
| `KATA_SN22_TEE_AGENT_IMAGE` | the container one submission runs in | `image@sha256:<64 hex>` |
| `KATA_SN22_TEE_RUNNER_IMAGE` | the trusted runner: room server + scorer | `image@sha256:<64 hex>` |
| `KATA_SN22_ROOM_MEASUREMENT` | the only attested room this lane accepts | 64 lowercase hex |
| `KATA_SN22_SCORER_POLICY_HASH` | what the scores mean | 64 lowercase hex |
| `KATA_SN22_ROUTE_POLICY_HASH` | which provider routes produced them | 64 lowercase hex |

Digests, not tags. A tag is a pointer somebody else can move, and both of these images either run
untrusted code or produce the score.

The installer rejects a placeholder, a mutable tag, a malformed digest and a mistyped hash. All four
are the same mistake — a deployment that cannot name what it is running — and every one of them
otherwise surfaces inside a sealed room, on a duel, reading like a broken lane.

---

## 2. Capture the question pool

**The lane cannot run a production round until you do this.** `kata_sn22/datasets/` ships only a
labelled `development` pool, and a production epoch refuses it by kind.

```bash
cd /srv/kata-sn22
uv run --extra snapshot python tools/snapshot_questions.py --out production
git add kata_sn22/datasets/production.jsonl kata_sn22/datasets/production.meta.json
git commit -m "chore(sn22): snapshot the upstream question pool"
```

Needs network; runs on your machine, never in a room. Every manifest records the pool's SHA-256, so
*"what were the two contestants asked"* has an answer that does not depend on when you look.

Upstream falls back to an LLM call when its dataset is unavailable — that call spends the
*validator's* money, and the validator has none. So there is no fallback here: a round that cannot
find its rows **fails before the duel**, while nobody has spent anything.

Re-snapshot only deliberately. It changes the question pool between one round and the next.

---

## 3. Build the two images

```bash
cd deploy/sn22-agent
PYTHON_BASE=python@sha256:<digest> ./build.sh v2 --push

cd ../sn22-runner
BASE=<registry>/kata-tee-runner@sha256:<digest> ./build.sh v2 --push
```

Both print the digest to put in the settings above.

**The agent image is built to contain nothing worth reaching.** `pip`, `ensurepip`, `setuptools`,
`apt`, `dpkg`, `curl`, `wget`, `git` and `ssh` are deleted, and the build **fails** if any survive.
An untrusted agent with a package manager is one network path away from running code that was never
reviewed, and the attested measurement would still be the approved one.

**The runner image carries the pinned upstream tree and exactly four third-party packages** —
`pydantic`, `numpy`, `pytz`, `tiktoken` — installed with `--require-hashes`. Those four are what
upstream's own scoring semantics depend on. `tiktoken`'s BPE table is baked in, because it downloads
on first use and a sealed room has no egress.

An SBOM and a vulnerability report are written beside each agent build if `syft` and `grype` are
installed. If they are not, the build says so loudly and records `"skipped"` in the summary — a
missing scanner must not become a missing report nobody noticed.

---

## 4. Deploy the rooms and record the measurement

Two Phala VMs, one per lane, from the runner image above. See `PHALA-DEPLOYMENT.md` for the compose
and provisioning detail.

After the room is up, read its measurement from `/pubkey` and put it in
`KATA_SN22_ROOM_MEASUREMENT`. Publish the same value to miners: it is what their sealing tool checks
before it will encrypt anything, and it is the only thing standing between a miner and sealing their
four keys to a room somebody else controls.

**Rebuilding either image changes the measurement.** That is the point — the measurement covers
every byte — but it means a rebuild invalidates every credential already sealed. Announce it, give
miners time to reseal, and expect a round of `credential_failure` from anyone who did not.

---

## 5. Verify the policy hashes

```bash
cd /srv/kata-sn22
uv run python -c "from kata_sn22.scorer_policy import policy_hash; print(policy_hash())"
```

Put that in `KATA_SN22_SCORER_POLICY_HASH`. It covers the upstream commit, the judge model and its
temperature, every judge prompt byte for byte, the pool weights, the exponents, the deep-sample rate
and the provider routes.

If it does not match what a room reports, the two are not scoring the same way and the duel is
refused rather than compared. That is a real refusal to expect after any deliberate change to the
scoring policy — and a surprising one is worth investigating before you override it.

---

## 6. Check it before you enable it

The whole suite runs offline. Provider calls are faked at the transport, the room is injected, and
the question pool is the labelled `development` one:

```bash
cd /srv/kata-sn22
uv sync --extra dev --extra upstream
uv run pytest -q
```

The `upstream` extra matters. Without it, every test that exercises the **real vendored upstream
scorer** skips rather than silently falling back to the port — and a suite that "passed" against the
port would be telling you nothing about the thing this lane actually scores with.

Then check the two images and the artifacts:

```bash
KATA_SN22_AGENT_IMAGE=<your agent digest> uv run pytest tests/test_sn22_agent_image.py
cd /srv/kata-subnets-deploy && uv run python installer/generate_lane_artifacts.py --check
```

The first runs the real image under the room's production restrictions — non-root, read-only,
no capabilities, no egress — and confirms it carries no installer. The second reports drift between
what the registry generates and what is checked in; it should print `OK`.

---

## 7. Canary, then enable

Keep `KATA_PROMOTION_DISABLED` set for the lane's first paid rounds. The lane scores normally and
publishes normally, and hands out no crown. The published result records what *would* have happened,
so the canary proves something about promotion rather than merely about scoring.

Watch for:

- **every contestant scoring zero.** Almost always a credential problem, not sixty bad agents. Check
  the published `failure_category`.
- **duels deferring.** A room, a quote or a provider — never a contestant. Deferred duels re-run;
  they do not decide anything.
- **the King never being challenged successfully.** Expected early. The score is normalised across
  both contestants, so a strong King is genuinely harder to beat.

Then enable the timer.

---

## 8. Reading a round

| What you see | Meaning | Who acts |
|---|---|---|
| `credential_missing`, `credential_invalid`, `credential_unauthorized`, `credential_payment_required`, `credential_rate_limited`, `credential_expired`, `credential_insufficient` | that contestant's own key failed; it scored zero | the miner |
| `infrastructure` | a room, a quote, or a confirmed provider outage | you; the duel re-runs |
| duel deferred | the two sides were not comparable, or a report was missing | you |
| `beats_king_pairwise: false` | the challenger genuinely lost | nobody |

Failure categories are published by **category only**. You will never see a provider's own error
text in a result file or a PR comment: a provider rejecting a request quotes that request, and the
request carried a contestant's key.

---

## 9. What you will not find, and why

**No provider budgets.** `KATA_SUBNET_BUDGET_INFERENCE_CALLS`, `DATA_API_CALLS` and `SCRAPE_UNITS`
are gone from this lane. They metered what the validator spent, and it spends nothing. A budget for
spending that cannot happen is a number that would only ever mislead whoever read it — an operator
seeing an inference-call cap would reasonably conclude the validator funds inference.

`KATA_SUBNET_BUDGET_TEE_RUNS` and `KATA_SUBNET_MAX_RUNTIME` remain. Those are safety ceilings on the
room, not cost controls on a provider.

**No promotion margins.** `KATA_PROMOTE_MARGINS` belonged to a seven-signal lexicographic
comparator that production no longer uses. A duel is decided on one number, `sn22_combined_score`,
from the pinned upstream's own aggregation. A margin on that would be Kata overriding upstream's
arithmetic about what counts as better. A tie keeps the King; strictly greater promotes.

**No `task_count` knob.** It is 60, and the lane refuses anything else in production rather than
rounding it up. Upstream deep-scores 20% of a pool and *drops* a contestant with fewer than three
deep samples, so 15 per pool is the smallest epoch that can be scored at all — the previously
deployed 8 would have zeroed every contestant for a reason unrelated to its answers, and it would
have looked like the agents were bad.

**No SN22 provider keys in the child environment.** The bot scopes paid credentials by lane: SN60
gets its scoring key, SN22 gets none. If you find yourself wanting to add one, re-read the top of
this page.
