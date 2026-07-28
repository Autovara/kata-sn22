# SN22 sealed-room runner

Three images, built in this order. Each is pinned to the previous one by **immutable digest** —
never a tag, because a tag is a pointer someone else can move, and these execute untrusted code
inside an attested VM.

```
python@sha256:…
  └─ kata-tee-runner:<tag>        the shared, subnet-blind base (../../../kata-tee-runner)
       └─ kata-sn22-runner:<tag>  this directory: base + tee_profile.py
            runs ↓
          kata-sn22-agent:<tag>   ../sn22-agent: the container ONE submission runs in
```

## Why SN22 and SN60 are separate images

The attested measurement covers the whole image. One image carrying both profiles would mean this
room attests to code that can also run SN60's project images — and the measurement would no longer
tell a validator which subnet's room it is talking to. Separate images means SN22's measurement
allowlist accepts exactly one measurement, and an SN60 room can never satisfy it.

## Build

```bash
# 1. the shared base (once, in kata-tee-runner/)
PYTHON_BASE=python:3.12-slim@sha256:… ./build.sh v1 --push

# 2. the agent image — the container a submission runs in
PYTHON_BASE=python:3.12-slim@sha256:… ../sn22-agent/build.sh v1 --push

# 3. this runner
BASE=docker.io/<org>/kata-tee-runner@sha256:… ./build.sh v1 --push
```

Every script refuses a mutable base and refuses any platform but `linux/amd64` (Phala rooms are
amd64 whatever the operator's laptop is).

## Deployment inputs

| Variable | Set on | Meaning |
|---|---|---|
| `KATA_SN22_TEE_AGENT_IMAGE` | the room | the **agent** image digest from step 2 |
| `KATA_SN22_TEE_AGENT_MEMORY` / `_CPUS` | the room | ceilings for one agent container (default 2g / 2) |
| `KATA_INFERENCE_GATEWAY_PROVIDER_ROUTES_JSON` | the room | the allowlisted provider routes the miner's key may reach |
| `KATA_SN22_ROOM_URL` | the lane | where this room answers |
| `KATA_SN22_ROOM_MEASUREMENTS` | the lane | the measurement of the image built above — **this** is what stops the lane talking to an SN60 room, or an unattested one |

The SN22 agent helper sends the versioned relay request
`{"op":"search","query":"…","limit":N}` and expects a JSON object with a `results` array. The
configured provider route must implement that contract, either directly or through a reviewed
adapter. A raw OpenAI-compatible chat endpoint does **not** implement it merely because the generic
gateway can forward bytes there. Treat the paid canary as failed unless both contestants retrieve
live results through the configured route.

## Who pays for what

The miner funds its own search and inference: its credential is sealed to its exact bundle,
decrypted only inside the room, and reaches the provider through the signed per-job gateway route.
The validator handles ciphertext only.

The lane pays for **verification** — the page fetches, the Apify re-scrapes and the LLM judging that
decide whether the miner's answer was real. That happens on the lane, not in this room.
