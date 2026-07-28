# The SN22 sealed-room runner = the generic kata-tee-runner base + the SN22 profile.
#
# The generic base provides sealing, the inference gateway, attestation, and the room server. This
# layer adds only SN22's execution profile.
#
# It is a SEPARATE image from kata-sn60-runner, and that is the point rather than an accident: the
# attested measurement covers the whole image, so one image carrying both profiles would mean an
# SN22 room attests to code that can also run SN60's project images, and the measurement would stop
# identifying which subnet's room a validator is talking to. Two images means each lane's
# measurement allowlist accepts exactly one measurement.
#
# A fail-closed placeholder keeps accidental direct builds from selecting a mutable base. build.sh
# requires the caller to replace it with an approved immutable kata-tee-runner digest.
ARG BASE=example.invalid/kata-tee-runner@sha256:0000000000000000000000000000000000000000000000000000000000000000
FROM ${BASE}
WORKDIR /app

# SN22-specific execution: the profile that runs one search task in the agent image and collects
# the agent's single JSON answer.
COPY tee_profile.py .

# The lane package AND the pinned upstream tree. The runner does not merely orchestrate: since
# Phase F it EXECUTES the real vendored SN22 validator to produce each pool score, so the tree it
# scores with has to be inside the attested measurement rather than fetched at run time.
COPY kata_sn22 /app/kata_sn22
COPY kata_sn22_upstream /app/kata_sn22/upstream

# The four packages upstream's own scoring semantics depend on, and no more.
#
#   pydantic  validates the protocol models the scorer reads
#   numpy     IS the arithmetic
#   pytz      decides what "within date range" means for the date-range penalty
#   tiktoken  counts the tokens the streaming penalty charges for
#
# Everything else upstream imports -- bittensor, wandb, aiohttp, openai, apify_client -- is
# infrastructure, replaced by kata_sn22.upstream_runtime. A chain client has no business inside an
# attested room, and the measurement covers every byte of what is here.
#
# `--require-hashes` is deliberate. An attested measurement covers every byte of the image, so
# "whatever the index served that day" is not a thing a room can be built from: the pinned file
# below names an exact artefact for each package, transitive dependencies included.
COPY requirements-upstream.txt /tmp/requirements-upstream.txt
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements-upstream.txt \
 && rm -f /tmp/requirements-upstream.txt \
 && python -c "import pydantic, numpy, pytz, tiktoken"

# tiktoken downloads its BPE table on first use, and a sealed room has no egress. Baked in here so
# the streaming penalty cannot fail on a duel for want of a file.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken
RUN mkdir -p /opt/tiktoken \
 && python -c "import tiktoken; tiktoken.get_encoding('o200k_base').encode('warm')" \
 && test -n "$(ls -A /opt/tiktoken)"

ENV PYTHONPATH=/app
# Tell the generic server which profile to load.
ENV KATA_TEE_PROFILE=tee_profile:Sn22TeeProfile
# Production gunicorn CMD is inherited from the base.
