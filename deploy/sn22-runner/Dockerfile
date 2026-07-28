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
# Tell the generic server which profile to load.
ENV KATA_TEE_PROFILE=tee_profile:Sn22TeeProfile
# CMD is inherited from the base: python -m room.server
