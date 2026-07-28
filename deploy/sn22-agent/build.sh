#!/usr/bin/env bash
# Build (and optionally push) the SN22 agent image — the container one submission runs in.
#
# It is built from the SAME `sn22_relay.py` the local sandbox serves, copied in at build time rather
# than vendored, so the two execution paths cannot drift: if they ever differ, a submission
# calibrated in the sandbox would behave differently in the room, and nothing would say why.
#
# Usage:
#   PYTHON_BASE=python:3.12-slim@sha256:<digest> ./build.sh v1
#   PYTHON_BASE=python:3.12-slim@sha256:<digest> ./build.sh v1 --push
set -euo pipefail

TAG="${1:?usage: ./build.sh <tag> [--push]}"
IMAGE="${IMAGE:-docker.io/carloscosimano/kata-sn22-agent:${TAG}}"
PYTHON_BASE="${PYTHON_BASE:?set PYTHON_BASE to an immutable Python image digest}"
PLATFORM="${PLATFORM:-linux/amd64}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIENT="${HERE}/../../kata_sn22/relay_client.py"

case "$PYTHON_BASE" in
  *@sha256:*) ;;
  *) echo "ERROR: PYTHON_BASE must be an immutable image digest (...@sha256:...)" >&2; exit 1 ;;
esac

case "${2:-}" in
  ""|--push) ;;
  *) echo "ERROR: usage: ./build.sh <tag> [--push]" >&2; exit 1 ;;
esac

case "$PLATFORM" in
  linux/amd64) ;;
  *) echo "ERROR: PLATFORM must be linux/amd64 for Phala rooms" >&2; exit 1 ;;
esac

[ -f "$CLIENT" ] || { echo "ERROR: relay client not found at $CLIENT" >&2; exit 1; }

# Staged into the build context under the name the agent imports. Copied, never symlinked: a build
# context follows no links, and a silently-empty file would produce an image whose agents all fail
# at import time.
trap 'rm -f "${HERE}/sn22_relay.py"' EXIT
cp "$CLIENT" "${HERE}/sn22_relay.py"

build_args=(
  --platform "$PLATFORM"
  --build-arg "PYTHON_BASE=$PYTHON_BASE"
  -t "$IMAGE"
)
if [ "${2:-}" = "--push" ]; then
  docker buildx build "${build_args[@]}" --push "$HERE"
else
  docker buildx build "${build_args[@]}" --load "$HERE"
fi
echo "built $IMAGE (relay client from $CLIENT)"
