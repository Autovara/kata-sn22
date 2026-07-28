#!/usr/bin/env bash
# Build (and optionally push) the SN22 agent image — the container one submission runs in.
#
# The SDK and the relay client are copied in from this repository at build time rather than vendored
# beside the Dockerfile, so the code a submission imports in the room is byte-identical to the code
# it imports in the sandbox. If they ever differed, an agent calibrated locally would behave
# differently in a real duel and nothing would say why.
#
# Every input is pinned by digest: the base image by `PYTHON_BASE`, and the result is reported as
# `image@sha256:...` because that is what the room's profile demands and what the attested
# measurement covers. A tag is a pointer somebody else can move, and this image executes code
# written by a stranger.
#
# Usage:
#   PYTHON_BASE=python:3.13-slim@sha256:<digest> ./build.sh v2
#   PYTHON_BASE=python:3.13-slim@sha256:<digest> ./build.sh v2 --push
set -euo pipefail

TAG="${1:?usage: ./build.sh <tag> [--push]}"
IMAGE="${IMAGE:-docker.io/carloscosimano/kata-sn22-agent:${TAG}}"
PYTHON_BASE="${PYTHON_BASE:?set PYTHON_BASE to an immutable Python image digest}"
PLATFORM="${PLATFORM:-linux/amd64}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${HERE}/../.."
SDK="${REPO}/kata_sn22_sdk"
CLIENT="${REPO}/kata_sn22/relay_client.py"
ARTIFACTS="${ARTIFACTS:-${HERE}/artifacts}"

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

[ -d "$SDK" ]    || { echo "ERROR: SDK not found at $SDK" >&2; exit 1; }
[ -f "$CLIENT" ] || { echo "ERROR: relay client not found at $CLIENT" >&2; exit 1; }

# Staged into the build context under the names the image expects. Copied, never symlinked: a build
# context follows no links, and a silently-empty file would produce an image whose agents all fail
# at import time — which reads as a bad submission rather than a bad build.
trap 'rm -rf "${HERE}/kata_sn22_sdk" "${HERE}/sn22_relay.py"' EXIT
rm -rf "${HERE}/kata_sn22_sdk"
cp -R "$SDK" "${HERE}/kata_sn22_sdk"
find "${HERE}/kata_sn22_sdk" -name '__pycache__' -type d -prune -exec rm -rf {} +
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

mkdir -p "$ARTIFACTS"

# The digest is the only identity worth recording. KATA_SN22_TEE_AGENT_IMAGE must be set to exactly
# this; the profile refuses anything that is not `...@sha256:...`.
if [ "${2:-}" = "--push" ]; then
  DIGEST="$(docker buildx imagetools inspect "$IMAGE" --format '{{.Manifest.Digest}}' 2>/dev/null || true)"
else
  DIGEST="$(docker image inspect "$IMAGE" --format '{{.Id}}' 2>/dev/null || true)"
fi

# --- software bill of materials and vulnerability report ------------------------------------------
#
# Generated at build time and kept beside the image, because "what is in the image that runs
# untrusted code" is a question that gets asked after an incident — when rebuilding it to find out
# is exactly the thing you cannot do. Both tools are optional, but a missing scanner must not
# silently become a missing report: their absence is warned about and recorded in the summary.
SBOM="${ARTIFACTS}/sbom-${TAG}.spdx.json"
VULN="${ARTIFACTS}/vulnerabilities-${TAG}.json"
SBOM_STATUS="skipped (syft not installed)"
VULN_STATUS="skipped (grype not installed)"

if command -v syft >/dev/null 2>&1; then
  syft "$IMAGE" -o spdx-json > "$SBOM"
  SBOM_STATUS="generated"
else
  echo "WARNING: syft not installed; no SBOM was generated for $IMAGE" >&2
fi

if command -v grype >/dev/null 2>&1; then
  # Never fail the build on a finding: the operator decides whether a CVE in a base image blocks a
  # release, and a build that refuses outright is a build that gets run with the check disabled.
  grype "$IMAGE" -o json > "$VULN" || true
  VULN_STATUS="generated"
else
  echo "WARNING: grype not installed; no vulnerability report was generated for $IMAGE" >&2
fi

cat > "${ARTIFACTS}/build-${TAG}.json" <<JSON
{
  "image": "${IMAGE}",
  "digest": "${DIGEST}",
  "python_base": "${PYTHON_BASE}",
  "platform": "${PLATFORM}",
  "sdk_source": "${SDK}",
  "relay_client_source": "${CLIENT}",
  "sbom": "${SBOM_STATUS}",
  "vulnerability_report": "${VULN_STATUS}"
}
JSON

echo "built $IMAGE"
echo "  digest: ${DIGEST:-<unknown>}"
echo "  sbom:   ${SBOM_STATUS}"
echo "  vulns:  ${VULN_STATUS}"
echo "  summary written to ${ARTIFACTS}/build-${TAG}.json"
