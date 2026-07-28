#!/usr/bin/env bash
# Build (and optionally push) the SN22 sealed-room runner image = kata-tee-runner base + the SN22
# profile. The generic base already contains the reviewed, subnet-neutral inference gateway. Phala
# rooms use amd64, so the build platform is explicit even on a different host architecture.
#
# Usage:
#   BASE=registry/kata-tee-runner@sha256:<digest> ./build.sh v9
#   BASE=registry/kata-tee-runner@sha256:<digest> ./build.sh v9 --push
set -euo pipefail

TAG="${1:?usage: ./build.sh <tag> [--push]}"
BASE="${BASE:?set BASE to the immutable kata-tee-runner image digest}"
IMAGE="${IMAGE:-docker.io/carloscosimano/kata-sn22-runner:${TAG}}"
PLATFORM="${PLATFORM:-linux/amd64}"
case "$BASE" in
  *@sha256:*) ;;
  *) echo "ERROR: BASE must be an immutable image digest (...@sha256:...)" >&2; exit 1 ;;
esac

case "${2:-}" in
  ""|--push) ;;
  *) echo "ERROR: usage: ./build.sh <tag> [--push]" >&2; exit 1 ;;
esac

case "$PLATFORM" in
  linux/amd64) ;;
  *) echo "ERROR: PLATFORM must be linux/amd64 for Phala rooms" >&2; exit 1 ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${HERE}/../.."

# Staged into the build context from this repository, so the code the room SCORES with is the code
# in this checkout. Since Phase F the runner executes the real vendored SN22 validator, so the
# pinned upstream tree has to be inside the attested image rather than fetched at run time.
[ -d "${REPO}/kata_sn22" ]   || { echo "ERROR: kata_sn22 not found" >&2; exit 1; }
[ -d "${REPO}/upstream" ]    || { echo "ERROR: pinned upstream tree not found" >&2; exit 1; }
[ -f "${HERE}/requirements-upstream.txt" ] || {
  echo "ERROR: requirements-upstream.txt missing; the image would install unpinned packages" >&2
  exit 1
}

trap 'rm -rf "${HERE}/kata_sn22" "${HERE}/kata_sn22_upstream"' EXIT
rm -rf "${HERE}/kata_sn22" "${HERE}/kata_sn22_upstream"
cp -R "${REPO}/kata_sn22" "${HERE}/kata_sn22"
cp -R "${REPO}/upstream"  "${HERE}/kata_sn22_upstream"
# The vendored tree already lives under kata_sn22/upstream at run time; staging it separately keeps
# the build context explicit about what is being attested.
rm -rf "${HERE}/kata_sn22/upstream"
find "${HERE}/kata_sn22" "${HERE}/kata_sn22_upstream" -name '__pycache__' -type d -prune \
  -exec rm -rf {} +

build_args=(
  --platform "$PLATFORM"
  --build-arg "BASE=$BASE"
  -t "$IMAGE"
)
if [ "${2:-}" = "--push" ]; then
  docker buildx build "${build_args[@]}" --push .
else
  docker buildx build "${build_args[@]}" --load .
fi
echo "built $IMAGE (FROM $BASE)"
echo "  scores with the pinned upstream at ${REPO}/upstream"
