#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
LAST_IMAGE_TAG_FILE="${SCRIPT_DIR}/.last-image-tag"

IMAGE_TAG="buckyball:$(git -C "${REPO_ROOT}" rev-parse --short=12 HEAD)"
GIT_REMOTE=$(git -C "${REPO_ROOT}" remote get-url origin)
GIT_COMMIT=$(git -C "${REPO_ROOT}" rev-parse HEAD)

echo "Building ${IMAGE_TAG}"
docker build \
  --tag "${IMAGE_TAG}" \
  --file "${SCRIPT_DIR}/Dockerfile" \
  --build-arg "GIT_REMOTE=${GIT_REMOTE}" \
  --build-arg "GIT_COMMIT=${GIT_COMMIT}" \
  "${SCRIPT_DIR}"

printf '%s\n' "${IMAGE_TAG}" > "${LAST_IMAGE_TAG_FILE}"
echo "Wrote ${LAST_IMAGE_TAG_FILE}"
