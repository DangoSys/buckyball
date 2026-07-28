#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'usage: %s IMAGE\n' "$0" >&2
  exit 1
fi

IMAGE="$1"

echo "Starting ${IMAGE}"
docker run \
  --rm \
  --interactive \
  --tty \
  --workdir /workspace/buckyball \
  "${IMAGE}"
