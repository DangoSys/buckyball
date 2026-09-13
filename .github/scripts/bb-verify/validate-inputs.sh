#!/usr/bin/env bash
# Validate dispatch inputs and reset the verdict output dir.
#
# Only the verify/* namespace is dispatchable — anything else (main & co.) is
# rejected here. sha is a race guard: prepare.sh compares it with the actual
# checked-out HEAD inside the flock critical section.
#
# Inputs arrive via env (never ${{ }} interpolation into the script body) and
# are charset-checked before landing in GITHUB_ENV — values written there skip
# shell parsing in later steps, so this step is the injection boundary.
set -euo pipefail

LAYER="${LAYER_INPUT:-merge}"
case "$LAYER" in
  merge|complete) : ;;
  *) echo "::error::layer must be merge or complete, got '$LAYER'"; exit 1 ;;
esac
echo "LAYER=$LAYER" >> "$GITHUB_ENV"

if [[ ! "$REF_INPUT" =~ ^verify/[A-Za-z0-9._-]+$ ]]; then
  echo "::error::ref must match ^verify/[A-Za-z0-9._-]+\$ (only the verify/* namespace is dispatchable), got '$REF_INPUT'"
  exit 1
fi
if [[ ! "$SHA_INPUT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "::error::sha must be 40 lowercase hex chars (compared with git rev-parse output in prepare), got '$SHA_INPUT'"
  exit 1
fi
echo "REF=$REF_INPUT" >> "$GITHUB_ENV"
echo "EXPECTED_SHA=$SHA_INPUT" >> "$GITHUB_ENV"

rm -rf "$VERIFY_OUT"
mkdir -p "$VERIFY_OUT"
echo "ref=$REF_INPUT sha=$SHA_INPUT layer=$LAYER"
