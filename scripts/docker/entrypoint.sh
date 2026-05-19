#!/usr/bin/env bash
set -euo pipefail

# BBDIR=/buckyball is set in the Docker image Env.
# sourceme.sh sets RISCV, BUDDY_MLIR_BUILD_DIR, PATH, etc.
source "${BBDIR}/sourceme.sh"

cd /workspace/bbdev/api
exec ./node_modules/.bin/motia start
