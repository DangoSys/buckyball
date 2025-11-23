#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
# HOST_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SPIKE_SRC="${SCRIPT_DIR}/riscv-isa-sim"
SPIKE_BUILD="${SPIKE_SRC}/build"
SPIKE_INSTALL="${SPIKE_SRC}/install"
# HOST_BUILD="${HOST_ROOT}/build"

mkdir -p "${SPIKE_BUILD}"
(
  cd "${SPIKE_BUILD}"
  ../configure --prefix="${SPIKE_INSTALL}" \
    --with-boost=no \
    --with-boost-asio=no \
    --with-boost-regex=no
  make -j$(nproc)
  make install
)

# cmake -S "${HOST_ROOT}" -B "${HOST_BUILD}"
# cmake --build "${HOST_BUILD}" -j$(nproc)
# cmake --install "${HOST_BUILD}"
