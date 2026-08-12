#!/usr/bin/env bash

set -e
set -o pipefail

BBDIR=$(git rev-parse --show-toplevel)

begin_step() {
  thisStepNum=$1
  thisStepDesc=$2

  local BLUE='\033[0;34m'
  local GREEN='\033[0;32m'
  local YELLOW='\033[1;33m'
  local NC='\033[0m'

  echo -e "${BLUE} ========================================================================="
  echo -e "${GREEN} ==== BUCKYBALL DOWNLOAD STEP ${YELLOW}$thisStepNum${GREEN}: ${YELLOW}$thisStepDesc${GREEN} "
  echo -e "${BLUE} ========================================================================="
  echo -e "${NC}"
}

begin_step "0-1" "submodules init"
cd ${BBDIR}
git submodule update --init --progress \
  arch/thirdparty/chipyard \
  arch/thirdparty/rocket-chip \
  arch/thirdparty/boom \
  arch/thirdparty/rocket-chip-inclusive-cache \
  arch/thirdparty/berkeley-hardfloat \
  bb-tests/workloads/lib/kernel \
  bbdev \
  bebop \
  compiler/thirdparty/buddy-mlir \
  docs \
  verify \
  thirdparty/firesim \
  thirdparty/soc-framework \
  thirdparty/waveform-mcp
git submodule update --init --depth 1 --single-branch --recommend-shallow --progress \
  bb-tests/thirdparty/linux \
  bb-tests/thirdparty/opensbi
git -C ${BBDIR}/thirdparty/firesim submodule update --init --progress \
  sim/rocket-chip \
  sim/berkeley-hardfloat \
  sim/diplomacy \
  sim/cde

# Rocket-Chip, BOOM, Inclusive Cache, and Hardfloat are provided by
# Buckyball's arch/thirdparty submodules above rather than Chipyard copies.
CYDIR=${BBDIR}/arch/thirdparty/chipyard
git -C ${CYDIR} submodule update --init --progress fpga/fpga-shells
git -C ${CYDIR} submodule update --init --progress \
  generators/diplomacy \
  generators/rocc-acc-utils \
  generators/bar-fetchers \
  generators/testchipip \
  generators/rocket-chip-blocks \
  generators/gemmini
git -C ${CYDIR} submodule update --init --progress tools/stage tools/cde tools/firrtl2 tools/rocket-dsp-utils tools/fixedpoint tools/dsptools
git -C ${CYDIR} submodule update --init --checkout --force tools/stage
git -C ${CYDIR} submodule update --init --checkout --force tools/cde
git -C ${CYDIR} submodule update --init --checkout --force tools/firrtl2
git -C ${CYDIR} submodule update --init --checkout --force tools/rocket-dsp-utils
git -C ${CYDIR} submodule update --init --checkout --force generators/rocc-acc-utils
git -C ${CYDIR} submodule update --init --checkout --force generators/bar-fetchers
git -C ${CYDIR} submodule update --init --checkout --force generators/testchipip
git -C ${CYDIR} submodule update --init --checkout --force generators/rocket-chip-blocks
git -C ${CYDIR} submodule update --init --checkout --force generators/gemmini

begin_step "0-2" "verify chipyard nested pins"
require_chipyard_nested() {
  local rel=$1
  shift
  local expected
  expected=$(git -C ${CYDIR} ls-tree HEAD "${rel}" | awk '{print $3}')
  if [[ -z "${expected}" ]]; then
    echo "error: chipyard does not record gitlink for ${rel}" >&2
    exit 1
  fi
  local actual
  actual=$(git -C ${CYDIR}/${rel} rev-parse HEAD)
  if [[ "${actual}" != "${expected}" ]]; then
    echo "error: ${rel} HEAD ${actual} != chipyard pin ${expected}; re-run nested submodule update" >&2
    exit 1
  fi
  local f
  for f in "$@"; do
    if [[ ! -e "${CYDIR}/${rel}/${f}" ]]; then
      echo "error: ${rel} is missing required file ${f} at ${actual}" >&2
      exit 1
    fi
  done
  echo "ok: ${rel} @ ${actual}"
}

require_chipyard_nested generators/testchipip \
  src/main/scala/ctc/CTC.scala \
  src/main/scala/dram/FastRAM.scala \
  src/main/scala/soc/SubsystemInjector.scala \
  src/main/scala/soc/OffchipBus.scala \
  src/main/scala/serdes/Parameters.scala
require_chipyard_nested generators/rocket-chip-blocks \
  src/main/scala/devices/chiplink/Bundles.scala
require_chipyard_nested generators/gemmini \
  src/main/scala/gemmini/MeshWithDelays.scala
require_chipyard_nested generators/diplomacy
require_chipyard_nested generators/rocc-acc-utils
require_chipyard_nested generators/bar-fetchers

begin_step "0-3" "buddy-mlir llvm init"
git -C ${BBDIR}/compiler/thirdparty/buddy-mlir submodule update --init --depth 1 --single-branch --recommend-shallow --progress llvm
