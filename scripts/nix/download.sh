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
git -C ${BBDIR}/arch/thirdparty/chipyard submodule update --init --progress fpga/fpga-shells
git -C ${BBDIR}/arch/thirdparty/chipyard submodule update --init --progress \
  generators/diplomacy \
  generators/rocc-acc-utils \
  generators/bar-fetchers \
  generators/testchipip \
  generators/rocket-chip-blocks
git -C ${BBDIR}/arch/thirdparty/chipyard submodule update --init --progress tools/stage tools/cde tools/firrtl2 tools/rocket-dsp-utils tools/fixedpoint tools/dsptools
git -C ${BBDIR}/arch/thirdparty/chipyard submodule update --init --checkout --force tools/stage
git -C ${BBDIR}/arch/thirdparty/chipyard submodule update --init --checkout --force tools/cde
git -C ${BBDIR}/arch/thirdparty/chipyard submodule update --init --checkout --force tools/firrtl2
git -C ${BBDIR}/arch/thirdparty/chipyard submodule update --init --checkout --force tools/rocket-dsp-utils
git -C ${BBDIR}/arch/thirdparty/chipyard submodule update --init --checkout --force generators/rocc-acc-utils
git -C ${BBDIR}/arch/thirdparty/chipyard submodule update --init --checkout --force generators/bar-fetchers
git -C ${BBDIR}/arch/thirdparty/chipyard submodule update --init --checkout --force generators/testchipip
git -C ${BBDIR}/arch/thirdparty/chipyard submodule update --init --checkout --force generators/rocket-chip-blocks

begin_step "0-3" "buddy-mlir llvm init"
git -C ${BBDIR}/compiler/thirdparty/buddy-mlir submodule update --init --depth 1 --single-branch --recommend-shallow --progress llvm
