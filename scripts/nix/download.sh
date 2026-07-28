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
  bb-tests/workloads/lib/kernel \
  bbdev \
  bebop \
  compiler/thirdparty/buddy-mlir \
  docs \
  verify \
  thirdparty/waveform-mcp
git submodule update --init --depth 1 --single-branch --recommend-shallow --progress \
  bb-tests/thirdparty/linux \
  bb-tests/thirdparty/opensbi

# I dont know why below is need for chipyard submodules, but it is
git -C ${BBDIR}/arch/thirdparty/chipyard submodule update --init --progress fpga/fpga-shells
git -C ${BBDIR}/arch/thirdparty/chipyard submodule update --init --progress generators/*
git -C ${BBDIR}/arch/thirdparty/chipyard submodule update --init --progress sims/firesim
git -C ${BBDIR}/arch/thirdparty/chipyard submodule update --init --progress tools/stage tools/cde tools/firrtl2 tools/rocket-dsp-utils tools/fixedpoint tools/dsptools
git -C ${BBDIR}/arch/thirdparty/chipyard submodule update --init --checkout --force tools/stage
git -C ${BBDIR}/arch/thirdparty/chipyard submodule update --init --checkout --force tools/cde
git -C ${BBDIR}/arch/thirdparty/chipyard submodule update --init --checkout --force tools/firrtl2
git -C ${BBDIR}/arch/thirdparty/chipyard submodule update --init --checkout --force tools/rocket-dsp-utils
git -C ${BBDIR}/arch/thirdparty/chipyard submodule update --init --checkout --force generators/rocc-acc-utils
git -C ${BBDIR}/arch/thirdparty/chipyard submodule update --init --checkout --force generators/bar-fetchers

# FireSim sim/ has its own submodules (cde, rocket-chip, diplomacy, berkeley-hardfloat)
rm -rf ${BBDIR}/arch/thirdparty/chipyard/sims/firesim/sim/cde \
       ${BBDIR}/arch/thirdparty/chipyard/sims/firesim/sim/rocket-chip \
       ${BBDIR}/arch/thirdparty/chipyard/sims/firesim/sim/diplomacy \
       ${BBDIR}/arch/thirdparty/chipyard/sims/firesim/sim/berkeley-hardfloat
git -C ${BBDIR}/arch/thirdparty/chipyard/sims/firesim submodule update --init --progress \
  sim/cde sim/rocket-chip sim/diplomacy sim/berkeley-hardfloat

begin_step "0-3" "buddy-mlir llvm init"
git -C ${BBDIR}/compiler/thirdparty/buddy-mlir submodule update --init --depth 1 --single-branch --recommend-shallow --progress llvm
