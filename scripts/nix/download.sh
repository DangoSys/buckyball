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

begin_step "0-2" "submodules init"
cd ${BBDIR}
git submodule update --init --progress \
  arch/thirdparty/gemmini \
  arch/thirdparty/rocket-chip \
  arch/thirdparty/berkeley-hardfloat \
  bb-tests/workloads/lib/kernel \
  bbdev \
  bebop \
  stack \
  docs \
  verify \
  thirdparty/firesim \
  thirdparty/soc-framework \
  thirdparty/waveform-mcp \
  .agents/skills \
  .agents/mcps
git -C ${BBDIR}/arch/thirdparty/rocket-chip submodule update --init --progress dependencies/cde dependencies/diplomacy
git submodule update --init --depth 1 --single-branch --recommend-shallow --progress \
  bb-tests/thirdparty/linux \
  bb-tests/thirdparty/opensbi
begin_step "0-4" "buddy-mlir llvm init"
git -C ${BBDIR}/stack submodule update --init --progress compiler/thirdparty/buddy-mlir
git -C ${BBDIR}/stack/compiler/thirdparty/buddy-mlir submodule update --init --depth 1 --single-branch --recommend-shallow --progress llvm
