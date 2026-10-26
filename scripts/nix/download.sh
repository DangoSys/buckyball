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
git submodule sync
git submodule update --init --progress
git -C ${BBDIR}/stack submodule update --init --progress engine os/linux
git -C ${BBDIR}/stack submodule update --init --depth 1 --single-branch --recommend-shallow --progress \
  os/thirdparty/linux \
  os/thirdparty/busybox \
  os/thirdparty/opensbi
begin_step "0-4" "buddy-mlir llvm init"
git -C ${BBDIR}/stack submodule update --init --progress thirdparty/buddy-mlir
git -C ${BBDIR}/stack/thirdparty/buddy-mlir submodule update --init --depth 1 --single-branch --recommend-shallow --progress llvm
