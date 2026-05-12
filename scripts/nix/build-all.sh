#!/usr/bin/env bash

# exit script if any command fails
set -e
set -o pipefail

BBDIR=$(git rev-parse --show-toplevel)

usage() {
  echo "Usage: ${0} [OPTIONS] "
  echo ""
  echo "Helper script to fully initialize repository that wraps other scripts."
  echo "By default it initializes/installs things in the following order:"
  echo "   1. bbdev install"
  echo "   2. Compiler installation"
  echo "   3. RTL pre-compile sources"
  echo "   4. bb-tests pre-compile sources"
  echo "   5. waveform-mcp build"
  echo "   6. pre-commit hooks installation"
  echo ""
  echo "**See below for options to skip parts of the setup. Skipping parts of the setup is not guaranteed to be tested/working.**"
  echo ""
  echo "Options"
  echo "  --help -h     : Display this message"
  echo "  --skip -s N   : Skip step N in the list above. Use multiple times to skip multiple steps ('-s N -s M ...')."
  exit "$1"
}

SKIP_LIST=()
VERBOSE_FLAG=""
INSTALL_IN_NIX=0

while [ "$1" != "" ];
do
  case $1 in
    -h | --help )
      usage 3 ;;
    --verbose | -v)
      VERBOSE_FLAG=$1
      set -x ;;
    --skip | -s)
      shift
      SKIP_LIST+=(${1}) ;;
    --install-in-nix)
      INSTALL_IN_NIX=1 ;;
    * )
      echo "Error: invalid option $1" >&2
      usage 1 ;;
  esac
  shift
done

# return true if the arg is not found in the SKIP_LIST
run_step() {
  local value=$1
  [[ ! " ${SKIP_LIST[*]} " =~ " ${value} " ]]
}

function begin_step
{
  thisStepNum=$1;
  thisStepDesc=$2;

  # Color codes
  local BLUE='\033[0;34m'
  local GREEN='\033[0;32m'
  local YELLOW='\033[1;33m'
  local NC='\033[0m' # No Color

  echo -e "${BLUE} ========================================================================="
  echo -e "${GREEN} ==== BUCKYBALL SETUP STEP ${YELLOW}$thisStepNum${GREEN}: ${YELLOW}$thisStepDesc${GREEN} "
  echo -e "${BLUE} ========================================================================="
  echo -e "${NC}"
}

begin_step "0-1" "submodules init"
# git submodule update --init --progress --jobs 8
cd ${BBDIR}
git submodule update --init --progress \
  arch/thirdparty/chipyard \
  bb-tests/workloads/lib/kernel \
  bbdev \
  bebop \
  compiler/thirdparty/buddy-mlir \
  docs \
  thirdparty/waveform-mcp

# I dont know why below is need for chipyard submodules, but it is
git -C ${BBDIR}/arch/thirdparty/chipyard submodule update --init --progress fpga/fpga-shells
git -C ${BBDIR}/arch/thirdparty/chipyard submodule update --init --progress generators/*
git -C ${BBDIR}/arch/thirdparty/chipyard submodule update --init --progress tools/*
git -C ${BBDIR}/arch/thirdparty/chipyard submodule update --init --progress sims/firesim
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
##########################################

begin_step "0-2" "Nix environment setup"
cd ${BBDIR}
nix build
if [ "${INSTALL_IN_NIX}" != "1" ]; then
  SKIP_ARGS=""
  for skip in "${SKIP_LIST[@]}"; do
    SKIP_ARGS="${SKIP_ARGS} -s ${skip}"
  done
  exec nix develop --command bash ${BBDIR}/scripts/nix/build-all.sh --install-in-nix ${SKIP_ARGS} ${VERBOSE_FLAG}
fi

if run_step "1"; then
  begin_step "1" "bbdev install"

  echo "Installing bbdev Python dependencies..."
  cd ${BBDIR}/bbdev/api
  uv venv .venv --python python3 --seed
  uv pip install --python .venv/bin/python -r pyproject.toml
fi

if run_step "2"; then
  begin_step "2" "Compiler installation"
  cd ${BBDIR}/compiler/thirdparty/buddy-mlir
  git submodule update --init --progress llvm

  mkdir -p llvm/build && cd llvm/build
  cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python3)
  ninja #check-mlir check-clang

  cd ${BBDIR}/compiler/thirdparty/buddy-mlir
  mkdir -p build && cd build
  cmake -G Ninja .. \
    -DBUDDY_EXTERNAL_DIALECTS_DIR=${BBDIR}/compiler/src \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
    -DPython3_EXECUTABLE=$(which python3) \
    -DPython_EXECUTABLE=$(which python3) \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  ninja # check-buddy
fi

if run_step "3"; then
  begin_step "3" "arch pre-compile sources"

  # Pre-compile chipyard dependencies in order
  # These must be compiled separately before arch/buckyball can use them
  cd ${BBDIR}/arch/thirdparty/chipyard

  # 1. cde provides org.chipsalliance.cde.config (required by rocketchip and firesim)
  # sbt -J-Xms512m -J-Xmx4g -J-XX:+UseG1GC "cde/compile"

  # 2. firrtl2 generates ANTLR parsers (required by chipyard)
  sbt -J-Xms512m -J-Xmx4g -J-XX:+UseG1GC "firrtl2/compile"

  # 3. chipyard includes tools/stage sources (required by firesim)
  # sbt -J-Xms512m -J-Xmx4g -J-XX:+UseG1GC "chipyard/compile"

  cd ${BBDIR}/arch
  bbdev verilator --verilog '--config sims.verilator.BuckyballToyVerilatorConfig'
fi

if run_step "4"; then
  begin_step "4" "bb-tests pre-compile sources"
  bbdev workload --build
fi

if run_step "5"; then
  begin_step "5" "waveform-mcp build"
  cd ${BBDIR}/thirdparty/waveform-mcp
  cargo build --release
fi

if run_step "6"; then
  begin_step "6" "pre-commit hooks installation"
  cd ${BBDIR}
  pre-commit install
  # Replace with wrapper so git commit gets nix env (result/bin in PATH)
  cp "${BBDIR}/scripts/pre-commit-hook.sh" "${BBDIR}/.git/hooks/pre-commit"
fi

begin_step "END" "Setup completed successfully!"
