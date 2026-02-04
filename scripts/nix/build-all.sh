#!/usr/bin/env bash

# exit script if any command fails
set -e
set -o pipefail

BBDIR=$(git rev-parse --show-toplevel)

source ${BBDIR}/scripts/utils.sh

usage() {
  echo "Usage: ${0} [OPTIONS] "
  echo ""
  echo "Helper script to fully initialize repository that wraps other scripts."
  echo "By default it initializes/installs things in the following order:"
  echo "   1. Compiler (buddy-mlir) pre-compile sources"
  echo "   2. bb-tests (workloads) pre-compile sources"
  echo "   3. Install document management system"
  echo "   4. Install workflow management system"
  echo "   5. Install pre-commit hooks"
  echo ""
  echo "**See below for options to skip parts of the setup. Skipping parts of the setup is not guaranteed to be tested/working.**"
  echo ""
  echo "Options"
  echo "  --help -h     : Display this message"
  echo "  --verbose -v  : Verbose printout"
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
git submodule update --init
cd ${BBDIR}/arch/thirdparty/chipyard && git submodule update --init generators/* tools/*

begin_step "0-2" "Nix environment setup"
cd ${BBDIR}
nix build

if [ "${INSTALL_IN_NIX}" = "0" ]; then
  SKIP_ARGS=""
  for skip in "${SKIP_LIST[@]}"; do
    SKIP_ARGS="${SKIP_ARGS} -s ${skip}"
  done
  exec nix develop --command bash ${BBDIR}/scripts/nix/build-all.sh --install-in-nix ${SKIP_ARGS} ${VERBOSE_FLAG}
fi

if run_step "1"; then
  begin_step "1" "riscv-tools setup"
  cd ${BBDIR}/thirdparty/libgloss
  mkdir -p build && cd build
  CC=riscv64-unknown-elf-gcc ../configure \
    --prefix=${RISCV}/lib \
    --host=riscv64-unknown-elf
  make
  make install
  # INSTALL_DIR=${RISCV}/lib
  # mkdir -p ${INSTALL_DIR}/lib
  # find . -name "libgloss_htif.a" | while read -r lib; do
  #   subdir=$(dirname "$lib" | sed 's|^\./||')
  #   [ "$subdir" = "build" ] && subdir=""
  #   dest=${INSTALL_DIR}/lib/${subdir}
  #   mkdir -p "$dest"
  #   cp "$lib" "$dest/"
  #   cp ../util/htif_nano.specs ../util/htif.ld ../util/htif.specs \
  #      ../util/htif_wrap.specs ../util/htif_argv.specs "$dest/" 2>/dev/null || true
  # done
fi

if run_step "2"; then
  begin_step "2" "Compiler (buddy-mlir) pre-compile sources"
  cd ${BBDIR}/compiler
	git submodule update --init llvm

	mkdir -p llvm/build && cd llvm/build
	cmake -G Ninja ../llvm \
			-DLLVM_ENABLE_PROJECTS="mlir;clang" \
			-DLLVM_TARGETS_TO_BUILD="host;RISCV" \
			-DLLVM_ENABLE_ASSERTIONS=ON \
			-DCMAKE_BUILD_TYPE=RELEASE \
			-DMLIR_ENABLE_BINDINGS_PYTHON=ON \
			-DPython3_EXECUTABLE=$(which python3)
	ninja #check-mlir check-clang

	cd ${BBDIR}/compiler
	mkdir -p build && cd build
	cmake -G Ninja .. \
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
  begin_step "3" "Install bebop"
  # ${BBDIR}/scripts/install-bebop.sh
  echo "bebop is not installed"
fi

if run_step "4"; then
  begin_step "4" "bb-tests (workloads) pre-compile sources"
  cd ${BBDIR}/bb-tests
  mkdir -p build && cd build
  cmake -G Ninja ..
  ninja
fi

if run_step "5"; then
  begin_step "5" "Install requirements for sardine"
  npm install --prefix ${BBDIR}/bb-tests/sardine allure-commandline
fi


if run_step "6"; then
  begin_step "6" "Install document management system"
  mdbook-mermaid install ${BBDIR}/docs/bb-note/
fi

if run_step "7"; then
  begin_step "7" "Init workflow management system"
  cd ${BBDIR}/workflow
  export USE_SYSTEMD=no
  npm init -y
  npm install motia@0.13.0-beta.161
  npx motia create -t python

  cd ${BBDIR}/workflow/steps && rm *.{py,json} || true
  cd ${BBDIR}/workflow/steps && rm -r src/ || true
  cd ${BBDIR}/workflow/steps && rm -r petstore/ || true
  cd ${BBDIR}/workflow && rm -r src/ || true
  cd ${BBDIR}/workflow && rm -r tutorial/ || true
  cd ${BBDIR}/workflow && rm *.{md,tsx,rdb} || true
fi

if run_step "8"; then
  begin_step "8" "Install pre-commit hooks"
  pre-commit install
fi

begin_step "END" "Setup completed successfully!"
