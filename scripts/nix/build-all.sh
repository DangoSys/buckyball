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
  echo "   6. bebop build"
  echo "   7. verify build"
  echo "   8. pre-commit hooks installation"
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

${BBDIR}/scripts/nix/download.sh

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

  bbdev config --install
fi

if run_step "2"; then
  begin_step "2" "Compiler installation"
  cd ${BBDIR}
  bbdev compiler --build '--chip toy'
fi

if run_step "3"; then
  begin_step "3" "RTL source pre-compile"
  bbdev verilator --verilog '--chip toy'
fi

if run_step "4"; then
  begin_step "4" "bb-tests pre-compile sources"
  bbdev workload --build '--chip toy'
fi

if run_step "5"; then
  begin_step "5" "waveform-mcp build"
  cd ${BBDIR}/thirdparty/waveform-mcp
  cargo build --release
fi

if run_step "6"; then
  begin_step "6" "bebop build"
  cd ${BBDIR}/bebop
  nix build
  nix develop -c echo "bebop built successfully"

  # check if spike source is downloaded, sometimes it's affected by network and will fail
  if [ ! -f ${BBDIR}/bebop/src/nodes/bemu/native/spike/configure.ac ]; then
    echo "ERROR: Spike source is missing: ${BBDIR}/bebop/src/nodes/bemu/native/spike/configure.ac" >&2
    exit 1
  fi
fi

if run_step "7"; then
  begin_step "7" "verify build"
  cd ${BBDIR}/verify
  nix develop -c echo "verify built successfully"
fi

if run_step "8"; then
  begin_step "8" "pre-commit hooks installation"
  cd ${BBDIR}
  pre-commit install
  # Replace with wrapper so git commit gets nix env (result/bin in PATH)
  cp "${BBDIR}/scripts/pre-commit-hook.sh" "${BBDIR}/.git/hooks/pre-commit"
fi

begin_step "END" "Setup completed successfully!"
