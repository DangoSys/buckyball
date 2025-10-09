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
  echo "   1. Setup workflow management system"
  echo "   2. Buckyball submodules"
  echo "   3. Toolchain installation"
  echo "   4. Compiler (buddy-mlir) pre-compile sources"
  echo "   5. bb-tests (workloads) pre-compile sources"
  echo "   6. Install Chipyard and Firesim"
  echo "   7. Buckyball pre-compile sources"
  echo "   8. Setup document management system"
  echo "   9. Install func-sim"
  echo "   10. Runs repository clean-up"
  echo ""
  echo "**See below for options to skip parts of the setup. Skipping parts of the setup is not guaranteed to be tested/working.**"
  echo ""
  echo "Options"
  echo "  --help -h     : Display this message"
  echo "  --verbose -v  : Verbose printout"
  echo "  --skip -s N   : Skip step N in the list above. Use multiple times to skip multiple steps ('-s N -s M ...')."
  echo "  --admin       : Add this option to install the admin tools (You dont need do this)."
  echo "  --conda-env-name <name> : Add this option to specify the conda environment name. Default is buckyball."

  exit "$1"
}

SKIP_LIST=()
VERBOSE_FLAG=""
ADMIN_MODE=false
CONDA_ENV_NAME="buckyball"

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
    --admin)
      ADMIN_MODE=true ;;
    --conda-env-name)
      shift
      CONDA_ENV_NAME=${1} ;;
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

if run_step "0"; then
  begin_step "0" "init env.sh"
  replace_content ${BBDIR}/env.sh base-conda-setup "source $(conda info --base)/etc/profile.d/conda.sh"
fi

if run_step "1"; then
  begin_step "1" "submodules init"
  git submodule update --init
  replace_content ${BBDIR}/arch/thirdparty/chipyard/env.sh base-conda-setup "source $(conda info --base)/etc/profile.d/conda.sh"
fi

# setup and install chipyard environment
if run_step "2"; then
  begin_step "2" "Chipyard environment setup"
  cd ${BBDIR}/arch/thirdparty/chipyard && ./build-setup.sh --conda-env-name ${CONDA_ENV_NAME}
  cp ${BBDIR}/arch/thirdparty/chipyard/env.sh ${BBDIR}/env.sh
  replace_content ${BBDIR}/env.sh build-setup-conda "conda activate ${CONDA_ENV_NAME}
source ${BBDIR}/arch/thirdparty/chipyard/scripts/fix-open-files.sh"
  replace_content ${BBDIR}/env.sh bb-dir-helper "BB_DIR=${BBDIR}"
fi

if run_step "3"; then
  begin_step "3" "Compiler (buddy-mlir) pre-compile sources"
  cd ${BBDIR}
  source ${BBDIR}/env.sh
  ./scripts/install-compiler.sh
fi

if run_step "4"; then
  begin_step "4" "Install simulator"
  # source ${BBDIR}/env.sh
  # ${BBDIR}/scripts/install-simulator.sh
  echo "Simulator installation is skipped"
fi

if run_step "5"; then
  begin_step "5" "bb-tests (workloads) pre-compile sources"
  source ${BBDIR}/env.sh
  cd ${BBDIR}/bb-tests
  mkdir -p build && cd build
  cmake -G Ninja ../
  ninja -j$(nproc)
fi

if run_step "6"; then
  begin_step "6" "Install requirements for sardine"
  source ${BBDIR}/env.sh
  pip install -r ${BBDIR}/bb-tests/sardine/requirements.txt
  npm install --prefix ${BBDIR}/bb-tests/sardine allure-commandline
fi


if run_step "7"; then
  begin_step "7" "Install document management system"
  source ${BBDIR}/env.sh
  ${BBDIR}/scripts/install-doc.sh
fi

if run_step "8"; then
  begin_step "8" "Init workflow management system"
  source ${BBDIR}/env.sh
  ${BBDIR}/scripts/install-workflow.sh
fi

if run_step "9"; then
  begin_step "9" "Install mill"
  source ${BBDIR}/env.sh
  cd ${BBDIR}/tools/mill
  ./install-mill.sh
fi

if run_step "10"; then
  begin_step "10" "pre-compile buckyball arch code"
  source ${BBDIR}/env.sh
  cd ${BBDIR}/
  bbdev verilator --verilog
fi

if run_step "11"; then
  begin_step "11" "Install pre-commit"
  source ${BBDIR}/env.sh
  pip install pre-commit
  cd ${BBDIR}
  pre-commit install
fi
