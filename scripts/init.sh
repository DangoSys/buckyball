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
    echo "   1. Setup workflow management system"
    echo "   2. Buckyball submodules"
    echo "   3. Toolchain installation"
    echo "   4. Compiler (buddy-mlir) pre-compile sources"
    echo "   5. bb-tests (workloads) pre-compile sources"
    echo "   6. Install Chipyard and Firesim"
    echo "   7. Buckyball pre-compile sources"
    echo "   8. Setup document management system"
    echo "   9. Runs repository clean-up"
    echo ""
    echo "**See below for options to skip parts of the setup. Skipping parts of the setup is not guaranteed to be tested/working.**"
    echo ""
    echo "Options"
    echo "  --help -h     : Display this message"
    echo "  --verbose -v  : Verbose printout"
    echo "  --skip -s N   : Skip step N in the list above. Use multiple times to skip multiple steps ('-s N -s M ...')."
    echo "  --admin       : Add this option to install the admin tools (You dont need do this)."

    exit "$1"
}

SKIP_LIST=()
VERBOSE_FLAG=""
ADMIN_MODE=false

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
        * )
            error "invalid option $1"
            usage 1 ;;
    esac
    shift
done

# return true if the arg is not found in the SKIP_LIST
run_step() {
    local value=$1
    [[ ! " ${SKIP_LIST[*]} " =~ " ${value} " ]]
}
