#!/bin/bash

set -e

BBDIR=$(git rev-parse --show-toplevel)

# git submodule update --init 
cd ${BBDIR}/thirdparty/chipyard
./build-setup.sh --skip-toolchain --skip-circt
