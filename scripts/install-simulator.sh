#!/bin/bash

BB_DIR=$(git rev-parse --show-toplevel)
RISCV_DIR=$RISCV

source $BB_DIR/scripts/env-exit.sh

# -------------------- func-sim --------------------
cd $BB_DIR/sims/func-sim

mkdir -p build
cd build
../configure --prefix=$RISCV_DIR
make -j$(nproc)
make install

ln -s spike bspike
# -------------------- func-sim --------------------