#!/bin/bash

BB_DIR=$(git rev-parse --show-toplevel)
# RISCV_DIR=$RISCV

source $BB_DIR/scripts/utils.sh
source $BB_DIR/scripts/env-exit.sh

# -------------------- func-sim --------------------
cd $BB_DIR/sims/func-sim

mkdir -p build && cd build
../configure --prefix=$BB_DIR/sims/func-sim/bspike
make -j$(nproc)
make install

cd $BB_DIR/sims/func-sim/bspike/bin
ln -s spike bspike
cd $BB_DIR

replace_content ${BB_DIR}/env.sh bspike "export PATH=$BB_DIR/sims/func-sim/bspike/bin:\$PATH"

# -------------------- func-sim --------------------
