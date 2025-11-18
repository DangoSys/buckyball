#!/bin/bash

set -e

ROOT="$(dirname "$(realpath "$0")")/.."

cd $ROOT
git submodule update --init

cd $ROOT/customext
mkdir -p build
cd build
cmake ..
make -j
make install

cd $ROOT/thirdparty/riscv-isa-sim
mkdir -p build
cd build
../configure --prefix=$RISCV --with-boost=no --with-boost-asio=no --with-boost-regex=no
make -j
make install
