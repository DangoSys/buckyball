#!/bin/bash

set -e

ROOT="$(dirname "$(realpath "$0")")/.."

cd $ROOT
git submodule update --init

# Install spike and integerate bebop into spike
cd $ROOT/thirdparty/riscv-isa-sim
mkdir -p build
cd build
../configure --prefix=$RISCV --with-boost=no --with-boost-asio=no --with-boost-regex=no
make -j
make install

cd $ROOT/customext
mkdir -p build
cd build
cmake ..
make -j
make install

# Install gem5 and integerate bebop into gem5
# sudo apt install build-essential git m4 scons zlib1g zlib1g-dev \
#     libprotobuf-dev protobuf-compiler libprotoc-dev libgoogle-perftools-dev \
#     python3-dev python-is-python3 libboost-all-dev pkg-config gcc-10 g++-10 \
#     python3-tk clang-format-18
cd $ROOT/thirdparty/gem5
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
scons build/RISCV/gem5.opt -j $(nproc) LIBS="absl_log_internal_check_op \
absl_log_internal_conditions \
absl_log_internal_message \
absl_base \
absl_raw_logging_internal \
absl_strings \
absl_throw_delegate \
absl_string_view \
absl_spinlock_wait \
absl_int128 \
absl_log_severity"
