#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
HOST_ROOT=${SCRIPT_DIR}/..
GEM5_ROOT=${SCRIPT_DIR}/gem5
HOST_BUILD=${HOST_ROOT}/build
IPC_BUILD_LIB=${HOST_BUILD}/ipc
IPC_INCLUDE=${HOST_ROOT}/ipc/include

cmake -S ${HOST_ROOT} -B ${HOST_BUILD}
cmake --build ${HOST_BUILD} --target bebop_ipc -j$(nproc)


# Install gem5 and integerate bebop into gem5
# sudo apt install build-essential git m4 scons zlib1g zlib1g-dev \
#     libprotobuf-dev protobuf-compiler libprotoc-dev libgoogle-perftools-dev \
#     python3-dev python-is-python3 libboost-all-dev pkg-config gcc-10 g++-10 \
#     python3-tk clang-format-18
# cd $ROOT/thirdparty/gem5
# export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
# scons build/RISCV/gem5.opt -j $(nproc) LIBS="absl_log_internal_check_op \

cd ${GEM5_ROOT}
# Apply the patch to gem5
git apply ${SCRIPT_DIR}/bebop.patch
# We need to update the patch in this way if we make changes to gem5
# git add -A && git diff --cached > ../bebop.patch

# Build gem5
export PKG_CONFIG_PATH=${CONDA_PREFIX:-}/lib/pkgconfig:${PKG_CONFIG_PATH:-}
BEBOP_IPC_LIB=${IPC_BUILD_LIB}/libbebop_ipc.a \
  BEBOP_IPC_INCLUDE=${IPC_INCLUDE} \
  scons build/RISCV/gem5.opt -j $(nproc) \
  LIBS="absl_log_internal_check_op \
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
