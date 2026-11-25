#!/usr/bin/env bash
# Source this file to add result/bin to PATH (requires 'nix build' first).
# This file is used to source the environment variables when you enter the
# buckyball environment ('nix develop' or just get environment variables).

BBDIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
RESULT_PATH="${BBDIR}/result"

#===----------------------------------------------------------------------------===
# Source Environment Variables
#===----------------------------------------------------------------------------===
export BUDDY_MLIR_BUILD_ROOT="${BBDIR}/stack/thirdparty/buddy-mlir/build"
export LLVM_MLIR_BUILD_DIR="${BBDIR}/stack/thirdparty/buddy-mlir/llvm/build"
export PYTHONPATH="${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${PYTHONPATH}"
export RISCV="${BBDIR}/result"
export PATH="${BUDDY_MLIR_BUILD_ROOT}/bin:${PATH}"

#===----------------------------------------------------------------------------===
# Export each submodule's PATH
#===----------------------------------------------------------------------------===
export PATH="${RESULT_PATH}/riscv64-unknown-elf/lib:${PATH}"
export PATH="${RESULT_PATH}/bin:${PATH}"

# bbdev CLI and Python utils
export PATH="${BBDIR}/bbdev:${PATH}"
export PYTHONPATH="${BBDIR}/bbdev/api:${PYTHONPATH}"
export PYTHONPATH="${BBDIR}/thirdparty/soc-framework/ip:${PYTHONPATH}"

# FireSim manager
export PATH="${BBDIR}/thirdparty/firesim/deploy:${PATH}"
