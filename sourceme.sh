#!/usr/bin/env bash
# Source this file to add result/bin to PATH (requires 'nix build' first).
# This file is used to source the environment variables when you enter the
# buckyball environment ('nix develop' or just get environment variables).

BBDIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
RESULT_PATH="${BBDIR}/result"

# if [ ! -d "$RESULT_PATH" ]; then
#   echo "Warning: result not found at $RESULT_PATH. Run 'nix build' first." >&2
#   return 1 2>/dev/null || exit 1
# fi

#===----------------------------------------------------------------------------===
# Source each submodule's ShellHooks
#===----------------------------------------------------------------------------===
# source "${BBDIR}/bbdev/nix/init.sh"

#===----------------------------------------------------------------------------===
# Source Environment Variables
#===----------------------------------------------------------------------------===
export BUDDY_MLIR_BUILD_ROOT="${BBDIR}/compiler/thirdparty/buddy-mlir/build"
export LLVM_MLIR_BUILD_DIR="${BBDIR}/compiler/thirdparty/buddy-mlir/llvm/build"
export PYTHONPATH="${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${PYTHONPATH}"
export RISCV="${BBDIR}/result"
export PATH="${BBDIR}/thirdparty/libgloss/install/lib:$PATH"
# Per-chip buddy-opt lives at compiler/thirdparty/buddy-mlir/build/<chip>/bin
export PATH="${BUDDY_MLIR_BUILD_ROOT}/bin:${PATH}"

# Optional interactive compiler session. Set before sourcing, e.g.:
#   export BUCKYBALL_COMPILER_CHIP=toy
# bbdev compiler/workload inject BUDDY_MLIR_BUILD_DIR per subprocess.
if [ -n "${BUCKYBALL_COMPILER_CHIP:-}" ]; then
  export BUDDY_MLIR_BUILD_DIR="${BUDDY_MLIR_BUILD_ROOT}/${BUCKYBALL_COMPILER_CHIP}"
  export BUDDY_BINARY_DIR="${BUDDY_MLIR_BUILD_DIR}/bin"
  if [ ! -d "${BUDDY_MLIR_BUILD_DIR}" ]; then
    echo "ERROR: missing compiler build for chip ${BUCKYBALL_COMPILER_CHIP}: ${BUDDY_MLIR_BUILD_DIR}" >&2
    return 1 2>/dev/null || exit 1
  fi
  export PYTHONPATH="${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}"
  export PATH="${BUDDY_BINARY_DIR}:${PATH}"
fi

#===----------------------------------------------------------------------------===
# Export each submodule's PATH
#===----------------------------------------------------------------------------===
export PATH="${RESULT_PATH}/riscv64-unknown-elf/lib:${PATH}"
export PATH="${RESULT_PATH}/bin:${PATH}"

# bbdev CLI and Python utils
export PATH="${BBDIR}/bbdev:${PATH}"
export PYTHONPATH="${BBDIR}/bbdev/api:${PYTHONPATH}"


# firesim manager
export PATH="${BBDIR}/arch/thirdparty/chipyard/sims/firesim/deploy:${PATH}"
