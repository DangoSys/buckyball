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
export BUDDY_MLIR_BUILD_DIR="${BBDIR}/compiler/build"
export LLVM_MLIR_BUILD_DIR="${BBDIR}/compiler/llvm/build"
export PYTHONPATH="${BBDIR}/compiler/llvm/build/tools/mlir/python_packages/mlir_core:${BBDIR}/compiler/build/python_packages:$PYTHONPATH"
export BUDDY_BINARY_DIR="${BBDIR}/compiler/build/bin"
export RISCV="${BBDIR}/result"
export PATH="${BBDIR}/thirdparty/libgloss/install/lib:$PATH"
export PATH="${BUDDY_BINARY_DIR}:${PATH}"

#===----------------------------------------------------------------------------===
# Export each submodule's PATH
#===----------------------------------------------------------------------------===
export PATH="${RESULT_PATH}/riscv64-unknown-elf/lib:${PATH}"
export PATH="${RESULT_PATH}/bin:${PATH}"

# bbdev CLI and Python utils
export PATH="${BBDIR}/bbdev:${PATH}"
export PYTHONPATH="${BBDIR}/bbdev/api:${PYTHONPATH}"

# sardine
export PYTHONPATH="${BBDIR}/lib/python3.13/site-packages:${PYTHONPATH}"
