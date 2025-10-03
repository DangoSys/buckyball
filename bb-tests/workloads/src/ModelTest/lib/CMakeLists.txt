# ==============================================================================
# ModelTest Common Libraries
# ==============================================================================

# Set RISC-V compiler for all targets in this directory
set(CMAKE_C_COMPILER ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)

set(LLVM_MLIR_EXECUTION_ENGINE_DIR ${BUDDY_MLIR_DIR}/llvm/mlir/include/mlir/ExecutionEngine)
set(DIP_MLIR_PATH ${BUDDY_MLIR_DIR}/frontend/Interfaces/lib/DIP.mlir)

# ==============================================================================
# CRunnerUtils Library - provides memrefCopy and other runtime utilities
# ==============================================================================

# Find CRunnerUtils.cpp in models directory (assuming ResNet18 as template)
set(CRUNNER_UTILS_SRC ${MODELTEST_LIB_DIR}/CRunnerUtils.cpp)

add_library(ModelTestCRunnerUtils STATIC ${CRUNNER_UTILS_SRC})

target_include_directories(ModelTestCRunnerUtils
  PUBLIC
  ${LLVM_MLIR_EXECUTION_ENGINE_DIR}
)

# ==============================================================================
# DIP Library for RISC-V - provides resize and other DIP operations
# ==============================================================================

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/DIP_riscv.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${DIP_MLIR_PATH}
          -lower-dip
          -arith-expand
          -lower-affine
          -convert-scf-to-cf
          -convert-math-to-llvm
          -convert-math-to-libm
          -convert-vector-to-llvm
          -finalize-memref-to-llvm
          -convert-func-to-llvm
          -convert-cf-to-llvm
          -convert-arith-to-llvm
          -reconcile-unrealized-casts |
          ${BUDDY_BINARY_DIR}/buddy-translate --mlir-to-llvmir |
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard
          -o ${CMAKE_CURRENT_BINARY_DIR}/DIP_riscv.o
  DEPENDS ${DIP_MLIR_PATH}
  COMMENT "Building DIP library for RISC-V"
  VERBATIM
)

add_library(ModelTestDIP_riscv STATIC ${CMAKE_CURRENT_BINARY_DIR}/DIP_riscv.o)
set_target_properties(ModelTestDIP_riscv PROPERTIES LINKER_LANGUAGE CXX)

# ==============================================================================
# Export variables for parent CMakeLists
# ==============================================================================

set(MODELTEST_CRUNNER_UTILS_LIB ModelTestCRunnerUtils PARENT_SCOPE)
set(MODELTEST_DIP_RISCV_LIB ModelTestDIP_riscv PARENT_SCOPE)
