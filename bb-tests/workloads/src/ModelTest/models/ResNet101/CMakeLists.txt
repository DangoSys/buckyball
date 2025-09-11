set(RESNET101_TOSA_PIPELINE "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-tensor),func.func(tosa-to-linalg),func.func(tosa-to-arith))")
set(BUDDY_MLIR_BINARY_DIR ${BUDDY_MLIR_BUILD_DIR}/bin)
set(CMAKE_CXX_COMPILER riscv64-unknown-linux-gnu-g++)
set(LLVM_MLIR_EXECUTION_ENGINE_DIR ${BUDDY_MLIR_BUILD_DIR}/../llvm/mlir/include/mlir/ExecutionEngine)
set(GEMMINI_CONFIG "dim=4 acc_t=f32 elem_t=f32")

if (NOT DEFINED ENV{RISCV})
  message(FATAL_ERROR "Can't find RISCV environment variable(missing: RISCV_TOOLCHAIN)")
endif()

add_custom_command(
OUTPUT
  resnet-101.o
COMMAND
  ${BUDDY_MLIR_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/ResNet101.mlir
  -pass-pipeline="${RESNET101_TOSA_PIPELINE}" |
  ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
  -linalg-bufferize
  -convert-linalg-to-gemmini="acc_t=f32"
  -llvm-request-c-wrappers
  -empty-tensor-to-alloc-tensor
  -arith-bufferize
  -tensor-bufferize
  -func-bufferize
  -convert-linalg-to-loops
  -convert-vector-to-scf
  -linalg-bufferize
  -buffer-deallocation
  -convert-vector-to-scf
  -convert-scf-to-cf
  -convert-vector-to-llvm
  -arith-expand
  -convert-math-to-llvm
  -convert-linalg-to-loops
  -expand-strided-metadata
  -lower-gemmini=${GEMMINI_CONFIG}
  -reconcile-unrealized-casts |
  ${BUDDY_MLIR_BINARY_DIR}/buddy-translate -buddy-to-llvmir |
  ${BUDDY_MLIR_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64
    -mattr=+buddyext,+D -float-abi=hard -o resnet-101.o
)

add_library(ResNet101 STATIC resnet-101.o)
# TODO: Add cross-compiled CRunnerUtils library.
add_library(CRunnerUtils STATIC CRunnerUtils.cpp)
target_include_directories(CRunnerUtils
  PRIVATE
  ${LLVM_MLIR_EXECUTION_ENGINE_DIR}
)
set_target_properties(ResNet101 PROPERTIES LINKER_LANGUAGE CXX)
add_executable(Gemmini-ResNet-101 Main.cpp)
target_include_directories(Gemmini-ResNet-101
  PRIVATE
  images
  include
)
target_link_libraries(Gemmini-ResNet-101
  -static
  ResNet101
  CRunnerUtils
)
