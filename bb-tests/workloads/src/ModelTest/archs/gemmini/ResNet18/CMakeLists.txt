set(CMAKE_CXX_COMPILER ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)

set(MODEL_RESNET18_DIR ${MODEL_DIR}/ResNet18)
set(GEMMINI_RESNET18_DIR ${GEMMINI_DIR}/ResNet18)
set(INTERFACE_DIR ${BUDDY_MLIR_DIR}/frontend/Interfaces)
set(INCLUDE_DIR ${GEMMINI_RESNET18_DIR}/include)


add_custom_command(
  OUTPUT  forward.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_RESNET18_DIR}/forward.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -buffer-deallocation-simplification
            -convert-linalg-to-loops
            -eliminate-empty-tensors
            -llvm-request-c-wrappers
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-scf-to-cf
            -convert-arith-to-llvm
            -expand-strided-metadata
            -finalize-memref-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O0 -mattr=+buddyext,+D -float-abi=hard -o forward.o
          DEPENDS ${MODEL_RESNET18_DIR}/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM)

add_custom_command(
  OUTPUT  subgraph0.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_RESNET18_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))"
            > subgraph0_linalg.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_RESNET18_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -convert-linalg-to-gemmini
            -batchmatmul-optimize
            > subgraph0_loops.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_RESNET18_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -convert-linalg-to-gemmini
            # -lower-gemmini
            > subgraph0_gemmini.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_RESNET18_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -convert-linalg-to-gemmini
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -lower-gemmini
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir
          > subgraph0.ll
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_RESNET18_DIR}/subgraph0.mlir
          -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
        ${BUDDY_BINARY_DIR}/buddy-opt
          -eliminate-empty-tensors
          -convert-elementwise-to-linalg
          -convert-tensor-to-linalg
          -one-shot-bufferize="bufferize-function-boundaries"
          -convert-linalg-to-gemmini
          -expand-strided-metadata
          -convert-linalg-to-loops
          -convert-scf-to-cf
          -llvm-request-c-wrappers
          -lower-gemmini
          -ownership-based-buffer-deallocation
          -buffer-deallocation-simplification
          -bufferization-lower-deallocations
          -convert-math-to-llvm
          -convert-math-to-libm
          -convert-arith-to-llvm
          -convert-func-to-llvm
          -finalize-memref-to-llvm
          -reconcile-unrealized-casts |
        ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
        ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard -o subgraph0.o
        DEPENDS ${MODEL_RESNET18_DIR}/subgraph0.mlir
  COMMENT "Building subgraph0.o"
  VERBATIM)


set(LLVM_MLIR_EXECUTION_ENGINE_DIR ${BUDDY_MLIR_DIR}/llvm/mlir/include/mlir/ExecutionEngine)
set(DIP_MLIR_PATH ${BUDDY_MLIR_DIR}/frontend/Interfaces/lib/DIP.mlir)

# Compile DIP.mlir to DIP.o for RISC-V
add_custom_command(
  OUTPUT DIP.o
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
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard -o DIP.o
  DEPENDS ${DIP_MLIR_PATH}
  COMMENT "Building DIP.o for RISC-V"
  VERBATIM)

add_library(GemminiRESNET18 STATIC subgraph0.o forward.o)
add_library(GemminiCRunnerUtils STATIC ${MODEL_RESNET18_DIR}/CRunnerUtils.cpp)
add_library(GemminiDIP STATIC DIP.o)

target_include_directories(GemminiCRunnerUtils
  PRIVATE
  ${LLVM_MLIR_EXECUTION_ENGINE_DIR}
)

set_target_properties(GemminiRESNET18 PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(GemminiDIP PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(GemminiRESNET18 resnet-model-build)

add_executable(buddy-gemmini-resnet-run ${MODEL_RESNET18_DIR}/buddy-resnet-main.cpp)
add_dependencies(buddy-gemmini-resnet-run GemminiRESNET18)

target_compile_definitions(buddy-gemmini-resnet-run PRIVATE
  RESNET_EXAMPLE_PATH="${MODEL_RESNET18_DIR}"
  RESNET_EXAMPLE_BUILD_PATH="${CMAKE_CURRENT_BINARY_DIR}"
)

target_include_directories(buddy-gemmini-resnet-run PRIVATE ${INTERFACE_DIR} ${INCLUDE_DIR})
set(BUDDY_RESNET18_LIBS GemminiRESNET18 GemminiCRunnerUtils GemminiDIP)
target_link_libraries(buddy-gemmini-resnet-run ${BUDDY_RESNET18_LIBS})
