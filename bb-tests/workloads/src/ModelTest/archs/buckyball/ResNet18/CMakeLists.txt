set(CMAKE_CXX_COMPILER ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)

set(MODEL_RESNET18_DIR ${MODEL_DIR}/ResNet18)
set(BUCKYBALL_RESNET18_DIR ${BUCKYBALL_DIR}/ResNet18)
set(INTERFACE_DIR ${BUDDY_MLIR_DIR}/frontend/Interfaces)
set(INCLUDE_DIR ${BUCKYBALL_RESNET18_DIR}/include)


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
            -convert-linalg-to-tile
            -batchmatmul-optimize
            > subgraph0_loops.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_RESNET18_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -convert-linalg-to-tile
            -convert-tile-to-buckyball
            # -lower-buckyball
            > subgraph0_buckyball.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_RESNET18_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -convert-linalg-to-tile
            -convert-tile-to-buckyball
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -lower-buckyball
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
          -convert-linalg-to-tile
          -convert-tile-to-buckyball
          -expand-strided-metadata
          -convert-linalg-to-loops
          -convert-scf-to-cf
          -llvm-request-c-wrappers
          -lower-buckyball
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

add_library(BuckyballRESNET18 STATIC subgraph0.o forward.o)
add_library(BuckyballCRunnerUtils STATIC ${MODEL_RESNET18_DIR}/CRunnerUtils.cpp)
add_library(BuckyballDIP STATIC DIP.o)

target_include_directories(BuckyballCRunnerUtils
  PRIVATE
  ${LLVM_MLIR_EXECUTION_ENGINE_DIR}
)

set_target_properties(BuckyballRESNET18 PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(BuckyballDIP PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(BuckyballRESNET18 resnet-model-build)

add_executable(buddy-buckyball-resnet-run ${MODEL_RESNET18_DIR}/buddy-resnet-main.cpp)
add_dependencies(buddy-buckyball-resnet-run BuckyballRESNET18)

target_compile_definitions(buddy-buckyball-resnet-run PRIVATE
  RESNET_EXAMPLE_PATH="${MODEL_RESNET18_DIR}"
  RESNET_EXAMPLE_BUILD_PATH="${CMAKE_CURRENT_BINARY_DIR}"
)

target_include_directories(buddy-buckyball-resnet-run PRIVATE ${INTERFACE_DIR} ${INCLUDE_DIR})
set(BUDDY_RESNET18_LIBS BuckyballRESNET18 BuckyballCRunnerUtils BuckyballDIP)
target_link_libraries(buddy-buckyball-resnet-run ${BUDDY_RESNET18_LIBS})
