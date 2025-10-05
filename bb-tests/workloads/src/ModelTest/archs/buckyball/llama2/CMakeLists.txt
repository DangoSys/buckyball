set(CMAKE_CXX_COMPILER ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)

set(MODEL_LLAMA2_DIR ${MODEL_DIR}/llama2)
set(BUCKYBALL_LLAMA2_DIR ${BUCKYBALL_DIR}/llama2)
set(INTERFACE_DIR ${BUDDY_MLIR_DIR}/frontend/Interfaces)
set(INCLUDE_DIR ${MODEL_LLAMA2_DIR}/include)

add_custom_command(
  OUTPUT forward.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_LLAMA2_DIR}/forward.mlir
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
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard -o forward.o
  DEPENDS ${MODEL_LLAMA2_DIR}/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM)

add_custom_command(
    OUTPUT subgraph0.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_LLAMA2_DIR}/subgraph0.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -one-shot-bufferize="bufferize-function-boundaries"
              -convert-linalg-to-tile
              -convert-tile-to-buckyball
              > subgraph0_buckyball1.mlir
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_LLAMA2_DIR}/subgraph0.mlir
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
              > subgraph0_buckyball2.mlir
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_LLAMA2_DIR}/subgraph0.mlir
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
              -convert-cf-to-llvm
              -convert-arith-to-llvm
              -convert-math-to-llvm
              -convert-math-to-libm
              -convert-func-to-llvm
              -finalize-memref-to-llvm
              -reconcile-unrealized-casts |
            ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir
              > subgraph0.ll
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_LLAMA2_DIR}/subgraph0.mlir
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
    DEPENDS ${MODEL_LLAMA2_DIR}/subgraph0.mlir
    COMMENT "Building subgraph0.o"
    VERBATIM)

add_library(BuckyballLLAMA2 STATIC forward.o subgraph0.o)
set_target_properties(BuckyballLLAMA2 PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(BuckyballLLAMA2 llama2-model-build)

add_executable(buddy-buckyball-llama2-run ${MODEL_LLAMA2_DIR}/llama-main.cpp)
add_dependencies(buddy-buckyball-llama2-run BuckyballLLAMA2)

target_compile_definitions(buddy-buckyball-llama2-run PRIVATE
  LLAMA_EXAMPLE_PATH="${MODEL_LLAMA2_DIR}"
  LLAMA_EXAMPLE_BUILD_PATH="${CMAKE_CURRENT_BINARY_DIR}"
)

target_include_directories(buddy-buckyball-llama2-run PRIVATE ${INTERFACE_DIR} ${INCLUDE_DIR})

set(BUDDY_LLAMA_LIBS BuckyballLLAMA2 ModelTestCRunnerUtils)

target_link_libraries(buddy-buckyball-llama2-run ${BUDDY_LLAMA_LIBS})
