set(CMAKE_CXX_COMPILER ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)

set(MODEL_DEEPSEEKR1_DIR ${MODEL_DIR}/DeepSeekR1)
set(LLVM_TOOLS_BINARY_DIR ${BUDDY_MLIR_DIR}/llvm/build/bin)
set(BUDDY_BINARY_DIR ${BUDDY_MLIR_DIR}/build/bin)
set(INTERFACE_DIR ${BUDDY_MLIR_DIR}/frontend/Interfaces)
set(INCLUDE_DIR ${GEMMINI_DEEPSEEKR1_DIR}/include)

add_custom_command(
  OUTPUT forward.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_DEEPSEEKR1_DIR}/forward.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize
            -matmul-parallel-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -lower-affine
            -buffer-deallocation
            -convert-vector-to-scf
            -expand-strided-metadata
            -convert-vector-to-llvm
            -memref-expand
            -arith-expand
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
        ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
        ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O0 -mattr=+buddyext,+D -float-abi=hard -o forward.o
  DEPENDS ${MODEL_DEEPSEEKR1_DIR}/forward.mlir
  COMMENT "Building forward.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph0.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${MODEL_DEEPSEEKR1_DIR}/subgraph0.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
              -convert-tensor-to-linalg
              > subgraph0_linalg.mlir
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${MODEL_DEEPSEEKR1_DIR}/subgraph0.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -one-shot-bufferize="bufferize-function-boundaries"
              -convert-linalg-to-gemmini
              -lower-gemmini
              > subgraph0_gemmini.mlir
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${MODEL_DEEPSEEKR1_DIR}/subgraph0.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -one-shot-bufferize="bufferize-function-boundaries"
              -convert-linalg-to-gemmini
              -lower-gemmini
              -buffer-deallocation
              -expand-strided-metadata
              -llvm-request-c-wrappers
              > subgraph0_test.mlir
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_DEEPSEEKR1_DIR}/subgraph0.mlir
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
              -convert-cf-to-llvm
              -convert-arith-to-llvm
              -convert-math-to-llvm
              -convert-math-to-libm
              -convert-func-to-llvm
              -finalize-memref-to-llvm
              -reconcile-unrealized-casts |
            ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir
            > subgraph0.ll
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_DEEPSEEKR1_DIR}/subgraph0.mlir
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
              -convert-cf-to-llvm
              -convert-arith-to-llvm
              -convert-math-to-llvm
              -convert-math-to-libm
              -convert-func-to-llvm
              -finalize-memref-to-llvm
              -reconcile-unrealized-casts |
            ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
            ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O3 -mattr=+buddyext,+D -float-abi=hard
              -o subgraph0.o
            DEPENDS ${MODEL_DEEPSEEKR1_DIR}/subgraph0.mlir
    COMMENT "Building subgraph0.o"
    VERBATIM)

add_library(GemminiDEEPSEEKR1 STATIC forward.o subgraph0.o)
set_target_properties(GemminiDEEPSEEKR1 PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(GemminiDEEPSEEKR1 deepseekr1-model-build)
add_executable(buddy-gemmini-deepseekr1-run ${MODEL_DEEPSEEKR1_DIR}/deepseek-r1-main.cpp)
add_dependencies(buddy-gemmini-deepseekr1-run GemminiDEEPSEEKR1)

target_include_directories(buddy-gemmini-deepseekr1-run PRIVATE ${INTERFACE_DIR} ${INCLUDE_DIR})
target_compile_definitions(buddy-gemmini-deepseekr1-run PRIVATE
  DEEPSEEKR1_EXAMPLE_PATH="${MODEL_DEEPSEEKR1_DIR}"
  DEEPSEEKR1_EXAMPLE_BUILD_PATH="${CMAKE_CURRENT_BINARY_DIR}"
)

if(BUDDY_MLIR_USE_MIMALLOC)
  list(APPEND GEMMINI_DEEPSEEKR1_LIBS mimalloc)
endif()

set(BUDDY_DEEPSEEKR1_LIBS GemminiDEEPSEEKR1 ModelTestCRunnerUtils)
target_link_libraries(buddy-gemmini-deepseekr1-run -static ${BUDDY_DEEPSEEKR1_LIBS})
