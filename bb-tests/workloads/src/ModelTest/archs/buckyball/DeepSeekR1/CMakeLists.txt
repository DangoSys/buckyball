set(LLVM_TOOLS_BINARY_DIR ${BUDDY_MLIR_DIR}/llvm/build/bin)
set(BUDDY_BINARY_DIR ${BUDDY_MLIR_DIR}/build/bin)
set(INTERFACE_DIR ${BUDDY_MLIR_DIR}/frontend/Interfaces)
set(INCLUDE_DIR ${BUCKYBALL_DEEPSEEKR1_DIR}/include)

add_custom_command(
  OUTPUT forward.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
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
            -func-bufferize
            -arith-bufferize
            -tensor-bufferize
            -buffer-deallocation
            -finalizing-bufferize
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
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
  COMMENT "Building forward.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph0.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
              -convert-tensor-to-linalg
              -linalg-bufferize
              > subgraph0_linalg.mlir
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
              -convert-tensor-to-linalg
              -linalg-bufferize
              -convert-linalg-to-tile
              -convert-tile-to-buckyball
              -lower-buckyball
              > subgraph0_buckyball.mlir
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -one-shot-bufferize
              -convert-linalg-to-tile
              -convert-tile-to-buckyball
              -lower-buckyball
              -func-bufferize
              -arith-bufferize
              -tensor-bufferize
              -buffer-deallocation
              -finalizing-bufferize
              -expand-strided-metadata
              -convert-scf-to-cf
              -llvm-request-c-wrappers
              > subgraph0_test.mlir
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize
            -convert-linalg-to-tile
            -convert-tile-to-buckyball
            -lower-buckyball
            -convert-linalg-to-loops
            -func-bufferize
            -arith-bufferize
            -tensor-bufferize
            -buffer-deallocation
            -finalizing-bufferize
            -expand-strided-metadata
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
            ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir
            > subgraph0.ll
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg
              -convert-tensor-to-linalg
              -one-shot-bufferize
              -convert-linalg-to-tile
              -convert-tile-to-buckyball
              -lower-buckyball
              -convert-linalg-to-loops
              -func-bufferize
              -arith-bufferize
              -tensor-bufferize
              -buffer-deallocation
              -finalizing-bufferize
              -expand-strided-metadata
              -convert-scf-to-cf
              -llvm-request-c-wrappers
              -convert-arith-to-llvm
              -convert-math-to-llvm
              -convert-math-to-libm
              -convert-func-to-llvm
              -finalize-memref-to-llvm
              -reconcile-unrealized-casts |
            ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
            ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O3 -mattr=+buddyext,+D -float-abi=hard
              -o subgraph0.o
            DEPENDS subgraph0.mlir
    COMMENT "Building subgraph0.o"
    VERBATIM)

add_library(BuckyballDEEPSEEKR1 STATIC forward.o subgraph0.o)
set_target_properties(BuckyballDEEPSEEKR1 PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(BuckyballDEEPSEEKR1 deepseekr1-model-build)
add_executable(buddy-buckyball-deepseek-r1-run ${MODEL_DEEPSEEKR1_DIR}/deepseek-r1-main.cpp)
add_dependencies(buddy-buckyball-deepseek-r1-run BuckyballDEEPSEEKR1)

target_include_directories(buddy-buckyball-deepseek-r1-run PRIVATE ${INTERFACE_DIR} ${INCLUDE_DIR})
target_compile_definitions(buddy-buckyball-deepseek-r1-run PRIVATE
  DEEPSEEKR1_EXAMPLE_PATH="${MODEL_DEEPSEEKR1_DIR}"
  DEEPSEEKR1_EXAMPLE_BUILD_PATH="${CMAKE_CURRENT_BINARY_DIR}"
)

if(BUDDY_MLIR_USE_MIMALLOC)
  list(APPEND BuckyballDEEPSEEKR1_LIBS mimalloc)
endif()

set(BUDDY_DEEPSEEKR1_LIBS BuckyballDEEPSEEKR1 ModelTestCRunnerUtils)
target_link_libraries(buddy-buckyball-deepseek-r1-run -static ${BUDDY_DEEPSEEKR1_LIBS})
