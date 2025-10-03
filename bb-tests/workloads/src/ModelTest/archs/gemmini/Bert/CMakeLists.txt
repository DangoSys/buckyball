set(CMAKE_CXX_COMPILER ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)

set(MODEL_BERT_DIR ${MODEL_DIR}/Bert)
set(GEMMINI_BERT_DIR ${GEMMINI_DIR}/Bert)
set(INTERFACE_DIR ${BUDDY_MLIR_DIR}/frontend/Interfaces)
set(INCLUDE_DIR ${GEMMINI_BERT_DIR}/include)

add_custom_command(
  OUTPUT forward.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_BERT_DIR}/forward.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith), empty-tensor-to-alloc-tensor, convert-elementwise-to-linalg)" |
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
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard -o forward.o
        DEPENDS ${MODEL_BERT_DIR}/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph0.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_BERT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -convert-linalg-to-gemmini
          > subgraph0_gemmini.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_BERT_DIR}/subgraph0.mlir
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
            -reconcile-unrealized-casts  |
        ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir
          > subgraph0.ll
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_BERT_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
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
          -reconcile-unrealized-casts  |
        ${BUDDY_BINARY_DIR}/buddy-translate -buddy-to-llvmir |
        ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard -o subgraph0.o
        DEPENDS ${MODEL_BERT_DIR}/subgraph0.mlir
  COMMENT "Building subgraph.o "
  VERBATIM)

add_library(GemminiBERT STATIC forward.o subgraph0.o)
set_target_properties(GemminiBERT PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(GemminiBERT bert-model-build)
add_executable(buddy-gemmini-bert-run ${MODEL_BERT_DIR}/bert-main.cpp)
add_dependencies(buddy-gemmini-bert-run GemminiBERT)

target_compile_definitions(buddy-gemmini-bert-run PRIVATE
  BERT_EXAMPLE_PATH="${MODEL_BERT_DIR}"
  BERT_EXAMPLE_BUILD_PATH="${CMAKE_CURRENT_BINARY_DIR}"
)

target_include_directories(buddy-gemmini-bert-run PRIVATE ${INTERFACE_DIR} ${INCLUDE_DIR})
set(BUDDY_BERT_LIBS GemminiBERT ModelTestCRunnerUtils ModelTestDIP_riscv)
target_link_libraries(buddy-gemmini-bert-run ${BUDDY_BERT_LIBS})
