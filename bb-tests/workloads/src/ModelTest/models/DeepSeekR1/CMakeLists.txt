set(LLVM_TOOLS_BINARY_DIR ${BUDDY_MLIR_DIR}/llvm/build/bin)
set(BUDDY_BINARY_DIR ${BUDDY_MLIR_DIR}/build/bin)
set(INTERFACE_DIR ${BUDDY_MLIR_DIR}/frontend/Interfaces)
set(MLIR_C_RUNNER_UTILS_DIR ${BUDDY_MLIR_DIR}/llvm/build/lib)

set(RISCV_GNU_TOOLCHAIN ${BUDDY_MLIR_DIR}/../.conda-env/riscv-tools)
set(CMAKE_CXX_COMPILER  ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir 
         ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir 
         ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/import-deepseek-r1.py
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating forward.mlir, subgraph0.mlir and arg0.data..."
)


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
            # -affine-parallelize  
            -lower-affine
            # -convert-scf-to-openmp  
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
            # -convert-openmp-to-llvm  
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
              -convert-linalg-to-gemmini
              -lower-gemmini
              > subgraph0_gemmini.mlir
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir 
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg 
              -convert-tensor-to-linalg
              -one-shot-bufferize
              -convert-linalg-to-gemmini
              -lower-gemmini 
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
            -convert-linalg-to-gemmini
            -lower-gemmini 
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
          # ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir 
            ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir 
            > subgraph0.ll
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir 
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
            ${BUDDY_BINARY_DIR}/buddy-opt
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg 
              -convert-tensor-to-linalg
              -one-shot-bufferize
              -convert-linalg-to-gemmini
              -lower-gemmini 
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


set(LLVM_MLIR_EXECUTION_ENGINE_DIR ${BUDDY_MLIR_DIR}/llvm/mlir/include/mlir/ExecutionEngine)


add_library(GEMMINI_DEEPSEEKR1 STATIC forward.o subgraph0.o)
add_library(CRunnerUtils STATIC CRunnerUtils.cpp)

target_include_directories(CRunnerUtils
  PRIVATE
  ${LLVM_MLIR_EXECUTION_ENGINE_DIR}
)

SET_TARGET_PROPERTIES(
  GEMMINI_DEEPSEEKR1
  PROPERTIES
  LINKER_LANGUAGE C)

add_executable(gemmini-deepseek-r1-run gemmini-deepseek-r1-main.cpp)
target_include_directories(gemmini-deepseek-r1-run PRIVATE ${INTERFACE_DIR})

set(DEEPSEEKR1_EXAMPLE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(DEEPSEEKR1_EXAMPLE_BUILD_PATH ${CMAKE_CURRENT_BINARY_DIR})

target_compile_definitions(gemmini-deepseek-r1-run PRIVATE 
  DEEPSEEKR1_EXAMPLE_PATH="${DEEPSEEKR1_EXAMPLE_PATH}" 
  DEEPSEEKR1_EXAMPLE_BUILD_PATH="${DEEPSEEKR1_EXAMPLE_BUILD_PATH}"
)

target_link_directories(gemmini-deepseek-r1-run PRIVATE ${LLVM_LIBRARY_DIR})

set(GEMMINI_DEEPSEEKR1_LIBS
  # DEEPSEEKR1
  GEMMINI_DEEPSEEKR1
  CRunnerUtils
  # mlir_c_runner_utils
  # omp
)
if(BUDDY_MLIR_USE_MIMALLOC)
  list(APPEND GEMMINI_DEEPSEEKR1_LIBS mimalloc)
endif()

target_link_libraries(gemmini-deepseek-r1-run -static ${GEMMINI_DEEPSEEKR1_LIBS})
