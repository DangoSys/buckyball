set(CMAKE_CXX_COMPILER ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)

set(MODEL_STABLEDIFFUSION_DIR ${MODEL_DIR}/StableDiffusion)
set(INTERFACE_DIR ${BUDDY_MLIR_DIR}/frontend/Interfaces)
set(INCLUDE_DIR ${MODEL_STABLEDIFFUSION_DIR}/include)

add_custom_command(
  OUTPUT forward_text_encoder.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_STABLEDIFFUSION_DIR}/forward_text_encoder.mlir
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
            > forward_text_encoder_gemmini.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_STABLEDIFFUSION_DIR}/forward_text_encoder.mlir
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
        ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard
          -o forward_text_encoder.o
  DEPENDS ${MODEL_STABLEDIFFUSION_DIR}/forward_text_encoder.mlir
  COMMENT "Building forward_text_encoder.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph0_text_encoder.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_text_encoder.mlir
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
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard
            -o subgraph0_text_encoder.o
    DEPENDS ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_text_encoder.mlir
    COMMENT "Building subgraph0_text_encoder.o "
    VERBATIM)

add_custom_command(
  OUTPUT forward_unet.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_STABLEDIFFUSION_DIR}/forward_unet.mlir
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
            > forward_unet_gemmini.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_STABLEDIFFUSION_DIR}/forward_unet.mlir
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
        ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard
          -o forward_unet.o
  DEPENDS ${MODEL_STABLEDIFFUSION_DIR}/forward_unet.mlir
  COMMENT "Building forward_unet.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph0_unet.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_unet.mlir
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
            > subgraph0_unet_gemmini.mlir
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_unet.mlir
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
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard
            -o subgraph0_unet.o
    DEPENDS ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_unet.mlir
    COMMENT "Building subgraph0_unet.o "
    VERBATIM)

add_custom_command(
  OUTPUT forward_vae.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_STABLEDIFFUSION_DIR}/forward_vae.mlir
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
            > forward_vae_gemmini.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_STABLEDIFFUSION_DIR}/forward_vae.mlir
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
        ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard
          -o forward_vae.o
  DEPENDS ${MODEL_STABLEDIFFUSION_DIR}/forward_vae.mlir
  COMMENT "Building forward_vae.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph0_vae.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_vae.mlir
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
              > subgraph0_vae_gemmini.mlir
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_vae.mlir
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
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard
            -o subgraph0_vae.o
    DEPENDS ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_vae.mlir
    COMMENT "Building subgraph0_vae.o "
    VERBATIM)


add_library(GemminiTEXTENCODER STATIC subgraph0_text_encoder.o forward_text_encoder.o)
add_library(GemminiUNET STATIC subgraph0_unet.o forward_unet.o)
add_library(GemminiVAE STATIC subgraph0_vae.o forward_vae.o)

set_target_properties(GemminiTEXTENCODER PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(GemminiUNET PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(GemminiVAE PROPERTIES LINKER_LANGUAGE CXX)

add_dependencies(GemminiTEXTENCODER stablediffusion-model-build)
add_dependencies(GemminiUNET stablediffusion-model-build)
add_dependencies(GemminiVAE stablediffusion-model-build)

add_executable(buddy-gemmini-stablediffusion-run ${MODEL_STABLEDIFFUSION_DIR}/buddy-stable-diffusion-main.cpp)

target_compile_definitions(buddy-gemmini-stablediffusion-run PRIVATE
  STABLE_DIFFUSION_EXAMPLE_PATH="${MODEL_STABLEDIFFUSION_DIR}"
  STABLE_DIFFUSION_EXAMPLE_BUILD_PATH="${CMAKE_CURRENT_BINARY_DIR}"
)

target_include_directories(buddy-gemmini-stablediffusion-run PRIVATE ${INTERFACE_DIR} ${INCLUDE_DIR})


if(BUDDY_MLIR_USE_MIMALLOC)
  list(APPEND BUDDY_STABLE_DIFFUSION_LIBS mimalloc)
endif()

set(BUDDY_STABLE_DIFFUSION_LIBS
  GemminiTEXTENCODER
  GemminiUNET
  GemminiVAE
  ModelTestCRunnerUtils
  ModelTestDIP_riscv
)

target_link_libraries(buddy-gemmini-stablediffusion-run ${BUDDY_STABLE_DIFFUSION_LIBS})
