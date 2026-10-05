get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../.." ABSOLUTE)
add_library(rvv_kernels STATIC dispatch.cpp matmul_dispatch.cpp norm_banks.cpp pack_banks.cpp)
target_link_libraries(rvv_kernels PRIVATE runtime rvv_images)
target_compile_options(rvv_kernels PRIVATE -O2)
target_compile_features(rvv_kernels PRIVATE cxx_std_20)
target_include_directories(rvv_kernels PRIVATE
  ${RUNTIME_PARAMS}
  "${REPO_ROOT}/bb-tests/workloads/lib"
  "${REPO_ROOT}/stack/compiler/thirdparty/buddy-mlir/llvm/mlir/include/mlir/ExecutionEngine")

set(IMAGE_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/images.cpp")
set(IMAGE_HEADER "${CMAKE_CURRENT_BINARY_DIR}/images.h")
add_custom_command(
  OUTPUT "${IMAGE_SOURCE}" "${IMAGE_HEADER}"
  BYPRODUCTS silu.o snake.o quant.o matmul.o norm.o softmax.o rope.o pack.o exp.o sin.o cos.o
    silu.elf swiglu.elf snake.elf quant.elf matmul.elf norm.elf softmax.elf attention_softmax.elf rope.elf pack.elf
  COMMAND "${PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/build_images.py"
    --output "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS build_images.py kernel.ld kernels.h silu.cpp snake.cpp quant.cpp matmul.cpp
    norm.cpp softmax.cpp rope.cpp pack.cpp
    math/exp.cpp math/sin.cpp math/cos.cpp math/math.h math/constants.h
  VERBATIM)
add_custom_target(rvv-images DEPENDS "${IMAGE_SOURCE}" "${IMAGE_HEADER}")
add_library(rvv_images STATIC "${IMAGE_SOURCE}")
target_compile_features(rvv_images PRIVATE cxx_std_20)
add_dependencies(rvv_images rvv-images)
target_include_directories(rvv_images PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_include_directories(rvv_kernels PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

add_library(rvv_quant_backend STATIC quant_dispatch.cpp)
target_link_libraries(rvv_quant_backend PRIVATE runtime rvv_images)
target_compile_options(rvv_quant_backend PRIVATE -O2)
target_compile_features(rvv_quant_backend PRIVATE cxx_std_20)
target_include_directories(rvv_quant_backend PRIVATE
  ${RUNTIME_PARAMS} "${REPO_ROOT}/bb-tests/workloads/lib"
  "${CMAKE_CURRENT_BINARY_DIR}")
add_dependencies(rvv_quant_backend rvv-images)
