get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)
set(RVV_SOURCE "${REPO_ROOT}/bb-tests/workloads/lib/bbsw/kernels/rvv")
set(IMAGES "${CMAKE_CURRENT_BINARY_DIR}/norm_window_image.h"
  "${CMAKE_CURRENT_BINARY_DIR}/packmm_image.h")
add_custom_command(OUTPUT ${IMAGES}
  COMMAND "${PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/build_image.py"
    --output "${CMAKE_CURRENT_BINARY_DIR}" --isa-dir "${RUNTIME_PARAMS}"
  DEPENDS build_image.py norm_window.cpp window.h packmm.cpp packmm.h
    "${RUNTIME_PARAMS}/ballISA.h" "${RVV_SOURCE}/build_images.py"
    "${RVV_SOURCE}/norm.cpp" "${RVV_SOURCE}/pack.cpp" "${RVV_SOURCE}/kernels.h" "${RVV_SOURCE}/kernel.ld"
  BYPRODUCTS norm_window.o norm.o norm_window.elf packmm.o pack.o packmm.elf
  VERBATIM)
add_custom_target(mxmm-images DEPENDS ${IMAGES})
add_library(mxmm_kernels STATIC dispatch.cpp pack_dispatch.cpp ${IMAGES})
target_link_libraries(mxmm_kernels PRIVATE runtime)
target_compile_options(mxmm_kernels PRIVATE -O2)
target_compile_features(mxmm_kernels PRIVATE cxx_std_20)
target_include_directories(mxmm_kernels PRIVATE "${CMAKE_CURRENT_BINARY_DIR}"
  "${RUNTIME_PARAMS}" "${REPO_ROOT}/bb-tests/workloads/lib")

add_library(mxmm_kernels_ant STATIC EXCLUDE_FROM_ALL dispatch.cpp pack_dispatch.cpp ${IMAGES})
target_link_libraries(mxmm_kernels_ant PRIVATE runtime_ant_tasks)
target_compile_options(mxmm_kernels_ant PRIVATE -O2)
target_compile_features(mxmm_kernels_ant PRIVATE cxx_std_20)
target_include_directories(mxmm_kernels_ant PRIVATE "${CMAKE_CURRENT_BINARY_DIR}"
  "${RUNTIME_PARAMS}" "${REPO_ROOT}/bb-tests/workloads/lib")
