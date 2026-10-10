set(KERNEL_DIR "${WORKLOAD_LIB_DIR}/bbsw/kernels/rvv")
set(IMAGE_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/images.cpp")
set(IMAGE_HEADER "${CMAKE_CURRENT_BINARY_DIR}/images.h")
file(READ "${BUCKYBALL_ISA_DIR}/${BUCKYBALL_CTEST_TARGET}/params.h" PARAMS)
foreach(PARAM BANK_NUM BANK_WIDTH BANK_LINES)
  string(REGEX MATCH "#define ${PARAM} ([0-9]+)" MATCH "${PARAMS}")
  set(${PARAM} "${CMAKE_MATCH_1}")
endforeach()
if(BANK_NUM LESS 4 OR NOT BANK_WIDTH EQUAL 128 OR BANK_LINES LESS 32)
  message(FATAL_ERROR
    "RVV kernel CTests require 4 banks, 128-bit rows and 512 bytes per bank; "
    "${BUCKYBALL_CTEST_TARGET} has ${BANK_NUM} banks, ${BANK_WIDTH}-bit rows, ${BANK_LINES} lines")
endif()
add_custom_command(
  OUTPUT "${IMAGE_SOURCE}" "${IMAGE_HEADER}"
  COMMAND "${Python3_EXECUTABLE}" "${KERNEL_DIR}/build_images.py"
    --output "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS "${KERNEL_DIR}/build_images.py" "${KERNEL_DIR}/kernel.ld"
    "${KERNEL_DIR}/kernels.h" "${KERNEL_DIR}/silu.cpp"
    "${KERNEL_DIR}/snake.cpp" "${KERNEL_DIR}/quant.cpp"
    "${KERNEL_DIR}/matmul.cpp"
    "${KERNEL_DIR}/math/exp.cpp" "${KERNEL_DIR}/math/sin.cpp"
    "${KERNEL_DIR}/math/math.h" "${KERNEL_DIR}/math/constants.h"
  VERBATIM)
buckyball_ctest_name(IMAGE_TARGET ${BUCKYBALL_CTEST_TARGET} rvv_images)
add_custom_target(${IMAGE_TARGET} DEPENDS "${IMAGE_SOURCE}" "${IMAGE_HEADER}")

foreach(KERNEL silu swiglu snake mxfp8_quant matmul)
  buckyball_enforce_ctest_line_limit("${KERNEL}.cpp")
  buckyball_ctest_name(NAME ${BUCKYBALL_CTEST_TARGET} rvv_${KERNEL})
  add_buckyball_linux_ctest(${NAME} ${CMAKE_CURRENT_SOURCE_DIR} ${KERNEL}.cpp)
  if(KERNEL STREQUAL "matmul")
    set(DISPATCH_SOURCE "${KERNEL_DIR}/matmul_dispatch.cpp")
  elseif(KERNEL STREQUAL "mxfp8_quant")
    set(DISPATCH_SOURCE "${KERNEL_DIR}/quant_dispatch.cpp")
  else()
    set(DISPATCH_SOURCE "${KERNEL_DIR}/dispatch.cpp")
  endif()
  target_sources(${NAME}-linux PRIVATE "${IMAGE_SOURCE}" "${DISPATCH_SOURCE}")
  target_include_directories(${NAME}-linux PRIVATE "${CMAKE_CURRENT_BINARY_DIR}"
    "${BUCKYBALL_REPO_ROOT}/stack/compiler/thirdparty/buddy-mlir/llvm/mlir/include/mlir/ExecutionEngine")
  target_compile_options(${NAME}-linux PRIVATE -O2)
  set_target_properties(${NAME}-linux PROPERTIES LINKER_LANGUAGE CXX)
  target_link_libraries(${NAME}-linux PRIVATE m)
  add_dependencies(${NAME}-linux ${IMAGE_TARGET})
  buckyball_ctest_deps(DEPS ${CMAKE_CURRENT_SOURCE_DIR} ${KERNEL}.cpp)
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${NAME}-baremetal"
    COMMAND ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-elf-g++
      ${BUCKYBALL_CTEST_C_FLAGS}
      -I${BUCKYBALL_REPO_ROOT}/stack/runtime/include -I${WORKLOAD_LIB_DIR}
      -I${BUCKYBALL_ISA_DIR}/chip -I${BUCKYBALL_ISA_DIR}/${BUCKYBALL_CTEST_TARGET}
      -I${CMAKE_CURRENT_BINARY_DIR}
      -I${BUCKYBALL_REPO_ROOT}/stack/compiler/thirdparty/buddy-mlir/llvm/mlir/include/mlir/ExecutionEngine
      -o "${NAME}-baremetal" ${BBSW_BAREMETAL_DIR}/crt0.S
      -x c ${WORKLOAD_LIB_DIR}/buckyball.c ${BBHW_MEM_C}
      ${BUCKYBALL_CHIP_COMMON_SOURCES}
      -x c++ "${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL}.cpp" "${IMAGE_SOURCE}" "${DISPATCH_SOURCE}" -lm
    DEPENDS ${DEPS} "${IMAGE_SOURCE}" "${IMAGE_HEADER}" "${DISPATCH_SOURCE}"
      ${BBSW_BAREMETAL_DIR}/crt0.S
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    VERBATIM)
  add_custom_target(${NAME}-ctest-build
    DEPENDS ${NAME}-linux-build "${CMAKE_CURRENT_BINARY_DIR}/${NAME}-baremetal")
  add_dependencies(${NAME}-ctest-build ${IMAGE_TARGET})
  add_dependencies(core-ctests-build ${NAME}-ctest-build)
endforeach()
