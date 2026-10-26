# Images: built by the thread stack (stack/kernels/rvv); dispatch code: built here.
set(KERNEL_DIR "${BUCKYBALL_REPO_ROOT}/stack/kernels/rvv")
set(IMAGE_DIR "${DESIGN_THREAD_BUILD_${BUCKYBALL_CTEST_TARGET}}/rvv")
file(READ "${BUCKYBALL_ISA_DIR}/${BUCKYBALL_CTEST_TARGET}/params.h" PARAMS)
foreach(PARAM BANK_NUM BANK_WIDTH BANK_LINES)
    string(REGEX MATCH "#define ${PARAM} ([0-9]+)" MATCH "${PARAMS}")
    set(${PARAM} "${CMAKE_MATCH_1}")
endforeach()
if(BANK_NUM LESS 4 OR NOT BANK_WIDTH EQUAL 128 OR BANK_LINES LESS 32)
    message(
        FATAL_ERROR
        "RVV kernel CTests require 4 banks, 128-bit rows and 512 bytes per bank; "
        "${BUCKYBALL_CTEST_TARGET} has ${BANK_NUM} banks, ${BANK_WIDTH}-bit rows, ${BANK_LINES} lines"
    )
endif()
foreach(
    KERNEL
    silu
    swiglu
    snake
    mxfp8_quant
    matmul
    norm
)
    buckyball_enforce_ctest_line_limit("${KERNEL}.cpp")
    set(NAME
        "${BUCKYBALL_WORKLOAD_DESIGN}-${BUCKYBALL_CTEST_TARGET}-ctest-rvv_${KERNEL}"
    )
    add_buckyball_linux_ctest(${NAME} ${CMAKE_CURRENT_SOURCE_DIR} ${KERNEL}.cpp)
    if(KERNEL STREQUAL "matmul")
        set(DISPATCH_SOURCE "${KERNEL_DIR}/matmul_dispatch.cpp")
    elseif(KERNEL STREQUAL "mxfp8_quant")
        set(DISPATCH_SOURCE "${KERNEL_DIR}/quant_dispatch.cpp")
    else()
        set(DISPATCH_SOURCE "${KERNEL_DIR}/dispatch.cpp")
    endif()
    target_sources(${NAME}-linux PRIVATE "${DISPATCH_SOURCE}")
    target_compile_options(${NAME}-linux PRIVATE "--embed-dir=${IMAGE_DIR}")
    target_include_directories(
        ${NAME}-linux
        PRIVATE
            "${KERNEL_DIR}"
            "${BUCKYBALL_REPO_ROOT}/stack/kernels"
            "${BUCKYBALL_REPO_ROOT}/stack/thirdparty/buddy-mlir/llvm/mlir/include/mlir/ExecutionEngine"
    )
    target_compile_options(${NAME}-linux PRIVATE -O2)
    set_target_properties(${NAME}-linux PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries(${NAME}-linux PRIVATE m)
    buckyball_ctest_deps(DEPS ${CMAKE_CURRENT_SOURCE_DIR} ${KERNEL}.cpp)
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${NAME}-baremetal"
        COMMAND
            ${ELF_CXX} ${BUCKYBALL_CTEST_C_FLAGS}
            -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
            -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
            -I${WORKLOAD_LIB_DIR} -I${KERNEL_DIR}
            -I${BUCKYBALL_REPO_ROOT}/stack/kernels --embed-dir=${IMAGE_DIR}
            -I${BUCKYBALL_ISA_DIR}/design
            -I${BUCKYBALL_ISA_DIR}/${BUCKYBALL_CTEST_TARGET}
            -I${BUCKYBALL_REPO_ROOT}/stack/thirdparty/buddy-mlir/llvm/mlir/include/mlir/ExecutionEngine
            -o "${NAME}-baremetal" ${BAREMETAL_DIR}/crt0.S -x c
            ${WORKLOAD_LIB_DIR}/buckyball.c ${BBHW_MEM_C}
            ${BUCKYBALL_DESIGN_COMMON_SOURCES} -x c++
            "${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL}.cpp" "${DISPATCH_SOURCE}" -lm
        DEPENDS ${DEPS} "${DISPATCH_SOURCE}" ${BAREMETAL_DIR}/crt0.S
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        VERBATIM
    )
    add_custom_target(
        ${NAME}-ctest-build
        DEPENDS
            ${NAME}-linux-build
            "${CMAKE_CURRENT_BINARY_DIR}/${NAME}-baremetal"
    )
    add_dependencies(component-ctests-build ${NAME}-ctest-build)
endforeach()
