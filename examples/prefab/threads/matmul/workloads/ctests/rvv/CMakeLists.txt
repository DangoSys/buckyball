# Images: built by the thread stack (stack/kernels/rvv); dispatch code: built here.
set(KERNEL_DIR "${BUCKYBALL_REPO_ROOT}/stack/kernels/rvv")
set(IMAGE_DIR "${DESIGN_THREAD_BUILD_${BUCKYBALL_CTEST_TARGET}}/rvv")
set(_BANK_NUM ${DESIGN_PROFILES_${BUCKYBALL_CTEST_TARGET}_THREAD_MEM_BANK_NUM})
set(_BANK_WIDTH ${DESIGN_PROFILES_${BUCKYBALL_CTEST_TARGET}_THREAD_MEM_BANK_WIDTH})
set(_BANK_ENTRIES ${DESIGN_PROFILES_${BUCKYBALL_CTEST_TARGET}_THREAD_MEM_BANK_ENTRIES})
if(_BANK_NUM LESS 4 OR NOT _BANK_WIDTH EQUAL 128 OR _BANK_ENTRIES LESS 32)
    message(
        FATAL_ERROR
        "RVV kernel CTests require 4 banks, 128-bit rows and 512 bytes per bank; "
        "${BUCKYBALL_CTEST_TARGET} has ${_BANK_NUM} banks, ${_BANK_WIDTH}-bit rows, ${_BANK_ENTRIES} lines"
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
        "${DESIGN_NAME}-${BUCKYBALL_CTEST_TARGET}-ctest-rvv_${KERNEL}"
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
    set(TARGET_DEFINES ${DESIGN_TARGET_DEFS_${BUCKYBALL_CTEST_TARGET}})
    list(TRANSFORM TARGET_DEFINES PREPEND "-D")
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${NAME}-baremetal"
        COMMAND
            ${ELF_CXX} ${BUCKYBALL_CTEST_C_FLAGS}
            -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
            -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
            -I${WORKLOAD_LIB_DIR} -I${KERNEL_DIR}
            -I${BUCKYBALL_REPO_ROOT}/stack/kernels --embed-dir=${IMAGE_DIR}
            ${DESIGN_DEFINES} ${TARGET_DEFINES}
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
