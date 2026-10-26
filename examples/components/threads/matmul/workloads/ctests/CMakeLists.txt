add_subdirectory(transpose_smatmul_quant_pipeline)
set(BUCKYBALL_CTEST_TARGET ${BUCKYBALL_COMPONENT_WORKLOAD_TARGET})
file(
    RELATIVE_PATH
    RVV_KERNEL_SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${WORKLOAD_SRC_DIR}/CTest/rvv/rvv_kernel.c
)
add_buckyball_ctest(${RVV_KERNEL_SOURCE})
set(RVV_KERNEL_TEST
    "${BUCKYBALL_WORKLOAD_DESIGN}-${BUCKYBALL_CTEST_TARGET}-ctest-rvv_kernel"
)
add_dependencies(component-ctests-build ${RVV_KERNEL_TEST}-ctest-build)
add_subdirectory(rvv)
