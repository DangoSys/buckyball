add_subdirectory(gemmini_relu_trace_pipeline)
set(BUCKYBALL_CTEST_TARGET toy)
set(BUCKYBALL_CTEST_GROUP_TARGET toy-ctests-build)
file(
    RELATIVE_PATH
    RVV_KERNEL_SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${WORKLOAD_SRC_DIR}/CTest/rvv/rvv_kernel.c
)
add_buckyball_ctest(${RVV_KERNEL_SOURCE})
set(RVV_KERNEL_TEST
    "${DESIGN_NAME}-${BUCKYBALL_CTEST_TARGET}-ctest-rvv_kernel"
)
add_dependencies(toy-ctests-build ${RVV_KERNEL_TEST}-ctest-build)

add_buckyball_ctest(rvv_ball.c)
set(RVV_BALL_TEST
    "${DESIGN_NAME}-${BUCKYBALL_CTEST_TARGET}-ctest-rvv_ball"
)
add_dependencies(toy-ctests-build ${RVV_BALL_TEST}-ctest-build)

add_buckyball_ctest(mmio_minimal_test.c)
