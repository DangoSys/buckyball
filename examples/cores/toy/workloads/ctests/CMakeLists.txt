add_subdirectory(gemmini_relu_trace_pipeline)
set(BUCKYBALL_CTEST_TARGET ${BUCKYBALL_CORE_WORKLOAD_TARGET})
file(RELATIVE_PATH RVV_KERNEL_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}
  ${WORKLOAD_SRC_DIR}/CTest/rvv/rvv_kernel.c)
add_buckyball_ctest(${RVV_KERNEL_SOURCE})
buckyball_ctest_name(RVV_KERNEL_TEST ${BUCKYBALL_CTEST_TARGET} rvv_kernel)
add_dependencies(core-ctests-build ${RVV_KERNEL_TEST}-ctest-build)

add_buckyball_ctest(rvv_ball.c)
buckyball_ctest_name(RVV_BALL_TEST ${BUCKYBALL_CTEST_TARGET} rvv_ball)
add_dependencies(core-ctests-build ${RVV_BALL_TEST}-ctest-build)
