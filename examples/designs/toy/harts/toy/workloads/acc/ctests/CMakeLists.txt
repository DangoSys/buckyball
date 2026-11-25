if("gemmini" IN_LIST DESIGN_PROFILES_toy_THREAD_BALLDOMAIN_BALLS AND
   "relu" IN_LIST DESIGN_PROFILES_toy_THREAD_BALLDOMAIN_BALLS AND
   "trace" IN_LIST DESIGN_PROFILES_toy_THREAD_BALLDOMAIN_BALLS)
  add_subdirectory(gemmini_relu_trace_pipeline)
endif()
set(CTEST_PROFILE toy)
set(CTEST_GROUP toy-ctests-build)
file(
    RELATIVE_PATH
    RVV_OPERATOR_SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${WORKLOAD_SRC_DIR}/CTest/rvv/rvv_operator.c
)
add_ctest(${RVV_OPERATOR_SOURCE})

add_ctest(rvv_ball.c)

add_ctest(mmio_minimal_test.c)
