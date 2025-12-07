set(CTEST_GEMMINI_WORKLOAD_DIR ${CTEST_WORKLOAD_DIR}/gemmini)
set(CTEST_TOY_WORKLOAD_DIR ${CTEST_WORKLOAD_DIR}/toy)
set(CTEST_RVV_WORKLOAD_DIR ${CTEST_WORKLOAD_DIR}/rvv)

add_subdirectory(gemmini)
add_subdirectory(toy)
add_subdirectory(rvv)
