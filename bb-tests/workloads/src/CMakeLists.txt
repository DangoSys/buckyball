set(CTEST_WORKLOAD_DIR ${WORKLOAD_SRC_DIR}/CTest)
set(MLIR_WORKLOAD_DIR ${WORKLOAD_SRC_DIR}/MLIRTest)
set(SOC_WORKLOAD_DIR ${WORKLOAD_SRC_DIR}/SocTest)

add_subdirectory(CTest)
add_subdirectory(MLIRTest)
add_subdirectory(SocTest)
