get_filename_component(REPO_ROOT ${WORKLOAD_DIR}/../.. ABSOLUTE)

set(CTEST_WORKLOAD_DIR ${WORKLOAD_SRC_DIR}/CTest)
set(MODELTEST_WORKLOAD_DIR ${WORKLOAD_SRC_DIR}/ModelTest)
set(MLIRTEST_WORKLOAD_DIR ${WORKLOAD_SRC_DIR}/MLIRTest)

add_subdirectory(CTest)
add_subdirectory(ModelTest)
add_subdirectory(MLIRTest)
