get_filename_component(REPO_ROOT ${WORKLOAD_DIR}/../.. ABSOLUTE)

set(CTEST_WORKLOAD_DIR ${WORKLOAD_SRC_DIR}/CTest)
set(MODELTEST_WORKLOAD_DIR ${WORKLOAD_SRC_DIR}/ModelTest)
set(MLIRTEST_WORKLOAD_DIR ${WORKLOAD_SRC_DIR}/MLIRTest)

include(${MLIRTEST_WORKLOAD_DIR}/CMakeLists.txt)
add_subdirectory(CTest)
add_subdirectory(ModelTest)
