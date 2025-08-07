set(CTEST_WORKLOAD_DIR ${WORKLOAD_DIR}/CTest)
set(E2ETEST_WORKLOAD_DIR ${WORKLOAD_DIR}/E2ETest)
set(OPTEST_WORKLOAD_DIR ${WORKLOAD_DIR}/OpTest)
set(TUTORIAL_WORKLOAD_DIR ${WORKLOAD_DIR}/tutorial)

add_subdirectory(CTest)
add_subdirectory(E2ETest)
add_subdirectory(OpTest)
add_subdirectory(tutorial)
