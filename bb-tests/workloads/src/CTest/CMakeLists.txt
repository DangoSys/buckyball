add_subdirectory(${CTEST_WORKLOAD_DIR}/base
  ${CMAKE_CURRENT_BINARY_DIR}/ctests/base)
add_subdirectory(${CTEST_WORKLOAD_DIR}/tlb
  ${CMAKE_CURRENT_BINARY_DIR}/ctests/tlb)
add_subdirectory(${CTEST_WORKLOAD_DIR}/tutorial
  ${CMAKE_CURRENT_BINARY_DIR}/ctests/tutorial)
set(BUCKYBALL_CHIP_WORKLOAD_DIR
  ${BUCKYBALL_REPO_ROOT}/examples/chips/${BUCKYBALL_WORKLOAD_CHIP}/workloads)
if(NOT EXISTS ${BUCKYBALL_CHIP_WORKLOAD_DIR}/CMakeLists.txt)
  message(FATAL_ERROR
    "Buckyball chip workloads do not exist: ${BUCKYBALL_CHIP_WORKLOAD_DIR}")
endif()
add_subdirectory(${BUCKYBALL_CHIP_WORKLOAD_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/chips/${BUCKYBALL_WORKLOAD_CHIP})
if(NOT TARGET chip-workloads-build)
  message(FATAL_ERROR
    "Buckyball chip workloads must define target chip-workloads-build: ${BUCKYBALL_WORKLOAD_CHIP}")
endif()
add_subdirectory(${CTEST_WORKLOAD_DIR}/embench
  ${CMAKE_CURRENT_BINARY_DIR}/ctests/embench)
add_subdirectory(${CTEST_WORKLOAD_DIR}/coremark
  ${CMAKE_CURRENT_BINARY_DIR}/ctests/coremark)
