get_filename_component(REPO_ROOT ${WORKLOAD_DIR}/../.. ABSOLUTE)

add_subdirectory(base)
set(BUCKYBALL_CHIP_WORKLOAD_DIR
  ${REPO_ROOT}/examples/chips/${BUCKYBALL_WORKLOAD_CHIP}/workloads)
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
add_subdirectory(embench)
add_subdirectory(coremark)
