get_filename_component(REPO_ROOT ${WORKLOAD_DIR}/../.. ABSOLUTE)

add_subdirectory(base)
if(BUCKYBALL_COMPILER_CHIP STREQUAL "toy")
  add_subdirectory(${REPO_ROOT}/examples/chips/toy/workloads
    ${CMAKE_CURRENT_BINARY_DIR}/toy)
else()
  add_custom_target(toy-workloads-build)
endif()
if(BUCKYBALL_COMPILER_CHIP STREQUAL "goban")
  add_subdirectory(${REPO_ROOT}/examples/chips/goban/workloads
    ${CMAKE_CURRENT_BINARY_DIR}/goban)
else()
  add_custom_target(goban-workloads-build)
endif()
add_subdirectory(embench)
add_subdirectory(coremark)
