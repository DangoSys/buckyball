get_filename_component(REPO_ROOT ${WORKLOAD_DIR}/../.. ABSOLUTE)

add_subdirectory(base)
add_subdirectory(${REPO_ROOT}/examples/chips/toy/workloads
  ${CMAKE_CURRENT_BINARY_DIR}/toy)
add_subdirectory(${REPO_ROOT}/examples/chips/goban/workloads
  ${CMAKE_CURRENT_BINARY_DIR}/goban)
add_subdirectory(embench)
add_subdirectory(coremark)
