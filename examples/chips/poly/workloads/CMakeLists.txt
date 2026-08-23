get_filename_component(BUCKYBALL_REPO_ROOT
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)

add_subdirectory(
  ${BUCKYBALL_REPO_ROOT}/examples/chips/goban/workloads
  ${CMAKE_CURRENT_BINARY_DIR}/goban-workloads
)
