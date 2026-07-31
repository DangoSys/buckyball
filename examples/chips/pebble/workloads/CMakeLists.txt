get_filename_component(BUCKYBALL_REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../.. ABSOLUTE)

set(BUCKYBALL_TOY_COMMON_DIR
  ${BUCKYBALL_REPO_ROOT}/examples/chips/toy/workloads/common)
set(BUCKYBALL_CHIP_COMMON_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/common/pebble.c)
set(BUCKYBALL_CHIP_COMMON_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/common/pebble.h)
set(BUCKYBALL_CHIP_COMMON_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/common)
include(${BUCKYBALL_REPO_ROOT}/examples/chips/toy/workloads/cmake/BuckyballWorkloads.cmake)

add_custom_target(pebble-ctests-build)
set(BUCKYBALL_CTEST_GROUP_TARGET pebble-ctests-build)

add_subdirectory(ctests)

set(BUCKYBALL_BALL_CTEST_DIRS
  ${BUCKYBALL_REPO_ROOT}/examples/balls/transpose/workloads/ctests
  ${BUCKYBALL_REPO_ROOT}/examples/balls/matrix/workloads/ctests
  ${BUCKYBALL_REPO_ROOT}/examples/balls/im2col/workloads/ctests
)
buckyball_add_ball_ctest_subdirs()

add_custom_target(chip-workloads-build ALL
  DEPENDS pebble-ctests-build
  COMMENT "Building pebble workloads"
  VERBATIM)
