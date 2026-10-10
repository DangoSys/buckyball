get_filename_component(BUCKYBALL_REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../.. ABSOLUTE)

set(BUCKYBALL_CHIP_COMMON_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/common/pebble.c)
set(BUCKYBALL_CHIP_COMMON_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/common/pebble.h)
set(BUCKYBALL_CHIP_COMMON_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/common)
set(BUCKYBALL_CTEST_TARGET pebble)

add_custom_target(pebble-ctests-build)
set(BUCKYBALL_CTEST_GROUP_TARGET pebble-ctests-build)

add_subdirectory(ctests)
buckyball_add_ball_workload_subdirs(ctests)

add_custom_target(chip-ctests-build DEPENDS pebble-ctests-build)

add_custom_target(chip-workloads-build ALL
  DEPENDS
    chip-ctests-build
    mlirtest-workloads-build
    soctests-build
  COMMENT "Building pebble workloads"
  VERBATIM)
