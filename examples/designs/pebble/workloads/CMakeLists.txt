get_filename_component(BUCKYBALL_REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../.. ABSOLUTE)

set(BUCKYBALL_DESIGN_COMMON_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/common/pebble.c)
set(BUCKYBALL_DESIGN_COMMON_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/common/pebble.h)
set(BUCKYBALL_DESIGN_COMMON_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/common)
set(BUCKYBALL_CTEST_TARGET pebble)

add_custom_target(pebble-ctests-build)
set(BUCKYBALL_CTEST_GROUP_TARGET pebble-ctests-build)

add_subdirectory(device)

add_custom_target(design-ctests-build DEPENDS pebble-ctests-build)

add_custom_target(design-workloads-build ALL
  DEPENDS
    design-ctests-build
    mlirtest-workloads-build
    soctests-build
  COMMENT "Building pebble workloads"
  VERBATIM)
