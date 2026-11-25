
set(BUCKYBALL_DESIGN_COMMON_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/common/pebble.c)
set(BUCKYBALL_DESIGN_COMMON_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/common/pebble.h)
set(BUCKYBALL_DESIGN_COMMON_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/common)
set(CTEST_PROFILE pebble)

add_custom_target(pebble-ctests-build)
set(CTEST_GROUP pebble-ctests-build)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../../harts/pebble/workloads" harts/pebble)

add_custom_target(design-ctests-build DEPENDS pebble-ctests-build)

add_custom_target(design-workloads-build ALL
  DEPENDS
    design-ctests-build
    mlirtest-workloads-build
    soctests-build
  COMMENT "Building pebble workloads"
  VERBATIM)
