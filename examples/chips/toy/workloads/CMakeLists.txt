get_filename_component(BUCKYBALL_REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../.. ABSOLUTE)

set(BUCKYBALL_TOY_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuckyballWorkloads.cmake)

add_custom_target(toy-ctests-build)
set(BUCKYBALL_CTEST_GROUP_TARGET toy-ctests-build)

add_subdirectory(ctests)
add_subdirectory(mlir_tests)

file(GLOB BUCKYBALL_BALL_CTEST_DIRS CONFIGURE_DEPENDS
  "${BUCKYBALL_REPO_ROOT}/examples/balls/*/workloads/ctests")
buckyball_add_ball_ctest_subdirs()

add_custom_target(chip-workloads-build ALL
  DEPENDS
    toy-ctests-build
    toy-mlir-tests-build
  COMMENT "Building toy workloads"
  VERBATIM)
