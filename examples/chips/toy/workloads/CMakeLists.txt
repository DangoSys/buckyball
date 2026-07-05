get_filename_component(BUCKYBALL_REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../.. ABSOLUTE)

set(BUCKYBALL_TOY_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuckyballWorkloads.cmake)

add_custom_target(toy-ctests-build)
set(BUCKYBALL_CTEST_GROUP_TARGET toy-ctests-build)

add_subdirectory(ctests)
add_subdirectory(mlir_tests)

file(GLOB BUCKYBALL_BALL_CTEST_DIRS CONFIGURE_DEPENDS
  "${BUCKYBALL_REPO_ROOT}/examples/balls/*/workloads/ctests")

foreach(BUCKYBALL_BALL_CTEST_DIR ${BUCKYBALL_BALL_CTEST_DIRS})
  get_filename_component(BUCKYBALL_BALL_WORKLOADS_DIR ${BUCKYBALL_BALL_CTEST_DIR} DIRECTORY)
  get_filename_component(BUCKYBALL_BALL_DIR ${BUCKYBALL_BALL_WORKLOADS_DIR} DIRECTORY)
  get_filename_component(BUCKYBALL_BALL_NAME ${BUCKYBALL_BALL_DIR} NAME)
  add_subdirectory(
    ${BUCKYBALL_BALL_CTEST_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/balls/${BUCKYBALL_BALL_NAME}/ctests)
endforeach()

add_custom_target(chip-workloads-build ALL
  DEPENDS
    toy-ctests-build
    toy-mlir-tests-build
  COMMENT "Building toy workloads"
  VERBATIM)
