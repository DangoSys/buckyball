get_filename_component(BUCKYBALL_REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../.. ABSOLUTE)

set(BUCKYBALL_TOY_COMMON_DIR
  ${BUCKYBALL_REPO_ROOT}/examples/chips/toy/workloads/common)
include(${BUCKYBALL_REPO_ROOT}/examples/chips/toy/workloads/cmake/BuckyballWorkloads.cmake)

add_custom_target(pebble-ctests-build)
set(BUCKYBALL_CTEST_GROUP_TARGET pebble-ctests-build)

set(PEBBLE_BALL_CTEST_DIRS
  ${BUCKYBALL_REPO_ROOT}/examples/balls/transpose/workloads/ctests
)

foreach(PEBBLE_BALL_CTEST_DIR ${PEBBLE_BALL_CTEST_DIRS})
  get_filename_component(PEBBLE_BALL_WORKLOADS_DIR
    ${PEBBLE_BALL_CTEST_DIR} DIRECTORY)
  get_filename_component(PEBBLE_BALL_DIR
    ${PEBBLE_BALL_WORKLOADS_DIR} DIRECTORY)
  get_filename_component(PEBBLE_BALL_NAME ${PEBBLE_BALL_DIR} NAME)
  add_subdirectory(
    ${PEBBLE_BALL_CTEST_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/balls/${PEBBLE_BALL_NAME}/ctests)
endforeach()

set(PEBBLE_SYSTOLIC_CTEST_DIR
  ${BUCKYBALL_REPO_ROOT}/examples/balls/systolicarray/workloads/ctests)
add_buckyball_ctests(
  ${PEBBLE_SYSTOLIC_CTEST_DIR}/bfp_test.c
  ${PEBBLE_SYSTOLIC_CTEST_DIR}/systolic_matmul_4x4_ones.c
  ${PEBBLE_SYSTOLIC_CTEST_DIR}/systolic_matmul_8x8_ones.c
  ${PEBBLE_SYSTOLIC_CTEST_DIR}/systolic_matmul_16x16_ones.c
)

add_custom_target(chip-workloads-build ALL
  DEPENDS pebble-ctests-build
  COMMENT "Building pebble workloads"
  VERBATIM)
