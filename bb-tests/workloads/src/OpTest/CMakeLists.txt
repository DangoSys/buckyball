# Ensure buddy-mlir build dir is available
if(NOT DEFINED ENV{BUDDY_MLIR_BUILD_DIR})
  message(FATAL_ERROR
    "BUDDY_MLIR_BUILD_DIR is not set. Enter `nix develop` shell first.")
endif()

set(OPTEST_BASE_DIR ${OPTEST_WORKLOAD_DIR}/base)
set(OPTEST_TILE_DIR ${OPTEST_WORKLOAD_DIR}/tile)
set(OPTEST_LINALG_DIR ${OPTEST_WORKLOAD_DIR}/linalg)

add_subdirectory(base)

add_custom_target(toy-mlir-tests-build)
if(BUCKYBALL_WORKLOAD_CHIP STREQUAL "toy")
  add_subdirectory(${REPO_ROOT}/examples/chips/toy/workloads/mlir_tests
    ${CMAKE_CURRENT_BINARY_DIR}/chips/toy/mlir_tests)
endif()

add_custom_target(balls-mlir-tests-build)
file(GLOB BUCKYBALL_BALL_MLIR_CMAKES CONFIGURE_DEPENDS
  "${REPO_ROOT}/examples/balls/*/workloads/mlir_tests/CMakeLists.txt")
foreach(BUCKYBALL_BALL_MLIR_CMAKE ${BUCKYBALL_BALL_MLIR_CMAKES})
  get_filename_component(BUCKYBALL_BALL_MLIR_DIR
    ${BUCKYBALL_BALL_MLIR_CMAKE} DIRECTORY)
  get_filename_component(BUCKYBALL_BALL_WORKLOADS_DIR
    ${BUCKYBALL_BALL_MLIR_DIR} DIRECTORY)
  get_filename_component(BUCKYBALL_BALL_DIR
    ${BUCKYBALL_BALL_WORKLOADS_DIR} DIRECTORY)
  get_filename_component(BUCKYBALL_BALL_NAME ${BUCKYBALL_BALL_DIR} NAME)
  add_subdirectory(
    ${BUCKYBALL_BALL_MLIR_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/balls/${BUCKYBALL_BALL_NAME}/mlir_tests)
endforeach()

if(BUCKYBALL_WORKLOAD_CHIP STREQUAL "toy")
  add_subdirectory(tile)
  add_subdirectory(linalg)
else()
  add_custom_target(tile-OpTest-build)
  add_custom_target(linalg-OpTest-build)
endif()

add_custom_target(OpTest-all ALL DEPENDS
    base-OpTest-build
    toy-mlir-tests-build
    balls-mlir-tests-build
    tile-OpTest-build
    linalg-OpTest-build
)
