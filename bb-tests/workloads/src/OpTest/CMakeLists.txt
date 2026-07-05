# Ensure buddy-mlir build dir is available
if(NOT DEFINED ENV{BUDDY_MLIR_BUILD_DIR})
  message(FATAL_ERROR
    "BUDDY_MLIR_BUILD_DIR is not set. Enter `nix develop` shell first.")
endif()

set(OPTEST_BASE_DIR ${OPTEST_WORKLOAD_DIR}/base)

add_subdirectory(base)

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

add_custom_target(OpTest-all ALL DEPENDS
  base-OpTest-build
  balls-mlir-tests-build
)
