# The workload root chooses the compiler instance for this chip's core.
if(NOT DEFINED BUDDY_BINARY_DIR)
  message(FATAL_ERROR
    "BUDDY_BINARY_DIR must be set by the workload root.")
endif()

set(MLIRTEST_BASE_DIR ${MLIRTEST_WORKLOAD_DIR}/base)

add_subdirectory(${MLIRTEST_WORKLOAD_DIR}/base
  ${CMAKE_CURRENT_BINARY_DIR}/mlir_tests/base)

add_custom_target(balls-mlir-tests-build)
file(GLOB BUCKYBALL_BALL_MLIR_CMAKES CONFIGURE_DEPENDS
  "${BUCKYBALL_REPO_ROOT}/examples/balls/*/workloads/mlir_tests/CMakeLists.txt")
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

add_custom_target(MLIRTest-all ALL DEPENDS
  base-MLIRTest-build
  balls-mlir-tests-build
)
