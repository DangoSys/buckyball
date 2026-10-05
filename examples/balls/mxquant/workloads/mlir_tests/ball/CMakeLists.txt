set(BUCKYBALL_MLIR_TEST_PREFIX ball)
add_buckyball_mlir_test(mxquant TARGET ${BUCKYBALL_MLIR_ACTIVE_TARGET})
buckyball_mlirtest_name(TEST_NAME ${BUCKYBALL_MLIR_ACTIVE_TARGET} ball_mxquant)
foreach(KIND baremetal linux)
  add_custom_command(OUTPUT ${TEST_NAME}-${KIND} APPEND
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../reference.h)
endforeach()
