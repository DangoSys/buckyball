set(MLIR_TEST_PREFIX ball)
add_mlir_test(mxint8quantize_random
  TARGET ${MLIR_TEST_TARGET}
  LIBRARIES bbsw-thread-${MLIR_TEST_TARGET}-linux
  BAREMETAL_LIBRARIES bbsw-thread-${MLIR_TEST_TARGET}-baremetal)
