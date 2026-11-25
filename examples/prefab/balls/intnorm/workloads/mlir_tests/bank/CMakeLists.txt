set(MLIR_TEST_PREFIX bank)
add_mlir_test(intnorm_round
  TARGET ${MLIR_TEST_TARGET}
  LIBRARIES bbsw-thread-${MLIR_TEST_TARGET}-linux
  BAREMETAL_LIBRARIES bbsw-thread-${MLIR_TEST_TARGET}-baremetal)
