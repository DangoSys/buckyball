set(MLIR_TEST_PREFIX ball)
# The main programs the banks through the C API.
add_mlir_test(f32add_groups
  TARGET ${MLIR_TEST_TARGET}
  LIBRARIES bbsw-thread-${MLIR_TEST_TARGET}-linux
  BAREMETAL_LIBRARIES bbsw-thread-${MLIR_TEST_TARGET}-baremetal)
