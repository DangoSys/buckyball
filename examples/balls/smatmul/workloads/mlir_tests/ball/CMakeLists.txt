set(BUCKYBALL_MLIR_GROUP_TARGET balls-mlir-tests-build)
set(BUCKYBALL_MLIR_TEST_PREFIX ball)

add_buckyball_mlir_test(smatmul_bias_accumulate
  TARGET ${BUCKYBALL_MLIR_ACTIVE_TARGET})
