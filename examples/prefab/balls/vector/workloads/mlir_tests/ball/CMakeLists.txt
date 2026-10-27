set(BUCKYBALL_MLIR_GROUP_TARGET balls-mlir-tests-build)
set(BUCKYBALL_MLIR_TEST_PREFIX ball)

add_buckyball_mlir_test(matmul_at_i8_16x16 TARGET ${BUCKYBALL_MLIR_ACTIVE_TARGET})
