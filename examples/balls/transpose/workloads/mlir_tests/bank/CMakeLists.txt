set(BUCKYBALL_MLIR_GROUP_TARGET balls-mlir-tests-build)
set(BUCKYBALL_MLIR_TEST_PREFIX bank)

add_buckyball_mlir_test(transpose_16x16_i8 TARGET ${BUCKYBALL_MLIR_ACTIVE_TARGET})
