set(BUCKYBALL_MLIR_GROUP_TARGET balls-mlir-tests-build)
set(BUCKYBALL_MLIR_TEST_PREFIX bank)

add_buckyball_mlir_test(relu_i32_mixed_sign TARGET ${BUCKYBALL_MLIR_ACTIVE_TARGET})
