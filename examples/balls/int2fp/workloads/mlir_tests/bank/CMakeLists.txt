set(BUCKYBALL_MLIR_GROUP_TARGET balls-mlir-tests-build)
set(BUCKYBALL_MLIR_TEST_PREFIX bank)

add_buckyball_mlir_test(int2fp_4x4 TARGET ${BUCKYBALL_MLIR_ACTIVE_TARGET})
add_buckyball_mlir_test(int2fp_relu_4x4 TARGET ${BUCKYBALL_MLIR_ACTIVE_TARGET})
