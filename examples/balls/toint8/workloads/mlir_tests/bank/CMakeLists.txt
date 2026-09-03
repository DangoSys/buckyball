set(BUCKYBALL_MLIR_GROUP_TARGET balls-mlir-tests-build)
set(BUCKYBALL_MLIR_TEST_PREFIX bank)

add_buckyball_mlir_test(quant_f32_to_i8
  TARGET ${BUCKYBALL_MLIR_ACTIVE_TARGET})
add_buckyball_mlir_test(quant_i32_to_i8
  TARGET ${BUCKYBALL_MLIR_ACTIVE_TARGET})
