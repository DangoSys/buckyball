set(BUCKYBALL_MLIR_GROUP_TARGET balls-mlir-tests-build)
set(BUCKYBALL_MLIR_TEST_PREFIX bank)

add_buckyball_mlir_test(matadd_bank_1024x8_i32 TARGET ${BUCKYBALL_MLIR_ACTIVE_TARGET})
