set(BUCKYBALL_MLIR_GROUP_TARGET balls-mlir-tests-build)
set(BUCKYBALL_MLIR_TEST_PREFIX bank)

add_buckyball_mlir_contract(transpose_16x16_i8_contract
  TARGET ${BUCKYBALL_MLIR_ACTIVE_TARGET}
  PASSES -assign-physical-banks)

add_buckyball_mlir_test(transpose_16x16_i8 TARGET ${BUCKYBALL_MLIR_ACTIVE_TARGET})
