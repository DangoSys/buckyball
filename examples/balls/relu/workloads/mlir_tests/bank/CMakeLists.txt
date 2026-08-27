set(BUCKYBALL_MLIR_GROUP_TARGET balls-mlir-tests-build)
set(BUCKYBALL_MLIR_TEST_PREFIX bank)

add_buckyball_mlir_contract(relu_bank_256x16_i32_contract
  TARGET ${BUCKYBALL_MLIR_ACTIVE_TARGET}
  PASSES -assign-physical-banks)

add_buckyball_mlir_test(relu_bank_256x16_i32 TARGET ${BUCKYBALL_MLIR_ACTIVE_TARGET})
