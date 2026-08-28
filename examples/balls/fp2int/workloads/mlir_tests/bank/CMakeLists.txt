set(BUCKYBALL_MLIR_GROUP_TARGET balls-mlir-tests-build)
set(BUCKYBALL_MLIR_TEST_PREFIX bank)

add_buckyball_mlir_contract(fp2int_4x4_contract
  TARGET ${BUCKYBALL_MLIR_ACTIVE_TARGET}
  PASSES -assign-physical-banks)

add_buckyball_mlir_test(fp2int_4x4 TARGET ${BUCKYBALL_MLIR_ACTIVE_TARGET})
