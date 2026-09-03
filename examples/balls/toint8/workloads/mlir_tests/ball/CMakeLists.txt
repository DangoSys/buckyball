set(BUCKYBALL_MLIR_GROUP_TARGET balls-mlir-tests-build)
set(BUCKYBALL_MLIR_TEST_PREFIX ball)

add_buckyball_mlir_contract(quant_modes_contract
  TARGET ${BUCKYBALL_MLIR_ACTIVE_TARGET}
  PASSES
    -assign-physical-banks
    ${BUCKYBALL_LOWER_BANK_SSA_TO_INTRINSICS}
    ${BUCKYBALL_LOWER_BUCKYBALL})
