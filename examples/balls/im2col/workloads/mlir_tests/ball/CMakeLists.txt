set(BUCKYBALL_MLIR_GROUP_TARGET balls-mlir-tests-build)
set(BUCKYBALL_MLIR_TEST_PREFIX ball)

add_buckyball_mlir_contract(im2col_6x6_k3_contract
  TARGET ${BUCKYBALL_MLIR_ACTIVE_TARGET}
  PASSES
    -assign-physical-banks
    ${BUCKYBALL_LOWER_BANK_SSA_TO_INTRINSICS}
    ${BUCKYBALL_LOWER_BUCKYBALL})

add_buckyball_mlir_test(im2col_6x6_k3 TARGET ${BUCKYBALL_MLIR_ACTIVE_TARGET})
