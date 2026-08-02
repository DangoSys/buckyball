include(${OPTEST_WORKLOAD_DIR}/cmake/BuckyballMlirTests.cmake)

set(BUCKYBALL_MLIR_GROUP_TARGET balls-mlir-tests-build)
set(BUCKYBALL_MLIR_TEST_PREFIX ball)

add_buckyball_mlir_contract(transpose_16x16_i8_contract
  PASSES
    "--assign-physical-banks=bank_num=16"
    ${BUCKYBALL_LOWER_BANK_SSA_TO_INTRINSICS}
    ${BUCKYBALL_LOWER_BUCKYBALL})

add_buckyball_mlir_optest(transpose_16x16_i8)
