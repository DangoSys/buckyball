include(${OPTEST_WORKLOAD_DIR}/cmake/BuckyballMlirTests.cmake)

set(BUCKYBALL_MLIR_GROUP_TARGET balls-mlir-tests-build)
set(BUCKYBALL_MLIR_TEST_PREFIX ball)

add_buckyball_mlir_contract(matrix_16x16_i8_contract
  PASSES
    "--assign-physical-banks=bank_num=${BUCKYBALL_MLIR_BANK_NUM}"
    ${BUCKYBALL_LOWER_BANK_SSA_TO_INTRINSICS}
    ${BUCKYBALL_LOWER_BUCKYBALL})

add_buckyball_mlir_optest(matrix_16x16_i8)
add_buckyball_mlir_optest(matrix_matmul_16x16_i8)
