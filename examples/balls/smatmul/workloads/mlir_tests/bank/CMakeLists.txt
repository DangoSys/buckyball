include(${MLIRTEST_WORKLOAD_DIR}/cmake/BuckyballMlirTests.cmake)

set(BUCKYBALL_MLIR_GROUP_TARGET balls-mlir-tests-build)
set(BUCKYBALL_MLIR_TEST_PREFIX bank)

add_buckyball_mlir_contract(smatmul_16x16_i8_contract
  PASSES "--assign-physical-banks=bank_num=${BUCKYBALL_MLIR_BANK_NUM}")

add_buckyball_mlir_test(smatmul_16x16_i8)
