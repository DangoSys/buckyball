include(${OPTEST_WORKLOAD_DIR}/cmake/BuckyballMlirTests.cmake)

set(BUCKYBALL_MLIR_GROUP_TARGET balls-mlir-tests-build)
set(BUCKYBALL_MLIR_TEST_PREFIX bank)

add_buckyball_mlir_contract(fp2int_4x4_contract
  PASSES "--assign-physical-banks=bank_num=${BUCKYBALL_MLIR_BANK_NUM}")

add_buckyball_mlir_optest(fp2int_4x4)
