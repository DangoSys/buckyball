include(${OPTEST_WORKLOAD_DIR}/cmake/BuckyballMlirTests.cmake)

set(BUCKYBALL_MLIR_GROUP_TARGET balls-mlir-tests-build)
set(BUCKYBALL_MLIR_TEST_PREFIX bank)

add_buckyball_mlir_contract(int2fp_4x4_contract
  PASSES "--assign-physical-banks=bank_num=${BUCKYBALL_MLIR_BANK_NUM}")

add_buckyball_mlir_optest(int2fp_4x4)
add_buckyball_mlir_optest(int32_to_int8_2x16)
