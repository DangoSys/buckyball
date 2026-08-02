include(${OPTEST_WORKLOAD_DIR}/cmake/BuckyballMlirTests.cmake)

set(BUCKYBALL_MLIR_GROUP_TARGET balls-mlir-tests-build)
set(BUCKYBALL_MLIR_TEST_PREFIX bank)

add_buckyball_mlir_contract(int2fp_4x4_contract
  PASSES "--assign-physical-banks=bank_num=16")

add_buckyball_mlir_optest(int2fp_4x4)
