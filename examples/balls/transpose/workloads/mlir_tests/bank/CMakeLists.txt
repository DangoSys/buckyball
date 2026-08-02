include(${OPTEST_WORKLOAD_DIR}/cmake/BuckyballMlirTests.cmake)

set(BUCKYBALL_MLIR_GROUP_TARGET balls-mlir-tests-build)
set(BUCKYBALL_MLIR_TEST_PREFIX transpose)

add_buckyball_mlir_contract(bank_ssa_contract
  PASSES "--assign-physical-banks=bank_num=16")

add_buckyball_mlir_optest(bank_ssa)
