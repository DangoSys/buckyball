set(BUCKYBALL_MLIR_GROUP_TARGET balls-mlir-tests-build)
set(BUCKYBALL_MLIR_TEST_PREFIX bank)

add_buckyball_mlir_contract(smatmul_ws_16x512_i8_contract
  TARGET ${BUCKYBALL_MLIR_ACTIVE_TARGET}
  PASSES -assign-physical-banks)

add_buckyball_mlir_test(smatmul_ws_16x512_i8 TARGET ${BUCKYBALL_MLIR_ACTIVE_TARGET})
