if(NOT DEFINED BUCKYBALL_MLIR_GROUP_TARGET)
  set(BUCKYBALL_MLIR_GROUP_TARGET balls-mlir-tests-build)
endif()
set(BUCKYBALL_MLIR_TEST_PREFIX bank)

buckyball_require_smatmul_bank_shape(${BUCKYBALL_MLIR_ACTIVE_TARGET})
add_buckyball_mlir_test(smatmul_bias_two_terms TARGET ${BUCKYBALL_MLIR_ACTIVE_TARGET})
