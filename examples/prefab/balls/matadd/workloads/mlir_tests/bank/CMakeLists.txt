if(NOT DEFINED BUCKYBALL_MLIR_GROUP_TARGET)
  set(BUCKYBALL_MLIR_GROUP_TARGET balls-mlir-tests-build)
endif()
set(BUCKYBALL_MLIR_TEST_PREFIX bank)

add_buckyball_mlir_test(matadd_i32_elementwise TARGET ${BUCKYBALL_MLIR_ACTIVE_TARGET})
