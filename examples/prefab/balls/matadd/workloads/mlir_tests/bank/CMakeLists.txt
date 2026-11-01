if(NOT DEFINED MLIR_TEST_GROUP)
  set(MLIR_TEST_GROUP balls-mlir-tests-build)
endif()
set(MLIR_TEST_PREFIX bank)

add_mlir_test(matadd_i32_elementwise TARGET ${MLIR_TEST_TARGET})
