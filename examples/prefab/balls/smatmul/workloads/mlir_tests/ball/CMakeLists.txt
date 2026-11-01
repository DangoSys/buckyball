if(NOT DEFINED MLIR_TEST_GROUP)
  set(MLIR_TEST_GROUP balls-mlir-tests-build)
endif()
set(MLIR_TEST_PREFIX ball)

add_mlir_test(smatmul_bias_accumulate
  TARGET ${MLIR_TEST_TARGET})
