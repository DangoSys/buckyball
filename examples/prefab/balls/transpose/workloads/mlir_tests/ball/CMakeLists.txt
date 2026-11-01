if(NOT DEFINED MLIR_TEST_GROUP)
  set(MLIR_TEST_GROUP balls-mlir-tests-build)
endif()
set(MLIR_TEST_PREFIX ball)

add_mlir_test(transpose_16x16_i8 TARGET ${MLIR_TEST_TARGET})
