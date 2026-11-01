if(NOT DEFINED MLIR_TEST_GROUP)
  set(MLIR_TEST_GROUP balls-mlir-tests-build)
endif()
set(MLIR_TEST_PREFIX ball)

add_mlir_test(int2fp_4x4 TARGET ${MLIR_TEST_TARGET})
add_mlir_test(int2fp_relu_4x4 TARGET ${MLIR_TEST_TARGET})
