if(NOT DEFINED MLIR_TEST_GROUP)
  set(MLIR_TEST_GROUP balls-mlir-tests-build)
endif()
set(MLIR_TEST_PREFIX ball)

add_mlir_test(quant_f32_to_i8
  TARGET ${MLIR_TEST_TARGET})
add_mlir_test(quant_i32_to_i8
  TARGET ${MLIR_TEST_TARGET})
