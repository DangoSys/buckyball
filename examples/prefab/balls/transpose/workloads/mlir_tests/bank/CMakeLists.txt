if(NOT DEFINED MLIR_TEST_GROUP)
  set(MLIR_TEST_GROUP balls-mlir-tests-build)
endif()
set(MLIR_TEST_PREFIX bank)

add_mlir_test(transpose_i8_packed_matrix TARGET ${MLIR_TEST_TARGET})
