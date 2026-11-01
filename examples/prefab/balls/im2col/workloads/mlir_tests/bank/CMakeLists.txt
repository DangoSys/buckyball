if(NOT DEFINED MLIR_TEST_GROUP)
  set(MLIR_TEST_GROUP balls-mlir-tests-build)
endif()
set(MLIR_TEST_PREFIX bank)

add_mlir_test(im2col_k3_two_blocks TARGET ${MLIR_TEST_TARGET})
