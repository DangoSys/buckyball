if(NOT DEFINED MLIR_TEST_GROUP)
  set(MLIR_TEST_GROUP balls-mlir-tests-build)
endif()
set(MLIR_TEST_PREFIX ball)

add_mlir_test(im2col_6x6_k3 TARGET ${MLIR_TEST_TARGET})
