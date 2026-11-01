set(MLIR_TEST_GROUP balls-mlir-tests-build)
set(MLIR_TEST_PREFIX ball)

add_mlir_test(matmul_at_i8_16x16 TARGET ${MLIR_TEST_TARGET})
