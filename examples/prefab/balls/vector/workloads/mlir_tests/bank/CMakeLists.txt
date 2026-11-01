set(MLIR_TEST_GROUP balls-mlir-tests-build)
set(MLIR_TEST_PREFIX bank)

add_mlir_test(matmul_at_i8_bank_depth TARGET ${MLIR_TEST_TARGET})
