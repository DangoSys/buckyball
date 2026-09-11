if(NOT DEFINED BUCKYBALL_MLIR_GROUP_TARGET)
  set(BUCKYBALL_MLIR_GROUP_TARGET balls-mlir-tests-build)
endif()
set(BUCKYBALL_MLIR_TEST_PREFIX bank)

buckyball_require_quant_i32_bank_shape(${BUCKYBALL_MLIR_ACTIVE_TARGET})
add_buckyball_mlir_test(quant_i32_pack16 TARGET ${BUCKYBALL_MLIR_ACTIVE_TARGET})
add_buckyball_mlir_test(quant_f32_pack16 TARGET ${BUCKYBALL_MLIR_ACTIVE_TARGET})
