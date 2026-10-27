if(NOT DEFINED BUCKYBALL_MLIR_GROUP_TARGET)
  set(BUCKYBALL_MLIR_GROUP_TARGET balls-mlir-tests-build)
endif()
set(BUCKYBALL_MLIR_TEST_PREFIX bank)

set(_lines ${DESIGN_PROFILES_${BUCKYBALL_MLIR_ACTIVE_TARGET}_THREAD_MEM_BANK_ENTRIES})
# int32_to_fp32 rows hold sixteen i32, four bank rows each.
math(EXPR BB_INT2FP_ROWS "${_lines} / 4")
add_buckyball_mlir_test(int2fp_unit_scale TARGET ${BUCKYBALL_MLIR_ACTIVE_TARGET})
