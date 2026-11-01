if(NOT DEFINED MLIR_TEST_GROUP)
  set(MLIR_TEST_GROUP balls-mlir-tests-build)
endif()
set(MLIR_TEST_PREFIX bank)

set(_lines ${DESIGN_PROFILES_${MLIR_TEST_TARGET}_THREAD_MEM_BANK_ENTRIES})
# int32_to_fp32 rows hold sixteen i32, four bank rows each.
math(EXPR BB_INT2FP_ROWS "${_lines} / 4")
add_mlir_test(int2fp_unit_scale TARGET ${MLIR_TEST_TARGET})
