if(NOT DEFINED MLIR_TEST_GROUP)
  set(MLIR_TEST_GROUP balls-mlir-tests-build)
endif()
set(MLIR_TEST_PREFIX bank)

set(_lines ${DESIGN_PROFILES_${MLIR_TEST_TARGET}_THREAD_MEM_BANK_ENTRIES})
math(EXPR _remainder "${_lines} % 4")
if(_lines LESS 4 OR NOT (_remainder EQUAL 0))
  message(FATAL_ERROR
    "ToInt8 INT32 BankOp test requires bank entries divisible by four "
    "(got lines=${_lines})")
endif()
# One QUANT_I32_TO_I8 instruction accepts at most 508 packed INT32 rows.
set(BB_QUANT_I32_INPUT_ROWS ${_lines})
if(BB_QUANT_I32_INPUT_ROWS GREATER 508)
  set(BB_QUANT_I32_INPUT_ROWS 508)
endif()
math(EXPR BB_QUANT_I32_OUTPUT_ROWS "${BB_QUANT_I32_INPUT_ROWS} / 4")
add_mlir_test(quant_i32_pack16 TARGET ${MLIR_TEST_TARGET})
add_mlir_test(quant_f32_pack16 TARGET ${MLIR_TEST_TARGET})
