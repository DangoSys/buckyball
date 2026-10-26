if(NOT DEFINED BUCKYBALL_MLIR_GROUP_TARGET)
  set(BUCKYBALL_MLIR_GROUP_TARGET balls-mlir-tests-build)
endif()
set(BUCKYBALL_MLIR_TEST_PREFIX bank)

set(_params "${BUCKYBALL_ISA_DIR}/${BUCKYBALL_MLIR_ACTIVE_TARGET}/params.h")
file(STRINGS "${_params}" _bank_lines REGEX "^#define BANK_LINES ")
string(REGEX REPLACE ".* " "" _lines "${_bank_lines}")
math(EXPR _remainder "${_lines} % 4")
if(_lines LESS 4 OR NOT (_remainder EQUAL 0))
  message(FATAL_ERROR
    "ToInt8 INT32 BankOp test requires BANK_LINES divisible by four "
    "(got lines=${_lines})")
endif()
# One QUANT_I32_TO_I8 instruction accepts at most 508 packed INT32 rows.
set(BB_QUANT_I32_INPUT_ROWS ${_lines})
if(BB_QUANT_I32_INPUT_ROWS GREATER 508)
  set(BB_QUANT_I32_INPUT_ROWS 508)
endif()
math(EXPR BB_QUANT_I32_OUTPUT_ROWS "${BB_QUANT_I32_INPUT_ROWS} / 4")
add_buckyball_mlir_test(quant_i32_pack16 TARGET ${BUCKYBALL_MLIR_ACTIVE_TARGET})
add_buckyball_mlir_test(quant_f32_pack16 TARGET ${BUCKYBALL_MLIR_ACTIVE_TARGET})
