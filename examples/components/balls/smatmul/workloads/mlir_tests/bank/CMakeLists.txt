if(NOT DEFINED BUCKYBALL_MLIR_GROUP_TARGET)
  set(BUCKYBALL_MLIR_GROUP_TARGET balls-mlir-tests-build)
endif()
set(BUCKYBALL_MLIR_TEST_PREFIX bank)

set(_params "${BUCKYBALL_ISA_DIR}/${BUCKYBALL_MLIR_ACTIVE_TARGET}/params.h")
file(STRINGS "${_params}" _bank_lines REGEX "^#define BANK_LINES ")
file(STRINGS "${_params}" _bank_width REGEX "^#define BANK_WIDTH ")
string(REGEX REPLACE ".* " "" _lines "${_bank_lines}")
string(REGEX REPLACE ".* " "" _width "${_bank_width}")
math(EXPR _line_remainder "${_lines} % 64")
if(NOT (_width EQUAL 128) OR _lines LESS 64 OR
   NOT (_line_remainder EQUAL 0))
  message(FATAL_ERROR
    "SMatMul BankOp test requires BANK_WIDTH=128 and BANK_LINES a multiple "
    "of 64 (got width=${_width}, lines=${_lines})")
endif()
math(EXPR BB_SMM_M "${_lines} / 4")
add_buckyball_mlir_test(smatmul_bias_two_terms TARGET ${BUCKYBALL_MLIR_ACTIVE_TARGET})
