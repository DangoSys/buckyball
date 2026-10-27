if(NOT DEFINED BUCKYBALL_MLIR_GROUP_TARGET)
  set(BUCKYBALL_MLIR_GROUP_TARGET balls-mlir-tests-build)
endif()
set(BUCKYBALL_MLIR_TEST_PREFIX bank)

set(_lines ${DESIGN_PROFILES_${BUCKYBALL_MLIR_ACTIVE_TARGET}_THREAD_MEM_BANK_ENTRIES})
set(_width ${DESIGN_PROFILES_${BUCKYBALL_MLIR_ACTIVE_TARGET}_THREAD_MEM_BANK_WIDTH})
math(EXPR _line_remainder "${_lines} % 64")
if(NOT (_width EQUAL 128) OR _lines LESS 64 OR
   NOT (_line_remainder EQUAL 0))
  message(FATAL_ERROR
    "SMatMul BankOp test requires 128-bit bank rows and bank entries a multiple "
    "of 64 (got width=${_width}, lines=${_lines})")
endif()
math(EXPR BB_SMM_M "${_lines} / 4")
add_buckyball_mlir_test(smatmul_bias_two_terms TARGET ${BUCKYBALL_MLIR_ACTIVE_TARGET})
