set(MLIR_TEST_GROUP balls-mlir-tests-build)
if(NOT DESIGN_PROFILES_${MLIR_TEST_TARGET}_THREAD_MEM_BANK_WIDTH EQUAL 128 OR
   DESIGN_PROFILES_${MLIR_TEST_TARGET}_THREAD_MEM_BANK_ENTRIES LESS 408)
  message(FATAL_ERROR "Mxint8quantize MLIR vectors require 128-bit bank rows and at least 408 rows")
endif()
add_subdirectory(ball)
add_subdirectory(bank)
