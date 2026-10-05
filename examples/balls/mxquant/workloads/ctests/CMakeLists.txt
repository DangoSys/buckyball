file(READ "${BUCKYBALL_ISA_DIR}/${BUCKYBALL_CTEST_TARGET}/params.h" MXQUANT_PARAMS)
string(REGEX MATCH "#define BANK_LINES ([0-9]+)" _ "${MXQUANT_PARAMS}")
set(MXQUANT_BANK_LINES ${CMAKE_MATCH_1})
string(REGEX MATCH "#define BANK_WIDTH ([0-9]+)" _ "${MXQUANT_PARAMS}")
math(EXPR MXQUANT_BANK_BYTES "${MXQUANT_BANK_LINES} * ${CMAKE_MATCH_1} / 8")
math(EXPR MXQUANT_BANK_TAIL "${MXQUANT_BANK_BYTES} % 128")
if(MXQUANT_BANK_BYTES LESS 3840 OR MXQUANT_BANK_TAIL)
  message(FATAL_ERROR
    "Mxquant CTests require banks >= 3840 bytes and a 32-element FP32 capacity; actual ${MXQUANT_BANK_BYTES} bytes")
endif()
add_buckyball_ctests(mxquant_values_test.c mxquant_capacity_test.c)
buckyball_ctest_name(TEST_NAME ${BUCKYBALL_CTEST_TARGET} mxquant_values_test)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}-baremetal APPEND
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../reference.h)
