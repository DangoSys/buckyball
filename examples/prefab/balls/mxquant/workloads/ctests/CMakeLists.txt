math(
    EXPR MXQUANT_BANK_BYTES
    "${DESIGN_PROFILES_${BUCKYBALL_CTEST_TARGET}_THREAD_MEM_BANK_ENTRIES} * ${DESIGN_PROFILES_${BUCKYBALL_CTEST_TARGET}_THREAD_MEM_BANK_WIDTH} / 8"
)
math(EXPR MXQUANT_BANK_TAIL "${MXQUANT_BANK_BYTES} % 128")
if(MXQUANT_BANK_BYTES LESS 3840 OR MXQUANT_BANK_TAIL)
    message(
        FATAL_ERROR
        "Mxquant CTests require banks >= 3840 bytes and a 32-element FP32 capacity; actual ${MXQUANT_BANK_BYTES} bytes"
    )
endif()
add_buckyball_ctest(mxquant_values_test.c)
add_buckyball_ctest(mxquant_capacity_test.c)
set(TEST_NAME
    "${DESIGN_NAME}-${BUCKYBALL_CTEST_TARGET}-ctest-mxquant_values_test"
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}-baremetal
    APPEND
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../reference.h
)
