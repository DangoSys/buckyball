add_buckyball_ctest(mxmm_fp32_test.c)
add_buckyball_ctest(mxmm_mxfp8_test.c)
add_buckyball_ctest(mxmm_bank_capacity_test.c)

math(
    EXPR MXMM_BANK_BYTES
    "${DESIGN_PROFILES_${BUCKYBALL_CTEST_TARGET}_THREAD_MEM_BANK_ENTRIES} * ${DESIGN_PROFILES_${BUCKYBALL_CTEST_TARGET}_THREAD_MEM_BANK_WIDTH} / 8"
)
if(MXMM_BANK_BYTES LESS 15840)
    message(
        FATAL_ERROR
        "Mxmm A-window row test requires 15840-byte banks; actual ${MXMM_BANK_BYTES} bytes"
    )
endif()
add_buckyball_ctest(mxmm_window_test.c)
add_buckyball_ctest(mxmm_window_rows_test.c)
add_buckyball_ctest(mxmm_window_chain_test.c)
