set(REDMULE_BANK_WIDTH ${DESIGN_PROFILES_${BUCKYBALL_CTEST_TARGET}_THREAD_MEM_BANK_WIDTH})
set(REDMULE_BANK_LINES ${DESIGN_PROFILES_${BUCKYBALL_CTEST_TARGET}_THREAD_MEM_BANK_ENTRIES})
if(NOT REDMULE_BANK_WIDTH EQUAL 128 OR REDMULE_BANK_LINES LESS 18)
    message(
        FATAL_ERROR
        "RedmuleBall tests require 128-bit rows and at least 18 bank rows, got ${REDMULE_BANK_WIDTH} bits and ${REDMULE_BANK_LINES} rows"
    )
endif()
add_buckyball_ctest(redmule_f16_test.c)
add_buckyball_ctest(redmule_f16_chain_test.c)
add_buckyball_ctest(redmule_f16_edges_test.c)
add_buckyball_ctest(redmule_f16_fused_test.c)
