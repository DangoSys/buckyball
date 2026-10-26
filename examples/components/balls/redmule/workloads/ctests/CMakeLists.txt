file(
    READ "${BUCKYBALL_ISA_DIR}/${BUCKYBALL_CTEST_TARGET}/params.h"
    REDMULE_PARAMS
)
string(REGEX MATCH "#define BANK_WIDTH ([0-9]+)" _ "${REDMULE_PARAMS}")
set(REDMULE_BANK_WIDTH ${CMAKE_MATCH_1})
string(REGEX MATCH "#define BANK_LINES ([0-9]+)" _ "${REDMULE_PARAMS}")
if(NOT REDMULE_BANK_WIDTH EQUAL 128 OR CMAKE_MATCH_1 LESS 18)
    message(
        FATAL_ERROR
        "RedmuleBall tests require 128-bit rows and at least 18 bank rows, got ${REDMULE_BANK_WIDTH} bits and ${CMAKE_MATCH_1} rows"
    )
endif()
add_buckyball_ctest(redmule_f16_test.c)
add_buckyball_ctest(redmule_f16_chain_test.c)
add_buckyball_ctest(redmule_f16_edges_test.c)
add_buckyball_ctest(redmule_f16_fused_test.c)
