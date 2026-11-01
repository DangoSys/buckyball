set(MLIR_TEST_GROUP balls-mlir-tests-build)
set(MLIR_TEST_PREFIX bank)

set(_lines ${DESIGN_PROFILES_${MLIR_TEST_TARGET}_THREAD_MEM_BANK_ENTRIES})
# mxfp4_i8 writes two bank rows per block; one scale row holds sixteen blocks.
math(EXPR BB_MXFP_BLOCKS "${_lines} / 2")
math(EXPR BB_MXFP_SCALE_ROWS "(${BB_MXFP_BLOCKS} + 15) / 16")
add_mlir_test(mxfp4_i8_bank_blocks TARGET ${MLIR_TEST_TARGET})
