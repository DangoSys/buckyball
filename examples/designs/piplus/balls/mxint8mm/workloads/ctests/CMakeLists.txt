# The mxint8mm C tests: packed operands built in C whose products are exact in
# FP32, compared bit for bit (the bit-exact reference is the BEMU model).
add_ctest(mm_small_test.c)
add_ctest(mm_accumulate_test.c)
add_ctest(mm_accumulate_update_test.c)
add_ctest(mm_extremes_test.c)
add_ctest(mm_bias_test.c)
add_ctest(mm_accumulate_bias_test.c)
add_ctest(mm_tiles_test.c)
add_ctest(mm_blocks_test.c)
add_ctest(mm_linear_sequence_test.c)
add_ctest(mm_rename_hang_test.c)
