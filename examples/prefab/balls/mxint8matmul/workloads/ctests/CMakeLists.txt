# The mxint8matmul C tests: packed operands built in C whose products are
# exact in FP32, compared bit for bit (the bit-exact reference is the BEMU
# model).
add_ctest(mm_small_test.c)
add_ctest(mm_accumulate_test.c)
add_ctest(mm_extremes_test.c)
add_ctest(mm_long_test.c)
