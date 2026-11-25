# The mxint8quant C tests: inputs built in C, the outputs checked against the
# definition (the bit-exact reference is the BEMU model).
add_ctest(quant_edges_test.c)
add_ctest(quant_random_test.c)
add_ctest(quant_update_test.c)
add_ctest(quant_columns_test.c)
add_ctest(quant_transpose_test.c)
