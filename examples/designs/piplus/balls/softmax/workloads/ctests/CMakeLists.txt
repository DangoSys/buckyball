# The softmax C tests: inputs built in C, the outputs checked for the
# properties of a softmax (the bit-exact reference is the BEMU model).
add_ctest(softmax_edges_test.c)
add_ctest(softmax_random_test.c)
add_ctest(softmax_scale_test.c)
