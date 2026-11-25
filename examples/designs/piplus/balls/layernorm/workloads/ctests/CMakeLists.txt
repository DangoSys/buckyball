# The layer normalization C tests: inputs built in C, the outputs checked
# against the normalization in double precision (the bit-exact reference is
# the BEMU model).
add_ctest(layernorm_test.c)
add_ctest(rmsnorm_test.c)
