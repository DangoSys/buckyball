set(CTEST_PROFILE main)
add_ctest(reduce_quant.c)
add_ctest(dma_unaligned_stream.c)
add_subdirectory(compiler)
