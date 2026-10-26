# Linux programs installed into the goban rootfs, each with the hart it runs on.
set(BUCKYBALL_KERNEL_PROGRAMS
  0 goban-matmul-ctest-rvv_kernel-linux
  0 goban-matmul-ctest-rvv_silu-linux
  0 goban-matmul-ctest-rvv_swiglu-linux
  0 goban-matmul-ctest-rvv_snake-linux
  0 goban-matmul-ctest-rvv_mxfp8_quant-linux
  0 goban-matmul-ctest-rvv_matmul-linux
  0 goban-matmul-ctest-mvin_mvout_test-linux
  0 goban-matmul-ctest-mvin_mvout_stride_test-linux
  PARENT_SCOPE)
