set(BUCKYBALL_MLIR_GROUP_TARGET linalg-mlir-tests-build)
set(BUCKYBALL_MLIR_TEST_PREFIX linalg)

add_custom_target(linalg-mlir-tests-build)

add_buckyball_mlir_test(conv2d_nhwc_hwcf
  TARGET toy
  PASSES
    -convert-linalg-to-buckyball
    -assign-physical-banks
    -lower-buckyball
    -convert-linalg-to-loops
    -expand-strided-metadata
    -lower-affine
    -convert-scf-to-cf
    -convert-cf-to-llvm
    -convert-arith-to-llvm
    -convert-math-to-llvm
    -finalize-memref-to-llvm
    -convert-func-to-llvm
    -reconcile-unrealized-casts)

add_buckyball_mlir_test(conv2d_nhwc_fhwc
  TARGET toy
  PASSES
    -convert-linalg-to-buckyball
    -assign-physical-banks
    -lower-buckyball
    -convert-linalg-to-loops
    -expand-strided-metadata
    -lower-affine
    -convert-scf-to-cf
    -convert-cf-to-llvm
    -convert-arith-to-llvm
    -convert-math-to-llvm
    -finalize-memref-to-llvm
    -convert-func-to-llvm
    -reconcile-unrealized-casts)

add_buckyball_mlir_test(conv2d_nchw_fchw
  TARGET toy
  PASSES
    -convert-linalg-to-buckyball
    -assign-physical-banks
    -lower-buckyball
    -convert-linalg-to-loops
    -expand-strided-metadata
    -lower-affine
    -convert-scf-to-cf
    -convert-cf-to-llvm
    -convert-arith-to-llvm
    -convert-math-to-llvm
    -finalize-memref-to-llvm
    -convert-func-to-llvm
    -reconcile-unrealized-casts)
