set(BUCKYBALL_MLIR_GROUP_TARGET linalg-mlir-tests-build)
set(BUCKYBALL_MLIR_TEST_PREFIX linalg)

add_custom_target(linalg-mlir-tests-build)

set(TOY_LINALG_FUNCTIONAL_PASSES
  -convert-linalg-to-tile
  -convert-tile-to-buckyball
  -lower-buckyball-to-bank-ssa
  -assign-physical-banks
  ${BUCKYBALL_LOWER_BANK_SSA_TO_INTRINSICS}
  -convert-linalg-to-loops
  -expand-strided-metadata
  -lower-affine
  -convert-scf-to-cf
  -convert-cf-to-llvm
  ${BUCKYBALL_LOWER_BUCKYBALL}
  -convert-arith-to-llvm
  -convert-math-to-llvm
  -finalize-memref-to-llvm
  -convert-func-to-llvm
  -reconcile-unrealized-casts)

foreach(test conv2d_nhwc_hwcf conv2d_nhwc_fhwc conv2d_nchw_fchw)
  add_buckyball_mlir_test(${test}
    TARGET toy
    PASSES ${TOY_LINALG_FUNCTIONAL_PASSES})
endforeach()
