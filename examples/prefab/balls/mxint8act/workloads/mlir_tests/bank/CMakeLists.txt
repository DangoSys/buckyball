set(MLIR_TEST_PREFIX bank)
add_mlir_test(mxint8act TARGET ${MLIR_TEST_TARGET})
add_mlir_test(mxint8act_reference TARGET ${MLIR_TEST_TARGET} PASSES
  -lower-frontend-fake-op "-one-shot-bufferize=bufferize-function-boundaries"
  -convert-linalg-to-loops -expand-strided-metadata -lower-affine
  -assign-physical-banks -lower-acc -convert-scf-to-cf -convert-cf-to-llvm
  -convert-arith-to-llvm -convert-math-to-llvm -finalize-memref-to-llvm
  -convert-func-to-llvm -reconcile-unrealized-casts)
