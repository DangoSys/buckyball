set(BUCKYBALL_MLIR_GROUP_TARGET pebble-mlir-tests-build)
set(BUCKYBALL_MLIR_TEST_PREFIX tile)

set(PEBBLE_TILE_FUNCTIONAL_PASSES
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

add_buckyball_mlir_test(mega_conv2d
  TARGET pebble
  PASSES ${PEBBLE_TILE_FUNCTIONAL_PASSES})

add_buckyball_mlir_test(mega_matmul
  TARGET pebble
  PASSES ${PEBBLE_TILE_FUNCTIONAL_PASSES})
