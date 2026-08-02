include(${OPTEST_WORKLOAD_DIR}/cmake/BuckyballMlirTests.cmake)

set(BUCKYBALL_MLIR_GROUP_TARGET pebble-mlir-tests-build)
set(BUCKYBALL_MLIR_TEST_PREFIX tile)

set(PEBBLE_TILE_TO_ELF_PASSES
  -convert-tile-to-buckyball
  -lower-buckyball-to-bank-ssa
  "--assign-physical-banks=bank_num=16"
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

add_buckyball_mlir_optest(matmul_16x16_i8 PASSES ${PEBBLE_TILE_TO_ELF_PASSES})
add_buckyball_mlir_optest(transpose_16x16_i8 PASSES ${PEBBLE_TILE_TO_ELF_PASSES})
add_buckyball_mlir_optest(conv2d_6x6_k3 PASSES ${PEBBLE_TILE_TO_ELF_PASSES})
