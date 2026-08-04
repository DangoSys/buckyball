include(${OPTEST_WORKLOAD_DIR}/cmake/BuckyballMlirTests.cmake)

set(BUCKYBALL_MLIR_GROUP_TARGET pebble-mlir-tests-build)
set(BUCKYBALL_MLIR_TEST_PREFIX tile)

set(PEBBLE_TILE_TO_ELF_PASSES
  -convert-tile-to-buckyball
  -lower-buckyball-to-bank-ssa
  "--assign-physical-banks=bank_num=${BUCKYBALL_MLIR_BANK_NUM}"
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

add_buckyball_mlir_optest(tile_matmul_16x16_i8 PASSES ${PEBBLE_TILE_TO_ELF_PASSES})
add_buckyball_mlir_optest(tile_matmul_16x16_f32 PASSES ${PEBBLE_TILE_TO_ELF_PASSES})
add_buckyball_mlir_optest(tile_matmul_1x256x120_f32 PASSES ${PEBBLE_TILE_TO_ELF_PASSES})
add_buckyball_mlir_optest(tile_transpose_16x16_i8 PASSES ${PEBBLE_TILE_TO_ELF_PASSES})
add_buckyball_mlir_optest(tile_transpose_120x256_f32 PASSES ${PEBBLE_TILE_TO_ELF_PASSES})
add_buckyball_mlir_optest(tile_transpose_128x128_dynstride_f32 PASSES ${PEBBLE_TILE_TO_ELF_PASSES})
add_buckyball_mlir_optest(tile_transpose_matmul_120x256_f32 PASSES ${PEBBLE_TILE_TO_ELF_PASSES})
add_buckyball_mlir_optest(tile_conv2d_6x6_k3 PASSES ${PEBBLE_TILE_TO_ELF_PASSES})
add_buckyball_mlir_optest(tile_conv2d_1x1_c16_oc8 PASSES ${PEBBLE_TILE_TO_ELF_PASSES})
add_buckyball_mlir_optest(tile_conv2d_1x1_c16_oc24 PASSES ${PEBBLE_TILE_TO_ELF_PASSES})
add_buckyball_mlir_optest(tile_conv2d_8x8_k3_c2_oc2 PASSES ${PEBBLE_TILE_TO_ELF_PASSES})
add_buckyball_mlir_optest(tile_conv2d_12x12_k5 PASSES ${PEBBLE_TILE_TO_ELF_PASSES})
add_buckyball_mlir_optest(tile_conv2d_28x28_k5 PASSES ${PEBBLE_TILE_TO_ELF_PASSES})
