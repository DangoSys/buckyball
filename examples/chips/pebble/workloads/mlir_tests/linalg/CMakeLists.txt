set(BUCKYBALL_MLIR_GROUP_TARGET pebble-mlir-tests-build)
set(BUCKYBALL_MLIR_TEST_PREFIX linalg)

add_buckyball_mlir_contract(quant_matmul_f32_i8_contract
  TARGET pebble
  PASSES
    -convert-tile-to-buckyball
    -lower-buckyball-to-bank-ssa
    -assign-physical-banks
    ${BUCKYBALL_LOWER_BANK_SSA_TO_INTRINSICS}
    ${BUCKYBALL_LOWER_BUCKYBALL})

add_buckyball_mlir_contract(quant_matmul_f32_i8_edge_contract
  TARGET pebble
  PASSES
    -convert-tile-to-buckyball
    -lower-buckyball-to-bank-ssa
    -assign-physical-banks
    ${BUCKYBALL_LOWER_BANK_SSA_TO_INTRINSICS}
    ${BUCKYBALL_LOWER_BUCKYBALL})

add_buckyball_mlir_contract(quant_matmul_f32_i8_tensor_contract
  TARGET pebble
  PASSES
    -convert-tile-to-buckyball
    -lower-buckyball-to-bank-ssa
    -assign-physical-banks
    ${BUCKYBALL_LOWER_BANK_SSA_TO_INTRINSICS}
    ${BUCKYBALL_LOWER_BUCKYBALL})

add_buckyball_mlir_contract(quantized_generic_f32_i8_contract
  TARGET pebble
  PASSES
    -convert-linalg-to-tile
    -convert-tile-to-buckyball
    -lower-buckyball-to-bank-ssa
    -assign-physical-banks
    ${BUCKYBALL_LOWER_BANK_SSA_TO_INTRINSICS}
    ${BUCKYBALL_LOWER_BUCKYBALL})

add_buckyball_mlir_test(linalg_matmul_16x16_i8
  TARGET pebble
  PASSES
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

add_buckyball_mlir_test(linalg_transpose_16x16_i8
  TARGET pebble
  PASSES
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

add_buckyball_mlir_test(linalg_conv_2d_nhwc_hwcf_6x6_k3
  TARGET pebble
  PASSES
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
