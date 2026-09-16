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

add_buckyball_mlir_test(mega_conv2d_s2_depthwise_s2
  TARGET pebble
  PASSES ${PEBBLE_TILE_FUNCTIONAL_PASSES})

file(READ ${CMAKE_CURRENT_SOURCE_DIR}/data/mega_global_avg_conv_exact_input.i8
  PEBBLE_GAP_CONV_INPUT_HEX HEX)
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/data/mega_global_avg_7x7_c288_input.i8
  PEBBLE_GAP_C288_INPUT_HEX HEX)
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/data/mega_global_avg_conv_exact_weight.i8
  PEBBLE_GAP_CONV_WEIGHT_HEX HEX)
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/data/mega_global_avg_conv2_exact_weight2.i8
  PEBBLE_GAP_CONV_WEIGHT2_HEX HEX)
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/data/mega_global_avg_conv2_exact_weight3.i8
  PEBBLE_GAP_CONV_WEIGHT3_HEX HEX)
add_buckyball_mlir_test(mega_global_avg_conv_exact
  TARGET pebble
  PASSES ${PEBBLE_TILE_FUNCTIONAL_PASSES})
add_buckyball_mlir_test(mega_global_avg_7x7_c288_exact
  TARGET pebble
  PASSES ${PEBBLE_TILE_FUNCTIONAL_PASSES})
add_buckyball_mlir_test(mega_global_avg_conv2_exact
  TARGET pebble
  PASSES ${PEBBLE_TILE_FUNCTIONAL_PASSES})
unset(PEBBLE_GAP_CONV_INPUT_HEX)
unset(PEBBLE_GAP_C288_INPUT_HEX)
unset(PEBBLE_GAP_CONV_WEIGHT_HEX)
unset(PEBBLE_GAP_CONV_WEIGHT2_HEX)
unset(PEBBLE_GAP_CONV_WEIGHT3_HEX)

add_buckyball_mlir_test(mega_se_gate_conv
  TARGET pebble
  PASSES ${PEBBLE_TILE_FUNCTIONAL_PASSES})

add_buckyball_mlir_test(mega_matmul
  TARGET pebble
  PASSES ${PEBBLE_TILE_FUNCTIONAL_PASSES})

foreach(test
    tile_matmul_16x16_i8
    tile_transpose_16x16_i8)
  add_buckyball_mlir_test(${test}
    TARGET pebble
    PASSES ${PEBBLE_TILE_FUNCTIONAL_PASSES})
endforeach()
