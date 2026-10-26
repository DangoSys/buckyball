set(BUCKYBALL_MLIR_GROUP_TARGET pebble-mlir-tests-build)
set(BUCKYBALL_MLIR_TEST_PREFIX linalg)

foreach(test
    linalg_relu_513x17_i32
    linalg_matmul_16x16_i8
    linalg_transpose_16x16_i8
    linalg_transpose_detect_1x84x100_f32
    linalg_transpose_nchw_nhwc_1x3x8x8_f32
    linalg_transpose_oihw_ohwi_4x8x3x3_f32
    linalg_mega_conv2d_k3_c16)
  add_buckyball_mlir_test(${test}
    TARGET pebble
    PASSES ${PEBBLE_CUSTOM_PASSES})
endforeach()
