// CHECK: %[[NHWC:.*]] = memref.alloc() : memref<1x8x8x16xf32>
// CHECK: %[[HWCF:.*]] = memref.alloc() : memref<3x3x16x32xf32>
// CHECK: %[[OUT:.*]] = memref.alloc() : memref<1x6x6x32xf32>
// CHECK: tile.tile_conv2d {{.*}} %[[HWCF]] %[[OUT]]
// CHECK: memref.dealloc %[[NHWC]] : memref<1x8x8x16xf32>
// CHECK: memref.dealloc %[[HWCF]] : memref<3x3x16x32xf32>
// CHECK: memref.dealloc %[[OUT]] : memref<1x6x6x32xf32>
// CHECK-NOT: linalg.conv_2d_nchw_fchw

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %zero_f32 = arith.constant 0.0 : f32
  %one_f32 = arith.constant 1.0 : f32

  %input = memref.alloc() : memref<1x16x8x8xf32>
  %filter = memref.alloc() : memref<32x16x3x3xf32>
  %output = memref.alloc() : memref<1x32x6x6xf32>

  linalg.fill ins(%one_f32 : f32) outs(%input : memref<1x16x8x8xf32>)
  linalg.fill ins(%one_f32 : f32) outs(%filter : memref<32x16x3x3xf32>)
  linalg.fill ins(%zero_f32 : f32) outs(%output : memref<1x32x6x6xf32>)

  linalg.conv_2d_nchw_fchw
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %filter : memref<1x16x8x8xf32>, memref<32x16x3x3xf32>)
    outs(%output : memref<1x32x6x6xf32>)

  memref.dealloc %input : memref<1x16x8x8xf32>
  memref.dealloc %filter : memref<32x16x3x3xf32>
  memref.dealloc %output : memref<1x32x6x6xf32>

  return %zero_i8 : i8
}
