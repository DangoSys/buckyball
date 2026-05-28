func.func private @check_result(memref<1x4x1x6xf32>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %zero_f32 = arith.constant 0.0 : f32
  %one_f32 = arith.constant 1.0 : f32

  %input = memref.alloc() {alignment = 16 : i64} : memref<1x4x1x8xf32>
  %filter = memref.alloc() {alignment = 16 : i64} : memref<1x1x8x6xf32>
  %output = memref.alloc() {alignment = 16 : i64} : memref<1x4x1x6xf32>

  linalg.fill ins(%one_f32 : f32) outs(%input : memref<1x4x1x8xf32>)
  linalg.fill ins(%one_f32 : f32) outs(%filter : memref<1x1x8x6xf32>)
  linalg.fill ins(%zero_f32 : f32) outs(%output : memref<1x4x1x6xf32>)

  linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %filter : memref<1x4x1x8xf32>, memref<1x1x8x6xf32>)
    outs(%output : memref<1x4x1x6xf32>)

  func.call @check_result(%output) : (memref<1x4x1x6xf32>) -> ()

  memref.dealloc %input : memref<1x4x1x8xf32>
  memref.dealloc %filter : memref<1x1x8x6xf32>
  memref.dealloc %output : memref<1x4x1x6xf32>

  return %zero_i8 : i8
}
