func.func private @check_result(memref<1x6x6x32xf32>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %zero_f32 = arith.constant 0.0 : f32
  %one_f32 = arith.constant 1.0 : f32

  %input = memref.alloc() : memref<1x8x8x16xf32>
  %filter = memref.alloc() : memref<32x3x3x16xf32>
  %output = memref.alloc() : memref<1x6x6x32xf32>

  linalg.fill ins(%one_f32 : f32) outs(%input : memref<1x8x8x16xf32>)
  linalg.fill ins(%one_f32 : f32) outs(%filter : memref<32x3x3x16xf32>)
  linalg.fill ins(%zero_f32 : f32) outs(%output : memref<1x6x6x32xf32>)

  linalg.conv_2d_nhwc_fhwc
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %filter : memref<1x8x8x16xf32>, memref<32x3x3x16xf32>)
    outs(%output : memref<1x6x6x32xf32>)

  func.call @check_result(%output) : (memref<1x6x6x32xf32>) -> ()

  memref.dealloc %input : memref<1x8x8x16xf32>
  memref.dealloc %filter : memref<32x3x3x16xf32>
  memref.dealloc %output : memref<1x6x6x32xf32>

  return %zero_i8 : i8
}
