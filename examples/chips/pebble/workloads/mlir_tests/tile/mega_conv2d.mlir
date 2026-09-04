func.func private @check_result(memref<1x16x2x2xi8>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %one_i8 = arith.constant 1 : i8
  %zero_i32 = arith.constant 0 : i32
  %one_f32 = arith.constant 1.0 : f32
  %input = memref.alloc() alignment = 64 : memref<1x4x4x2xi8>
  %conv_weight = memref.alloc() alignment = 64 : memref<1x2x16x16xi8>
  %conv_bias = memref.alloc() alignment = 64 : memref<16xi32>
  %conv_scale = memref.alloc() alignment = 64 : memref<16xf32>
  %lut = memref.alloc() alignment = 64 : memref<1xi8>
  %intermediate = memref.alloc() alignment = 64 : memref<1x4x4x16xi8>
  %depthwise_weight = memref.alloc() alignment = 64 : memref<3x3x16x1xi8>
  %depthwise_bias = memref.alloc() alignment = 64 : memref<16xi32>
  %depthwise_scale = memref.alloc() alignment = 64 : memref<16xf32>
  %depthwise_output = memref.alloc() alignment = 64 : memref<1x4x4x16xi8>
  %output = memref.alloc() alignment = 64 : memref<1x16x2x2xi8>
  linalg.fill ins(%one_i8 : i8) outs(%input : memref<1x4x4x2xi8>)
  linalg.fill ins(%one_i8 : i8) outs(%conv_weight : memref<1x2x16x16xi8>)
  linalg.fill ins(%zero_i32 : i32) outs(%conv_bias : memref<16xi32>)
  linalg.fill ins(%one_f32 : f32) outs(%conv_scale : memref<16xf32>)
  linalg.fill ins(%zero_i8 : i8) outs(%lut : memref<1xi8>)
  linalg.fill ins(%zero_i8 : i8) outs(%intermediate : memref<1x4x4x16xi8>)
  linalg.fill ins(%one_i8 : i8) outs(%depthwise_weight : memref<3x3x16x1xi8>)
  linalg.fill ins(%zero_i32 : i32) outs(%depthwise_bias : memref<16xi32>)
  linalg.fill ins(%one_f32 : f32) outs(%depthwise_scale : memref<16xf32>)
  linalg.fill ins(%zero_i8 : i8) outs(%depthwise_output : memref<1x4x4x16xi8>)
  linalg.fill ins(%zero_i8 : i8) outs(%output : memref<1x16x2x2xi8>)
  tile.mega_kernel %input %output
      : memref<1x4x4x2xi8> memref<1x16x2x2xi8> {
    tile.mega_conv2d %input %conv_weight %conv_bias %conv_scale %lut
        %intermediate {activation = 1 : i64, kernel = 3 : i64,
                       outputScale = 1.0 : f32, padHigh = 1 : i64,
                       padLow = 1 : i64, stride = 1 : i64}
        : memref<1x4x4x2xi8> memref<1x2x16x16xi8> memref<16xi32>
          memref<16xf32> memref<1xi8> memref<1x4x4x16xi8>
    tile.mega_conv2d_depthwise %intermediate %depthwise_weight
        %depthwise_bias %depthwise_scale %lut %depthwise_output
        {activation = 0 : i64, kernel = 3 : i64,
         outputScale = 1.0 : f32, padHigh = 1 : i64,
         padLow = 1 : i64, stride = 1 : i64}
        : memref<1x4x4x16xi8> memref<3x3x16x1xi8> memref<16xi32>
          memref<16xf32> memref<1xi8> memref<1x4x4x16xi8>
    tile.mega_max_pool2d %depthwise_output %output
        {finalOutput = true, kernel = 2 : i64, padding = 0 : i64,
         stride = 2 : i64}
        : memref<1x4x4x16xi8> memref<1x16x2x2xi8>
  }
  func.call @check_result(%output) : (memref<1x16x2x2xi8>) -> ()
  memref.dealloc %input : memref<1x4x4x2xi8>
  memref.dealloc %conv_weight : memref<1x2x16x16xi8>
  memref.dealloc %conv_bias : memref<16xi32>
  memref.dealloc %conv_scale : memref<16xf32>
  memref.dealloc %lut : memref<1xi8>
  memref.dealloc %intermediate : memref<1x4x4x16xi8>
  memref.dealloc %depthwise_weight : memref<3x3x16x1xi8>
  memref.dealloc %depthwise_bias : memref<16xi32>
  memref.dealloc %depthwise_scale : memref<16xf32>
  memref.dealloc %depthwise_output : memref<1x4x4x16xi8>
  memref.dealloc %output : memref<1x16x2x2xi8>
  return %zero_i8 : i8
}
