func.func private @check_result(memref<1x14x14x40xi8>) -> ()

func.func @main() -> i8 {
  %i8_0 = arith.constant 0 : i8
  %i8_1 = arith.constant 1 : i8
  %i32_0 = arith.constant 0 : i32
  %i32_1 = arith.constant 1 : i32
  %f32_001 = arith.constant 0.01 : f32
  %f32_01 = arith.constant 0.1 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index
  %c14 = arith.constant 14 : index
  %c96 = arith.constant 96 : index
  %c256 = arith.constant 256 : index
  %input = memref.alloc() alignment = 64 : memref<1x14x14x96xi8>
  %gap = memref.alloc() alignment = 64 : memref<1x1x1x96xi8>
  %w1raw = memref.alloc() alignment = 64 : memref<49168xi8>
  %w1 = memref.reinterpret_cast %w1raw to offset: [16], sizes: [2, 96, 16, 16], strides: [24576, 256, 16, 1] : memref<49168xi8> to memref<2x96x16x16xi8, strided<[24576, 256, 16, 1], offset: 16>>
  %b1 = memref.alloc() alignment = 64 : memref<24xi32>
  %s1 = memref.alloc() alignment = 64 : memref<24xf32>
  %h1 = memref.alloc() alignment = 64 : memref<1x1x1x24xi8>
  %w2raw = memref.alloc() alignment = 64 : memref<36880xi8>
  %w2 = memref.reinterpret_cast %w2raw to offset: [16], sizes: [6, 24, 16, 16], strides: [6144, 256, 16, 1] : memref<36880xi8> to memref<6x24x16x16xi8, strided<[6144, 256, 16, 1], offset: 16>>
  %b2 = memref.alloc() alignment = 64 : memref<96xi32>
  %s2 = memref.alloc() alignment = 64 : memref<96xf32>
  %lut = memref.alloc() alignment = 64 : memref<256xi8>
  %gate = memref.alloc() alignment = 64 : memref<1x1x1x96xi8>
  %mul = memref.alloc() alignment = 64 : memref<1x14x14x96xi8>
  %w3raw = memref.alloc() alignment = 64 : memref<73744xi8>
  %w3 = memref.reinterpret_cast %w3raw to offset: [16], sizes: [3, 96, 16, 16], strides: [24576, 256, 16, 1] : memref<73744xi8> to memref<3x96x16x16xi8, strided<[24576, 256, 16, 1], offset: 16>>
  %b3 = memref.alloc() alignment = 64 : memref<40xi32>
  %s3 = memref.alloc() alignment = 64 : memref<40xf32>
  %none = memref.alloc() alignment = 64 : memref<1xi8>
  %output = memref.alloc() alignment = 64 : memref<1x14x14x40xi8>
  scf.for %y = %c0 to %c14 step %c1 {
    scf.for %x = %c0 to %c14 step %c1 {
      scf.for %c = %c0 to %c96 step %c1 {
        %a = arith.addi %y, %x : index
        %d = arith.addi %a, %c : index
        %r = arith.remui %d, %c5 : index
        %v = arith.addi %r, %c1 : index
        %q = arith.index_cast %v : index to i8
        memref.store %q, %input[%c0, %y, %x, %c] : memref<1x14x14x96xi8>
      }
    }
  }
  linalg.fill ins(%i8_1 : i8) outs(%w1 : memref<2x96x16x16xi8, strided<[24576, 256, 16, 1], offset: 16>>)
  linalg.fill ins(%i8_1 : i8) outs(%w2 : memref<6x24x16x16xi8, strided<[6144, 256, 16, 1], offset: 16>>)
  linalg.fill ins(%i8_1 : i8) outs(%w3 : memref<3x96x16x16xi8, strided<[24576, 256, 16, 1], offset: 16>>)
  linalg.fill ins(%i32_1 : i32) outs(%b1 : memref<24xi32>)
  linalg.fill ins(%i32_0 : i32) outs(%b2 : memref<96xi32>)
  linalg.fill ins(%i32_0 : i32) outs(%b3 : memref<40xi32>)
  linalg.fill ins(%f32_001 : f32) outs(%s1 : memref<24xf32>)
  linalg.fill ins(%f32_01 : f32) outs(%s2 : memref<96xf32>)
  linalg.fill ins(%f32_001 : f32) outs(%s3 : memref<40xf32>)
  linalg.fill ins(%i8_0 : i8) outs(%none : memref<1xi8>)
  scf.for %i = %c0 to %c256 step %c1 {
    %v = arith.index_cast %i : index to i8
    memref.store %v, %lut[%i] : memref<256xi8>
  }
  linalg.fill ins(%i8_0 : i8) outs(%output : memref<1x14x14x40xi8>)
  scf.for %iteration = %c0 to %c2 step %c1 {
    tile.mega_kernel %input %output : memref<1x14x14x96xi8> memref<1x14x14x40xi8> {
    tile.mega_global_avg_pool %input %gap {inputScale = 1.0 : f32, outputScale = 1.0 : f32} : memref<1x14x14x96xi8> memref<1x1x1x96xi8>
    tile.mega_conv2d %gap %w1 %b1 %s1 %none %h1 {activation = 1 : i64, kernel = 1 : i64, outputScale = 1.0 : f32, padHigh = 0 : i64, padLow = 0 : i64, stride = 1 : i64} : memref<1x1x1x96xi8> memref<2x96x16x16xi8, strided<[24576, 256, 16, 1], offset: 16>> memref<24xi32> memref<24xf32> memref<1xi8> memref<1x1x1x24xi8>
    tile.mega_conv2d %h1 %w2 %b2 %s2 %lut %gate {activation = 2 : i64, kernel = 1 : i64, outputScale = 1.0 : f32, padHigh = 0 : i64, padLow = 0 : i64, stride = 1 : i64} : memref<1x1x1x24xi8> memref<6x24x16x16xi8, strided<[6144, 256, 16, 1], offset: 16>> memref<96xi32> memref<96xf32> memref<256xi8> memref<1x1x1x96xi8>
    tile.mega_int8_mul %gate %input %mul {activation = 0 : i64, lhsScale = 0.1 : f32, outputScale = 1.0 : f32, rhsScale = 1.0 : f32} : memref<1x1x1x96xi8> memref<1x14x14x96xi8> memref<1x14x14x96xi8>
    tile.mega_conv2d %mul %w3 %b3 %s3 %none %output {activation = 0 : i64, kernel = 1 : i64, outputScale = 1.0 : f32, padHigh = 0 : i64, padLow = 0 : i64, stride = 1 : i64} : memref<1x14x14x96xi8> memref<3x96x16x16xi8, strided<[24576, 256, 16, 1], offset: 16>> memref<40xi32> memref<40xf32> memref<1xi8> memref<1x14x14x40xi8>
    }
  }
  func.call @check_result(%output) : (memref<1x14x14x40xi8>) -> ()
  return %i8_0 : i8
}
