func.func private @check_result(memref<1x16x2x2xi8>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %zero_i32 = arith.constant 0 : i32
  %one_f32 = arith.constant 1.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c7 = arith.constant 7 : index
  %c16 = arith.constant 16 : index
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
  scf.for %h = %c0 to %c4 step %c1 {
    scf.for %w = %c0 to %c4 step %c1 {
      scf.for %ic = %c0 to %c2 step %c1 {
        %h3 = arith.muli %h, %c3 : index
        %w2 = arith.muli %w, %c2 : index
        %sum0 = arith.addi %h3, %w2 : index
        %sum = arith.addi %sum0, %ic : index
        %rem = arith.remui %sum, %c7 : index
        %signed = arith.subi %rem, %c3 : index
        %value = arith.index_cast %signed : index to i8
        memref.store %value, %input[%c0, %h, %w, %ic] : memref<1x4x4x2xi8>
      }
    }
  }
  scf.for %ic = %c0 to %c2 step %c1 {
    scf.for %p = %c0 to %c16 step %c1 {
      scf.for %oc = %c0 to %c16 step %c1 {
        %oc3 = arith.muli %oc, %c3 : index
        %sum0 = arith.addi %p, %ic : index
        %sum = arith.addi %sum0, %oc3 : index
        %rem = arith.remui %sum, %c5 : index
        %signed = arith.subi %rem, %c2 : index
        %value = arith.index_cast %signed : index to i8
        memref.store %value, %conv_weight[%c0, %ic, %p, %oc] : memref<1x2x16x16xi8>
      }
    }
  }
  scf.for %oc = %c0 to %c16 step %c1 {
    %signed = arith.subi %oc, %c4 : index
    %value = arith.index_cast %signed : index to i32
    memref.store %value, %conv_bias[%oc] : memref<16xi32>
  }
  linalg.fill ins(%one_f32 : f32) outs(%conv_scale : memref<16xf32>)
  linalg.fill ins(%zero_i8 : i8) outs(%lut : memref<1xi8>)
  linalg.fill ins(%zero_i8 : i8) outs(%intermediate : memref<1x4x4x16xi8>)
  scf.for %kh = %c0 to %c3 step %c1 {
    scf.for %kw = %c0 to %c3 step %c1 {
      scf.for %oc = %c0 to %c16 step %c1 {
        %kw2 = arith.muli %kw, %c2 : index
        %sum0 = arith.addi %kh, %kw2 : index
        %sum = arith.addi %sum0, %oc : index
        %rem = arith.remui %sum, %c5 : index
        %signed = arith.subi %rem, %c2 : index
        %value = arith.index_cast %signed : index to i8
        memref.store %value, %depthwise_weight[%kh, %kw, %oc, %c0] : memref<3x3x16x1xi8>
      }
    }
  }
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
  return %zero_i8 : i8
}
