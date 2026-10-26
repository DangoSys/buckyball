func.func private @check_result(memref<1x6x6x40xi8>) -> ()

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
  %c6 = arith.constant 6 : index
  %c96 = arith.constant 96 : index
  %c256 = arith.constant 256 : index
  %input = memref.alloc() alignment = 64 : memref<1x6x6x96xi8>
  %w1raw = memref.alloc() alignment = 64 : memref<49168xi8>
  %w1 = memref.reinterpret_cast %w1raw to offset: [16], sizes: [2, 96, 16, 16], strides: [24576, 256, 16, 1] : memref<49168xi8> to memref<2x96x16x16xi8, strided<[24576, 256, 16, 1], offset: 16>>
  %b1 = memref.alloc() alignment = 64 : memref<24xi32>
  %s1 = memref.alloc() alignment = 64 : memref<24xf32>
  %w2raw = memref.alloc() alignment = 64 : memref<36880xi8>
  %w2 = memref.reinterpret_cast %w2raw to offset: [16], sizes: [6, 24, 16, 16], strides: [6144, 256, 16, 1] : memref<36880xi8> to memref<6x24x16x16xi8, strided<[6144, 256, 16, 1], offset: 16>>
  %b2 = memref.alloc() alignment = 64 : memref<96xi32>
  %s2 = memref.alloc() alignment = 64 : memref<96xf32>
  %lut = memref.alloc() alignment = 64 : memref<256xi8>
  %w3raw = memref.alloc() alignment = 64 : memref<73744xi8>
  %w3 = memref.reinterpret_cast %w3raw to offset: [16], sizes: [3, 96, 16, 16], strides: [24576, 256, 16, 1] : memref<73744xi8> to memref<3x96x16x16xi8, strided<[24576, 256, 16, 1], offset: 16>>
  %b3 = memref.alloc() alignment = 64 : memref<40xi32>
  %s3 = memref.alloc() alignment = 64 : memref<40xf32>
  scf.for %y = %c0 to %c6 step %c1 {
    scf.for %x = %c0 to %c6 step %c1 {
      scf.for %c = %c0 to %c96 step %c1 {
        %a = arith.addi %y, %x : index
        %d = arith.addi %a, %c : index
        %r = arith.remui %d, %c5 : index
        %v = arith.addi %r, %c1 : index
        %q = arith.index_cast %v : index to i8
        memref.store %q, %input[%c0, %y, %x, %c] : memref<1x6x6x96xi8>
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
  scf.for %i = %c0 to %c256 step %c1 {
    %v = arith.index_cast %i : index to i8
    memref.store %v, %lut[%i] : memref<256xi8>
  }
  %source_tensor = bufferization.to_tensor %input restrict : memref<1x6x6x96xi8> to tensor<1x6x6x96xi8>
  %source_matrix = tensor.collapse_shape %source_tensor [[0, 1, 2], [3]] : tensor<1x6x6x96xi8> into tensor<36x96xi8>
  %w1_raw = bufferization.to_tensor %w1 restrict : memref<2x96x16x16xi8, strided<[24576, 256, 16, 1], offset: 16>> to tensor<2x96x16x16xi8>
  %w1_empty = tensor.empty() : tensor<96x24xi8>
  %weights1 = linalg.generic {indexing_maps = [affine_map<(k,n)->(n floordiv 16,k,0,n mod 16)>, affine_map<(k,n)->(k,n)>], iterator_types = ["parallel", "parallel"]} ins(%w1_raw: tensor<2x96x16x16xi8>) outs(%w1_empty: tensor<96x24xi8>) {
  ^bb0(%v: i8, %o: i8): linalg.yield %v : i8
  } -> tensor<96x24xi8>
  %b1_raw = bufferization.to_tensor %b1 restrict : memref<24xi32> to tensor<24xi32>
  %bias_tensor1 = tensor.expand_shape %b1_raw [[0,1]] output_shape [1,24] : tensor<24xi32> into tensor<1x24xi32>
  %s1_raw = bufferization.to_tensor %s1 restrict : memref<24xf32> to tensor<24xf32>
  %scale_tensor1 = tensor.expand_shape %s1_raw [[0,1]] output_shape [1,24] : tensor<24xf32> into tensor<1x24xf32>
  %w2_raw = bufferization.to_tensor %w2 restrict : memref<6x24x16x16xi8, strided<[6144, 256, 16, 1], offset: 16>> to tensor<6x24x16x16xi8>
  %w2_empty = tensor.empty() : tensor<24x96xi8>
  %weights2 = linalg.generic {indexing_maps = [affine_map<(k,n)->(n floordiv 16,k,0,n mod 16)>, affine_map<(k,n)->(k,n)>], iterator_types = ["parallel", "parallel"]} ins(%w2_raw: tensor<6x24x16x16xi8>) outs(%w2_empty: tensor<24x96xi8>) {
  ^bb0(%v: i8, %o: i8): linalg.yield %v : i8
  } -> tensor<24x96xi8>
  %b2_raw = bufferization.to_tensor %b2 restrict : memref<96xi32> to tensor<96xi32>
  %bias_tensor2 = tensor.expand_shape %b2_raw [[0,1]] output_shape [1,96] : tensor<96xi32> into tensor<1x96xi32>
  %s2_raw = bufferization.to_tensor %s2 restrict : memref<96xf32> to tensor<96xf32>
  %scale_tensor2 = tensor.expand_shape %s2_raw [[0,1]] output_shape [1,96] : tensor<96xf32> into tensor<1x96xf32>
  %w3_raw = bufferization.to_tensor %w3 restrict : memref<3x96x16x16xi8, strided<[24576, 256, 16, 1], offset: 16>> to tensor<3x96x16x16xi8>
  %w3_empty = tensor.empty() : tensor<96x40xi8>
  %weights3 = linalg.generic {indexing_maps = [affine_map<(k,n)->(n floordiv 16,k,0,n mod 16)>, affine_map<(k,n)->(k,n)>], iterator_types = ["parallel", "parallel"]} ins(%w3_raw: tensor<3x96x16x16xi8>) outs(%w3_empty: tensor<96x40xi8>) {
  ^bb0(%v: i8, %o: i8): linalg.yield %v : i8
  } -> tensor<96x40xi8>
  %b3_raw = bufferization.to_tensor %b3 restrict : memref<40xi32> to tensor<40xi32>
  %bias_tensor3 = tensor.expand_shape %b3_raw [[0,1]] output_shape [1,40] : tensor<40xi32> into tensor<1x40xi32>
  %s3_raw = bufferization.to_tensor %s3 restrict : memref<40xf32> to tensor<40xf32>
  %scale_tensor3 = tensor.expand_shape %s3_raw [[0,1]] output_shape [1,40] : tensor<40xf32> into tensor<1x40xf32>
  %table = bufferization.to_tensor %lut restrict : memref<256xi8> to tensor<256xi8>
  scf.for %iteration = %c0 to %c2 step %c1 {
  %average = custom.op "global_avg_i8"(%source_tensor) {input_scale = 1.0 : f32, output_scale = 1.0 : f32} : (tensor<1x6x6x96xi8>) -> tensor<1x1x1x96xi8>
  %a1 = tensor.collapse_shape %average [[0,1,2],[3]] : tensor<1x1x1x96xi8> into tensor<1x96xi8>
  %v1 = custom.op "linear_i8_requant"(%a1, %weights1, %bias_tensor1, %scale_tensor1) {relu = true} : (tensor<1x96xi8>, tensor<96x24xi8>, tensor<1x24xi32>, tensor<1x24xf32>) -> tensor<1x24xi8>
  %v2 = custom.op "linear_i8_requant"(%v1, %weights2, %bias_tensor2, %scale_tensor2) {relu = false} : (tensor<1x24xi8>, tensor<24x96xi8>, tensor<1x96xi32>, tensor<1x96xf32>) -> tensor<1x96xi8>
  %looked_up = custom.op "lookup_i8"(%v2, %table) : (tensor<1x96xi8>, tensor<256xi8>) -> tensor<1x96xi8>
  %multiplied = custom.op "gate_mul_i8"(%looked_up, %source_matrix) {ratio = 1.000000015e-01 : f32} : (tensor<1x96xi8>, tensor<36x96xi8>) -> tensor<36x96xi8>
  %v3 = custom.op "linear_i8_requant"(%multiplied, %weights3, %bias_tensor3, %scale_tensor3) {relu = false} : (tensor<36x96xi8>, tensor<96x40xi8>, tensor<1x40xi32>, tensor<1x40xf32>) -> tensor<36x40xi8>
  %final = tensor.expand_shape %v3 [[0,1,2],[3]] output_shape [1,6,6,40] : tensor<36x40xi8> into tensor<1x6x6x40xi8>
  %memory = bufferization.to_buffer %final : tensor<1x6x6x40xi8> to memref<1x6x6x40xi8>
  func.call @check_result(%memory) : (memref<1x6x6x40xi8>) -> ()
  }
  return %i8_0 : i8
}
