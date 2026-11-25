func.func private @check_result(memref<1x16x5x5xf32>)
func.func @main() -> i8 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %c7 = arith.constant 7 : index
  %c16 = arith.constant 16 : index
  %c25 = arith.constant 25 : index
  %zi8 = arith.constant 0 : i8
  %zi32 = arith.constant 0 : i32
  %input = memref.alloc() alignment = 64 : memref<1x5x5x2xi8>
  %weight = memref.alloc() alignment = 64 : memref<1x6400xi8>
  %bias = memref.alloc() alignment = 64 : memref<16xi32>
  linalg.fill ins(%zi8 : i8) outs(%weight : memref<1x6400xi8>)
  linalg.fill ins(%zi32 : i32) outs(%bias : memref<16xi32>)
  scf.for %y = %c0 to %c5 step %c1 {
    scf.for %x = %c0 to %c5 step %c1 {
      scf.for %c = %c0 to %c2 step %c1 {
        %y3 = arith.muli %y, %c3 : index
        %x2 = arith.muli %x, %c2 : index
        %s0 = arith.addi %y3, %x2 : index
        %s = arith.addi %s0, %c : index
        %r = arith.remui %s, %c7 : index
        %v = arith.subi %r, %c3 : index
        %i = arith.index_cast %v : index to i8
        memref.store %i, %input[%c0, %y, %x, %c] : memref<1x5x5x2xi8>
      }
    }
  }
  scf.for %c = %c0 to %c2 step %c1 {
    scf.for %k = %c0 to %c25 step %c1 {
      scf.for %o = %c0 to %c16 step %c1 {
        %s0 = arith.addi %k, %c : index
        %s = arith.addi %s0, %o : index
        %r = arith.remui %s, %c3 : index
        %v = arith.subi %r, %c1 : index
        %i = arith.index_cast %v : index to i8
        %kc = arith.muli %k, %c16 : index
        %channel = arith.addi %kc, %c : index
        %line = arith.muli %channel, %c16 : index
        %at = arith.addi %line, %o : index
        memref.store %i, %weight[%c0, %at] : memref<1x6400xi8>
      }
    }
  }
  %input_tensor = bufferization.to_tensor %input restrict : memref<1x5x5x2xi8> to tensor<1x5x5x2xi8>
  %input_step = arith.constant dense<1.0> : tensor<1x1x1x1xf32>
  %input_float = "frontend.int8_to_fp32"(%input_tensor, %input_step) : (tensor<1x5x5x2xi8>, tensor<1x1x1x1xf32>) -> tensor<1x5x5x2xf32>
  %packed = bufferization.to_tensor %weight restrict : memref<1x6400xi8> to tensor<1x6400xi8>
  %expanded = tensor.expand_shape %packed [[0], [1, 2, 3, 4, 5]] output_shape [1, 1, 5, 5, 16, 16] : tensor<1x6400xi8> into tensor<1x1x5x5x16x16xi8>
  %weight_empty = tensor.empty() : tensor<1x16x1x16x5x5xi8>
  %transposed = linalg.transpose ins(%expanded : tensor<1x1x5x5x16x16xi8>) outs(%weight_empty : tensor<1x16x1x16x5x5xi8>) permutation = [0, 5, 1, 4, 2, 3]
  %full = tensor.collapse_shape %transposed [[0, 1], [2, 3], [4], [5]] : tensor<1x16x1x16x5x5xi8> into tensor<16x16x5x5xi8>
  %weight_tensor = tensor.extract_slice %full[0, 0, 0, 0] [16, 2, 5, 5] [1, 1, 1, 1] : tensor<16x16x5x5xi8> to tensor<16x2x5x5xi8>
  %bias_tensor = bufferization.to_tensor %bias restrict : memref<16xi32> to tensor<16xi32>
  %weight_step = arith.constant dense<1.0> : tensor<16x1x1x1xf32>
  %result = bbacc.custom "conv2d_i8"(%input_float, %weight_tensor, %bias_tensor, %input_step, %input_step, %weight_step) {strides = array<i64: 1, 1>, dilations = array<i64: 1, 1>, padding = array<i64: 2, 2, 2, 2>} : (tensor<1x5x5x2xf32>, tensor<16x2x5x5xi8>, tensor<16xi32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<16x1x1x1xf32>) -> tensor<1x5x5x16xf32>
  %empty = tensor.empty() : tensor<1x16x5x5xf32>
  %nchw = linalg.transpose ins(%result: tensor<1x5x5x16xf32>) outs(%empty: tensor<1x16x5x5xf32>) permutation = [0, 3, 1, 2]
  %memory = bufferization.to_buffer %nchw : tensor<1x16x5x5xf32> to memref<1x16x5x5xf32>
  func.call @check_result(%memory) : (memref<1x16x5x5xf32>) -> ()
  %ok = arith.constant 0 : i8
  return %ok : i8
}
