func.func private @check_result(memref<1x1x1x16xf32>)
func.func @main() -> i8 {
  %success = arith.constant 0 : i8
  %input = arith.constant dense<1.0> : tensor<1x11x11x3xf32>
  %packed = arith.constant dense<1> : tensor<1x30976xi8>
  %expanded = tensor.expand_shape %packed [[0, 1], [2, 3, 4, 5]] output_shape [1, 1, 11, 11, 16, 16] : tensor<1x30976xi8> into tensor<1x1x11x11x16x16xi8>
  %empty = tensor.empty() : tensor<1x16x1x16x11x11xi8>
  %transposed = linalg.transpose ins(%expanded : tensor<1x1x11x11x16x16xi8>) outs(%empty : tensor<1x16x1x16x11x11xi8>) permutation = [0, 5, 1, 4, 2, 3]
  %flat = tensor.collapse_shape %transposed [[0, 1, 2], [3], [4], [5]] : tensor<1x16x1x16x11x11xi8> into tensor<16x16x11x11xi8>
  %weight = tensor.extract_slice %flat[0, 0, 0, 0] [16, 3, 11, 11] [1, 1, 1, 1] : tensor<16x16x11x11xi8> to tensor<16x3x11x11xi8>
  %bias = arith.constant dense<0> : tensor<16xi32>
  %unit = arith.constant dense<1.0> : tensor<1x1x1x1xf32>
  %steps = arith.constant dense<1.0> : tensor<16x1x1x1xf32>
  %result = bbacc.custom "conv2d_i8"(%input, %weight, %bias, %unit, %unit, %steps) {strides = array<i64: 1, 1>, dilations = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>} : (tensor<1x11x11x3xf32>, tensor<16x3x11x11xi8>, tensor<16xi32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<16x1x1x1xf32>) -> tensor<1x1x1x16xf32>
  %memory = bufferization.to_buffer %result : tensor<1x1x1x16xf32> to memref<1x1x1x16xf32>
  func.call @check_result(%memory) : (memref<1x1x1x16xf32>) -> ()
  return %success : i8
}
