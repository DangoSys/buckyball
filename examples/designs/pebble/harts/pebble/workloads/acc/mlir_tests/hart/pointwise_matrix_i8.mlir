#bias = affine_map<(n, h, w, c) -> (c)>
#id = affine_map<(n, h, w, c) -> (n, h, w, c)>
func.func private @check_result(memref<1x1x1x16xi32>)
func.func @main() -> i8 {
  %success = arith.constant 0 : i8
  %codes = arith.constant dense<1> : tensor<1x1x1x16xi32>
  %scale = arith.constant dense<1.0> : tensor<1x1x1x16xf32>
  %input = "frontend.int32_to_int8"(%codes, %scale) : (tensor<1x1x1x16xi32>, tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xi8>
  %packed = arith.constant dense<1> : tensor<1x256xi8>
  %expanded = tensor.expand_shape %packed [[0], [1, 2, 3, 4, 5]] output_shape [1, 1, 1, 1, 16, 16] : tensor<1x256xi8> into tensor<1x1x1x1x16x16xi8>
  %empty = tensor.empty() : tensor<1x16x1x16x1x1xi8>
  %transposed = linalg.transpose ins(%expanded : tensor<1x1x1x1x16x16xi8>) outs(%empty : tensor<1x16x1x16x1x1xi8>) permutation = [0, 5, 1, 4, 2, 3]
  %weight = tensor.collapse_shape %transposed [[0, 1], [2, 3], [4], [5]] : tensor<1x16x1x16x1x1xi8> into tensor<16x16x1x1xi8>
  %bias = arith.constant dense<[-17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2]> : tensor<16xi32>
  %unit = arith.constant dense<1.0> : tensor<1x1x1x16xf32>
  %multipliers = arith.constant dense<3> : tensor<16xi32>
  %shifts = arith.constant dense<1> : tensor<16xi32>
  frontend.trace_start {cycle = false, id = 0 : i64, id_path = [0], tag = "conv", tensor = false}
  %weights_empty = tensor.empty() : tensor<16x1x1x16xi8>
  %weights = linalg.transpose ins(%weight : tensor<16x16x1x1xi8>) outs(%weights_empty : tensor<16x1x1x16xi8>) permutation = [0, 2, 3, 1]
  %initial_empty = tensor.empty() : tensor<1x1x1x16xi32>
  %initial = linalg.generic {indexing_maps = [#bias, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bias : tensor<16xi32>) outs(%initial_empty : tensor<1x1x1x16xi32>) {
  ^bb0(%value: i32, %out: i32):
    linalg.yield %value : i32
  } -> tensor<1x1x1x16xi32>
  %raw = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%input, %weights : tensor<1x1x1x16xi8>, tensor<16x1x1x16xi8>) outs(%initial : tensor<1x1x1x16xi32>) -> tensor<1x1x1x16xi32>
  %observed = frontend.trace_end %raw {cycle = false, id = 0 : i64, id_path = [0], tag = "conv", tensor = false} : tensor<1x1x1x16xi32>
  %result = "frontend.fixed_requant"(%observed, %multipliers, %shifts) <{fraction = 24 : i64}> : (tensor<1x1x1x16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<1x1x1x16xi32>
  %memory = bufferization.to_buffer %result : tensor<1x1x1x16xi32> to memref<1x1x1x16xi32>
  func.call @check_result(%memory) : (memref<1x1x1x16xi32>) -> ()
  return %success : i8
}
