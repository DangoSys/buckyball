// BANK: smatmul_ball.bank_matmul_i8
// BANK: fixed_ball.bank_fixed {{.*}}mode = 0
// BANK-NEXT: {{.*}}fixed_ball.bank_fixed {{.*}}mode = 4
// BANK: frontend.trace_end {{.*}}tensor<1x2x2x16xi32>
func.func private @check_result(memref<1x2x2x16xi32>)
func.func @main() -> i8 {
  %success = arith.constant 0 : i8
  %input = arith.constant dense<1> : tensor<1x3x3x16xi8>
  %packed = arith.constant dense<1> : tensor<1x256xi8>
  %expanded = tensor.expand_shape %packed [[0], [1, 2, 3, 4, 5]] output_shape [1, 1, 1, 1, 16, 16] : tensor<1x256xi8> into tensor<1x1x1x1x16x16xi8>
  %empty = tensor.empty() : tensor<1x16x1x16x1x1xi8>
  %transposed = linalg.transpose ins(%expanded : tensor<1x1x1x1x16x16xi8>) outs(%empty : tensor<1x16x1x16x1x1xi8>) permutation = [0, 5, 1, 4, 2, 3]
  %weight = tensor.collapse_shape %transposed [[0, 1], [2, 3], [4], [5]] : tensor<1x16x1x16x1x1xi8> into tensor<16x16x1x1xi8>
  %bias = arith.constant dense<[-17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2]> : tensor<16xi32>
  %multipliers = arith.constant dense<3> : tensor<16xi32>
  %shifts = arith.constant dense<1> : tensor<16xi32>
  frontend.trace_start {cycle = false, id = 0 : i64, id_path = [0], tag = "conv", tensor = false}
  %filter_empty = tensor.empty() : tensor<16x1x1x16xi8>
  %filter = linalg.transpose ins(%weight : tensor<16x16x1x1xi8>) outs(%filter_empty : tensor<16x1x1x16xi8>) permutation = [0, 2, 3, 1]
  %output = tensor.empty() : tensor<1x3x3x16xi32>
  %initial = linalg.generic {indexing_maps = [affine_map<(n,h,w,c)->(c)>, affine_map<(n,h,w,c)->(n,h,w,c)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bias : tensor<16xi32>) outs(%output : tensor<1x3x3x16xi32>) {
  ^bb0(%b: i32, %out: i32):
    linalg.yield %b : i32
  } -> tensor<1x3x3x16xi32>
  %raw = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%input, %filter : tensor<1x3x3x16xi8>, tensor<16x1x1x16xi8>) outs(%initial : tensor<1x3x3x16xi32>) -> tensor<1x3x3x16xi32>
  %observed = frontend.trace_end %raw {cycle = false, id = 0 : i64, id_path = [0], tag = "conv", tensor = false} : tensor<1x3x3x16xi32>
  %cropped = tensor.extract_slice %observed[0, 0, 0, 0] [1, 2, 2, 16] [1, 1, 1, 1] : tensor<1x3x3x16xi32> to tensor<1x2x2x16xi32>
  %result = "frontend.fixed_requant"(%cropped, %multipliers, %shifts) <{fraction = 24 : i64}> : (tensor<1x2x2x16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<1x2x2x16xi32>
  %activated = "frontend.fixed_unary"(%result) <{fraction = 24 : i64, kind = "relu"}> : (tensor<1x2x2x16xi32>) -> tensor<1x2x2x16xi32>
  %memory = bufferization.to_buffer %activated : tensor<1x2x2x16xi32> to memref<1x2x2x16xi32>
  func.call @check_result(%memory) : (memref<1x2x2x16xi32>) -> ()
  return %success : i8
}
