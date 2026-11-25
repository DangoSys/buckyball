// BANK: int8add_ball.bank_add_i8
// BANK-NOT: frontend.scaled_add_i8
func.func private @check_result(memref<2x16xi8>)
func.func @main() -> i8 {
  %a = arith.constant dense<[[1, 3, 5, 7, -1, -3, -5, -7, 127, -128, 80, -80, 0, 2, -2, 6], [1, 3, 5, 7, -1, -3, -5, -7, 127, -128, 80, -80, 0, 2, -2, 6]]> : tensor<2x16xi8>
  %b = arith.constant dense<[[0, 0, 0, 0, 0, 0, 0, 0, 127, -128, 80, -80, 0, 2, -2, -2], [0, 0, 0, 0, 0, 0, 0, 0, 127, -128, 80, -80, 0, 2, -2, -2]]> : tensor<2x16xi8>
  %r = "frontend.scaled_add_i8"(%a, %b) {lhs_ratio = 0.5 : f32, rhs_ratio = 1.0 : f32} : (tensor<2x16xi8>, tensor<2x16xi8>) -> tensor<2x16xi8>
  %buffer = bufferization.to_buffer %r : tensor<2x16xi8> to memref<2x16xi8>
  func.call @check_result(%buffer) : (memref<2x16xi8>) -> ()
  %ok = arith.constant 0 : i8
  return %ok : i8
}
