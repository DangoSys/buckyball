func.func private @check_result(memref<3xi32>)
func.func @main() -> i8 {
  %input = arith.constant dense<[[0, 1, -1, 7, 3, -1, 0, 2], [0, 2147483647, -2147483648, 1, 2, 3, 4, 5], [7, 7, 7, 7, 7, 7, 7, 7]]> : tensor<3x8xi32>
  %result = "frontend.row_max_u32"(%input) : (tensor<3x8xi32>) -> tensor<3xi32>
  %buffer = bufferization.to_buffer %result : tensor<3xi32> to memref<3xi32>
  func.call @check_result(%buffer) : (memref<3xi32>) -> ()
  %ok = arith.constant 0 : i8
  return %ok : i8
}
