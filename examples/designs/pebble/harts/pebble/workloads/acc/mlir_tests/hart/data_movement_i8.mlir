func.func private @check_result(memref<4x32xi8>)
func.func @main() -> i8 {
  %a = arith.constant dense<[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]> : tensor<3x16xi8>
  %b = arith.constant dense<[[-128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128], [-64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64], [127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127]]> : tensor<3x16xi8>
  %joined = "frontend.concat_columns"(%a, %b) : (tensor<3x16xi8>, tensor<3x16xi8>) -> tensor<3x32xi8>
  %indices = arith.constant dense<[2, 0, 2, 1]> : tensor<4xi64>
  %gathered = "frontend.gather_rows"(%indices, %joined) : (tensor<4xi64>, tensor<3x32xi8>) -> tensor<4x32xi8>
  %memory = bufferization.to_buffer %gathered : tensor<4x32xi8> to memref<4x32xi8>
  func.call @check_result(%memory) : (memref<4x32xi8>) -> ()
  %ok = arith.constant 0 : i8
  return %ok : i8
}
