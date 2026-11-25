func.func private @check_result(memref<5x32xi8>)
func.func @main() -> i8 {
  %a = arith.constant dense<1> : tensor<513x16xi8>
  %b = arith.constant dense<2> : tensor<513x16xi8>
  %joined = "frontend.concat_columns"(%a, %b) : (tensor<513x16xi8>, tensor<513x16xi8>) -> tensor<513x32xi8>
  %indices = arith.constant dense<[0, 255, 383, 384, 512]> : tensor<5xi64>
  %selected = "frontend.gather_rows"(%indices, %joined) : (tensor<5xi64>, tensor<513x32xi8>) -> tensor<5x32xi8>
  %memory = bufferization.to_buffer %selected : tensor<5x32xi8> to memref<5x32xi8>
  func.call @check_result(%memory) : (memref<5x32xi8>) -> ()
  %ok = arith.constant 0 : i8
  return %ok : i8
}
