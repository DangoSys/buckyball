func.func private @check_result(memref<1x1x2x10xi8>)
func.func @main() -> i8 {
  %zero = arith.constant 0 : i8
  %input = arith.constant dense<[[[[1, 3, -1, -3, 5, 7, -5, -7, 255, -257], [254, -256, 0, 2, 4, -2, -4, 6, 8, -6]]]]> : tensor<1x1x2x10xi32>
  %padded = arith.constant dense<0.5> : tensor<1x1x1x16xf32>
  %multipliers = tensor.extract_slice %padded[0, 0, 0, 0] [1, 1, 1, 10] [1, 1, 1, 1] : tensor<1x1x1x16xf32> to tensor<1x1x1x10xf32>
  %result = "frontend.int32_to_int8"(%input, %multipliers) : (tensor<1x1x2x10xi32>, tensor<1x1x1x10xf32>) -> tensor<1x1x2x10xi8>
  %memory = bufferization.to_buffer %result : tensor<1x1x2x10xi8> to memref<1x1x2x10xi8>
  func.call @check_result(%memory) : (memref<1x1x2x10xi8>) -> ()
  return %zero : i8
}
