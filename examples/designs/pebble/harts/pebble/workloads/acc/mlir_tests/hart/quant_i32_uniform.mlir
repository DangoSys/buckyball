func.func private @check_result(memref<1x32xi8>)
func.func @main() -> i8 {
  %zero = arith.constant 0 : i8
  %input = arith.constant dense<[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]]> : tensor<1x32xi32>
  %physical = arith.constant dense<0.5> : tensor<1x32xf32>
  %scale = tensor.extract_slice %physical[0, 0] [1, 1] [1, 1] : tensor<1x32xf32> to tensor<1x1xf32>
  %result = "frontend.int32_to_int8"(%input, %scale) : (tensor<1x32xi32>, tensor<1x1xf32>) -> tensor<1x32xi8>
  %memory = bufferization.to_buffer %result : tensor<1x32xi8> to memref<1x32xi8>
  func.call @check_result(%memory) : (memref<1x32xi8>) -> ()
  return %zero : i8
}
