func.func private @check_result(memref<1x16xi32>)
func.func @main() -> i8 {
  %zero = arith.constant 0 : i8
  %input = arith.constant dense<[[1,3,-1,-3,5,7,-5,-7,127,128,-127,-128,2147483647,-2147483648,17,-17]]> : tensor<1x16xi32>
  %m = arith.constant dense<[1,1,1,1,3,3,3,3,1,1,1,1,1,1,1,1]> : tensor<16xi32>
  %s = arith.constant dense<[1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1]> : tensor<16xi32>
  %result = "frontend.fixed_requant"(%input, %m, %s) <{fraction = 24 : i64}> : (tensor<1x16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<1x16xi32>
  %memory = bufferization.to_buffer %result : tensor<1x16xi32> to memref<1x16xi32>
  func.call @check_result(%memory) : (memref<1x16xi32>) -> ()
  return %zero : i8
}
