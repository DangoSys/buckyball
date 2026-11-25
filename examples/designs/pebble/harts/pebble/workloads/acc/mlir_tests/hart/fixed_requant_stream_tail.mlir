func.func private @init(memref<2064xi32>, memref<2064xi32>)
func.func private @check_result(memref<1x2064xi32>)
func.func @main() -> i8 {
  %input = arith.constant dense<3> : tensor<1x2064xi32>
  %m = memref.alloc() : memref<2064xi32>
  %s = memref.alloc() : memref<2064xi32>
  func.call @init(%m, %s) : (memref<2064xi32>, memref<2064xi32>) -> ()
  %multiplier = bufferization.to_tensor %m restrict : memref<2064xi32> to tensor<2064xi32>
  %shift = bufferization.to_tensor %s restrict : memref<2064xi32> to tensor<2064xi32>
  %result = "frontend.fixed_requant"(%input, %multiplier, %shift) <{fraction = 24 : i64}> : (tensor<1x2064xi32>, tensor<2064xi32>, tensor<2064xi32>) -> tensor<1x2064xi32>
  %memory = bufferization.to_buffer %result : tensor<1x2064xi32> to memref<1x2064xi32>
  func.call @check_result(%memory) : (memref<1x2064xi32>) -> ()
  %ok = arith.constant 0 : i8
  return %ok : i8
}
