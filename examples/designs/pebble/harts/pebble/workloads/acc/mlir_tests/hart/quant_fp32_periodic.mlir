func.func private @init_input(memref<1x2x16x3xf32>)
func.func private @check_result(memref<1x2x16x3xi8>)
func.func @main() -> i8 {
  %zero = arith.constant 0 : i8
  %input = memref.alloc() : memref<1x2x16x3xf32>
  func.call @init_input(%input) : (memref<1x2x16x3xf32>) -> ()
  %tensor = bufferization.to_tensor %input restrict : memref<1x2x16x3xf32> to tensor<1x2x16x3xf32>
  %physical = arith.constant dense<[[[[0.5,0.25,2.0,0.5,0.25,2.0,0.5,0.25,2.0,0.5,0.25,2.0,0.5,0.25,2.0,0.5,0.25,2.0,0.5,0.25,2.0,0.5,0.25,2.0,0.5,0.25,2.0,0.5,0.25,2.0,0.5,0.25,2.0,0.5,0.25,2.0,0.5,0.25,2.0,0.5,0.25,2.0,0.5,0.25,2.0,0.5,0.25,2.0]]]]> : tensor<1x1x1x48xf32>
  %scale = tensor.extract_slice %physical[0,0,0,0] [1,1,1,3] [1,1,1,1] : tensor<1x1x1x48xf32> to tensor<1x1x1x3xf32>
  %result = "frontend.fp32_to_int8"(%tensor, %scale) : (tensor<1x2x16x3xf32>, tensor<1x1x1x3xf32>) -> tensor<1x2x16x3xi8>
  %memory = bufferization.to_buffer %result : tensor<1x2x16x3xi8> to memref<1x2x16x3xi8>
  func.call @check_result(%memory) : (memref<1x2x16x3xi8>) -> ()
  return %zero : i8
}
