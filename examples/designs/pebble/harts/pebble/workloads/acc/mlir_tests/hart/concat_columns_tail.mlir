func.func private @check_f32(memref<3x6xf32>)
func.func private @check_i8(memref<3x5xi8>)
func.func @main() -> i8 {
  %a = arith.constant dense<[[1.0, 2.0, 3.0, 4.0], [-4.0, -3.0, -2.0, -1.0], [2.5, 3.5, 4.5, 5.5]]> : tensor<3x4xf32>
  %b = arith.constant dense<[[0.5], [0.25], [0.75]]> : tensor<3x1xf32>
  %c = arith.constant dense<[[1.0], [2.0], [3.0]]> : tensor<3x1xf32>
  %joined = "frontend.concat_columns"(%a, %b, %c) : (tensor<3x4xf32>, tensor<3x1xf32>, tensor<3x1xf32>) -> tensor<3x6xf32>
  %memory = bufferization.to_buffer %joined : tensor<3x6xf32> to memref<3x6xf32>
  func.call @check_f32(%memory) : (memref<3x6xf32>) -> ()
  %x = arith.constant dense<[[-128, 0, 127], [-3, -2, -1], [1, 2, 3]]> : tensor<3x3xi8>
  %y = arith.constant dense<[[4, 5], [6, 7], [8, 9]]> : tensor<3x2xi8>
  %bytes = "frontend.concat_columns"(%x, %y) : (tensor<3x3xi8>, tensor<3x2xi8>) -> tensor<3x5xi8>
  %byte_memory = bufferization.to_buffer %bytes : tensor<3x5xi8> to memref<3x5xi8>
  func.call @check_i8(%byte_memory) : (memref<3x5xi8>) -> ()
  %ok = arith.constant 0 : i8
  return %ok : i8
}
