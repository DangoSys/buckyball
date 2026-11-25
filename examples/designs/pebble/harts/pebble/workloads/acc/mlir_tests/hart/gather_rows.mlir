func.func private @check_i32(memref<6x4xi32>)
func.func private @check_f32(memref<2x4xf32>)
func.func @main() -> i8 {
  %indices = arith.constant dense<[5, 0, 2, 3, 2, 1]> : tensor<6xi64>
  %a = arith.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi32>
  %b = arith.constant dense<[[8, 9, 10, 11]]> : tensor<1x4xi32>
  %c = arith.constant dense<[[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, -1]]> : tensor<3x4xi32>
  %gathered = "frontend.gather_rows"(%indices, %a, %b, %c) : (tensor<6xi64>, tensor<2x4xi32>, tensor<1x4xi32>, tensor<3x4xi32>) -> tensor<6x4xi32>
  %memory = bufferization.to_buffer %gathered : tensor<6x4xi32> to memref<6x4xi32>
  func.call @check_i32(%memory) : (memref<6x4xi32>) -> ()
  %float_indices = arith.constant dense<[2, 0]> : tensor<2xi64>
  %f = arith.constant dense<[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]]> : tensor<3x4xf32>
  %floats = "frontend.gather_rows"(%float_indices, %f) : (tensor<2xi64>, tensor<3x4xf32>) -> tensor<2x4xf32>
  %float_memory = bufferization.to_buffer %floats : tensor<2x4xf32> to memref<2x4xf32>
  func.call @check_f32(%float_memory) : (memref<2x4xf32>) -> ()
  %ok = arith.constant 0 : i8
  return %ok : i8
}
