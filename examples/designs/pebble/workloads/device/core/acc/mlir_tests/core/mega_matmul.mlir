func.func private @check_result(memref<1x16xf32>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %three = arith.constant 3 : index
  %five = arith.constant 5 : index
  %seven = arith.constant 7 : index
  %eight = arith.constant 8 : index
  %dim = arith.constant 16 : index
  %input = memref.alloc() alignment = 64 : memref<1x16xi8>
  %weight = memref.alloc() alignment = 64 : memref<16x16xi8>
  %bias = memref.alloc() alignment = 64 : memref<16xi32>
  scf.for %k = %zero to %dim step %one {
    %irem = arith.remui %k, %seven : index
    %ival = arith.subi %irem, %three : index
    %iv = arith.index_cast %ival : index to i8
    memref.store %iv, %input[%zero, %k] : memref<1x16xi8>
    %biasv = arith.subi %k, %eight : index
    %bias32 = arith.index_cast %biasv : index to i32
    memref.store %bias32, %bias[%k] : memref<16xi32>
    scf.for %column = %zero to %dim step %one {
      %k2 = arith.muli %k, %two : index
      %c3 = arith.muli %column, %three : index
      %sum = arith.addi %k2, %c3 : index
      %rem = arith.remui %sum, %five : index
      %value = arith.subi %rem, %two : index
      %wv = arith.index_cast %value : index to i8
      memref.store %wv, %weight[%k, %column] : memref<16x16xi8>
    }
  }
  %it = bufferization.to_tensor %input restrict : memref<1x16xi8> to tensor<1x16xi8>
  %wt = bufferization.to_tensor %weight restrict : memref<16x16xi8> to tensor<16x16xi8>
  %bt = bufferization.to_tensor %bias restrict : memref<16xi32> to tensor<16xi32>
  %bias_tensor = tensor.expand_shape %bt [[0, 1]] output_shape [1, 16] : tensor<16xi32> into tensor<1x16xi32>
  %unit = arith.constant dense<1.0> : tensor<1x1xf32>
  %weight_step = arith.constant dense<1.0> : tensor<1x16xf32>
  %input_float = "frontend.int8_to_fp32"(%it, %unit) : (tensor<1x16xi8>, tensor<1x1xf32>) -> tensor<1x16xf32>
  %result = custom.op "linear_i8"(%input_float, %wt, %bias_tensor, %unit, %unit, %weight_step) : (tensor<1x16xf32>, tensor<16x16xi8>, tensor<1x16xi32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x16xf32>) -> tensor<1x16xf32>
  %memory = bufferization.to_buffer %result : tensor<1x16xf32> to memref<1x16xf32>
  func.call @check_result(%memory) : (memref<1x16xf32>) -> ()
  return %zero_i8 : i8
}
