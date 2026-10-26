func.func private @check_result(memref<17x17xf32>) -> ()
func.func @main() -> i8 {
  %z8 = arith.constant 0 : i8
  %z = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %three = arith.constant 3 : index
  %five = arith.constant 5 : index
  %seven = arith.constant 7 : index
  %eight = arith.constant 8 : index
  %dim = arith.constant 17 : index
  %inner = arith.constant 65 : index
  %input = memref.alloc() alignment = 64 : memref<17x65xi8>
  %weight = memref.alloc() alignment = 64 : memref<65x17xi8>
  %bias = memref.alloc() alignment = 64 : memref<17xi32>
  scf.for %r = %z to %dim step %one {
    scf.for %k = %z to %inner step %one {
      %sum = arith.addi %r, %k : index
      %rem = arith.remui %sum, %seven : index
      %v = arith.subi %rem, %three : index
      %i8 = arith.index_cast %v : index to i8
      memref.store %i8, %input[%r, %k] : memref<17x65xi8>
    }
    %b = arith.subi %r, %eight : index
    %b32 = arith.index_cast %b : index to i32
    memref.store %b32, %bias[%r] : memref<17xi32>
  }
  scf.for %k = %z to %inner step %one {
    scf.for %c = %z to %dim step %one {
      %k2 = arith.muli %k, %two : index
      %c3 = arith.muli %c, %three : index
      %sum = arith.addi %k2, %c3 : index
      %rem = arith.remui %sum, %five : index
      %v = arith.subi %rem, %two : index
      %i8 = arith.index_cast %v : index to i8
      memref.store %i8, %weight[%k, %c] : memref<65x17xi8>
    }
  }
  %it = bufferization.to_tensor %input restrict : memref<17x65xi8> to tensor<17x65xi8>
  %wt = bufferization.to_tensor %weight restrict : memref<65x17xi8> to tensor<65x17xi8>
  %bt = bufferization.to_tensor %bias restrict : memref<17xi32> to tensor<17xi32>
  %bias_tensor = tensor.expand_shape %bt [[0, 1]] output_shape [1, 17] : tensor<17xi32> into tensor<1x17xi32>
  %unit = arith.constant dense<1.0> : tensor<1x1xf32>
  %weight_step = arith.constant dense<0.5> : tensor<1x17xf32>
  %input_float = "frontend.int8_to_fp32"(%it, %unit) : (tensor<17x65xi8>, tensor<1x1xf32>) -> tensor<17x65xf32>
  %result = custom.op "linear_i8"(%input_float, %wt, %bias_tensor, %unit, %unit, %weight_step) : (tensor<17x65xf32>, tensor<65x17xi8>, tensor<1x17xi32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x17xf32>) -> tensor<17x17xf32>
  %memory = bufferization.to_buffer %result : tensor<17x17xf32> to memref<17x17xf32>
  func.call @check_result(%memory) : (memref<17x17xf32>) -> ()
  return %z8 : i8
}
