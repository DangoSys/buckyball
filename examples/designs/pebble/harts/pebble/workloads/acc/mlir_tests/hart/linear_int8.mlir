func.func private @check_result(memref<1x32xf32>)

func.func @main() -> i8 {
  %exit = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %c13 = arith.constant 13 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %scale_unit = arith.constant 0.0078125 : f32
  %input = arith.constant dense<[[
    -300.0, -255.0, -5.0, -3.0, -1.0, 0.0, 1.0, 3.0,
    5.0, 254.0, 255.0, 300.0, 2.0, 6.0, 10.0, 14.0,
    -257.0, -253.0, -7.0, -9.0, -11.0, -13.0, -15.0, 7.0,
    9.0, 11.0, 13.0, 15.0, 253.0, 257.0, -254.0, -256.0
  ]]> : tensor<1x32xf32>
  %multiplier = arith.constant dense<0.5> : tensor<1x1xf32>
  %input_step = arith.constant dense<2.0> : tensor<1x1xf32>
  %zero = arith.constant dense<0.0> : tensor<1x32xf32>
  %weights = memref.alloc() alignment = 64 : memref<32x32xi8>
  %steps = memref.alloc() alignment = 64 : memref<1x32xf32>
  %bias = memref.alloc() alignment = 64 : memref<1x32xi32>
  scf.for %column = %c0 to %c32 step %c1 {
    %number = arith.addi %column, %c1 : index
    %number_i32 = arith.index_cast %number : index to i32
    %number_f32 = arith.sitofp %number_i32 : i32 to f32
    %step = arith.mulf %number_f32, %scale_unit : f32
    memref.store %step, %steps[%c0, %column] : memref<1x32xf32>
    %shifted = arith.subi %column, %c16 : index
    %shifted_i32 = arith.index_cast %shifted : index to i32
    memref.store %shifted_i32, %bias[%c0, %column] : memref<1x32xi32>
    scf.for %k = %c0 to %c32 step %c1 {
      %k7 = arith.muli %k, %c7 : index
      %column3 = arith.muli %column, %c3 : index
      %sum = arith.addi %k7, %column3 : index
      %mod = arith.remui %sum, %c13 : index
      %signed = arith.subi %mod, %c6 : index
      %weight = arith.index_cast %signed : index to i8
      memref.store %weight, %weights[%k, %column] : memref<32x32xi8>
    }
  }
  %weight_codes = bufferization.to_tensor %weights restrict : memref<32x32xi8> to tensor<32x32xi8>
  %weight_step = bufferization.to_tensor %steps restrict : memref<1x32xf32> to tensor<1x32xf32>
  %bias_codes = bufferization.to_tensor %bias restrict : memref<1x32xi32> to tensor<1x32xi32>
  %codes = "frontend.fp32_to_int8"(%input, %multiplier) : (tensor<1x32xf32>, tensor<1x1xf32>) -> tensor<1x32xi8>
  %activation = "frontend.int8_to_fp32"(%codes, %input_step) : (tensor<1x32xi8>, tensor<1x1xf32>) -> tensor<1x32xf32>
  %weight_values = "frontend.int8_to_fp32"(%weight_codes, %weight_step) : (tensor<32x32xi8>, tensor<1x32xf32>) -> tensor<32x32xf32>
  %product = linalg.matmul ins(%activation, %weight_values : tensor<1x32xf32>, tensor<32x32xf32>) outs(%zero : tensor<1x32xf32>) -> tensor<1x32xf32>
  %empty = tensor.empty() : tensor<1x32xf32>
  %bias_step = linalg.generic {
    indexing_maps = [affine_map<(m,n)->(m,0)>, affine_map<(m,n)->(m,n)>, affine_map<(m,n)->(m,n)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input_step, %weight_step: tensor<1x1xf32>, tensor<1x32xf32>) outs(%empty: tensor<1x32xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %unused: f32):
    %scale = arith.mulf %lhs, %rhs : f32
    linalg.yield %scale : f32
  } -> tensor<1x32xf32>
  %bias_tensor = "frontend.int32_to_fp32"(%bias_codes, %bias_step) : (tensor<1x32xi32>, tensor<1x32xf32>) -> tensor<1x32xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(m, n) -> (m, n)>, affine_map<(m, n) -> (m, n)>, affine_map<(m, n) -> (m, n)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%product, %bias_tensor : tensor<1x32xf32>, tensor<1x32xf32>) outs(%empty : tensor<1x32xf32>) {
  ^bb0(%value: f32, %addend: f32, %unused: f32):
    %added = arith.addf %value, %addend : f32
    linalg.yield %added : f32
  } -> tensor<1x32xf32>
  %output = bufferization.to_buffer %result : tensor<1x32xf32> to memref<1x32xf32>
  func.call @check_result(%output) : (memref<1x32xf32>) -> ()
  return %exit : i8
}
