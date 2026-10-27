// FORM: custom.op "linear_i8"{{.*}}custom.resident, relu = true{{.*}}-> tensor<1x32xi8>
// BANK: toint8_ball.bank_int32_to_int8
// BANK: buckyball.bank_transfer
// BANK-NOT: buckyball.bank_mvout
// BANK: smatmul_ball.bank_matmul_i8 {{.*}}k = 16>
func.func private @check_result(memref<1x16xf32>)

func.func @main() -> i8 {
  %exit = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c7 = arith.constant 7 : index
  %c8 = arith.constant 8 : index
  %c9 = arith.constant 9 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c99 = arith.constant 99 : i32
  %unit1 = arith.constant 0.015625 : f32
  %unit2 = arith.constant 0.0078125 : f32
  %multiplier = arith.constant dense<0.5> : tensor<1x1xf32>
  %input_step = arith.constant dense<2.0> : tensor<1x1xf32>
  %zero = arith.constant dense<0.0> : tensor<1x32xf32>
  %x = memref.alloc() alignment = 64 : memref<1x32xf32>
  %w1 = memref.alloc() alignment = 64 : memref<32x32xi8>
  %b1 = memref.alloc() alignment = 64 : memref<1x32xi32>
  %s1 = memref.alloc() alignment = 64 : memref<1x32xf32>
  %w2 = memref.alloc() alignment = 64 : memref<32x16xi8>
  %b2 = memref.alloc() alignment = 64 : memref<1x16xi32>
  %s2 = memref.alloc() alignment = 64 : memref<1x16xf32>
  scf.for %k = %c0 to %c32 step %c1 {
    %rem = arith.remui %k, %c7 : index
    %shift = arith.subi %rem, %c3 : index
    %integer = arith.index_cast %shift : index to i32
    %scaled = arith.muli %integer, %c99 : i32
    %value = arith.sitofp %scaled : i32 to f32
    memref.store %value, %x[%c0, %k] : memref<1x32xf32>
    %bias = arith.subi %k, %c16 : index
    %bias32 = arith.index_cast %bias : index to i32
    memref.store %bias32, %b1[%c0, %k] : memref<1x32xi32>
    %number = arith.addi %k, %c1 : index
    %number32 = arith.index_cast %number : index to i32
    %numberf = arith.sitofp %number32 : i32 to f32
    %scale = arith.mulf %numberf, %unit1 : f32
    memref.store %scale, %s1[%c0, %k] : memref<1x32xf32>
    scf.for %column = %c0 to %c32 step %c1 {
      %k3 = arith.muli %k, %c3 : index
      %column5 = arith.muli %column, %c5 : index
      %sum = arith.addi %k3, %column5 : index
      %mod = arith.remui %sum, %c9 : index
      %signed = arith.subi %mod, %c4 : index
      %weight = arith.index_cast %signed : index to i8
      memref.store %weight, %w1[%k, %column] : memref<32x32xi8>
    }
    scf.for %column = %c0 to %c16 step %c1 {
      %k5 = arith.muli %k, %c5 : index
      %column3 = arith.muli %column, %c3 : index
      %sum = arith.addi %k5, %column3 : index
      %mod = arith.remui %sum, %c7 : index
      %signed = arith.subi %mod, %c3 : index
      %weight = arith.index_cast %signed : index to i8
      memref.store %weight, %w2[%k, %column] : memref<32x16xi8>
    }
  }
  scf.for %column = %c0 to %c16 step %c1 {
    %bias = arith.subi %column, %c8 : index
    %bias32 = arith.index_cast %bias : index to i32
    memref.store %bias32, %b2[%c0, %column] : memref<1x16xi32>
    %number = arith.addi %column, %c1 : index
    %number32 = arith.index_cast %number : index to i32
    %numberf = arith.sitofp %number32 : i32 to f32
    %scale = arith.mulf %numberf, %unit2 : f32
    memref.store %scale, %s2[%c0, %column] : memref<1x16xf32>
  }
  %input = bufferization.to_tensor %x restrict : memref<1x32xf32> to tensor<1x32xf32>
  %weight1 = bufferization.to_tensor %w1 restrict : memref<32x32xi8> to tensor<32x32xi8>
  %bias1 = bufferization.to_tensor %b1 restrict : memref<1x32xi32> to tensor<1x32xi32>
  %step1 = bufferization.to_tensor %s1 restrict : memref<1x32xf32> to tensor<1x32xf32>
  %weight2 = bufferization.to_tensor %w2 restrict : memref<32x16xi8> to tensor<32x16xi8>
  %bias2 = bufferization.to_tensor %b2 restrict : memref<1x16xi32> to tensor<1x16xi32>
  %step2 = bufferization.to_tensor %s2 restrict : memref<1x16xf32> to tensor<1x16xf32>
  %first = custom.op "linear_i8"(%input, %weight1, %bias1, %multiplier, %input_step, %step1) : (tensor<1x32xf32>, tensor<32x32xi8>, tensor<1x32xi32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x32xf32>) -> tensor<1x32xf32>
  %empty = tensor.empty() : tensor<1x32xf32>
  %relu = linalg.generic {
    indexing_maps = [affine_map<(m,n)->(m,n)>, affine_map<(m,n)->(m,n)>, affine_map<(m,n)->(m,n)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%first, %zero: tensor<1x32xf32>, tensor<1x32xf32>) outs(%empty: tensor<1x32xf32>) {
  ^bb0(%value: f32, %zero_value: f32, %unused: f32):
    %positive = arith.maximumf %value, %zero_value : f32
    linalg.yield %positive : f32
  } -> tensor<1x32xf32>
  %second = custom.op "linear_i8"(%relu, %weight2, %bias2, %multiplier, %input_step, %step2) : (tensor<1x32xf32>, tensor<32x16xi8>, tensor<1x16xi32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x16xf32>) -> tensor<1x16xf32>
  %output = bufferization.to_buffer %second : tensor<1x16xf32> to memref<1x16xf32>
  func.call @check_result(%output) : (memref<1x16xf32>) -> ()
  return %exit : i8
}
