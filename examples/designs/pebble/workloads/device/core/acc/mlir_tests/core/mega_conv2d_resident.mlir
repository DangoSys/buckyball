// FORM: %[[FIRST:.*]] = custom.op "conv2d_i8"{{.*}}custom.resident{{.*}}relu = true{{.*}}-> tensor<1x4x4x16xi8>
// BANK: %[[CODES:.*]] = toint8_ball.bank_int32_to_int8 {{.*}}relu>
// BANK-NOT: buckyball.bank_mvout
// BANK: im2col_ball.bank_im2col_i8 %[[CODES]] ->{{.*}}padding = 1,
// BANK: maxpool_ball.bank_maxpool_i8
// BANK: return
func.func private @check_result(memref<1x16x2x2xi8>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %zero_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c7 = arith.constant 7 : index
  %c9 = arith.constant 9 : index
  %c16 = arith.constant 16 : index
  %input = memref.alloc() alignment = 64 : memref<1x4x4x2xi8>
  %conv_weight = memref.alloc() alignment = 64 : memref<16x2x3x3xi8>
  %conv_bias = memref.alloc() alignment = 64 : memref<16xi32>
  %second_weight = memref.alloc() alignment = 64 : memref<16x16x3x3xi8>
  %second_bias = memref.alloc() alignment = 64 : memref<16xi32>
  scf.for %h = %c0 to %c4 step %c1 {
    scf.for %w = %c0 to %c4 step %c1 {
      scf.for %ic = %c0 to %c2 step %c1 {
        %h3 = arith.muli %h, %c3 : index
        %w2 = arith.muli %w, %c2 : index
        %sum0 = arith.addi %h3, %w2 : index
        %sum = arith.addi %sum0, %ic : index
        %rem = arith.remui %sum, %c7 : index
        %signed = arith.subi %rem, %c3 : index
        %value = arith.index_cast %signed : index to i8
        memref.store %value, %input[%c0, %h, %w, %ic] : memref<1x4x4x2xi8>
      }
    }
  }
  scf.for %ic = %c0 to %c2 step %c1 {
    scf.for %p = %c0 to %c9 step %c1 {
      scf.for %oc = %c0 to %c16 step %c1 {
        %oc3 = arith.muli %oc, %c3 : index
        %sum0 = arith.addi %p, %ic : index
        %sum = arith.addi %sum0, %oc3 : index
        %rem = arith.remui %sum, %c5 : index
        %signed = arith.subi %rem, %c2 : index
        %value = arith.index_cast %signed : index to i8
        %ky_index = arith.divui %p, %c3 : index
        %kx_index = arith.remui %p, %c3 : index
        memref.store %value, %conv_weight[%oc, %ic, %ky_index, %kx_index] : memref<16x2x3x3xi8>
      }
    }
  }
  scf.for %oc = %c0 to %c16 step %c1 {
    %signed = arith.subi %oc, %c4 : index
    %value = arith.index_cast %signed : index to i32
    memref.store %value, %conv_bias[%oc] : memref<16xi32>
  }
  scf.for %p = %c0 to %c9 step %c1 {
    scf.for %ic = %c0 to %c16 step %c1 {
      scf.for %oc = %c0 to %c16 step %c1 {
        %sum0 = arith.addi %p, %ic : index
        %sum = arith.addi %sum0, %oc : index
        %rem = arith.remui %sum, %c5 : index
        %signed = arith.subi %rem, %c2 : index
        %value = arith.index_cast %signed : index to i8
        %kh = arith.divui %p, %c3 : index
        %kw = arith.remui %p, %c3 : index
        memref.store %value, %second_weight[%oc, %ic, %kh, %kw] : memref<16x16x3x3xi8>
      }
    }
  }
  linalg.fill ins(%zero_i32 : i32) outs(%second_bias : memref<16xi32>)
  %unit = arith.constant dense<1.0> : tensor<1x1x1x1xf32>
  %weight_step = arith.constant dense<1.0> : tensor<16x1x1x1xf32>
  %it = bufferization.to_tensor %input restrict : memref<1x4x4x2xi8> to tensor<1x4x4x2xi8>
  %input_float = "frontend.int8_to_fp32"(%it, %unit) : (tensor<1x4x4x2xi8>, tensor<1x1x1x1xf32>) -> tensor<1x4x4x2xf32>
  %wt = bufferization.to_tensor %conv_weight restrict : memref<16x2x3x3xi8> to tensor<16x2x3x3xi8>
  %bt = bufferization.to_tensor %conv_bias restrict : memref<16xi32> to tensor<16xi32>
  %dt = bufferization.to_tensor %second_weight restrict : memref<16x16x3x3xi8> to tensor<16x16x3x3xi8>
  %dbt = bufferization.to_tensor %second_bias restrict : memref<16xi32> to tensor<16xi32>
  %first = custom.op "conv2d_i8"(%input_float, %wt, %bt, %unit, %unit, %weight_step) {strides = array<i64: 1, 1>, dilations = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>} : (tensor<1x4x4x2xf32>, tensor<16x2x3x3xi8>, tensor<16xi32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<16x1x1x1xf32>) -> tensor<1x4x4x16xf32>
  %zero = arith.constant dense<0.0> : tensor<1x4x4x16xf32>
  %relu = linalg.generic {indexing_maps = [affine_map<(n,h,w,c)->(n,h,w,c)>, affine_map<(n,h,w,c)->(n,h,w,c)>, affine_map<(n,h,w,c)->(n,h,w,c)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%first, %zero: tensor<1x4x4x16xf32>, tensor<1x4x4x16xf32>) outs(%zero: tensor<1x4x4x16xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %r = arith.maximumf %a, %b : f32
    linalg.yield %r : f32
  } -> tensor<1x4x4x16xf32>
  %second = custom.op "conv2d_i8"(%relu, %dt, %dbt, %unit, %unit, %weight_step) {strides = array<i64: 1, 1>, dilations = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>} : (tensor<1x4x4x16xf32>, tensor<16x16x3x3xi8>, tensor<16xi32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<16x1x1x1xf32>) -> tensor<1x4x4x16xf32>
  %filter = tensor.empty() : tensor<2x2xf32>
  %minimum = arith.constant dense<-3.40282347E+38> : tensor<1x2x2x16xf32>
  %pooled = linalg.pooling_nhwc_max {strides = dense<2> : vector<2xi64>, dilations = dense<1> : vector<2xi64>} ins(%second, %filter: tensor<1x4x4x16xf32>, tensor<2x2xf32>) outs(%minimum: tensor<1x2x2x16xf32>) -> tensor<1x2x2x16xf32>
  %q = "frontend.fp32_to_int8"(%pooled, %unit) : (tensor<1x2x2x16xf32>, tensor<1x1x1x1xf32>) -> tensor<1x2x2x16xi8>
  %empty = tensor.empty() : tensor<1x16x2x2xi8>
  %nchw = linalg.transpose ins(%q: tensor<1x2x2x16xi8>) outs(%empty: tensor<1x16x2x2xi8>) permutation = [0, 3, 1, 2]
  %memory = bufferization.to_buffer %nchw : tensor<1x16x2x2xi8> to memref<1x16x2x2xi8>
  func.call @check_result(%memory) : (memref<1x16x2x2xi8>) -> ()
  return %zero_i8 : i8
}
