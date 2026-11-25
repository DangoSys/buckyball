// BANK: toint8_ball.bank_int32_to_int8 {{.*}}relu>
// BANK: im2col_ball.bank_im2col_i8
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
  %conv_weight = memref.alloc() alignment = 64 : memref<1x2304xi8>
  linalg.fill ins(%zero_i8 : i8) outs(%conv_weight : memref<1x2304xi8>)
  %conv_bias = memref.alloc() alignment = 64 : memref<16xi32>
  %depthwise_weight = memref.alloc() alignment = 64 : memref<16x1x3x3xi8>
  %depthwise_bias = memref.alloc() alignment = 64 : memref<16xi32>
  scf.for %h = %c0 to %c4 step %c1 {
    scf.for %w = %c0 to %c4 step %c1 {
      scf.for %ic = %c0 to %c2 step %c1 {
        %sum = affine.apply affine_map<(h, w, ic) -> (h * 3 + w * 2 + ic)>(%h, %w, %ic)
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
        %sum = affine.apply affine_map<(p, ic, oc) -> (p + ic + oc * 3)>(%p, %ic, %oc)
        %rem = arith.remui %sum, %c5 : index
        %signed = arith.subi %rem, %c2 : index
        %value = arith.index_cast %signed : index to i8
        %offset = affine.apply affine_map<(p, ic, oc) -> ((p * 16 + ic) * 16 + oc)>(%p, %ic, %oc)
        memref.store %value, %conv_weight[%c0, %offset] : memref<1x2304xi8>
      }
    }
  }
  scf.for %oc = %c0 to %c16 step %c1 {
    %signed = arith.subi %oc, %c4 : index
    %value = arith.index_cast %signed : index to i32
    memref.store %value, %conv_bias[%oc] : memref<16xi32>
  }
  scf.for %kh = %c0 to %c3 step %c1 {
    scf.for %kw = %c0 to %c3 step %c1 {
      scf.for %oc = %c0 to %c16 step %c1 {
        %kw2 = arith.muli %kw, %c2 : index
        %sum0 = arith.addi %kh, %kw2 : index
        %sum = arith.addi %sum0, %oc : index
        %rem = arith.remui %sum, %c5 : index
        %signed = arith.subi %rem, %c2 : index
        %value = arith.index_cast %signed : index to i8
        memref.store %value, %depthwise_weight[%oc, %c0, %kh, %kw] : memref<16x1x3x3xi8>
      }
    }
  }
  linalg.fill ins(%zero_i32 : i32) outs(%depthwise_bias : memref<16xi32>)
  %unit = arith.constant dense<1.0> : tensor<1x1x1x1xf32>
  %weight_step = arith.constant dense<1.0> : tensor<16x1x1x1xf32>
  %it = bufferization.to_tensor %input restrict : memref<1x4x4x2xi8> to tensor<1x4x4x2xi8>
  %input_float = "frontend.int8_to_fp32"(%it, %unit) : (tensor<1x4x4x2xi8>, tensor<1x1x1x1xf32>) -> tensor<1x4x4x2xf32>
  %wt_packed = bufferization.to_tensor %conv_weight restrict : memref<1x2304xi8> to tensor<1x2304xi8>
  %wt_expanded = tensor.expand_shape %wt_packed [[0], [1, 2, 3, 4, 5]] output_shape [1, 1, 3, 3, 16, 16] : tensor<1x2304xi8> into tensor<1x1x3x3x16x16xi8>
  %wt_empty = tensor.empty() : tensor<1x16x1x16x3x3xi8>
  %wt_transposed = linalg.transpose ins(%wt_expanded : tensor<1x1x3x3x16x16xi8>) outs(%wt_empty : tensor<1x16x1x16x3x3xi8>) permutation = [0, 5, 1, 4, 2, 3]
  %wt_full = tensor.collapse_shape %wt_transposed [[0, 1], [2, 3], [4], [5]] : tensor<1x16x1x16x3x3xi8> into tensor<16x16x3x3xi8>
  %wt = tensor.extract_slice %wt_full[0, 0, 0, 0] [16, 2, 3, 3] [1, 1, 1, 1] : tensor<16x16x3x3xi8> to tensor<16x2x3x3xi8>
  %bt = bufferization.to_tensor %conv_bias restrict : memref<16xi32> to tensor<16xi32>
  %dt = bufferization.to_tensor %depthwise_weight restrict : memref<16x1x3x3xi8> to tensor<16x1x3x3xi8>
  %dbt = bufferization.to_tensor %depthwise_bias restrict : memref<16xi32> to tensor<16xi32>
  %first = bbacc.custom "conv2d_i8"(%input_float, %wt, %bt, %unit, %unit, %weight_step) {strides = array<i64: 1, 1>, dilations = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>} : (tensor<1x4x4x2xf32>, tensor<16x2x3x3xi8>, tensor<16xi32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<16x1x1x1xf32>) -> tensor<1x4x4x16xf32>
  %zero = arith.constant dense<0.0> : tensor<1x4x4x16xf32>
  %relu = linalg.generic {indexing_maps = [affine_map<(n,h,w,c)->(n,h,w,c)>, affine_map<(n,h,w,c)->(n,h,w,c)>, affine_map<(n,h,w,c)->(n,h,w,c)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%first, %zero: tensor<1x4x4x16xf32>, tensor<1x4x4x16xf32>) outs(%zero: tensor<1x4x4x16xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %r = arith.maximumf %a, %b : f32
    linalg.yield %r : f32
  } -> tensor<1x4x4x16xf32>
  %second = bbacc.custom "depthwise_conv2d_i8"(%relu, %dt, %dbt, %unit, %unit, %weight_step) {strides = array<i64: 1, 1>, dilations = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>} : (tensor<1x4x4x16xf32>, tensor<16x1x3x3xi8>, tensor<16xi32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<16x1x1x1xf32>) -> tensor<1x4x4x16xf32>
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
