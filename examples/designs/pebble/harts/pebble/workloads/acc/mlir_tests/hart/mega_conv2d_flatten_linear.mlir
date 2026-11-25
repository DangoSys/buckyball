// The pooled map stays in its bank and feeds the Linear transposed.
// BANK: maxpool_ball.bank_maxpool_i8
// BANK-NOT: bbacc.bank_mvout
// BANK: transpose_ball.bank_transpose_i8
// BANK: smatmul_ball.bank_matmul_i8
func.func private @check_result(memref<1x16xf32>) -> ()
func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %c7 = arith.constant 7 : index
  %c8 = arith.constant 8 : index
  %c9 = arith.constant 9 : index
  %c16 = arith.constant 16 : index
  %c256 = arith.constant 256 : index
  %input = memref.alloc() alignment = 64 : memref<1x8x8x2xi8>
  %conv_weight = memref.alloc() alignment = 64 : memref<1x2304xi8>
  linalg.fill ins(%zero_i8 : i8) outs(%conv_weight : memref<1x2304xi8>)
  %conv_bias = memref.alloc() alignment = 64 : memref<16xi32>
  %linear_weight = memref.alloc() alignment = 64 : memref<256x16xi8>
  %linear_bias = memref.alloc() alignment = 64 : memref<1x16xi32>
  scf.for %h = %c0 to %c8 step %c1 {
    scf.for %w = %c0 to %c8 step %c1 {
      scf.for %ic = %c0 to %c2 step %c1 {
        %sum = affine.apply affine_map<(h, w, ic) -> (h * 3 + w * 2 + ic)>(%h, %w, %ic)
        %rem = arith.remui %sum, %c7 : index
        %signed = arith.subi %rem, %c3 : index
        %value = arith.index_cast %signed : index to i8
        memref.store %value, %input[%c0, %h, %w, %ic] : memref<1x8x8x2xi8>
      }
    }
  }
  scf.for %oc = %c0 to %c16 step %c1 {
    scf.for %ic = %c0 to %c2 step %c1 {
      scf.for %p = %c0 to %c9 step %c1 {
        %sum = affine.apply affine_map<(p, ic, oc) -> (p + ic + oc * 3)>(%p, %ic, %oc)
        %rem = arith.remui %sum, %c5 : index
        %signed = arith.subi %rem, %c2 : index
        %value = arith.index_cast %signed : index to i8
        %offset = affine.apply affine_map<(p, ic, oc) -> ((p * 16 + ic) * 16 + oc)>(%p, %ic, %oc)
        memref.store %value, %conv_weight[%c0, %offset] : memref<1x2304xi8>
      }
    }
    %bias = arith.index_cast %oc : index to i32
    memref.store %bias, %conv_bias[%oc] : memref<16xi32>
    memref.store %bias, %linear_bias[%c0, %oc] : memref<1x16xi32>
    scf.for %k = %c0 to %c256 step %c1 {
      %sum = arith.addi %k, %oc : index
      %rem = arith.remui %sum, %c5 : index
      %signed = arith.subi %rem, %c2 : index
      %value = arith.index_cast %signed : index to i8
      memref.store %value, %linear_weight[%k, %oc] : memref<256x16xi8>
    }
  }
  %unit = arith.constant dense<1.0> : tensor<1x1x1x1xf32>
  %step = arith.constant dense<1.0> : tensor<1x1xf32>
  %conv_step = arith.constant dense<1.0> : tensor<16x1x1x1xf32>
  %linear_step = arith.constant dense<1.0> : tensor<1x16xf32>
  %it = bufferization.to_tensor %input restrict : memref<1x8x8x2xi8> to tensor<1x8x8x2xi8>
  %x = "frontend.int8_to_fp32"(%it, %unit) : (tensor<1x8x8x2xi8>, tensor<1x1x1x1xf32>) -> tensor<1x8x8x2xf32>
  %wt_packed = bufferization.to_tensor %conv_weight restrict : memref<1x2304xi8> to tensor<1x2304xi8>
  %wt_expanded = tensor.expand_shape %wt_packed [[0], [1, 2, 3, 4, 5]] output_shape [1, 1, 3, 3, 16, 16] : tensor<1x2304xi8> into tensor<1x1x3x3x16x16xi8>
  %wt_empty = tensor.empty() : tensor<1x16x1x16x3x3xi8>
  %wt_transposed = linalg.transpose ins(%wt_expanded : tensor<1x1x3x3x16x16xi8>) outs(%wt_empty : tensor<1x16x1x16x3x3xi8>) permutation = [0, 5, 1, 4, 2, 3]
  %wt_full = tensor.collapse_shape %wt_transposed [[0, 1], [2, 3], [4], [5]] : tensor<1x16x1x16x3x3xi8> into tensor<16x16x3x3xi8>
  %wt = tensor.extract_slice %wt_full[0, 0, 0, 0] [16, 2, 3, 3] [1, 1, 1, 1] : tensor<16x16x3x3xi8> to tensor<16x2x3x3xi8>
  %bt = bufferization.to_tensor %conv_bias restrict : memref<16xi32> to tensor<16xi32>
  %lw = bufferization.to_tensor %linear_weight restrict : memref<256x16xi8> to tensor<256x16xi8>
  %lb = bufferization.to_tensor %linear_bias restrict : memref<1x16xi32> to tensor<1x16xi32>
  %conv = bbacc.custom "conv2d_i8"(%x, %wt, %bt, %unit, %unit, %conv_step) {strides = array<i64: 1, 1>, dilations = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>} : (tensor<1x8x8x2xf32>, tensor<16x2x3x3xi8>, tensor<16xi32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<16x1x1x1xf32>) -> tensor<1x8x8x16xf32>
  %zero = arith.constant dense<0.0> : tensor<1x8x8x16xf32>
  %relu = linalg.generic {indexing_maps = [affine_map<(n,h,w,c)->(n,h,w,c)>, affine_map<(n,h,w,c)->(n,h,w,c)>, affine_map<(n,h,w,c)->(n,h,w,c)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%conv, %zero: tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>) outs(%zero: tensor<1x8x8x16xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %r = arith.maximumf %a, %b : f32
    linalg.yield %r : f32
  } -> tensor<1x8x8x16xf32>
  %filter = tensor.empty() : tensor<2x2xf32>
  %minimum = arith.constant dense<-3.40282347E+38> : tensor<1x4x4x16xf32>
  %pooled = linalg.pooling_nhwc_max {strides = dense<2> : vector<2xi64>, dilations = dense<1> : vector<2xi64>} ins(%relu, %filter: tensor<1x8x8x16xf32>, tensor<2x2xf32>) outs(%minimum: tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
  %empty = tensor.empty() : tensor<1x16x4x4xf32>
  %nchw = linalg.transpose ins(%pooled: tensor<1x4x4x16xf32>) outs(%empty: tensor<1x16x4x4xf32>) permutation = [0, 3, 1, 2]
  %flat = tensor.collapse_shape %nchw [[0], [1, 2, 3]] : tensor<1x16x4x4xf32> into tensor<1x256xf32>
  %result = bbacc.custom "linear_i8"(%flat, %lw, %lb, %step, %step, %linear_step) : (tensor<1x256xf32>, tensor<256x16xi8>, tensor<1x16xi32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x16xf32>) -> tensor<1x16xf32>
  %memory = bufferization.to_buffer %result : tensor<1x16xf32> to memref<1x16xf32>
  func.call @check_result(%memory) : (memref<1x16xf32>) -> ()
  return %zero_i8 : i8
}
