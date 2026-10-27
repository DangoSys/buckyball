// CHECK-LABEL: func.func @depthwise
// CHECK: custom.op "depthwise_conv2d_i8"
// CHECK-SAME: padding = array<i64: 1, 1, 1, 1>
// CHECK: return
func.func @depthwise(%x: tensor<1x4x4x16xf32>, %w: tensor<16x1x3x3xi8>, %bias_codes: tensor<16xi32>, %input_step: tensor<1x1x1x1xf32>, %multiplier: tensor<1x1x1x1xf32>, %weight_step: tensor<16x1x1x1xf32>) -> tensor<1x4x4x16xf32> {
  %q = "frontend.fp32_to_int8"(%x,%multiplier) : (tensor<1x4x4x16xf32>,tensor<1x1x1x1xf32>) -> tensor<1x4x4x16xi8>
  %values = "frontend.int8_to_fp32"(%q,%input_step) : (tensor<1x4x4x16xi8>,tensor<1x1x1x1xf32>) -> tensor<1x4x4x16xf32>
  %zero = arith.constant 0.0 : f32
  %padded = tensor.pad %values low[0,1,1,0] high[0,1,1,0] {
  ^bb0(%n: index, %h: index, %wi: index, %c: index):
    tensor.yield %zero : f32
  } : tensor<1x4x4x16xf32> to tensor<1x6x6x16xf32>
  %wf = "frontend.int8_to_fp32"(%w,%weight_step) : (tensor<16x1x3x3xi8>,tensor<16x1x1x1xf32>) -> tensor<16x1x3x3xf32>
  %we = tensor.empty() : tensor<3x3x16x1xf32>
  %weights = linalg.transpose ins(%wf: tensor<16x1x3x3xf32>) outs(%we: tensor<3x3x16x1xf32>) permutation = [2,3,0,1]
  %scale = linalg.generic {indexing_maps = [affine_map<(f,c,h,w)->(0,c,h,w)>,affine_map<(f,c,h,w)->(f,c,h,w)>,affine_map<(f,c,h,w)->(f,c,h,w)>], iterator_types = ["parallel","parallel","parallel","parallel"]} ins(%input_step,%weight_step: tensor<1x1x1x1xf32>,tensor<16x1x1x1xf32>) outs(%weight_step: tensor<16x1x1x1xf32>) {
  ^bb0(%a: f32,%b: f32,%o: f32):
    %r = arith.mulf %a,%b : f32
    linalg.yield %r : f32
  } -> tensor<16x1x1x1xf32>
  %bs = tensor.collapse_shape %scale [[0,1,2,3]] : tensor<16x1x1x1xf32> into tensor<16xf32>
  %bias = "frontend.int32_to_fp32"(%bias_codes,%bs) : (tensor<16xi32>,tensor<16xf32>) -> tensor<16xf32>
  %initial = arith.constant dense<0.0> : tensor<1x4x4x16x1xf32>
  %convolution = linalg.depthwise_conv_2d_nhwc_hwcm {strides = dense<1> : tensor<2xi64>, dilations = dense<1> : tensor<2xi64>} ins(%padded,%weights: tensor<1x6x6x16xf32>,tensor<3x3x16x1xf32>) outs(%initial: tensor<1x4x4x16x1xf32>) -> tensor<1x4x4x16x1xf32>
  %collapsed = tensor.collapse_shape %convolution [[0],[1],[2],[3,4]] : tensor<1x4x4x16x1xf32> into tensor<1x4x4x16xf32>
  %empty = tensor.empty() : tensor<1x4x4x16xf32>
  %result = linalg.generic {indexing_maps = [affine_map<(n,h,w,c)->(c)>,affine_map<(n,h,w,c)->(n,h,w,c)>,affine_map<(n,h,w,c)->(n,h,w,c)>], iterator_types = ["parallel","parallel","parallel","parallel"]} ins(%bias,%collapsed: tensor<16xf32>,tensor<1x4x4x16xf32>) outs(%empty: tensor<1x4x4x16xf32>) {
  ^bb0(%a: f32,%b: f32,%o: f32):
    %r = arith.addf %a,%b : f32
    linalg.yield %r : f32
  } -> tensor<1x4x4x16xf32>
  return %result : tensor<1x4x4x16xf32>
}
