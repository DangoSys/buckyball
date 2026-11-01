// CHECK: frontend.trace_start {{.*}}id = 0
// CHECK: frontend.trace_start {{.*}}id = 1
// CHECK: im2col_ball.bank_im2col_i8
// CHECK: frontend.trace_end {{.*}}id = 0
// CHECK: frontend.trace_end {{.*}}id = 1
// CHECK-NOT: frontend.trace_start
// CHECK: return
func.func @conv(%x: tensor<1x4x4x2xf32>, %w1: tensor<16x2x3x3xi8>, %b1: tensor<16xi32>, %w2: tensor<16x16x3x3xi8>, %b2: tensor<16xi32>) -> tensor<1x4x4x16xf32> {
  %unit = arith.constant dense<1.0> : tensor<1x1x1x1xf32>
  %scale = arith.constant dense<1.0> : tensor<16x1x1x1xf32>
  %zero = arith.constant dense<0.0> : tensor<1x4x4x16xf32>
  frontend.trace_start {cycle = true, tensor = false, id = 0 : i64, id_path = [0]}
  %first = bbacc.custom "conv2d_i8"(%x, %w1, %b1, %unit, %unit, %scale) {strides = array<i64: 1, 1>, dilations = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>} : (tensor<1x4x4x2xf32>, tensor<16x2x3x3xi8>, tensor<16xi32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<16x1x1x1xf32>) -> tensor<1x4x4x16xf32>
  %first_trace = frontend.trace_end %first {cycle = true, tensor = false, id = 0 : i64, id_path = [0]} : tensor<1x4x4x16xf32>
  frontend.trace_start {cycle = true, tensor = false, id = 1 : i64, id_path = [1]}
  %relu = linalg.generic {indexing_maps = [affine_map<(n,h,w,c)->(n,h,w,c)>, affine_map<(n,h,w,c)->(n,h,w,c)>, affine_map<(n,h,w,c)->(n,h,w,c)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%first_trace, %zero: tensor<1x4x4x16xf32>, tensor<1x4x4x16xf32>) outs(%zero: tensor<1x4x4x16xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %positive = arith.maximumf %a, %b : f32
    linalg.yield %positive : f32
  } -> tensor<1x4x4x16xf32>
  %relu_trace = frontend.trace_end %relu {cycle = true, tensor = false, id = 1 : i64, id_path = [1]} : tensor<1x4x4x16xf32>
  %second = bbacc.custom "conv2d_i8"(%relu_trace, %w2, %b2, %unit, %unit, %scale) {strides = array<i64: 1, 1>, dilations = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>} : (tensor<1x4x4x16xf32>, tensor<16x16x3x3xi8>, tensor<16xi32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<16x1x1x1xf32>) -> tensor<1x4x4x16xf32>
  return %second : tensor<1x4x4x16xf32>
}
