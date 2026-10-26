// CHECK-LABEL: func.func @linear
// CHECK: buddy_trace.start
// CHECK: %[[R:.*]] = custom.op "linear_i8"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5)
// CHECK: %[[T:.*]] = buddy_trace.end %[[R]]
// CHECK-NOT: arith.addf
// CHECK-NOT: frontend.int32_to_fp32
// CHECK: return %[[T]]
func.func @linear(%a: tensor<1x32xf32>, %w: tensor<32x32xi8>, %bias: tensor<1x32xi32>, %multiplier: tensor<1x1xf32>, %input_step: tensor<1x1xf32>, %weight_step: tensor<1x32xf32>) -> tensor<1x32xf32> {
  %zero = arith.constant dense<0.0> : tensor<1x32xf32>
  %q = "frontend.fp32_to_int8"(%a, %multiplier) : (tensor<1x32xf32>, tensor<1x1xf32>) -> tensor<1x32xi8>
  %x = "frontend.int8_to_fp32"(%q, %input_step) : (tensor<1x32xi8>, tensor<1x1xf32>) -> tensor<1x32xf32>
  %y = "frontend.int8_to_fp32"(%w, %weight_step) : (tensor<32x32xi8>, tensor<1x32xf32>) -> tensor<32x32xf32>
  buddy_trace.start {cycle = true, id = 0 : i64, id_path = [0], tag = "linear", tensor = false}
  %product = linalg.matmul ins(%x, %y: tensor<1x32xf32>, tensor<32x32xf32>) outs(%zero: tensor<1x32xf32>) -> tensor<1x32xf32>
  %traced = buddy_trace.end %product {cycle = true, id = 0 : i64, id_path = [0], tag = "linear", tensor = false} : tensor<1x32xf32>
  %empty = tensor.empty() : tensor<1x32xf32>
  %scale = linalg.generic {
    indexing_maps = [affine_map<(m,n)->(m,0)>, affine_map<(m,n)->(m,n)>, affine_map<(m,n)->(m,n)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input_step, %weight_step: tensor<1x1xf32>, tensor<1x32xf32>) outs(%empty: tensor<1x32xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %unused: f32):
    %s = arith.mulf %lhs, %rhs : f32
    linalg.yield %s : f32
  } -> tensor<1x32xf32>
  %bias_f32 = "frontend.int32_to_fp32"(%bias, %scale) : (tensor<1x32xi32>, tensor<1x32xf32>) -> tensor<1x32xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(m,n)->(m,n)>, affine_map<(m,n)->(m,n)>, affine_map<(m,n)->(m,n)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%traced, %bias_f32: tensor<1x32xf32>, tensor<1x32xf32>) outs(%empty: tensor<1x32xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %unused: f32):
    %s = arith.addf %lhs, %rhs : f32
    linalg.yield %s : f32
  } -> tensor<1x32xf32>
  return %result : tensor<1x32xf32>
}

// A bias in a different scale domain cannot initialize this accumulator.
// CHECK-LABEL: func.func @unrelated_scale
// CHECK: %[[ZERO:.*]] = arith.constant dense<0> : tensor<1x32xi32>
// CHECK: custom.op "linear_i8"(%arg0, %arg1, %[[ZERO]], %arg3, %arg4, %arg5)
// CHECK: "frontend.int32_to_fp32"(%arg2, %arg6)
// CHECK: arith.addf
// CHECK: return
func.func @unrelated_scale(%a: tensor<1x32xf32>, %w: tensor<32x32xi8>, %bias: tensor<1x32xi32>, %multiplier: tensor<1x1xf32>, %input_step: tensor<1x1xf32>, %weight_step: tensor<1x32xf32>, %bias_step: tensor<1x32xf32>) -> tensor<1x32xf32> {
  %zero = arith.constant dense<0.0> : tensor<1x32xf32>
  %q = "frontend.fp32_to_int8"(%a, %multiplier) : (tensor<1x32xf32>, tensor<1x1xf32>) -> tensor<1x32xi8>
  %x = "frontend.int8_to_fp32"(%q, %input_step) : (tensor<1x32xi8>, tensor<1x1xf32>) -> tensor<1x32xf32>
  %y = "frontend.int8_to_fp32"(%w, %weight_step) : (tensor<32x32xi8>, tensor<1x32xf32>) -> tensor<32x32xf32>
  %product = linalg.matmul ins(%x, %y: tensor<1x32xf32>, tensor<32x32xf32>) outs(%zero: tensor<1x32xf32>) -> tensor<1x32xf32>
  %bias_f32 = "frontend.int32_to_fp32"(%bias, %bias_step) : (tensor<1x32xi32>, tensor<1x32xf32>) -> tensor<1x32xf32>
  %empty = tensor.empty() : tensor<1x32xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(m,n)->(m,n)>, affine_map<(m,n)->(m,n)>, affine_map<(m,n)->(m,n)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%product, %bias_f32: tensor<1x32xf32>, tensor<1x32xf32>) outs(%empty: tensor<1x32xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %unused: f32):
    %s = arith.addf %lhs, %rhs : f32
    linalg.yield %s : f32
  } -> tensor<1x32xf32>
  return %result : tensor<1x32xf32>
}
