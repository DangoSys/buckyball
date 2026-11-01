// CHECK: frontend.trace_start {{.*}}id = 0
// CHECK: frontend.trace_start {{.*}}id = 1
// CHECK: smatmul_ball.bank_matmul_i8
// CHECK: frontend.trace_end {{.*}}id = 0
// CHECK: frontend.trace_end {{.*}}id = 1
// CHECK-NOT: frontend.trace_start
// CHECK: return
func.func @linear(%x: tensor<1x32xf32>, %w1: tensor<32x32xi8>, %b1: tensor<1x32xi32>, %s1: tensor<1x32xf32>, %w2: tensor<32x16xi8>, %b2: tensor<1x16xi32>, %s2: tensor<1x16xf32>) -> tensor<1x16xf32> {
  %unit = arith.constant dense<1.0> : tensor<1x1xf32>
  %zero = arith.constant dense<0.0> : tensor<1x32xf32>
  frontend.trace_start {cycle = true, tensor = false, id = 0 : i64, id_path = [0]}
  %first = bbacc.custom "linear_i8"(%x, %w1, %b1, %unit, %unit, %s1) : (tensor<1x32xf32>, tensor<32x32xi8>, tensor<1x32xi32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x32xf32>) -> tensor<1x32xf32>
  %first_trace = frontend.trace_end %first {cycle = true, tensor = false, id = 0 : i64, id_path = [0]} : tensor<1x32xf32>
  frontend.trace_start {cycle = true, tensor = false, id = 1 : i64, id_path = [1]}
  %relu = linalg.generic {indexing_maps = [affine_map<(m,n)->(m,n)>, affine_map<(m,n)->(m,n)>, affine_map<(m,n)->(m,n)>], iterator_types = ["parallel", "parallel"]} ins(%first_trace, %zero: tensor<1x32xf32>, tensor<1x32xf32>) outs(%zero: tensor<1x32xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %positive = arith.maximumf %a, %b : f32
    linalg.yield %positive : f32
  } -> tensor<1x32xf32>
  %relu_trace = frontend.trace_end %relu {cycle = true, tensor = false, id = 1 : i64, id_path = [1]} : tensor<1x32xf32>
  %second = bbacc.custom "linear_i8"(%relu_trace, %w2, %b2, %unit, %unit, %s2) : (tensor<1x32xf32>, tensor<32x16xi8>, tensor<1x16xi32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x16xf32>) -> tensor<1x16xf32>
  return %second : tensor<1x16xf32>
}
