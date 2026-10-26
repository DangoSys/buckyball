// CHECK-LABEL: func.func @named
// CHECK-NOT: linalg.matmul
// CHECK: custom.op "matmul_i8"(%arg0, %arg1, %arg2)
// CHECK: return
func.func @named(%a: tensor<1x32xi8>, %b: tensor<32x32xi8>, %initial: tensor<1x32xi32>) -> tensor<1x32xi32> {
  %product = linalg.matmul ins(%a, %b: tensor<1x32xi8>, tensor<32x32xi8>) outs(%initial: tensor<1x32xi32>) -> tensor<1x32xi32>
  return %product : tensor<1x32xi32>
}

// CHECK-LABEL: func.func @generic
// CHECK-NOT: linalg.generic
// CHECK: custom.op "matmul_i8"(%arg0, %arg1, %arg2)
// CHECK: return
func.func @generic(%a: tensor<1x32xi8>, %b: tensor<32x32xi8>, %initial: tensor<1x32xi32>) -> tensor<1x32xi32> {
  %product = linalg.generic {
    indexing_maps = [affine_map<(m,n,k)->(m,k)>, affine_map<(m,n,k)->(k,n)>, affine_map<(m,n,k)->(m,n)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%a, %b: tensor<1x32xi8>, tensor<32x32xi8>) outs(%initial: tensor<1x32xi32>) {
  ^bb0(%lhs: i8, %rhs: i8, %acc: i32):
    %x = arith.extsi %lhs : i8 to i32
    %y = arith.extsi %rhs : i8 to i32
    %p = arith.muli %x, %y : i32
    %s = arith.addi %acc, %p : i32
    linalg.yield %s : i32
  } -> tensor<1x32xi32>
  return %product : tensor<1x32xi32>
}

// CHECK-LABEL: func.func @linear
// CHECK: %[[Q:.*]] = "frontend.fp32_to_int8"
// CHECK: %[[D:.*]] = "frontend.int8_to_fp32"(%[[Q]],
// CHECK-NOT: linalg.generic
// CHECK: %[[R:.*]] = custom.op "linear_i8"(%arg0, %arg1, %{{.*}}, %arg2, %arg3, %arg4)
// CHECK: return %[[R]], %[[D]]
func.func @linear(%a: tensor<1x32xf32>, %w: tensor<32x32xi8>, %multiplier: tensor<1x1xf32>, %input_step: tensor<1x1xf32>, %weight_step: tensor<1x32xf32>) -> (tensor<1x32xf32>, tensor<1x32xf32>) {
  %zero = arith.constant dense<0.0> : tensor<1x32xf32>
  %q = "frontend.fp32_to_int8"(%a, %multiplier) : (tensor<1x32xf32>, tensor<1x1xf32>) -> tensor<1x32xi8>
  %x = "frontend.int8_to_fp32"(%q, %input_step) : (tensor<1x32xi8>, tensor<1x1xf32>) -> tensor<1x32xf32>
  %y = "frontend.int8_to_fp32"(%w, %weight_step) : (tensor<32x32xi8>, tensor<1x32xf32>) -> tensor<32x32xf32>
  %product = linalg.generic {
    indexing_maps = [affine_map<(m,n,k)->(m,k)>, affine_map<(m,n,k)->(k,n)>, affine_map<(m,n,k)->(m,n)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%x, %y: tensor<1x32xf32>, tensor<32x32xf32>) outs(%zero: tensor<1x32xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
    %p = arith.mulf %lhs, %rhs : f32
    %s = arith.addf %acc, %p : f32
    linalg.yield %s : f32
  } -> tensor<1x32xf32>
  return %product, %x : tensor<1x32xf32>, tensor<1x32xf32>
}

// CHECK-LABEL: func.func @different_body
// CHECK-NOT: custom.op
// CHECK: linalg.generic
// CHECK: arith.subi
// CHECK: return
func.func @different_body(%a: tensor<1x32xi8>, %b: tensor<32x32xi8>, %initial: tensor<1x32xi32>) -> tensor<1x32xi32> {
  %product = linalg.generic {
    indexing_maps = [affine_map<(m,n,k)->(m,k)>, affine_map<(m,n,k)->(k,n)>, affine_map<(m,n,k)->(m,n)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%a, %b: tensor<1x32xi8>, tensor<32x32xi8>) outs(%initial: tensor<1x32xi32>) {
  ^bb0(%lhs: i8, %rhs: i8, %acc: i32):
    %x = arith.extsi %lhs : i8 to i32
    %y = arith.extsi %rhs : i8 to i32
    %p = arith.muli %x, %y : i32
    %s = arith.subi %acc, %p : i32
    linalg.yield %s : i32
  } -> tensor<1x32xi32>
  return %product : tensor<1x32xi32>
}

// CHECK-LABEL: func.func @shared_producer
// CHECK: %[[CODES:.*]] = "frontend.fp32_to_int8"
// CHECK-NOT: "frontend.int8_to_fp32"
// CHECK: custom.op "linear_i8"(%arg0, %[[CODES]], %{{.*}}, %arg1, %arg2, %arg2)
// CHECK: return
func.func @shared_producer(%a: tensor<1x1xf32>, %multiplier: tensor<1x1xf32>, %step: tensor<1x1xf32>) -> tensor<1x1xf32> {
  %zero = arith.constant dense<0.0> : tensor<1x1xf32>
  %q = "frontend.fp32_to_int8"(%a, %multiplier) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xi8>
  %x = "frontend.int8_to_fp32"(%q, %step) : (tensor<1x1xi8>, tensor<1x1xf32>) -> tensor<1x1xf32>
  %product = linalg.matmul ins(%x, %x: tensor<1x1xf32>, tensor<1x1xf32>) outs(%zero: tensor<1x1xf32>) -> tensor<1x1xf32>
  return %product : tensor<1x1xf32>
}
