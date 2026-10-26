// RUN: buddy-opt %s -outline-rvv-kernels -canonicalize | FileCheck %s
// CHECK-LABEL: func.func @tanh
// CHECK: call @rvv_tanh
// CHECK-NOT: math.tanh
// CHECK: return
// CHECK-LABEL: func.func @mixed_expression
// CHECK-NOT: call @rvv_tanh
// CHECK: math.tanh
// CHECK: arith.addf
// CHECK: return
// CHECK-LABEL: func.func @sin
// CHECK: call @rvv_sin
// CHECK-NOT: math.sin
// CHECK: return
// CHECK-LABEL: func.func @cos
// CHECK: call @rvv_cos
// CHECK-NOT: math.cos
// CHECK: return
#identity = affine_map<(d0) -> (d0)>
module {
  func.func @tanh(%x: tensor<17xf32>) -> tensor<17xf32> {
    %empty = tensor.empty() : tensor<17xf32>
    %result = linalg.generic {indexing_maps = [#identity, #identity], iterator_types = ["parallel"]} ins(%x : tensor<17xf32>) outs(%empty : tensor<17xf32>) {
    ^bb0(%xval: f32, %out: f32):
      %t = math.tanh %xval : f32
      linalg.yield %t : f32
    } -> tensor<17xf32>
    return %result : tensor<17xf32>
  }
  func.func @mixed_expression(%x: tensor<17xf32>) -> tensor<17xf32> {
    %empty = tensor.empty() : tensor<17xf32>
    %one = arith.constant 1.0 : f32
    %result = linalg.generic {indexing_maps = [#identity, #identity], iterator_types = ["parallel"]} ins(%x : tensor<17xf32>) outs(%empty : tensor<17xf32>) {
    ^bb0(%xval: f32, %out: f32):
      %t = math.tanh %xval : f32
      %a = arith.addf %t, %one : f32
      linalg.yield %a : f32
    } -> tensor<17xf32>
    return %result : tensor<17xf32>
  }
  func.func @sin(%x: tensor<17xf32>) -> tensor<17xf32> {
    %empty = tensor.empty() : tensor<17xf32>
    %result = linalg.generic {indexing_maps = [#identity, #identity], iterator_types = ["parallel"]} ins(%x : tensor<17xf32>) outs(%empty : tensor<17xf32>) {
    ^bb0(%xval: f32, %out: f32):
      %t = math.sin %xval : f32
      linalg.yield %t : f32
    } -> tensor<17xf32>
    return %result : tensor<17xf32>
  }

  func.func @cos(%x: tensor<17xf32>) -> tensor<17xf32> {
    %empty = tensor.empty() : tensor<17xf32>
    %result = linalg.generic {indexing_maps = [#identity, #identity], iterator_types = ["parallel"]} ins(%x : tensor<17xf32>) outs(%empty : tensor<17xf32>) {
    ^bb0(%xval: f32, %out: f32):
      %t = math.cos %xval : f32
      linalg.yield %t : f32
    } -> tensor<17xf32>
    return %result : tensor<17xf32>
  }

}
