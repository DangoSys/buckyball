// RUN: buddy-opt %s -outline-rvv-kernels -canonicalize | FileCheck %s
// CHECK-LABEL: func.func @weighted_residual
// CHECK: call @rvv_norm(
// CHECK-NOT: math.powf
// CHECK-NOT: linalg.reduce
// CHECK: linalg.generic
// CHECK: arith.addf
// CHECK-NEXT: {{.*}} = arith.mulf
// CHECK-NEXT: linalg.yield
// CHECK-LABEL: func.func @unweighted
// CHECK: call @rvv_norm_no_weight(
// CHECK-NOT: math.powf
// CHECK-NOT: linalg.reduce
// CHECK: return
#identity = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0, 0)>
#weight = affine_map<(d0, d1) -> (0, d1)>
module {
  func.func @weighted_residual(%x: tensor<2x4xf32>, %w: tensor<4xf32>, %residual: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %zero = arith.constant 0.0 : f32
    %mean = arith.constant 0.25 : f32
    %eps = arith.constant 0.000001 : f32
    %power = arith.constant -0.5 : f32
    %two = arith.constant 2 : i32
    %scale = arith.constant 0.5 : f32
    %empty = tensor.empty() : tensor<2x4xf32>
    %square = linalg.generic {indexing_maps=[#identity,#identity],iterator_types=["parallel","parallel"]} ins(%x:tensor<2x4xf32>) outs(%empty:tensor<2x4xf32>) {
    ^bb0(%v:f32,%o:f32):
      %s = math.fpowi %v, %two : f32, i32
      linalg.yield %s : f32
    } -> tensor<2x4xf32>
    %rows = tensor.empty() : tensor<2xf32>
    %init = linalg.fill ins(%zero:f32) outs(%rows:tensor<2xf32>) -> tensor<2xf32>
    %sum = linalg.reduce ins(%square:tensor<2x4xf32>) outs(%init:tensor<2xf32>) dimensions=[1] (%v:f32,%acc:f32) {
      %s = arith.addf %v, %acc : f32
      linalg.yield %s : f32
    }
    %expanded = tensor.expand_shape %sum [[0,1]] output_shape [2,1] : tensor<2xf32> into tensor<2x1xf32>
    %inverseEmpty = tensor.empty() : tensor<2x1xf32>
    %inverse = linalg.generic {indexing_maps=[#identity,#identity],iterator_types=["parallel","parallel"]} ins(%expanded:tensor<2x1xf32>) outs(%inverseEmpty:tensor<2x1xf32>) {
    ^bb0(%v:f32,%o:f32):
      %a = arith.mulf %mean, %v : f32
      %b = arith.addf %a, %eps : f32
      %c = math.powf %b, %power : f32
      linalg.yield %c : f32
    } -> tensor<2x1xf32>
    %weights = tensor.expand_shape %w [[0,1]] output_shape [1,4] : tensor<4xf32> into tensor<1x4xf32>
    %result = linalg.generic {indexing_maps=[#identity,#row,#weight,#identity,#identity],iterator_types=["parallel","parallel"]} ins(%x,%inverse,%weights,%residual:tensor<2x4xf32>,tensor<2x1xf32>,tensor<1x4xf32>,tensor<2x4xf32>) outs(%empty:tensor<2x4xf32>) {
    ^bb0(%v:f32,%inv:f32,%gamma:f32,%res:f32,%o:f32):
      %a = arith.mulf %v, %inv : f32
      %b = arith.mulf %a, %gamma : f32
      %c = arith.addf %b, %res : f32
      %d = arith.mulf %c, %scale : f32
      linalg.yield %d : f32
    } -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
  func.func @unweighted(%x: tensor<2x4xf32>, %w: tensor<4xf32>, %residual: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %zero = arith.constant 0.0 : f32
    %mean = arith.constant 0.25 : f32
    %eps = arith.constant 0.000001 : f32
    %power = arith.constant -0.5 : f32
    %two = arith.constant 2 : i32
    %scale = arith.constant 0.5 : f32
    %empty = tensor.empty() : tensor<2x4xf32>
    %square = linalg.generic {indexing_maps=[#identity,#identity],iterator_types=["parallel","parallel"]} ins(%x:tensor<2x4xf32>) outs(%empty:tensor<2x4xf32>) {
    ^bb0(%v:f32,%o:f32):
      %s = math.fpowi %v, %two : f32, i32
      linalg.yield %s : f32
    } -> tensor<2x4xf32>
    %rows = tensor.empty() : tensor<2xf32>
    %init = linalg.fill ins(%zero:f32) outs(%rows:tensor<2xf32>) -> tensor<2xf32>
    %sum = linalg.reduce ins(%square:tensor<2x4xf32>) outs(%init:tensor<2xf32>) dimensions=[1] (%v:f32,%acc:f32) {
      %s = arith.addf %v, %acc : f32
      linalg.yield %s : f32
    }
    %expanded = tensor.expand_shape %sum [[0,1]] output_shape [2,1] : tensor<2xf32> into tensor<2x1xf32>
    %inverseEmpty = tensor.empty() : tensor<2x1xf32>
    %inverse = linalg.generic {indexing_maps=[#identity,#identity],iterator_types=["parallel","parallel"]} ins(%expanded:tensor<2x1xf32>) outs(%inverseEmpty:tensor<2x1xf32>) {
    ^bb0(%v:f32,%o:f32):
      %a = arith.mulf %mean, %v : f32
      %b = arith.addf %a, %eps : f32
      %c = math.powf %b, %power : f32
      linalg.yield %c : f32
    } -> tensor<2x1xf32>
    %weights = tensor.expand_shape %w [[0,1]] output_shape [1,4] : tensor<4xf32> into tensor<1x4xf32>
    %result = linalg.generic {indexing_maps=[#identity,#row,#weight,#identity,#identity],iterator_types=["parallel","parallel"]} ins(%x,%inverse,%weights,%residual:tensor<2x4xf32>,tensor<2x1xf32>,tensor<1x4xf32>,tensor<2x4xf32>) outs(%empty:tensor<2x4xf32>) {
    ^bb0(%v:f32,%inv:f32,%gamma:f32,%res:f32,%o:f32):
      %a = arith.mulf %v, %inv : f32
      linalg.yield %a : f32
    } -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
}
