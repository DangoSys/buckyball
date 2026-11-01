// CHECK-LABEL: func.func @outputs
// CHECK: %[[FIRST:.*]] = bufferization.materialize_in_destination
// CHECK: %[[SECOND:.*]] = bufferization.materialize_in_destination %[[FIRST]]
// CHECK: bufferization.materialize_in_destination %[[SECOND]]
#map = affine_map<(d0) -> (d0)>
func.func @outputs(%a: tensor<4xf32>, %b: tensor<4xf32>, %x: tensor<4xf32>, %y: tensor<4xf32>, %z: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
  %sum = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
      ins(%a, %b : tensor<4xf32>, tensor<4xf32>) outs(%x : tensor<4xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %old: f32):
      %value = arith.addf %lhs, %rhs : f32
      linalg.yield %value : f32
  } -> tensor<4xf32>
  %first = bufferization.materialize_in_destination %sum in %x : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %second = bufferization.materialize_in_destination %sum in %y : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %third = bufferization.materialize_in_destination %sum in %z : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %first, %second, %third : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
}
