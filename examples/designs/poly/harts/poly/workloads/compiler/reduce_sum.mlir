#id = affine_map<(d0,d1)->(d0,d1)>
func.func @reduce_sum(%a: tensor<?x4xf32>, %b: tensor<?x4xf32>, %c: tensor<?x4xf32>, %out: tensor<?x4xf32>) -> tensor<?x4xf32> attributes {llvm.emit_c_interface} {
  %zero = arith.constant 0 : index
  %n = tensor.dim %a, %zero : tensor<?x4xf32>
  %empty = tensor.empty(%n) : tensor<?x4xf32>
  %ab = linalg.generic {indexing_maps=[#id,#id,#id], iterator_types=["parallel","parallel"]} ins(%a,%b: tensor<?x4xf32>,tensor<?x4xf32>) outs(%empty:tensor<?x4xf32>) {
  ^bb0(%x:f32,%y:f32,%z:f32):
    %sum = arith.addf %x,%y : f32
    linalg.yield %sum : f32
  } -> tensor<?x4xf32>
  %abc = linalg.generic {indexing_maps=[#id,#id,#id], iterator_types=["parallel","parallel"]} ins(%ab,%c: tensor<?x4xf32>,tensor<?x4xf32>) outs(%empty:tensor<?x4xf32>) {
  ^bb0(%x:f32,%y:f32,%z:f32):
    %sum = arith.addf %x,%y : f32
    linalg.yield %sum : f32
  } -> tensor<?x4xf32>
  %written = bufferization.materialize_in_destination %abc in %out : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
  return %written : tensor<?x4xf32>
}
