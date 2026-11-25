#id = affine_map<(d0, d1) -> (d0, d1)>
#bias = affine_map<(d0, d1) -> (0, d1)>
#scale = affine_map<(d0, d1) -> (d0, 0)>
func.func @compute(%a: memref<4x516xf32>, %b: memref<1x516xf32>, %s: memref<4x1xf32>, %out: memref<4x516xf32>) attributes {llvm.emit_c_interface} {
  %at = bufferization.to_tensor %a restrict : memref<4x516xf32> to tensor<4x516xf32>
  %bt = bufferization.to_tensor %b restrict : memref<1x516xf32> to tensor<1x516xf32>
  %st = bufferization.to_tensor %s restrict : memref<4x1xf32> to tensor<4x1xf32>
  %e0 = tensor.empty() : tensor<4x516xf32>
  %x = linalg.generic {indexing_maps = [#id, #bias, #scale, #id], iterator_types = ["parallel", "parallel"]}
      ins(%at, %bt, %st : tensor<4x516xf32>, tensor<1x516xf32>, tensor<4x1xf32>) outs(%e0 : tensor<4x516xf32>) {
  ^bb0(%av: f32, %bv: f32, %sv: f32, %unused: f32):
    %sum = arith.addf %av, %bv : f32
    %scaled = arith.mulf %sum, %sv : f32
    %added = arith.addf %scaled, %av : f32
    %again = arith.mulf %added, %sv : f32
    linalg.yield %again : f32
  } -> tensor<4x516xf32>
  %e1 = tensor.empty() : tensor<4x516xf32>
  %y = linalg.generic {indexing_maps = [#id, #bias, #id], iterator_types = ["parallel", "parallel"]}
      ins(%x, %bt : tensor<4x516xf32>, tensor<1x516xf32>) outs(%e1 : tensor<4x516xf32>) {
  ^bb0(%xv: f32, %bv: f32, %unused: f32):
    %difference = arith.subf %xv, %bv : f32
    linalg.yield %difference : f32
  } -> tensor<4x516xf32>
  bufferization.materialize_in_destination %y in writable %out : (tensor<4x516xf32>, memref<4x516xf32>) -> ()
  return
}
