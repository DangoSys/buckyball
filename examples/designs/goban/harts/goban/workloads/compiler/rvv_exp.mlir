#id = affine_map<(d0, d1) -> (d0, d1)>
func.func @compute(%a: memref<2x36xf32>, %out: memref<2x36xf32>) attributes {llvm.emit_c_interface} {
  %input = bufferization.to_tensor %a restrict : memref<2x36xf32> to tensor<2x36xf32>
  %empty = tensor.empty() : tensor<2x36xf32>
  %result = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%input : tensor<2x36xf32>) outs(%empty : tensor<2x36xf32>) {
  ^bb0(%value: f32, %unused: f32):
    %exp = math.exp %value : f32
    linalg.yield %exp : f32
  } -> tensor<2x36xf32>
  bufferization.materialize_in_destination %result in writable %out : (tensor<2x36xf32>, memref<2x36xf32>) -> ()
  return
}
