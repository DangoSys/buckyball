#id = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @compute(%a: memref<1x2x32xf32>, %out: memref<1x2x16xf32>) attributes {llvm.emit_c_interface} {
  %input = bufferization.to_tensor %a restrict : memref<1x2x32xf32> to tensor<1x2x32xf32>
  %left = tensor.extract_slice %input[0, 0, 0] [1, 2, 16] [1, 1, 1] : tensor<1x2x32xf32> to tensor<1x2x16xf32>
  %right = tensor.extract_slice %input[0, 0, 16] [1, 2, 16] [1, 1, 1] : tensor<1x2x32xf32> to tensor<1x2x16xf32>
  %empty = tensor.empty() : tensor<1x2x16xf32>
  %result = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%left, %right : tensor<1x2x16xf32>, tensor<1x2x16xf32>) outs(%empty : tensor<1x2x16xf32>) {
  ^bb0(%x: f32, %y: f32, %unused: f32):
    %value = arith.mulf %x, %y : f32
    linalg.yield %value : f32
  } -> tensor<1x2x16xf32>
  bufferization.materialize_in_destination %result in writable %out : (tensor<1x2x16xf32>, memref<1x2x16xf32>) -> ()
  return
}
