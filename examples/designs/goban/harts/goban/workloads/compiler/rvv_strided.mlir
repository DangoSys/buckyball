#id = affine_map<(d0, d1) -> (d0, d1)>
func.func @compute(%a: memref<4x32xf32>, %out: memref<4x32xf32>) attributes {llvm.emit_c_interface} {
  %at = bufferization.to_tensor %a restrict : memref<4x32xf32> to tensor<4x32xf32>
  %slice = tensor.extract_slice %at[0, 16] [4, 16] [1, 1] : tensor<4x32xf32> to tensor<4x16xf32>
  %empty = tensor.empty() : tensor<4x16xf32>
  %result = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%slice : tensor<4x16xf32>) outs(%empty : tensor<4x16xf32>) {
  ^bb0(%input: f32, %unused: f32):
    %one = arith.constant 1.0 : f32
    %value = arith.addf %input, %one : f32
    linalg.yield %value : f32
  } -> tensor<4x16xf32>
  %target = memref.subview %out[0, 16] [4, 16] [1, 1] : memref<4x32xf32> to memref<4x16xf32, strided<[32, 1], offset: 16>>
  bufferization.materialize_in_destination %result in writable %target : (tensor<4x16xf32>, memref<4x16xf32, strided<[32, 1], offset: 16>>) -> ()
  return
}
